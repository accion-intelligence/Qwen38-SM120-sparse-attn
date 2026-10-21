"""Capture exact block attention mass of the full-attention layers with vLLM.

Runs the real model in-process, hooks FlashAttentionImpl.forward to collect the
post-norm/post-RoPE q and k of every full-attention layer across prefill
chunks, then computes per-head 64x64 block mass (exact causal softmax) and
three cheap block estimators. Only block-level tensors are saved.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
import torch  # noqa: E402

MODEL = os.environ.get('MODEL_PATH', 'Qwen/Qwen3.8-27B')   # set MODEL_PATH to a local snapshot
STATE = {'active': False, 'store': {}}


def install_hook():
    import vllm.v1.attention.backends.flash_attn as fa_mod
    orig = fa_mod.FlashAttentionImpl.forward

    def hooked(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        if STATE['active'] and attn_metadata is not None:
            n = attn_metadata.num_actual_tokens
            name = getattr(layer, 'layer_name', None) or str(id(layer))
            STATE['store'].setdefault(name, []).append(
                (query[:n].detach().to('cpu'), key[:n].detach().to('cpu')))
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)

    fa_mod.FlashAttentionImpl.forward = hooked


@torch.no_grad()
def block_mass(q_rows, row_pos, k, block, rows_per_block, chunk=1024):
    """Exact causal softmax mass per (head, query block, key block).

    q_rows: [R, H, D] rows at absolute positions row_pos [R]; k: [L, Hk, D].
    Returns [H, nB, nB] where each query block row averages its sampled rows.
    """
    R, H, D = q_rows.shape
    L, Hk, _ = k.shape
    g = H // Hk
    nB = L // block
    scale = D ** -0.5
    mass = torch.zeros(H, nB, nB, device=k.device, dtype=torch.float32)
    for h in range(H):
        kh = k[:, h // g]
        qh = q_rows[:, h]
        for r0 in range(0, R, chunk):
            r1 = min(r0 + chunk, R)
            pos = row_pos[r0:r1]
            kmax = int(pos.max().item()) + 1
            kmax = ((kmax + block - 1) // block) * block
            s = (qh[r0:r1] @ kh[:kmax].T).float() * scale
            s.masked_fill_(torch.arange(kmax, device=s.device)[None, :] > pos[:, None], float('-inf'))
            p = torch.softmax(s, dim=-1)
            pb = p.view(r1 - r0, kmax // block, block).sum(-1)
            mass[h, :, :kmax // block].index_add_(0, pos // block, pb)
    return mass / rows_per_block


@torch.no_grad()
def estimators(q, k, block, stride):
    """Block-level scores from pooled q/k: mean-pool logits and XAttention antidiagonal sums."""
    L, H, D = q.shape
    Hk = k.shape[1]
    g = H // Hk
    nB = L // block
    scale = D ** -0.5
    qm = q.view(nB, block, H, D).float().mean(1)
    km = k.view(nB, block, Hk, D).float().mean(1)
    qs = q.view(nB, block // stride, stride, H, D).float().sum(1)
    ks = k.view(nB, block // stride, stride, Hk, D).float().sum(1).flip(1)
    causal = torch.ones(nB, nB, device=q.device, dtype=torch.bool).tril()
    est_mean = torch.empty(H, nB, nB, device=q.device, dtype=torch.float16)
    est_anti = torch.empty_like(est_mean)
    for h in range(H):
        a = (qm[:, h] @ km[:, h // g].T) * scale
        b = torch.einsum('isd,jsd->ij', qs[:, :, h], ks[:, :, h // g]) * scale
        est_mean[h] = a.masked_fill(~causal, float('-inf')).half()
        est_anti[h] = b.masked_fill(~causal, float('-inf')).half()
    return est_mean, est_anti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--max-model-len', type=int, default=66560)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    install_hook()
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, dtype='bfloat16', max_model_len=args.max_model_len, enforce_eager=True,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_batched_tokens=8192,
              max_num_seqs=1, enable_chunked_prefill=True, enable_prefix_caching=False,
              seed=20261015)
    print(f'model loaded in {time.time() - t0:.0f}s', flush=True)
    prompts = json.loads(args.prompts.read_text())
    status = []
    for row in prompts:
        kind, length, ids = row['kind'], row['length'], row['prompt_token_ids']
        out_dir = args.output / f'{kind}-{length}'
        if (out_dir / 'done.json').exists():
            print(f'skip {out_dir}', flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        STATE['store'].clear()
        STATE['active'] = True
        t1 = time.time()
        res = llm.generate([{'prompt_token_ids': ids}], SamplingParams(max_tokens=1, temperature=0.0),
                           use_tqdm=False)
        STATE['active'] = False
        prefill_s = time.time() - t1
        first_token = res[0].outputs[0].token_ids[0] if res[0].outputs[0].token_ids else None
        layers = sorted(STATE['store'])
        print(f'{kind}-{length}: prefill {prefill_s:.1f}s, {len(layers)} attention layers captured, '
              f'first token {first_token}', flush=True)
        t2 = time.time()
        meta = {k: v for k, v in row.items() if k != 'prompt_token_ids'}
        meta.update({'layers': layers, 'block': args.block, 'stride': args.stride})
        for name in layers:
            q = torch.cat([x[0] for x in STATE['store'][name]]).cuda()
            k = torch.cat([x[1] for x in STATE['store'][name]]).cuda()
            assert q.shape[0] == length and k.shape[0] == length, (q.shape, k.shape, length)
            L = length
            full_pos = torch.arange(L, device='cuda')
            mass = block_mass(q, full_pos, k, args.block, args.block)
            sub = torch.arange(args.stride // 2, L, args.stride, device='cuda')
            est_rows = block_mass(q[sub], sub, k, args.block, args.block // args.stride).half()
            est_mean, est_anti = estimators(q, k, args.block, args.stride)
            torch.save({'mass': mass.cpu(), 'est_rows': est_rows.cpu(), 'est_mean': est_mean.cpu(),
                        'est_anti': est_anti.cpu(), 'layer': name, 'H': q.shape[1], 'Hk': k.shape[1],
                        'D': q.shape[2], 'L': L}, out_dir / f'{name.replace("/", "_")}.pt')
            del q, k, mass, est_rows, est_mean, est_anti
            torch.cuda.empty_cache()
        STATE['store'].clear()
        meta['analysis_s'] = time.time() - t2
        meta['prefill_s'] = prefill_s
        meta['first_token'] = first_token
        (out_dir / 'done.json').write_text(json.dumps(meta, indent=1))
        status.append(meta)
        print(f'  block mass saved in {meta["analysis_s"]:.0f}s', flush=True)
    (args.output / 'status.json').write_text(json.dumps(status, indent=1))
    print('all done', flush=True)


if __name__ == '__main__':
    main()
