"""Numerics of FP8 (e4m3) attention operands on the real model: output error vs BF16 attention.

Hooks FlashAttentionImpl.forward (like capture_block_mass) on a long prompt, and for a few
full-attention layers compares a BF16 reference attention of a 1024-query slice against variants
with K/V (and Q, P) quantized to e4m3 at different scale granularities.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
import torch  # noqa: E402

MODEL = os.environ.get('MODEL_PATH', 'Qwen/Qwen3.8-27B')   # set MODEL_PATH to a local snapshot
STATE = {'active': False, 'store': {}}
E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0


def install_hook():
    import vllm.v1.attention.backends.flash_attn as fa_mod
    orig = fa_mod.FlashAttentionImpl.forward

    def hooked(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        if STATE['active'] and attn_metadata is not None:
            n = attn_metadata.num_actual_tokens
            name = getattr(layer, 'layer_name', None) or str(id(layer))
            STATE['store'].setdefault(name, []).append((query[:n].detach().to('cpu'), key[:n].detach().to('cpu'), value[:n].detach().to('cpu')))
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
    fa_mod.FlashAttentionImpl.forward = hooked


def quant(x, axis_scale):
    """Quantize to e4m3 with amax scaling over all dims except those in axis_scale (kept per-element scales)."""
    red = [d for d in range(x.dim()) if d not in axis_scale]
    amax = x.abs().amax(dim=red, keepdim=True).clamp_min(1e-6)
    s = amax / E4M3_MAX
    return (x / s).to(E4M3).to(torch.float32) * s


def quant_mx(x, block_dim, block=32):
    """MX-style quantization: e4m3 values with a power-of-two (UE8M0) scale per contiguous block of `block`
    elements along `block_dim` (per row of every other dim).  Emulates mma.sync.kind::mxf8f6f4 block_scale."""
    xt = x.movedim(block_dim, -1)
    shp = xt.shape
    xb = xt.reshape(-1, shp[-1] // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / E4M3_MAX))                 # 2^e * 448 >= amax
    s = torch.exp2(e)
    y = (xb / s).to(E4M3).to(torch.float32) * s
    return y.reshape(shp).movedim(-1, block_dim)


@torch.no_grad()
def attention_ref(q, k, v, q_start, p_quant=None):
    """q [n,H,D] fp32, k/v [T,Hk,D] fp32; causal; returns [n,H,D] fp32. p_quant: None|'row'|'block64'."""
    n, H, D = q.shape
    T, Hk, _ = k.shape
    g = H // Hk
    pos_q = q_start + torch.arange(n, device=q.device)
    pos_k = torch.arange(T, device=q.device)
    mask = pos_k[None, :] <= pos_q[:, None]
    outs = []
    for h in range(H):
        s = (q[:, h] @ k[:, h // g].T) * D ** -0.5
        s.masked_fill_(~mask, float('-inf'))
        p = torch.softmax(s, -1)
        if p_quant == 'row':
            p = quant(p, axis_scale=(0,))                       # per query row scale
        elif p_quant == 'block64':
            pb = p.view(n, T // 64, 64)
            p = quant(pb, axis_scale=(0, 1)).view(n, T)         # per (row, 64-key block)
        elif p_quant == 'mx32':
            p = quant_mx(p, 1, 32)                                # UE8M0 per (row, 32-key block)
        outs.append(p @ v[:, h // g])
    return torch.stack(outs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--length', type=int, default=65536)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--layers', default='11,35,63')
    args = ap.parse_args()
    install_hook()
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype='bfloat16', max_model_len=66560, enforce_eager=True, gpu_memory_utilization=0.7,
              max_num_batched_tokens=8192, max_num_seqs=1, enable_chunked_prefill=True, enable_prefix_caching=False, seed=20261015)
    row = [r for r in json.loads(args.prompts.read_text()) if r['length'] == args.length][0]
    STATE['active'] = True
    llm.generate([{'prompt_token_ids': row['prompt_token_ids']}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    STATE['active'] = False
    layers = [l for l in sorted(STATE['store']) if any(f'layers.{i}.' in l for i in args.layers.split(','))]
    results = {}
    for name in layers:
        q = torch.cat([x[0] for x in STATE['store'][name]]).cuda().float()
        k = torch.cat([x[1] for x in STATE['store'][name]]).cuda().float()
        v = torch.cat([x[2] for x in STATE['store'][name]]).cuda().float()
        L = q.shape[0]
        qs = L - 1024                                             # last 1024 queries attend to the whole context
        qq = q[qs:]
        ref = attention_ref(qq, k, v, qs)
        variants = {
            'kv_per_tensor': (quant(k, ()), quant(v, ())),
            'kv_per_head': (quant(k, (1,)), quant(v, (1,))),
            'kv_per_token': (quant(k, (0, 1)), quant(v, (0, 1))),
            'k_per_token_v_per_channel': (quant(k, (0, 1)), quant(v, (1, 2))),
            'k_per_token_v_block64_channel': (quant(k, (0, 1)), quant(v.view(L // 64, 64, *v.shape[1:]), (0, 2, 3)).view_as(v)),
            # hardware MXF8 emulation: K scaled per (token, 32-d block), V per (channel, 32-key block)
            'kv_mx32': (quant_mx(k, 2, 32), quant_mx(v, 0, 32)),
            'k_mx32_v_per_channel': (quant_mx(k, 2, 32), quant(v, (1, 2))),
        }
        stats = {'kv_tensor_rms': {'k': k.pow(2).mean().sqrt().item(), 'v': v.pow(2).mean().sqrt().item(), 'k_amax': k.abs().max().item(), 'v_amax': v.abs().max().item()}}
        for vname, (kq, vq) in variants.items():
            o = attention_ref(qq, kq, vq, qs)
            stats[vname] = {'rel_l2': ((o - ref).norm() / ref.norm()).item(),
                            'max_abs_over_rms': ((o - ref).abs().max() / ref.pow(2).mean().sqrt()).item()}
        # Q quantized per row too, plus P quantized per row / per block (full fp8 pipeline)
        kq, vq = variants['k_per_token_v_per_channel']
        qq8 = quant(qq, (0, 1))
        for pq in (None, 'row', 'block64'):
            o = attention_ref(qq8, kq, vq, qs, p_quant=pq)
            stats[f'q_per_token+kv_as_above+p_{pq}'] = {'rel_l2': ((o - ref).norm() / ref.norm()).item(),
                                                        'max_abs_over_rms': ((o - ref).abs().max() / ref.pow(2).mean().sqrt()).item()}
        # current production kernel: everything per tensor (amax*1.5/448 margin), P per row*448
        kq, vq = quant(k, ()), quant(v, ())
        o = attention_ref(quant(qq, ()), kq, vq, qs, p_quant='row')
        stats['prod_all_per_tensor+p_row'] = {'rel_l2': ((o - ref).norm() / ref.norm()).item(),
                                              'max_abs_over_rms': ((o - ref).abs().max() / ref.pow(2).mean().sqrt()).item()}
        # full MXF8 pipeline: Q/K per (token, 32-d block), V per (channel, 32-key block), P per (row, 32-key block)
        kq, vq = variants['kv_mx32']
        o = attention_ref(quant_mx(qq, 2, 32), kq, vq, qs, p_quant='mx32')
        stats['mxf8_all'] = {'rel_l2': ((o - ref).norm() / ref.norm()).item(),
                             'max_abs_over_rms': ((o - ref).abs().max() / ref.pow(2).mean().sqrt()).item()}
        # software alternative: Q/K per token, V per channel (epilogue), P per row
        kq, vq = variants['k_per_token_v_per_channel']
        o = attention_ref(qq8, kq, vq, qs, p_quant='row')
        stats['sw_qk_per_token_v_per_channel+p_row'] = {'rel_l2': ((o - ref).norm() / ref.norm()).item(),
                                                        'max_abs_over_rms': ((o - ref).abs().max() / ref.pow(2).mean().sqrt()).item()}
        results[name] = stats
        print(name, json.dumps({k2: (round(v2['rel_l2'], 5) if isinstance(v2, dict) and 'rel_l2' in v2 else v2) for k2, v2 in stats.items()}), flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
