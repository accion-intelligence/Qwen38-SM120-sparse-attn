"""Correctness and timing checks for the SM120 FlashAttention kernel (dense + block-sparse)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm120_attn.kernels.kai_fa_bf16 import attention_sm120  # noqa: E402

BLOCK = 64


def reference(q, k, v, q_start, masks=None):
    """fp32 causal attention; q [n,H,D] at positions q_start.., k/v [kv,Hk,D]; masks [H, MB, NB] bool (optional)."""
    n, H, D = q.shape
    kv, Hk, _ = k.shape
    g = H // Hk
    pos_q = q_start + torch.arange(n, device=q.device)
    pos_k = torch.arange(kv, device=q.device)
    allowed = pos_k[None, :] <= pos_q[:, None]
    outs = []
    for h in range(H):
        s = (q[:, h].float() @ k[:, h // g].float().T) * D ** -0.5
        m = allowed.clone()
        if masks is not None:
            m &= masks[h].repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :kv]
        s.masked_fill_(~m, float('-inf'))
        outs.append(torch.softmax(s, -1) @ v[:, h // g].float())
    return torch.stack(outs, 1)


def random_masks(n, kv, q_start, H, density, device):
    """Per-head block masks keeping the diagonal and block 0; q_start must be a multiple of BLOCK."""
    MB = -(-n // BLOCK)
    NB = -(-kv // BLOCK)
    gi = q_start // BLOCK + torch.arange(MB, device=device)
    j = torch.arange(NB, device=device)
    future = j[None, :] > gi[:, None]
    masks = torch.rand(H, MB, NB, device=device) < density
    masks[:, torch.arange(MB, device=device), gi] = True
    masks[:, :, 0] = True
    masks &= ~future[None]
    return masks


def to_lists_m128(masks):
    """Block lists for 128-row query blocks from 64-row masks: union of each adjacent pair of rows."""
    H, MB, NB = masks.shape
    if MB % 2:
        masks = torch.nn.functional.pad(masks, (0, 0, 0, 1))
    m2 = masks.view(H, -1, 2, NB).any(2)
    return to_lists(m2)


def to_lists(masks):
    """Padded block lists without host syncs: selected blocks first in ascending order."""
    H, MB, NB = masks.shape
    counts = masks.sum(2).to(torch.int32).flatten()
    order = torch.argsort((~masks).to(torch.int8), dim=-1, stable=True)
    return counts, order.to(torch.int32).reshape(H * MB, NB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m-block', type=int, default=64)
    ap.add_argument('--n-block', type=int, default=64)
    ap.add_argument('--threads', type=int, default=128)
    ap.add_argument('--big', action='store_true', help='also time 8192x65536')
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = 'cuda'
    H, Hk, D = 24, 4, 256
    kw = dict(m_block=args.m_block, n_block=args.n_block, num_threads=args.threads)
    print('--- dense causal (q_start = kv - n)')
    for n, kv in ((256, 1024), (1000, 4024), (64, 4096), (2048, 2048)):
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        o = attention_sm120(q, k, v, **kw)
        ref = reference(q, k, v, kv - n)
        err = (o.float() - ref).abs().max().item() / ref.abs().max().item()
        print(f'  n={n:5d} kv={kv:5d}: max rel err {err:.2e}')
    print('--- block-sparse (per-head BSR), q_start aligned to 64')
    for n, kv, density in ((512, 2048, 0.3), (1024, 8192, 0.2), (2048, 8192, 0.5)):
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        q_start = kv - n
        masks = random_masks(n, kv, q_start, H, density, dev)
        counts, lists = to_lists(masks)
        o = attention_sm120(q, k, v, q_start, counts=counts, lists=lists, **kw)
        ref = reference(q, k, v, q_start, masks)
        err = (o.float() - ref).abs().max().item() / ref.abs().max().item()
        kept = counts.sum().item() / (H * sum(q_start // BLOCK + i + 1 for i in range(-(-n // BLOCK))))
        print(f'  n={n:5d} kv={kv:5d} kept={kept:.3f}: max rel err vs masked reference {err:.2e}')
    if args.big:
        n, kv = 8192, 65536
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
        for label, extra in (('dense', {}),):
            for _ in range(3): attention_sm120(q, k, v, **kw, **extra)
            torch.cuda.synchronize(); t = time.time(); iters = 5
            for _ in range(iters): attention_sm120(q, k, v, **kw, **extra)
            torch.cuda.synchronize(); ms = (time.time() - t) / iters * 1000
            print(f'  {label} 8192x65536 m{args.m_block} n{args.n_block} t{args.threads}: {ms:.2f} ms  {fl/ms/1e9:.1f} TFLOPS (FA2 non-paged 338, paged 305)')
        for density in (0.4, 0.2):
            masks = random_masks(n, kv, kv - n, H, density, dev)
            counts, lists = to_lists(masks)
            kept = counts.sum().item() / (H * sum((kv - n) // BLOCK + i + 1 for i in range(n // BLOCK)))
            for _ in range(2): attention_sm120(q, k, v, kv - n, counts=counts, lists=lists, **kw)
            torch.cuda.synchronize(); t = time.time(); iters = 5
            for _ in range(iters): attention_sm120(q, k, v, kv - n, counts=counts, lists=lists, **kw)
            torch.cuda.synchronize(); ms = (time.time() - t) / iters * 1000
            print(f'  sparse kept={kept:.3f}: {ms:.2f} ms  (effective {fl*kept/ms/1e9:.1f} TFLOPS on kept blocks)')


if __name__ == '__main__':
    main()
