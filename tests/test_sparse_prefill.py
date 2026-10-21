"""Standalone checks for the SM120 glue v2: cached block means and sync-free lists match the direct path."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sm120_attn.sparse_prefill as ksp  # noqa: E402
from test_fa_bf16 import reference  # noqa: E402


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    H, Hk, D = 24, 4, 256
    kv = 24576
    k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    # make attention structured so that selection is sparse
    u = torch.randn(Hk, 1, D, device=dev)
    hot = (torch.rand(kv // 64 + 1, device=dev) < 0.1).repeat_interleave(64)[:kv]
    k = (k.float() * 0.5 + hot[:, None, None].float() * u.transpose(0, 1) * 3.0).to(torch.bfloat16)
    sp = ksp.SparsePrefill(H, Hk, D, dev, tau=0.95, min_kv=0, min_q=0, kernel='sm120')
    # chunked prefill: 3 chunks of 8192; cached means built incrementally from each chunk's keys
    for c, (q_start, n) in enumerate(((0, 8192), (8192, 8192), (16384, 8192))):
        q = (torch.randn(n, H, D, device=dev) + u.repeat_interleave(H // Hk, 0).transpose(0, 1)).to(torch.bfloat16)
        km = ksp._cached_block_means('layer0', 7, k[q_start:q_start + n], q_start, q_start + n)
        nb = -(-(q_start + n) // 64)
        km_direct = k[:nb * 64].view(nb, 64, Hk, D).float().mean(1).transpose(0, 1)
        print(f'chunk {c}: cached means max|diff| = {(km - km_direct).abs().max().item():.2e}')
        o_cached = sp.forward_sm120(q, k[:q_start + n], v[:q_start + n], q_start, km=km)
        o_direct = sp.forward_sm120(q, k[:q_start + n], v[:q_start + n], q_start)
        masks = sp.block_masks(q, None, q_start, union=False, km=km)
        ref = reference(q, k[:q_start + n], v[:q_start + n], q_start, masks)
        err = (o_cached.float() - ref).abs().max().item() / ref.abs().max().item()
        print(f'         kept={sp.kept_fraction():.3f}  cached-vs-direct max|diff|={(o_cached.float()-o_direct.float()).abs().max().item():.2e}  vs masked ref {err:.2e}')
    # unaligned chunk boundary
    sp2 = ksp.SparsePrefill(H, Hk, D, dev, tau=0.95, min_kv=0, min_q=0, kernel='sm120')
    for q_start, n in ((0, 5000), (5000, 3192), (8192, 1000)):
        q = (torch.randn(n, H, D, device=dev) + u.repeat_interleave(H // Hk, 0).transpose(0, 1)).to(torch.bfloat16)
        km = ksp._cached_block_means('layer1', 9, k[q_start:q_start + n], q_start, q_start + n)
        o = sp2.forward_sm120(q, k[:q_start + n], v[:q_start + n], q_start, km=km)
        masks = sp2.block_masks(torch.cat([q.new_zeros(q_start % 64, H, D), q, q.new_zeros((-(q_start % 64 + n)) % 64, H, D)]), None, q_start - q_start % 64, valid=(q_start % 64, n), union=False, km=km)
        ref = reference(torch.cat([q.new_zeros(q_start % 64, H, D), q]), k[:q_start + n], v[:q_start + n], q_start - q_start % 64, masks)[q_start % 64:]
        print(f'unaligned q_start={q_start} n={n}: vs masked ref {(o.float()-ref).abs().max().item()/ref.abs().max().item():.2e}')
    # timing of the glue at 8192 x 65536 with cached means
    kv = 65536
    k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(8192, H, D, device=dev, dtype=torch.bfloat16)
    km = k.view(kv // 64, 64, Hk, D).float().mean(1).transpose(0, 1)
    sp3 = ksp.SparsePrefill(H, Hk, D, dev, tau=0.95, min_kv=0, min_q=0, kernel='sm120'); sp3.timing = True
    for _ in range(3): sp3.forward_sm120(q, k, v, kv - 8192, km=km)
    sp3.stats.update(select_ms=0.0, run_ms=0.0)
    for _ in range(5): sp3.forward_sm120(q, k, v, kv - 8192, km=km)
    print(f'8192x65536 random (dense-ish): select {sp3.stats["select_ms"]/5:.2f} ms, run {sp3.stats["run_ms"]/5:.2f} ms, kept {sp3.kept_fraction():.3f}')


if __name__ == '__main__':
    main()


def test_stale_first_page():
    """A request whose first page was recycled from an earlier request, and whose own first chunk was too
    short for the sparse path, must not inherit the earlier request's block means or FP8 scales."""
    import sm120_attn.sparse_prefill as ksp
    dev = 'cuda'
    H, Hk, D = 24, 4, 256
    torch.manual_seed(1)
    kA = torch.randn(16384, Hk, D, device=dev, dtype=torch.bfloat16)
    for q_start, n in ((0, 8192), (8192, 8192)):
        ksp._cached_block_means('L2', 3, kA[q_start:q_start + n], q_start, q_start + n)
    qA = torch.randn(8192, H, D, device=dev, dtype=torch.bfloat16)
    sA = ksp._cached_scales('L2', 3, qA, kA[:8192], kA[:8192] * 0.5, 0)
    # request B reuses first page 3; its first chunk (1000 rows) never reached the cache
    kB = torch.randn(8192, Hk, D, device=dev, dtype=torch.bfloat16) * 3.0
    km = ksp._cached_block_means('L2', 3, kB[1000:8192], 1000, 8192, key_all=kB[:8192])
    direct = kB.view(128, 64, Hk, D).float().mean(1).transpose(0, 1)
    d = (km - direct).abs().max().item()
    print(f'stale first page: rebuilt means max|diff| = {d:.2e} (A would give {(kA[:8192].view(128, 64, Hk, D).float().mean(1).transpose(0, 1) - direct).abs().max().item():.2e})')
    assert d < 1e-2
    qB = torch.randn(7192, H, D, device=dev, dtype=torch.bfloat16) * 4.0
    sB = ksp._cached_scales('L2', 3, qB, kB, kB * 0.5, 1000)
    fresh = tuple(t.float().abs().amax().item() * ksp.SCALE_MARGIN / 448.0 for t in (qB, kB, kB * 0.5))
    print(f'stale first page: scales B {tuple(round(x, 4) for x in sB)} fresh {tuple(round(x, 4) for x in fresh)} (A: {tuple(round(x, 4) for x in sA)})')
    assert all(abs(a - b) < 1e-6 for a, b in zip(sB, fresh))
    # continuity: the next chunk of B reuses the entry without rebuilding
    km2 = ksp._cached_block_means('L2', 3, kB[:0].new_zeros(0, Hk, D) if False else torch.randn(64, Hk, D, device=dev, dtype=torch.bfloat16), 8192, 8256)
    assert ksp._KMEAN_CACHE[('L2', 3)]['last_kv'] == 8256
    print('stale first page: ok')


if __name__ == '__main__' and 'stale' in __import__('sys').argv:
    test_stale_first_page()
