"""TMA-pipelined FP8 kernel vs the cp.async FP8 kernel on identical quantised operands; then timing."""
import sys, time
import torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from sm120_attn.kernels.kai_fa_fp8 import attention_sm120_fp8, quantize_e4m3, permute_vt_keys
from sm120_attn.kernels.kai_fa_fp8_tma import attention_sm120_fp8_tma
from test_fa_bf16 import random_masks, to_lists_m128

torch.manual_seed(0)
H, Hk, D = 24, 4, 256
def prep(n, kv):
    q = torch.randn(n, H, D, device='cuda', dtype=torch.bfloat16); k = torch.randn(kv, Hk, D, device='cuda', dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device='cuda', dtype=torch.bfloat16)
    q8, sq = quantize_e4m3(q); k8, sk = quantize_e4m3(k); v8, sv = quantize_e4m3(v); vT8 = permute_vt_keys(v8.permute(1, 2, 0))
    return q, k, v, (q8, sq, k8, sk, vT8, sv)
print('--- dense causal: tma vs cp.async kernel (same operands)')
for n, kv in ((256, 1024), (1000, 4032), (64, 4096), (2048, 2048), (8192, 16384)):
    q, k, v, pq = prep(n, kv)
    ref = attention_sm120_fp8(q, k, v, prequant=pq, m_block=128)
    out = attention_sm120_fp8_tma(q, k, v, prequant=pq, m_block=128)
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    print(f'  n={n:5d} kv={kv:6d}: rel L2 {err.item():.3e}  max abs {(out.float()-ref.float()).abs().max().item():.3e}')
print('--- block-sparse')
for n, kv in ((512, 2048), (1024, 8192)):
    q, k, v, pq = prep(n, kv)
    masks = random_masks(n, kv, kv - n, H, 0.5, 'cuda')
    counts, lists = to_lists_m128(masks)
    ref = attention_sm120_fp8(q, k, v, counts=counts, lists=lists, prequant=pq, m_block=128)
    out = attention_sm120_fp8_tma(q, k, v, counts=counts, lists=lists, prequant=pq, m_block=128)
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    print(f'  n={n:5d} kv={kv:6d}: rel L2 {err.item():.3e}')
if '--big' in sys.argv:
    n, kv = 8192, 65536
    q, k, v, pq = prep(n, kv)
    fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
    for name, fn in (('fp8 cp.async m128', lambda: attention_sm120_fp8(q, k, v, prequant=pq, m_block=128)),
                     ('fp8 tma m128', lambda: attention_sm120_fp8_tma(q, k, v, prequant=pq, m_block=128))):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10): fn()
        torch.cuda.synchronize(); ms = (time.time() - t) / 10 * 1000
        print(f'  {name}: {ms:.2f} ms  {fl/ms/1e9:.1f} TFLOPS')
