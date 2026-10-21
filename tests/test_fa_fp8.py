import sys, time, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from sm120_attn.kernels.kai_fa_fp8 import attention_sm120_fp8, quantize_e4m3, permute_vt_keys
from sm120_attn.kernels.kai_fa_bf16 import attention_sm120
from test_fa_bf16 import reference, random_masks, to_lists, to_lists_m128
torch.manual_seed(0); dev = 'cuda'; H, Hk, D = 24, 4, 256

def ref_from_quant(q, k, v, q_start):
    """fp32 reference using the same e4m3-rounded operands (isolates kernel error from quantisation error)."""
    q8, sq = quantize_e4m3(q); k8, sk = quantize_e4m3(k); v8, sv = quantize_e4m3(v)
    return reference((q8.float() * sq).to(torch.bfloat16), (k8.float() * sk).to(torch.bfloat16), (v8.float() * sv).to(torch.bfloat16), q_start)

print('--- dense causal (fp8 kernel)')
for n, kv in ((256, 1024), (1000, 4024), (64, 4096), (2048, 2048)):
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16); k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    o = attention_sm120_fp8(q, k, v, m_block=128)
    ref_q = ref_from_quant(q, k, v, kv - n); ref = reference(q, k, v, kv - n)
    print(f'  n={n:5d} kv={kv:5d}: vs quantised-operand ref {(o.float()-ref_q).abs().max().item()/ref_q.abs().max().item():.2e} | vs bf16 ref rel L2 {(o.float()-ref).norm().item()/ref.norm().item():.3e}')
print('--- block-sparse (fp8 kernel)')
for n, kv, density in ((512, 2048, 0.3), (1024, 8192, 0.2)):
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16); k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    masks = random_masks(n, kv, kv - n, H, density, dev); counts, lists = to_lists_m128(masks)
    o = attention_sm120_fp8(q, k, v, kv - n, counts=counts, lists=lists, m_block=128)
    masks = masks.view(H, -1, 2, masks.shape[2]).any(2).repeat_interleave(2, 1)[:, :masks.shape[1]]  # reference sees the union too
    q8, sq = quantize_e4m3(q); k8, sk = quantize_e4m3(k); v8, sv = quantize_e4m3(v)
    ref = reference((q8.float()*sq).to(torch.bfloat16), (k8.float()*sk).to(torch.bfloat16), (v8.float()*sv).to(torch.bfloat16), kv - n, masks)
    print(f'  n={n:5d} kv={kv:5d}: vs quantised-operand masked ref {(o.float()-ref).abs().max().item()/ref.abs().max().item():.2e}')
if '--big' in sys.argv:
    n, kv = 8192, 65536
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16); k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    q8, sq = quantize_e4m3(q); k8, sk = quantize_e4m3(k); v8, sv = quantize_e4m3(v); vT8 = permute_vt_keys(v8.permute(1, 2, 0))
    fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
    for name, fn in (('bf16 v2', lambda: attention_sm120(q, k, v)), ('fp8 m64 t128', lambda: attention_sm120_fp8(q, k, v, prequant=(q8, sq, k8, sk, vT8, sv), m_block=64)), ('fp8 m128 t256', lambda: attention_sm120_fp8(q, k, v, prequant=(q8, sq, k8, sk, vT8, sv), m_block=128))):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.time(); iters = 5
        for _ in range(iters): fn()
        torch.cuda.synchronize(); ms = (time.time() - t) / iters * 1000
        print(f'  {name}: {ms:.2f} ms  {fl/ms/1e9:.1f} TFLOPS')
