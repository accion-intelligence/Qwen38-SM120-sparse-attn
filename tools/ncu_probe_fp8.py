import importlib, sys, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'sm120_attn.kernels.kai_fa_fp8')
fn = getattr(mod, sys.argv[2] if len(sys.argv) > 2 else 'attention_sm120_fp8')
from sm120_attn.kernels.kai_fa_fp8 import quantize_e4m3, permute_vt_keys
H, Hk, D = 24, 4, 256; n, kv = 8192, 16384
q = torch.randn(n, H, D, device='cuda', dtype=torch.bfloat16); k = torch.randn(kv, Hk, D, device='cuda', dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device='cuda', dtype=torch.bfloat16)
q8, sq = quantize_e4m3(q); k8, sk = quantize_e4m3(k); v8, sv = quantize_e4m3(v); vT8 = permute_vt_keys(v8.permute(1, 2, 0))
for _ in range(2): fn(q, k, v, prequant=(q8, sq, k8, sk, vT8, sv))
torch.cuda.synchronize()
