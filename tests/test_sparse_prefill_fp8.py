import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sm120_attn.sparse_prefill as ksp
from test_fa_bf16 import reference
torch.manual_seed(0); dev = 'cuda'; H, Hk, D = 24, 4, 256
kv = 24576
k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16); v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
u = torch.randn(Hk, 1, D, device=dev); hot = (torch.rand(kv // 64 + 1, device=dev) < 0.1).repeat_interleave(64)[:kv]
k = (k.float() * 0.5 + hot[:, None, None].float() * u.transpose(0, 1) * 3.0).to(torch.bfloat16)
sp = ksp.SparsePrefill(H, Hk, D, dev, tau=0.95, min_kv=0, min_q=0, kernel='sm120fp8')
sp16 = ksp.SparsePrefill(H, Hk, D, dev, tau=0.95, min_kv=0, min_q=0, kernel='sm120')
for q_start, n in ((0, 8192), (8192, 8192), (16384, 8192), (5000, 3000)):
    q = (torch.randn(n, H, D, device=dev) + u.repeat_interleave(H // Hk, 0).transpose(0, 1)).to(torch.bfloat16)
    o8 = sp.forward_sm120fp8(q, k[:q_start + n], v[:q_start + n], q_start)
    o16 = sp16.forward_sm120(q, k[:q_start + n], v[:q_start + n], q_start)
    ref = reference(q, k[:q_start + n], v[:q_start + n], q_start)
    print(f'q_start={q_start:5d} n={n}: fp8 vs bf16-sparse rel L2 {(o8.float()-o16.float()).norm().item()/o16.float().norm().item():.3e} | fp8 vs dense ref {(o8.float()-ref).norm().item()/ref.norm().item():.3e} | bf16-sparse vs dense {(o16.float()-ref).norm().item()/ref.norm().item():.3e} | kept fp8 {sp.kept_fraction():.3f} bf16 {sp16.kept_fraction():.3f}')
# direct-kernel cross-check with identical masks and scales (isolates glue plumbing from quantisation error)
from sm120_attn.kernels.kai_fa_fp8 import attention_sm120_fp8, permute_vt_keys
from test_fa_bf16 import to_lists_m128
q_start, n = 8192, 8192
q = (torch.randn(n, H, D, device=dev) + u.repeat_interleave(H // Hk, 0).transpose(0, 1)).to(torch.bfloat16)
kk, vv = k[:q_start + n], v[:q_start + n]
km = kk.view(-1, 64, Hk, D).float().mean(1).transpose(0, 1)
masks = sp.block_masks(q, None, q_start, union=False, km=km)
counts, lists = to_lists_m128(masks)
sq, sk, sv = (float(t.float().abs().amax().item()) / 448.0 for t in (q, kk, vv))
o_direct = attention_sm120_fp8(q, kk, vv, q_start, counts=counts, lists=lists, prequant=((q.float()/sq).to(torch.float8_e4m3fn), sq, (kk.float()/sk).to(torch.float8_e4m3fn), sk, permute_vt_keys((vv.float()/sv).permute(1,2,0).to(torch.float8_e4m3fn)), sv), m_block=128)
o_glue = sp.forward_sm120fp8(q, kk, vv, q_start, scales=(sq, sk, sv))
print(f'glue vs direct kernel max|diff| = {(o_glue.float()-o_direct.float()).abs().max().item():.3e}  (should be ~0)')
