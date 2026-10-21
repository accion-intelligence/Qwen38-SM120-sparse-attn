"""SM120 (RTX PRO 6000 Blackwell, CC 12.0) FlashAttention prefill kernels written in the CUTLASS CuTe DSL.

kai_fa_bf16     BF16 operands, 64-row query blocks, 4 warps, cp.async (derived from the CUTLASS Ampere FA2 CuTe DSL example)
kai_fa_fp8      e4m3 operands, 128-row query blocks, 8 warps, cp.async, K double buffer, P converted in registers
kai_fa_fp8_tma  e4m3 operands, TMA + mbarrier two-stage pipelines, 8 MMA warps + producer warp group (fastest)
All three support GQA, chunked-prefill causal masking with a query offset, and per-(head, query block) block lists.
"""
from .kai_fa_bf16 import attention_sm120
from .kai_fa_fp8 import attention_sm120_fp8, quantize_e4m3, permute_vt_keys, vt_key_index, VT_KEY_PERM
from .kai_fa_fp8_tma import attention_sm120_fp8_tma

__all__ = ['attention_sm120', 'attention_sm120_fp8', 'attention_sm120_fp8_tma', 'quantize_e4m3', 'permute_vt_keys',
           'vt_key_index', 'VT_KEY_PERM']
