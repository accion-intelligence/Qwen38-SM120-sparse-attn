# Optional: BF16 gate/up GEMM + SwiGLU fusion for the Qwen3.8 dense MLP

A persistent CuTe DSL BF16 GEMM (`kai_swiglu_vendor.py`, derived from the CUTLASS `blackwell_geforce` dense GEMM
example, BSD-3-Clause) that computes the gate and up projections in one pass with SwiGLU applied in registers
(`kai_persistent_mlp.py`), replacing `Qwen2MoeMLP` in `qwen3_5.py` (`qwen3_5.patch`). Measured gain: 2.5-3%
TTFT with BF16 weights. It is **not** used by the recommended all-FP8 configuration: with `--quantization fp8`
vLLM's quantized linear layers (CUTLASS FP8 scaled_mm) are faster and this fusion would bypass them.

Install with `python tools/patch_vllm.py --target <vllm package dir> --with-mlp-fusion`.
