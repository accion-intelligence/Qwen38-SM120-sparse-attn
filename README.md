# sm120-sparse-attn

Dynamic block-sparse FP8 prefill attention for long-context inference on **RTX PRO 6000 Blackwell (SM120)**,
packaged as a vLLM hook. With `--quantization fp8` for the GEMMs it brings the time to first token of
Qwen3.8-27B on one card from 12.1 / 30.2 / 81.7 s down to 6.7 / 15.2 / 36.2 s for 64K / 128K / 256K-token
inputs (1.81x / 1.98x / 2.26x), with needle retrieval intact and a per-token output perturbation at the level of
the sparsity itself. Full numbers: [docs/results.md](docs/results.md); the study write-up (Chinese):
[docs/zh/deployment-study.md](docs/zh/deployment-study.md); interactive chart: `docs/speedup.html`.

## What is in the box

| component | file | role |
|---|---|---|
| FP8 attention kernel, TMA pipeline | `sm120_attn/kernels/kai_fa_fp8_tma.py` | production kernel: e4m3 `mma.sync` m16n8k32, 128-row query blocks, 8 MMA warps + a producer warp group feeding two-stage K / V^T mbarrier pipelines by TMA, P converted in registers. 688 TFLOPS dense (vLLM's FA2: 305) |
| FP8 attention kernel, cp.async | `sm120_attn/kernels/kai_fa_fp8.py` | same math with cp.async and CTA barriers (635 TFLOPS); also holds the quantisation helpers |
| BF16 attention kernel | `sm120_attn/kernels/kai_fa_bf16.py` | 64-row blocks, 4 warps, 340 TFLOPS; derived from the CUTLASS Ampere FA2 CuTe DSL example |
| sparse prefill glue + vLLM hook | `sm120_attn/sparse_prefill.py` | per-head dynamic 64x64 block selection (block-mean scores, tau-mass), cached block means and FP8 scales per request, mixed-batch handling, `try_sparse_forward` entry point |
| vLLM patch | `tools/patch_vllm.py`, `patches/` | five lines in `FlashAttentionImpl.forward` |
| evaluation | `scripts/` | prompt builder, greedy validation (needle), teacher-forced error scoring, TTFT benchmark, sparsity/numerics probes |
| optional | `extras/bf16_mlp_fusion/` | BF16 gate/up GEMM + SwiGLU fusion for BF16-weight deployments (not used with FP8 GEMM) |

All kernels are written in the CUTLASS CuTe DSL (no C++ build); they JIT-compile on first use for CC 12.0.

## Requirements

- One RTX PRO 6000 Blackwell (or another CC 12.0 part with >= 99 KB shared memory per block); the kernels use
  `mma.sync`, TMA, mbarriers and `setmaxnreg` as exposed by the CuTe DSL for `sm_120a`.
- CUDA 13, PyTorch 2.13 (cu130), `nvidia-cutlass-dsl` 4.7.1, `cuda-python` 13.x, vLLM **0.30.0**
  (the hook anchors on its `flash_attn.py`). Tested with Python 3.12.
- Model: Qwen3.8-27B (hybrid GDN + full attention, head_dim 256, 24 query / 4 KV heads). The kernels support
  head_dim 256 only; the glue is generic for vLLM's FlashAttention backend with causal full attention.

## Install

```bash
git clone --recursive <this repo>
cd sm120-sparse-attn
pip install -e .                      # the sm120_attn package

# vLLM: either the matching wheel ...
pip install vllm==0.30.0
python tools/patch_vllm.py --target "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
# ... or build the submodule from source after patching it
python tools/patch_vllm.py --target vllm/vllm && pip install -e vllm/
```

`tools/patch_vllm.py --check` only reports. The patch is idempotent and is the whole vLLM modification.

## Run

```bash
export SM120_SPARSE_ATTN=1 SM120_SPARSE_KERNEL=sm120fp8 SM120_SPARSE_TAU=0.95
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --quantization fp8 \
    --max-model-len 262144 --max-num-batched-tokens 12032 --max-num-seqs 4 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

Configuration (environment, or a JSON file named by `SM120_SPARSE_CONFIG`):

| variable | default | meaning |
|---|---|---|
| `SM120_SPARSE_ATTN` | 1 | enable the hook |
| `SM120_SPARSE_KERNEL` | `sm120fp8` | `sm120fp8` (FP8 kernels), `sm120` (BF16 kernel), `flashinfer` (retired reference) |
| `SM120_SPARSE_FP8_PIPELINE` | `tma` | `tma` or `cpasync` FP8 kernel |
| `SM120_SPARSE_TAU` | 0.95 | attention mass kept per head and query block (0.90 and 0.85 also retrieve the needle; fewer blocks) |
| `SM120_SPARSE_MIN_Q`, `SM120_SPARSE_MIN_KV` | 1024, 8192 | a prefill chunk takes the sparse path only above these lengths |
| `SM120_SPARSE_DEBUG` | 0 | number of diagnostic lines to print |

`--max-num-batched-tokens 12032` makes the attention grid (24 heads x 94 query blocks) a whole number of waves on
the 188 SMs; it is worth about 2% over 8192. The three deployment variants measured in `docs/results.md`:
A = FP8 attention + BF16 GEMM (drop `--quantization fp8`), B = BF16 attention (`SM120_SPARSE_KERNEL=sm120`) +
FP8 GEMM, C = all FP8 (above). C is the recommended one.

## Evaluate

```bash
# prompts: exact-length long documents with a planted fact, N distinct prompts per length
python scripts/build_prompts.py --model $MODEL --corpus path/to/*.md --lengths 65536,131072,260096 --concurrency 2 --output prompts.json
# greedy generations + needle (dense reference, then a variant)
SM120_SPARSE_ATTN=0 python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-dense.json
python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-c.json --quantization fp8
# teacher-forced error vs the dense continuation
SM120_SPARSE_ATTN=0 python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --output tfe-dense.json
python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --reference-scores tfe-dense.json --output tfe-c.json --quantization fp8
python scripts/compare_fp8_variants.py --validation . --variants 'dense=dense' 'C=c:c' --output error.json
# time to first token (launches vllm serve)
python scripts/bench_ttft.py --model $MODEL --prompts prompts.json --output ttft-c --serve-args "--quantization fp8" --env SM120_SPARSE_KERNEL=sm120fp8
python scripts/summarize_ttft.py --runs stock=ttft-stock C=ttft-c --output summary.json
```

Kernel tests (run from `tests/`, GPU required): `test_fa_bf16.py`, `test_fa_fp8.py --big`, `test_fa_fp8_tma.py --big`,
`test_sparse_prefill.py [stale]`, `test_sparse_prefill_fp8.py`. `tools/ncu_probe_fp8.py` is an NCU target.

## How it works

1. **Block selection.** For each prefill chunk the glue keeps per-request means of every 64-key block of K
   (updated incrementally), scores them against the chunk's 64-row query blocks, and keeps per head the
   smallest set of blocks that carries `tau` of the attention mass (diagonal and sink blocks always).
   The result is a padded block list per (head, 128-row query block); no host synchronisation.
2. **FP8 operands.** Q, K, V are quantised per request and layer to e4m3 with amax scales taken from the
   request's first sparse chunk; V is supplied transposed (keys contiguous) with the keys of every 32-key block
   permuted so the kernel can turn the S accumulator into the P operand in registers.
3. **Kernel.** `kai_fa_fp8_tma`: a producer warp group streams K and V^T tiles with TMA into two-stage mbarrier
   pipelines, eight MMA warps run online softmax per 128x64 tile, consumer warps never synchronise with each
   other, and the epilogue writes bf16 directly. Details and the rejected variants (split-D, ping-pong
   scheduling, S double buffering, explicit prefetch, MXF8 block scaling) are in [docs/kernels.md](docs/kernels.md).
4. **vLLM integration.** The hook runs before the normal FlashAttention call; batches without a qualifying
   long prefill chunk return immediately, mixed batches are split per request (dense FlashAttention for the
   others), CUDA-graph capture is never touched.

## Limitations

- SM120 / CC 12.0 and head_dim 256 only; KV cache stays BF16 (vLLM's FA2 backend has no FP8 KV cache on this
  architecture), so each chunk re-quantises and gathers the request's K/V.
- Prefill only; decode uses vLLM's kernels.
- Quality was checked with needle retrieval and teacher-forced scoring on long technical documents; run a
  broader long-context evaluation (RULER-style) before production use.

## License

Apache-2.0 for this project. The kernels derive from NVIDIA CUTLASS examples (BSD-3-Clause, notices retained);
vLLM is included as a submodule under its own Apache-2.0 license. See `THIRD_PARTY_NOTICES.md`.
