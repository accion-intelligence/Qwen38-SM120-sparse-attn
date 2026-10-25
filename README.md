# Qwen3.8 on RTX PRO 6000 (SM120): sparse FP8 attention

**Built by KAI-Pikachu.** An optimised single-GPU deployment of **Qwen3.8-27B** on the
**RTX PRO 6000 Blackwell (SM120)**: hand-written CuTe DSL FP8 attention kernels and dynamic block-sparse
prefill, plugged into vLLM 0.30.0 through a five-line hook.

Time to first token, single request, BF16 GEMMs on both sides (only attention differs):

| input | stock vLLM | + KAI-Pikachu sparse FP8 attention | speedup |
|---|---:|---:|---:|
| 64K | 12.06 s | 9.99 s | **1.21×** |
| 128K | 30.17 s | 21.97 s | **1.37×** |
| 256K | 81.66 s | 49.45 s | **1.65×** |

Needle retrieval intact; mean ΔNLL vs BF16 dense +0.003 nat. Attention kernel: 688 TFLOPS dense vs 305 for
vLLM's FA2.

The recommended deployment also enables vLLM's built-in FP8 W8A8 GEMM (`--quantization fp8`, not part of this
work), reaching 6.68 / 15.20 / 36.16 s. The matching baseline (dense attention + vLLM FP8 GEMM) has not been
measured yet, so the attention share of that total is not isolated; see [deployment notes](docs/deployment.md).

Docs: [deployment notes](docs/deployment.md) · [full results](docs/results.md) ·
[kernel details](docs/kernels.md) · interactive chart `docs/speedup.html`

## Roadmap

- [x] Prefill: dynamic block-sparse FP8 attention
- [ ] Measure the dense-attention + vLLM FP8 GEMM baseline (TODO)
- [ ] Decode (TODO)
- [ ] FP8 KV cache read directly by the kernel (TODO)

## Layout

| path | contents |
|---|---|
| `sm120_attn/kernels/kai_fa_fp8_tma.py` | production FP8 kernel: TMA + mbarrier pipelines, 8 MMA warps + producer warp group |
| `sm120_attn/kernels/kai_fa_fp8.py` | cp.async FP8 kernel and quantisation helpers |
| `sm120_attn/kernels/kai_fa_bf16.py` | BF16 kernel |
| `sm120_attn/sparse_prefill.py` | block selection, FP8 scale/mean caches, vLLM entry point |
| `tools/patch_vllm.py`, `patches/` | the vLLM patch |
| `scripts/` | prompt builder, needle validation, teacher-forced scoring, TTFT benchmark |
| `extras/bf16_mlp_fusion/` | optional gate/up + SwiGLU fusion for BF16-weight deployments |

Kernels JIT-compile on first use; no C++ build.

## Requirements

RTX PRO 6000 Blackwell (CC 12.0), CUDA 13, PyTorch 2.13 (cu130), `nvidia-cutlass-dsl` 4.7.1,
`cuda-python` 13.x, vLLM 0.30.0, Python 3.12.

## Install

```bash
git clone --recursive <this repo> && cd Qwen38-SM120-sparse-attn
pip install -e .
pip install vllm==0.30.0
python tools/patch_vllm.py --target "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
# or patch and build the submodule: python tools/patch_vllm.py --target vllm/vllm && pip install -e vllm/
```

The patch is idempotent; `--check` only reports.

## Run

```bash
export SM120_SPARSE_ATTN=1 SM120_SPARSE_KERNEL=sm120fp8 SM120_SPARSE_TAU=0.95
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --quantization fp8 \
    --max-model-len 262144 --max-num-batched-tokens 12032 --max-num-seqs 4 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

| variable | default | meaning |
|---|---|---|
| `SM120_SPARSE_ATTN` | 1 | enable the hook |
| `SM120_SPARSE_KERNEL` | `sm120fp8` | `sm120fp8` (FP8) or `sm120` (BF16) |
| `SM120_SPARSE_FP8_PIPELINE` | `tma` | `tma` or `cpasync` |
| `SM120_SPARSE_TAU` | 0.95 | attention mass kept per head and query block |
| `SM120_SPARSE_MIN_Q` / `_MIN_KV` | 1024 / 8192 | minimum chunk / context length for the sparse path |
| `SM120_SPARSE_DEBUG` | 0 | diagnostic lines to print |

Settings can also come from a JSON file named by `SM120_SPARSE_CONFIG`.

## Evaluate

```bash
python scripts/build_prompts.py --model $MODEL --corpus path/to/*.md --lengths 65536,131072,260096 --concurrency 2 --output prompts.json
SM120_SPARSE_ATTN=0 python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-dense.json
python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-c.json --quantization fp8
SM120_SPARSE_ATTN=0 python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --output tfe-dense.json
python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --reference-scores tfe-dense.json --output tfe-c.json --quantization fp8
python scripts/bench_ttft.py --model $MODEL --prompts prompts.json --output ttft-c --serve-args "--quantization fp8" --env SM120_SPARSE_KERNEL=sm120fp8
```

Kernel tests (GPU, from `tests/`): `test_fa_bf16.py`, `test_fa_fp8.py --big`, `test_fa_fp8_tma.py --big`,
`test_sparse_prefill_fp8.py`.

## License

Apache-2.0. Kernels derive from NVIDIA CUTLASS examples (BSD-3-Clause, notices retained); vLLM is a submodule
under Apache-2.0. See `THIRD_PARTY_NOTICES.md`.
