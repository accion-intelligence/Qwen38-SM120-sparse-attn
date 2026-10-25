# KAI-Pikachu · One Day of Autonomous Optimization on SM120

**These results were produced by KAI-Pikachu during one day of autonomous operation, without human supervision.**

The target was long-context prefill for **Qwen3.8-27B** on a single **RTX PRO 6000 Blackwell (SM120)**.
The resulting implementation combines custom CuTe DSL FP8 attention kernels, dynamic block sparsity,
and a small vLLM integration hook to accelerate 64K–256K-token inputs. This repository shares the
optimization results, source code, evaluation scripts, and recorded measurements from that run.

At 256K input length, the optimized BF16 GEMM configuration reduces time to first token from
**81.66 s to 49.45 s (1.65×)**. Combining the attention kernels with vLLM's built-in FP8 GEMM brings
that to **36.16 s**. The configurations and contributions behind these numbers are detailed below.

[Deployment guide](docs/deployment.md) · [Full results](docs/results.md) ·
[Kernel design](docs/kernels.md) · [Performance chart](docs/speedup.html)

## What's included

- **Custom FP8 and BF16 attention kernels.** The FP8 TMA kernel uses staged K/V transfers, mbarrier
  pipelines, and register-resident attention probabilities, reaching **688 TFLOPS** in the dense
  8K-query × 64K-key causal benchmark.
- **Dynamic block-sparse prefill.** Attention blocks are selected per head and query block, with a
  configurable attention-mass threshold and incremental caches for chunked prefill.
- **vLLM integration.** A small, idempotent patch connects the kernels to vLLM 0.30.0. The release
  targets prefill; decode uses vLLM's existing kernels.
- **Evaluation tools.** Prompt construction, needle retrieval, teacher-forced output comparison,
  and time-to-first-token benchmarks are included alongside the recorded validation results.

## Measured performance

Time to first token (TTFT) for **Qwen3.8-27B on one RTX PRO 6000 Blackwell**, one request,
prefill chunk 8192:

| Input length | Stock vLLM | KAI-Pikachu, BF16 GEMM¹ | Speedup vs stock | KAI-Pikachu + vLLM FP8 GEMM² |
|---|---:|---:|---:|---:|
| 64K | 12.06 s | 9.99 s | **1.21×** | 6.68 s |
| 128K | 30.17 s | 21.97 s | **1.37×** | 15.20 s |
| 256K | 81.66 s | 49.45 s | **1.65×** | 36.16 s |

¹ Uses KAI-Pikachu sparse FP8 attention and the optional BF16 MLP fusion. Both this configuration
and stock vLLM use BF16 GEMMs; the reported speedup includes the fusion contribution.

² Uses KAI-Pikachu sparse FP8 attention with vLLM's built-in FP8 W8A8 GEMM (`--quantization fp8`).
The FP8 GEMM implementation belongs to vLLM. A dense-attention + FP8 GEMM baseline has not yet been
measured, so this column reports combined deployment latency.

In the recorded teacher-forced evaluation, the sparse FP8 attention / BF16 GEMM variant has a mean
**ΔNLL of +0.0032 nat** and **98.7% top-1 agreement** against the BF16 dense reference. Needle retrieval
passed the single-request tests at 64K, 128K, and 256K. See [full results](docs/results.md) for methodology, concurrent requests,
and output comparisons; raw records are in [`docs/validation/`](docs/validation/).

## Requirements

The measured environment uses **RTX PRO 6000 Blackwell (CC 12.0)**, CUDA 13, PyTorch 2.13 (cu130),
`nvidia-cutlass-dsl` 4.7.1, `cuda-python` 13.x, vLLM 0.30.0, and Python 3.12.

Kernels JIT-compile on first use; no separate C++ build is required for `sm120_attn`.

## Quick start

### Install

```bash
git clone --recursive <this repo> && cd Qwen38-SM120-sparse-attn
pip install -e .
pip install vllm==0.30.0
python tools/patch_vllm.py --target "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
# or patch and build the submodule: python tools/patch_vllm.py --target vllm/vllm && pip install -e vllm/
```

The patch is idempotent. Add `--check` to inspect its status without changing files.
For the BF16 GEMM configuration, see the [optional MLP fusion](extras/bf16_mlp_fusion/README.md).

### Serve

The recommended deployment combines sparse FP8 attention with vLLM's FP8 GEMM. It uses a prefill
chunk of 12032; the table above uses 8192. See the [deployment guide](docs/deployment.md) for the
chunk-size comparison and configuration choices.

```bash
export SM120_SPARSE_ATTN=1 SM120_SPARSE_KERNEL=sm120fp8 SM120_SPARSE_TAU=0.95
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --quantization fp8 \
    --max-model-len 262144 --max-num-batched-tokens 12032 --max-num-seqs 4 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

### Configure

| Variable | Default | Meaning |
|---|---|---|
| `SM120_SPARSE_ATTN` | 1 | enable the sparse attention hook |
| `SM120_SPARSE_KERNEL` | `sm120fp8` | `sm120fp8` (FP8) or `sm120` (BF16) |
| `SM120_SPARSE_FP8_PIPELINE` | `tma` | `tma` or `cpasync` |
| `SM120_SPARSE_TAU` | 0.95 | target attention mass retained per head and query block |
| `SM120_SPARSE_MIN_Q` / `_MIN_KV` | 1024 / 8192 | minimum chunk / context length for the sparse path |
| `SM120_SPARSE_DEBUG` | 0 | diagnostic lines to print |

Settings can also come from a JSON file named by `SM120_SPARSE_CONFIG`.

## Reproduce the results

Install the evaluation dependencies with `pip install -e '.[eval]'`, set `MODEL` to your model
name or local path, and provide a corpus for prompt construction:

```bash
python scripts/build_prompts.py --model $MODEL --corpus path/to/*.md --lengths 65536,131072,260096 --concurrency 2 --output prompts.json
SM120_SPARSE_ATTN=0 python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-dense.json
python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-c.json --quantization fp8
SM120_SPARSE_ATTN=0 python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --output tfe-dense.json
python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --reference-scores tfe-dense.json --output tfe-c.json --quantization fp8
python scripts/bench_ttft.py --model $MODEL --prompts prompts.json --output ttft-c --serve-args "--quantization fp8" --env SM120_SPARSE_KERNEL=sm120fp8
```

Run kernel checks from the repository root on a compatible GPU:

```bash
python tests/test_fa_bf16.py
python tests/test_fa_fp8.py --big
python tests/test_fa_fp8_tma.py --big
python tests/test_sparse_prefill_fp8.py
```

## Repository layout

| Path | Contents |
|---|---|
| `sm120_attn/kernels/kai_fa_fp8_tma.py` | FP8 attention kernel with TMA and mbarrier pipelines |
| `sm120_attn/kernels/kai_fa_fp8.py` | cp.async FP8 kernel and quantization helpers |
| `sm120_attn/kernels/kai_fa_bf16.py` | BF16 attention kernel |
| `sm120_attn/sparse_prefill.py` | Block selection, FP8 scale/mean caches, and vLLM entry point |
| `tools/patch_vllm.py`, `patches/` | vLLM integration patch and notes |
| `scripts/`, `tests/` | Evaluation scripts and GPU kernel checks |
| `docs/` | Deployment guide, kernel design, and validation records |
| `extras/bf16_mlp_fusion/` | Optional gate/up + SwiGLU fusion for BF16 deployments |

## Roadmap

- [x] Dynamic block-sparse FP8 prefill attention on SM120
- [x] vLLM integration and evaluation scripts
- [ ] Measure the dense-attention + vLLM FP8 GEMM baseline
- [ ] Optimize the decode path
- [ ] Read an FP8 KV cache directly in the sparse kernel

## License and acknowledgments

Released under [Apache-2.0](LICENSE). The kernels build on NVIDIA CUTLASS examples under BSD-3-Clause,
with the original notices retained. vLLM is included as a submodule under Apache-2.0.
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and license details.
