# Results (Qwen3.8-27B, one RTX PRO 6000 Blackwell, vLLM 0.30.0, 2026-10-19/20)

All numbers were measured on the same card (GPU7 of a shared node unless stated), search split of the
long-document prompts (`scripts/build_prompts.py` semantics), prefill chunk 8192, one warmup + two measured runs.
Raw files: `docs/validation/`.

## Attention kernels (dense, 8192 query rows x 65536 keys, causal, head_dim 256, GQA 24/4)

| kernel | ms | TFLOPS | notes |
|---|---:|---:|---|
| vLLM FlashAttention-2 (paged) | - | 305 | baseline (NCU 8K x 16K) |
| `kai_fa_bf16` (64-row blocks, 4 warps) | 36.6 | 340 | |
| `kai_fa_fp8` single buffer, P via smem | 23.0 | 538 | |
| `kai_fa_fp8` + K double buffer + P in registers | 19.5 | 635 | tensor pipe 75% |
| `kai_fa_fp8_tma` (TMA, 2-stage K/V^T, producer warp group) | **18.0** | **688** | tensor pipe 84% |

Block-sparse (per-head block lists, 40% of blocks): `kai_fa_fp8_tma` 11.5 ms (64K), 24.3 ms (8K x 128K), 26.8 ms (4K x 256K).

## Time to first token (ms)

| case | stock vLLM | ① MLP fusion | ② + sparse BF16 attn | ③ + FP8 attn | A: ④ FP8 TMA attn | B: sparse BF16 attn + FP8 GEMM | C: all FP8 | C vs stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K, 1 request | 12063 | 11715 | 10540 | 10144 | 9994 | 7215 | **6677** | 1.81x |
| 64K, 2 concurrent | 18698 | 18137 | 16425 | 15986 | 15780 | 11168 | **10481** | 1.78x |
| 128K, 1 request | 30171 | 29430 | 24014 | 22628 | 21973 | 17329 | **15203** | 1.98x |
| 128K, 2 concurrent | 45771 | 44682 | 36148 | 34365 | 33467 | 25768 | **23144** | 1.98x |
| 256K, 1 request | 81662 | 80373 | 57597 | 52025 | 49446 | 44330 | **36156** | 2.26x |
| 256K, 2 concurrent | 172197* | 121635 | 82446 | 76414 | 73331 | 62222 | **52799** | 2.30x (vs ①) |

\* the stock instance cannot hold two 256K contexts in its KV cache and queues the second request.
Prefill chunk 12032 instead of 8192 (attention grid = whole waves on 188 SMs) gains another 1.9/1.9/2.6% at
64K/128K/256K single request (`summary-chunk-gpu0.md`).

## Output error (teacher-forced, 256-token continuation of the BF16 dense output)

| variant | mean ΔNLL | top-1 agreement | mean \|Δlogprob\| 64K/128K/256K | needle |
|---|---:|---:|---|---|
| sparse BF16 attention | −0.0001 | 98.8% | 0.015 / 0.015 / 0.053 | ok |
| A: FP8 sparse attention + BF16 GEMM | +0.0032 | 98.7% | 0.015 / 0.015 / 0.046 | ok |
| B: BF16 sparse attention + FP8 GEMM | +0.0023 | 97.4% | 0.029 / 0.028 / 0.054 | ok |
| C: all FP8 | +0.0026 | 97.7% | 0.028 / 0.024 / 0.052 | ok |

FP8 attention adds essentially nothing on top of sparsity; FP8 GEMM doubles the per-token perturbation; all-FP8
equals FP8-GEMM-only. Per-layer attention output error of the e4m3 kernel on real activations: 1.1-2.5%.
Emulated MXF8 block scaling (power-of-two scales per 32 elements) was *worse* than per-tensor real scales
(3.04/3.75/1.53% vs 2.73/3.21/1.33% for layers 11/35/63) and was not implemented.

## Sparsity (real documents / code, 64 x 64 blocks)

Blocks needed for 95% / 99% of the attention mass: documents 64K 34% / 62%, code 64K 34% / 61%. Only 6-11 of 384
heads are strongly local; block selection is dynamic per head and per query block (tau = 0.95 keeps 44-57%).
