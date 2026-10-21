# Deployment notes: Qwen3.8-27B on one RTX PRO 6000 Blackwell (SM120)

Goal: minimum time to first token for 64K / 128K / 256K-token inputs on a single SM120 card.
Full numbers are in [results.md](results.md), kernel internals in [kernels.md](kernels.md).

## Recommended configuration

All-FP8 (variant C): FP8 block-sparse attention + vLLM FP8 W8A8 GEMMs, prefill chunk 12032.

```bash
export SM120_SPARSE_ATTN=1 SM120_SPARSE_KERNEL=sm120fp8 SM120_SPARSE_TAU=0.95
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --quantization fp8 \
    --max-model-len 262144 --max-num-batched-tokens 12032 --max-num-seqs 4 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

| variant | attention | GEMM | TTFT 64K / 128K / 256K (s) | mean ΔNLL | top-1 agreement |
|---|---|---|---|---:|---:|
| stock vLLM | BF16 dense | BF16 | 12.06 / 30.17 / 81.66 | 0 | — |
| A | FP8 sparse | BF16 | 9.99 / 21.97 / 49.45 | +0.0032 | 98.7% |
| B | BF16 sparse | FP8 | 7.22 / 17.33 / 44.33 | +0.0023 | 97.4% |
| **C** | **FP8 sparse** | **FP8** | **6.68 / 15.20 / 36.16** | **+0.0026** | **97.7%** |

Single request, prefill chunk 8192; chunk 12032 adds another ~2%. Needle retrieval passes for every variant.

- FP8 GEMM gives the larger speedup: at 64K attention is only ~22% of TTFT.
- FP8 attention adds almost no error on top of sparsity (per-token |Δlogprob| 0.015 vs 0.015 at 64K);
  FP8 GEMM roughly doubles it, and C ≈ B in error. Pick A if output fidelity matters more than speed.
- `--max-num-batched-tokens 12032` makes the attention grid (24 heads × 94 query blocks = 2256 CTAs) exactly
  12 waves on 188 SMs. At 8192 rows the last partial wave costs ~5% kernel time.

## Hardware facts that shaped the design

| measurement | value | consequence |
|---|---|---|
| sustained BF16 GEMM | 600 W cap, ~2.15 GHz, ~414 TFLOPS | cuBLAS already hits 392 TFLOPS on the MLP shapes; no room in dense GEMM |
| DSMEM | 3–6 GB/s per SM, 5–10× slower than L2 via TMA | no cross-CTA fusion; MLP megakernel abandoned |
| `mma.sync` HMMA, 4 / 8 warps per SM | 422 / 455 TFLOPS | need 2 warps per scheduler to saturate the tensor pipe |
| vLLM FA2 paged attention, head_dim 256 | 305 TFLOPS, HMMA 66% | attention is where the headroom is |

## Attention sparsity

On real long documents and code, 64×64 blocks covering 95% / 99% of attention mass are 34% / 62% of all
blocks at 64K. Only 6–11 of 384 heads are strongly local, so selection is dynamic per head and per query
block. Block-mean scoring ranks within ~1.5 points of exact row sampling. τ = 0.95 keeps 44–50% of blocks;
τ = 0.90 / 0.85 keep 32–37% / 24–29% and still retrieve the needle.

## Pipeline

1. **Block selection.** Per-request means of every 64-key block of K (updated incrementally per chunk) are
   scored against 64-row query blocks; each head keeps the smallest block set carrying τ of the mass, plus
   diagonal and sink blocks. Output is a padded list per (head, 128-row block), no host sync.
2. **FP8 operands.** Q/K/V are quantised to e4m3 per request and layer, with amax scales from the first
   sparse chunk. V is passed transposed with keys permuted per 32-key block so the S accumulator can be
   reused as the P operand in registers.
3. **Kernel** (`kai_fa_fp8_tma`). A producer warp streams K and Vᵀ via TMA into two-stage mbarrier
   pipelines; 8 MMA warps run online softmax on 128×64 tiles with no CTA-wide barrier in the main loop.
4. **vLLM hook.** Runs before the normal FlashAttention call, only for prefill chunks with q ≥ 1024 and
   kv ≥ 8192. Mixed batches are split per request; CUDA graphs are untouched.

Caches are keyed by (layer, first KV page) and record where the previous chunk ended; a discontinuity
(e.g. a recycled page under concurrency) triggers a rebuild from the request's full K history.

## Kernel progression (dense, 8K × 64K causal)

| step | ms | TFLOPS |
|---|---:|---:|
| vLLM FA2 (paged) | — | 305 |
| `kai_fa_bf16`, 64-row blocks, 4 warps | 36.6 | 340 |
| FP8, 128-row blocks, 8 warps, P via smem | 23.0 | 538 |
| + K double buffer, P in registers (`kai_fa_fp8`) | 19.5 | 635 |
| + TMA, 2-stage mbarrier pipelines, producer warp group (`kai_fa_fp8_tma`) | **18.0** | **688** |

Tried and dropped: split-D warp pairs (−28%), FA3-style warp-group ping-pong, intra-warp S double buffering,
explicit ldmatrix prefetch, and MXF8 block scaling (emulated error 3.0–3.8% vs 1.3–3.2% for per-tensor scales).
Details in [kernels.md](kernels.md).

## TODO

- [ ] Decode path optimisation (currently vLLM's stock decode kernels).
- [ ] FP8 KV cache, read directly by the sparse kernel (removes the per-chunk gather + quantise + transpose).
- [ ] Static per-layer FP8 scales from calibration.
- [ ] Fuse block-mean updates into the KV-cache write.
- [ ] Fuse small memory-bound kernels around the GEMMs (norm, GDN conv, gating), ~2–3%.
