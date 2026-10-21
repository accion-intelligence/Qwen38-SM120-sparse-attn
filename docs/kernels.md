# SM120 attention kernels

`kai_fa_bf16.py` is derived from the CUTLASS CuTe DSL Ampere FlashAttention-2
example (`cute/ampere/kernel/attention/flash_attention_v2.py`, BSD-3-Clause)
and targets RTX PRO 6000 Blackwell (SM120, `mma.sync` m16n8k16 BF16, cp.async).
Changes so far: SM120 shared-memory capacity, GQA head mapping
(`qhead_per_kvhead`), chunked-prefill causal masking with an absolute query
offset, and an optional per-(head, query block) BSR block list for dynamic
block-sparse attention (diagonal block last in each list). `attention_sm120`
is a torch driver: `q [n, H, D]`, `k/v [kv_len, Hk, D]` (NHD, contiguous).

```bash
PY=/home/li004074/miniconda3/envs/kai-vllm030/bin/python
CUDA_VISIBLE_DEVICES=0 $PY scripts/qwen_sm120_attention/test_fa_bf16.py --big
```

Planned next steps: measure dense 8K x 64K against FA2 (338 TFLOPS non-paged,
305 paged), then TMA + mbarrier producer warp, two consumer warp groups with
softmax/MMA ping-pong, and a 128-row / 32-key tile configuration for head_dim
256 within the 100 KiB shared-memory budget. The vLLM glue in
`scripts/qwen_sparse_attn_study/sparse_prefill.py` selects this kernel with
`SM120_SPARSE_KERNEL=sm120` (per-query-head block lists) instead of the FlashInfer
BSR kernel (per-KV-head union).

## v3 split-D experiment (2026-10-19, rejected)

`v3 split-D experiment (not shipped)` pairs two warps per 16 query rows, each owning one half of
head_dim (O accumulator 64 registers, Q resident in registers, partial S
exchanged through shared memory). It is correct (dense and block-sparse) but
slower (51 ms vs 40 ms at 8K x 64K on a shared GPU0): shared-memory traffic for
K/V is proportional to the number of 16-row groups, not to the head_dim split,
so only the Q reload (~11% of ldmatrix traffic) was saved while the exchange
added 400M scalar shared loads/stores, bank conflicts and pair barriers (NCU:
HMMA 57% vs 74%). A pure-HMMA microbenchmark (`roofline-calibration`
`hmma_rate.cu`) shows 4 warps/SM reach 422 TFLOPS and 8 warps 455, and removing
the softmax exponentials from v2 saves only 3%, so the remaining BF16 headroom
is in the ldmatrix->mma dependency chain of a single warp per scheduler, not in
softmax overlap. Kept for reference; the production kernel is v2.

## FP8 (e4m3) kernel (2026-10-19, `kai_fa_fp8.py`)

Same FA2 structure as v2 with `warp.MmaFP8Op` (m16n8k32, fp32 accumulate), Q/K in e4m3 and
V supplied transposed (`V^T [Hk, D, T]`, keys contiguous) because the B operand must be K-major
and 8-bit `ldmatrix.trans` does not exist. P is scaled by 448, written as e4m3 to a CTA-private
shared tile and re-read with `ldmatrix` as the A operand (no fp8 fragment re-packing). All
quantisation scales fold into `softmax_scale * s_q * s_k` and `s_v / 448`.

Lessons: computing the masking coordinates inside the main loop (runtime `if n_tile < mask_steps`)
raised register pressure to 255 and spilled (4.5 GB local traffic); specialising the loop into
constexpr masked / unmasked steps like v2 removed most of it. The 128-row / 256-thread
configuration (one CTA, 8 warps, 72 KB smem, impossible for BF16 within 100 KB) removes the
spills entirely and uses the tensor pipe at 57% (fp8 rate, i.e. ~114% of the BF16 rate):

| kernel (8192 x 16384, NCU-locked clocks) | regs | local loads | HMMA % | time |
|---|---:|---:|---:|---:|
| BF16 v2, m64 t128 | 248 | 0 | 74.4 | 9.36 ms |
| FP8 m64 t128 | 255 | 6.0 GB | 46.8 | 7.44 ms |
| FP8 m128 t256 | 255 | 24 MB | 57.2 | 6.10 ms |

At 8192 x 65536 on a shared GPU0: BF16 v2 39.8 ms, FP8 m64 30.9 ms, FP8 m128 25.7 ms.
Block-sparse lists for 128-row blocks are the union of the two 64-row lists
(`tests/test_fa_bf16.to_lists_m128`). Kernel error vs an fp32 reference on the same
e4m3 operands is 1-2e-2 (P rounding); on real activations the e4m3 probe gave 1-2.5% output
error, and needle retrieval survives (see the deployment doc).

## FP8 kernel optimisation round (2026-10-19, GPU7, 8192 x 65536 causal, dense)

| step | file / switch | 8K x 64K | NCU 8K x 16K (locked clocks) |
|---|---|---:|---|
| single-buffered K/V, P via smem | `kai_fa_fp8 (single-buffer revision, not shipped)` | 23.00 ms (538 TF) | 6.10 ms, 255 regs, HMMA 57% |
| + K double buffer (pointer arithmetic) | `kai_fa_fp8 (K double buffer revision, not shipped)` | 21.68 ms (571 TF) | 5.76 ms, 2.4 GB local loads |
| + P converted in registers (V^T keys permuted) | `kai_fa_fp8.py` (cp.async kernel) | 19.47 ms (635 TF) | 5.10 ms, 247 regs, 0 local, tensor pipe 75% |
| + TMA loads, 2-stage K and V^T mbarrier pipelines, producer warp group | `kai_fa_fp8_tma.py` | 17.98 ms (688 TF) | 4.56 ms, tensor pipe 84%, 0 local |
| + intra-warp S double buffer (QK^T(n+1) before softmax(n)) | `s_pipeline=True` | 19.07 ms | slower: rejected (default off) |

Other sizes (TMA vs cp.async kernel): 8K x 128K 37.85 vs 40.41 ms; 4K x 256K 41.79 vs 45.41 ms;
40%-dense block-sparse 11.5 vs 13.0 (64K), 24.3 vs 27.1 (128K), 26.8 vs 30.1 ms (256K).

P in registers: the `m16n8k32` e4m3 A fragment holds keys `{4t..4t+3, 16+4t..16+4t+3}` per thread while the
S accumulator holds keys `{8j+2t, 8j+2t+1}`; storing V^T's keys in the order `VT_KEY_PERM` inside every
32-key block makes the two agree, so P never leaves registers (no smem tile, no ldmatrix, no warp syncs)
and 8 KB of smem is freed. `permute_vt_keys()` applies the permutation on the host (`index_select`, folded
into the V^T transpose copy); every caller that builds `vT8` must use it.

TMA kernel: 8 MMA warps + 4 producer warps (384 threads; only one producer warp issues TMA, the other three
exit after `setmaxnreg.dec`). `min_blocks_per_mp=1` is required, otherwise ptxas ignores `setmaxnreg` and the
kernel runs with 40 registers (175 GB of spills). The register split must balance exactly per CTA
(4 x (168 - 24) == 8 x (240 - 168)); 8 + 1 warp layouts (248/24, 232/40) deadlock in `setmaxnreg.inc`. Consumer
warps wait on "full" mbarriers and release stages per warp; the only CTA-wide barrier is pipeline init, so the
two warps of each SM sub-partition drift and overlap softmax with the other warp's MMAs, which is also why the
explicit S double buffer did not help. OOB rows/keys are zero-filled by TMA. Global operands are (rows, d, heads)
views: Q (Sq, D, H), K (Sk, D, Hk), V^T (D, Sk, Hk); output (Sq, H, D) bf16.

Rejected along the way: a dynamic `if kb == 0` K-buffer branch (32.7 ms, both code paths compiled);
unroll-by-2 with static parity (22.85 ms, more instructions than pointer arithmetic).

## Where the remaining tensor-pipe idle goes (2026-10-19 night, GPU0)

NCU on the TMA kernel (8K x 16K): tensor pipe active 84% of active cycles, but `sm__throughput` is 76.7% of
elapsed; the 0.91 ratio is the last-wave tail (1536 CTAs on 188 SMs = 8.17 waves). Measured by varying the query
rows (24 heads x m_blocks CTAs, 64K keys, dense):

| CTAs | waves | TFLOPS |
|---:|---:|---:|
| 1128 | 6.00 | 701 |
| 1152 | 6.13 | 654 |
| 1488 | 7.91 | 701 |
| 1512 | 8.04 | 670 |
| 1536 (8192 rows, the vLLM chunk) | 8.17 | 668 |
| 2256 (12032 rows) | 12.00 | 691 |

So ~5% of the 8192-row kernel time is wave quantisation; the fix is the prefill chunk size (6016 or 12032 rows
make heads x m_blocks a multiple of 188), not the kernel. The remaining ~10% is intrinsic to `mma.sync`
(ldmatrix -> MMA issue bubbles inside a warp). Three source-level attempts, all bit-identical and all rejected:

| attempt | switch | 8K x 64K | why it does not help |
|---|---|---:|---|
| FA3-style ping-pong between warp groups (named barriers) | `pingpong=True` | 19.3 ms vs 18.3 | with `mma.sync` a single warp cannot keep the pipe full; the second warp's interleaved MMAs were hiding the first warp's ldmatrix bubbles |
| explicit ldmatrix prefetch of slice k+1 before MMA k | `prefetch=True` | 18.4 vs 18.3 | ptxas already hoists the loads |
| skip the O rescale when the row max is unchanged | (reverted) | 18.1-18.5 vs 18.3 | the rescale is not on the critical path; the other warp's MMAs cover it |

The mbarrier (`SYNCS`) stall samples (8%) are the producer warp spinning in `producer_acquire`, not consumers
starving. Near-term headroom for this design is therefore the chunk-size alignment (~5%) and, for the last
partial wave, split-KV tiles; both are scheduling, not kernel code. End to end (task C, GPU0), prefill chunk
12032 vs 8192 gave +1.9/+1.9/+2.6% TTFT at 64K/128K/256K single-request (mean +1.8% over six cases).
