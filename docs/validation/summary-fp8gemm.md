| case | stock ms | fused ms | sparse-sm120 ms | sparse-sm120fp8 ms | sparse-sm120fp8-tma ms | fp8gemm-sm120 ms | fp8gemm-sm120fp8 ms | fused vs stock | sparse-sm120 vs stock | sparse-sm120fp8 vs stock | sparse-sm120fp8-tma vs stock | fp8gemm-sm120 vs stock | fp8gemm-sm120fp8 vs stock | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| in65536-out65536-c1 | 12063 | 11715 | 10540 | 10144 | 9994 | 7215 | 6677 | +2.9% | +12.6% | +15.9% | +17.1% | +40.2% | +44.6% | ok / ok / ok / ok / ok / ok / ok |
| in65536-out65536-c2 | 18698 | 18137 | 16425 | 15986 | 15780 | 11168 | 10481 | +3.0% | +12.2% | +14.5% | +15.6% | +40.3% | +43.9% | ok / ok / ok / ok / ok / ok / ok |
| in131072-out131072-c1 | 30171 | 29430 | 24014 | 22628 | 21973 | 17329 | 15203 | +2.5% | +20.4% | +25.0% | +27.2% | +42.6% | +49.6% | ok / ok / ok / ok / ok / ok / ok |
| in131072-out131072-c2 | 45771 | 44682 | 36148 | 34365 | 33467 | 25768 | 23144 | +2.4% | +21.0% | +24.9% | +26.9% | +43.7% | +49.4% | ok / ok / ok / ok / ok / ok / ok |
| in260096-out2048-c1 | 81662 | 80373 | 57597 | 52025 | 49446 | 44330 | 36156 | +1.6% | +29.5% | +36.3% | +39.5% | +45.7% | +55.7% | ok / ok / ok / ok / ok / ok / ok |
| in260096-out2048-c2 | 172197 | 121635 | 82446 | 76414 | 73331 | 62222 | 52799 | +29.4% | +52.1% | +55.6% | +57.4% | +63.9% | +69.3% | ok / ok / ok / ok / FAIL / ok / FAIL |
| equal-weight mean | 60094 | 50996 | 37861 | 35260 | 33998 | 28006 | 24077 | +15.1% | +37.0% | +41.3% | +43.4% | +53.4% | +59.9% | |

Notes (2026-10-19 evening):
- sparse-sm120fp8-tma: FP8 sparse attention with the TMA/mbarrier kernel (`kai_sm120_fa_fp8_tma.py`, now the default
  for `kernel: sm120fp8`); fp8gemm-sm120 (B): BF16 sparse attention + vLLM `--quantization fp8` GEMM (W8A8 dynamic);
  fp8gemm-sm120fp8 (C): FP8 sparse attention + FP8 GEMM. B/C derive from task-stock (no BF16 MLP fusion).
- FAIL = the first generated token differed from the stock oracle in one of the two repetitions (256K, concurrency 2,
  one request); warmup and the other repetition matched. Both failures are FP8-attention variants whose per-request
  scales come from the first prefill chunk, which depends on the scheduler's chunk split under concurrency.
- 256K c2 stock is KV-capacity bound (second request waits for the first to decode); fused is the fair reference.
- Output-error (teacher-forced vs BF16 dense) in validation/fp8-variants-error.md.
