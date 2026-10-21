# Qwen3.8-27B on SM120：稀疏注意力部署栈与自定义 attention kernel

日期：2026-10-19。目标从任务契约转为"单卡 RTX PRO 6000 Blackwell SE（SM120）上的高效部署方案"：
计算保持 BF16，允许 FP8 KV cache，允许 sparse / mask attention，优化对象为 64K/128K/256K 输入。
脚本入口见 [scripts/qwen_sparse_attn_study](../../scripts/)
与 [scripts/qwen_sm120_attention](../kernels.md)。原始产物在
`runs/qwen38-sparse-attn-gpu0-20261019/`、`runs/qwen38-sparse-baseline-gpu0-20261019/`、
`runs/qwen38-ultralong-gpu7-20261019/`，不进入 Git。

## 硬件事实（GPU0/GPU7 实测，决定了方向）

| 事实 | 数值 | 含义 |
|---|---|---|
| 持续 BF16 GEMM 的功耗墙 | 600 W，SM 时钟降到约 2.15 GHz | 峰值约 414 TFLOPS，不是按 2.43 GHz 算的 467.8 |
| cuBLAS 对本模型 MLP 形状 | 392 TFLOPS，NCU HMMA 管线 94.8% | dense GEMM 没有 kernel 空间 |
| L2 读带宽 / DRAM 读带宽 | 约 6.5～7 TB/s / 1535 GB/s | 算力 bound 的 tile 需要 ≥52 FLOP/B，即 token tile ≥128 |
| DSMEM（cluster 共享内存） | 每 SM 3～6 GB/s，比 L2 经 TMA 慢 5～10 倍 | SM120 无硬件 DSMEM，跨 CTA 交换激活不可行 |
| 纯 HMMA 链 4 / 8 warp 每 SM | 422 / 455 TFLOPS | mma.sync 需要 2 warp/调度器才能喂满 tensor 管线 |
| vLLM 实际的 FA2 分页 attention（head_dim 256） | 305 TFLOPS，HMMA 66% | 这是 attention 侧可改进的起点 |

由此，先前的 MLP megakernel / DSMEM 路线被放弃（见 shared-mlp 报告），工作转向 attention。

## 注意力稀疏度（真实长输入，16 层 full attention）

对 64×64 block 的精确注意力质量，oracle 保留比例（覆盖 95% / 99% 质量）：技术文档 64K 0.34 / 0.62，
代码 64K 0.34 / 0.61，文档 16K 0.46 / 0.72；任务原合成记录 64K 0.65 / 0.87，不具代表性。
384 个 head 中只有 6～11 个强局部头，head 局部性在文档与代码间相关 0.88～0.90、与合成记录 0.70，
因此采用按 head、按 query block 的动态选块。块均值打分的排序质量与精确行抽样只差约 1.5 个点。
数据：`runs/qwen38-sparse-attn-gpu0-20261019/summary.md`、`head-patterns.md`。

## 稀疏 prefill 的实现

- 胶水层 `sm120_attn/sparse_prefill.py` 挂在 vLLM FlashAttention backend 上：仅对 q_len ≥1024、kv_len ≥8192 的
  prefill chunk 启用；块均值打分、按 head 的 τ 质量选块、对角与 sink 块常驻、定长块列表（无主机同步）；
  K 的块均值按 (层, 首页) 缓存并由当前 chunk 增量更新；混合批次按请求拆分，不合格的请求走逐请求 dense。
- 自定义 SM120 kernel `kai_fa_bf16.py`：从 CUTLASS 的 Ampere FA2 CuTe DSL 示例改造，sm_120 共享内存容量、
  GQA 映射、chunked prefill 的 q 偏移因果掩码、按 (head, m_block) 的块列表遍历。dense 8K×64K：
  36.4 ms = 340 TFLOPS（FA2 非分页 338，vLLM 分页路径 305）；稀疏耗时随保留比例线性缩放（40% → 15.1 ms）。
  任意 q 起始位置与残块对 fp32 参考误差约 3e-3。
- 被否决的 v3（split-D warp 对）：正确但慢 28%。K/V 的 smem 读取量由 16 行组的数量决定，拆分 head_dim
  只省掉 Q 重读（约 11% 的 ldmatrix），却增加 4 亿次标量 smem 读写与配对 barrier（HMMA 57% 对 74%）。
  去掉 v2 的 softmax 指数运算只快 3%，剩余 stall 来自单 warp 的 ldmatrix→HMMA 依赖链。

## FP8（e4m3）attention kernel

`kai_fa_fp8.py`：Q/K 为 e4m3，V 以 d-major 的转置布局提供（B 操作数必须 K-major，8 位没有
`ldmatrix.trans`），`warp.MmaFP8Op` m16n8k32、FP32 累加；P 乘 448 后以 e4m3 写入 CTA 私有的 smem tile，
再用 ldmatrix 读回作为 A 操作数；量化尺度折叠进 softmax 的 log2 系数与输出系数。FP8 使 tile 减半，
"128 行 query block、256 线程"的 8 warp 配置在 72 KB smem 内可行（BF16 需要 128 KB），
消除了 4 warp 下的发射率上限和寄存器溢出：

| 8K×16K，NCU 锁频 | 寄存器 | local 访存 | HMMA | 用时 |
|---|---:|---:|---:|---:|
| BF16 v2（m64，128 线程） | 248 | 0 | 74.4% | 9.36 ms |
| FP8 m64，128 线程 | 255 | 6.0 GB | 46.8% | 7.44 ms |
| FP8 m128，256 线程 | 255 | 24 MB | 57.2%（FP8 口径） | 6.10 ms |

8K×64K（GPU0，有干扰）：BF16 v2 39.8 ms，FP8 m128 25.7 ms。胶水层 `kernel: sm120fp8`：每请求每层
的尺度取首个 chunk 的 amax×1.5，每层每 chunk 用 bf16 运算量化 K/V/Q 并转置出 Vᵀ，128 行块的列表取
相邻两个 64 行块的并集。真实激活上 e4m3 的输出误差 1.1%～2.5%（Q/K/V/P 全 FP8 约 1.2%～2.5%）。

### FP8 kernel 优化轮（2026-10-19 晚，GPU7，8K×64K 因果、dense）

| 步骤 | 文件 / 开关 | 8K×64K | NCU 8K×16K（锁频） |
|---|---|---:|---|
| 单缓冲 K/V，P 经 smem | `kai_fa_fp8（单缓冲版本，未收录）` | 23.00 ms（538 TF） | 6.10 ms，255 寄存器，HMMA 57% |
| + K 双缓冲（指针算术选缓冲） | `kai_fa_fp8（K 双缓冲版本，未收录）` | 21.68 ms（571 TF） | 5.76 ms，2.4 GB local 读 |
| + P 在寄存器内转换（Vᵀ 键序按 32 键块置换） | `kai_fa_fp8.py`（cp.async 版） | 19.47 ms（635 TF） | 5.10 ms，247 寄存器，0 local，tensor 管线 75% |
| + TMA 加载、K/Vᵀ 各 2 级 mbarrier 流水、生产者 warp 组 | `kai_fa_fp8_tma.py` | 17.98 ms（688 TF） | 4.56 ms，tensor 管线 84%，0 local |
| + warp 内 S 双缓冲（先发 QKᵀ(n+1) 再做 softmax(n)） | `s_pipeline=True` | 19.07 ms | 变慢，默认关闭 |

其他尺寸（TMA 版对 cp.async 版）：8K×128K 37.85 对 40.41 ms；4K×256K 41.79 对 45.41 ms；40% 块稀疏
11.5 对 13.0（64K）、24.3 对 27.1（128K）、26.8 对 30.1 ms（256K）。胶水层 `kernel: sm120fp8` 默认走 TMA 版
（`fp8_pipeline: tma|cpasync`）。TMA 版为 8 个 MMA warp + 4 个生产者 warp（384 线程，启动 168 寄存器，
`setmaxnreg` 调成 240/24，必须按 CTA 精确平衡，8+1 warp 的布局会死锁；`min_blocks_per_mp=1` 不设则 ptxas
忽略 setmaxnreg、按 40 寄存器编译并溢出 175 GB）。消费者 warp 只等 "full" mbarrier、按 warp 释放缓冲，
整个主循环没有 CTA 级 barrier，同一 SM 子分区的两个 warp 自然错相，这也是显式 S 双缓冲没有收益的原因。
P 留在寄存器：m16n8k32 e4m3 的 A 片段每线程持有键 {4t..4t+3, 16+4t..}，S 累加器持有 {8j+2t, 8j+2t+1}，
把 Vᵀ 每 32 键块内的键按 `VT_KEY_PERM` 重排后二者一致，P 不再经过 smem（省 8 KB、两次 warp 同步和一次 ldmatrix）。
`s_pipeline`（S 双缓冲）与早先的 v3 一样作为否定结果保留。

**MXF8 硬件块缩放（`MmaMXF8Op`，每 32 元素一个 UE8M0 尺度）没有实现**：用真实激活在 torch 中仿真
（`fp8_numerics_probe.py`，`validation/fp8-numerics-mx.json`），attention 输出相对 L2 误差（层 11/35/63）：

| 量化方案 | 层 11 | 层 35 | 层 63 |
|---|---:|---:|---:|
| 生产方案：Q/K/V 按张量 amax 尺度，P 按行 ×448 | 2.73% | 3.21% | 1.33% |
| 软件细粒度：Q/K 按 token、V 按通道、P 按行 | 2.56% | 2.69% | 1.31% |
| MXF8 全流程：Q/K 按 (token, 32-d 块)、V 按 (通道, 32 键块)、P 按 (行, 32 键块)，2 的幂尺度 | 3.04% | 3.75% | 1.53% |
| 仅 K/V 按张量 | 2.31% | 2.32% | 1.09% |
| 仅 K/V MX32 | 2.36% | 2.80% | 1.09% |

块缩放的 2 的幂尺度平均浪费半个比特的量程，而误差本身由 e4m3 的 3 位尾数主导，所以 MXF8 比现在的实数尺度
更差；细粒度软件尺度也只减少 5%～15% 的误差。FP8 attention 的精度下限就是 e4m3 本身，要再降误差只能把
PV 保持 BF16（混合精度 kernel）或者不用 FP8 attention——这正是 A/B/C 对比实验要回答的问题。

### FP8 GEMM 变体（A/B/C 对比实验，进行中）

`task-fp8gemm-sm120`（B：BF16 稀疏 attention + FP8 GEMM）与 `task-fp8gemm-sm120fp8`（C：FP8 稀疏 attention +
FP8 GEMM）从 `task-stock` 派生（不带 BF16 gate/SwiGLU 融合，否则会绕过量化线性层），`serving.json` 加
`quantization: fp8`（vLLM Fp8 W8A8，权重加载时量化、激活按 token 动态尺度、CUTLASS scaled_mm；SM120 上
实测 `torch._scaled_mm` 对本模型 MLP 形状 1.84～1.89 倍）。误差评测用 `teacher_forced_eval.py`：把 BF16 dense
的贪心续写（256 token）接在 prompt 后，以 `prompt_logprobs` 取每个参考 token 的 NLL 与 top-1 一致率。
队列脚本 `run_fp8gemm_queue_gpu7.sh`：先 needle 验证 B/C，再对 stock / sparse-sm120 / sparse-sm120fp8(TMA) /
B / C 做 teacher-forced 评分，最后测 TTFT。

### A/B/C 输出误差结果（teacher-forced，GPU7，2026-10-19）

参考为 BF16 dense（stock）。每个 prompt 接上 stock 自己的 256 token 贪心续写，报告参考 token 的平均 NLL
（括号内为相对 stock 的差）、top-1 一致率（该 token 是否仍是本变体的 argmax）和 needle 命中：

| variant | 65536 tok NLL (Δ) / top-1 / needle | 131072 tok NLL (Δ) / top-1 / needle | 260096 tok NLL (Δ) / top-1 / needle | mean Δ NLL | mean top-1 |
|---|---|---|---|---:|---:|
| stock (BF16 dense) | 0.2588 (+0.0000) / 99.6% / ? | 0.2352 (+0.0000) / 100.0% / ? | 0.2160 (+0.0000) / 100.0% / ? | +0.0000 | 99.9% |
| BF16 sparse attn (sparse-sm120) | 0.2568 (-0.0020) / 100.0% / ✓ | 0.2360 (+0.0008) / 99.2% / ✓ | 0.2169 (+0.0010) / 97.3% / ✓ | -0.0001 | 98.8% |
| A: FP8 sparse attn + BF16 GEMM | 0.2613 (+0.0025) / 99.2% / ✓ | 0.2397 (+0.0045) / 98.8% / ✓ | 0.2185 (+0.0025) / 98.0% / ✓ | +0.0032 | 98.7% |
| B: BF16 sparse attn + FP8 GEMM | 0.2671 (+0.0083) / 95.7% / ✓ | 0.2348 (-0.0004) / 98.8% / ✓ | 0.2151 (-0.0009) / 97.7% / ✓ | +0.0023 | 97.4% |
| C: FP8 sparse attn + FP8 GEMM | 0.2669 (+0.0081) / 97.3% / ✓ | 0.2376 (+0.0024) / 98.4% / ✓ | 0.2132 (-0.0028) / 97.3% / ✓ | +0.0026 | 97.7% |

每 token logprob 相对 stock 的扰动（平均 / 最大）：

| 变体 | 64K 平均/最大 \|Δlogprob\| | 128K | 260096 |
|---|---|---|---|
| BF16 稀疏 attention | 0.0150 / 0.251 | 0.0145 / 0.206 | 0.0533 / 0.713 |
| A：FP8 稀疏 attention + BF16 GEMM | 0.0149 / 0.145 | 0.0145 / 0.191 | 0.0456 / 0.713 |
| B：BF16 稀疏 attention + FP8 GEMM | 0.0294 / 0.358 | 0.0278 / 0.310 | 0.0544 / 0.756 |
| C：FP8 稀疏 attention + FP8 GEMM | 0.0279 / 0.337 | 0.0240 / 0.370 | 0.0519 / 0.891 |

结论：
- 三个变体的 NLL 差都在 0.01 nat 以内，top-1 一致率 97%～99%，needle 全部命中；按这个口径都可用。
- **FP8 attention（A）的误差小于 FP8 GEMM（B）**：A 的每 token logprob 扰动（0.015 / 0.015 / 0.046）与纯 BF16
  稀疏 attention（0.015 / 0.015 / 0.053）几乎相同，即在稀疏化之上 FP8 attention 几乎不再增加误差；
  B 在 64K/128K 的扰动约为 A 的两倍（0.029 / 0.028），top-1 一致率也更低（95.7% 对 99.2%）。
- **全 FP8（C）≈ B**：误差由 FP8 GEMM 主导，把 attention 也换成 FP8 不再额外增加（0.028 / 0.024 / 0.052）。
- 256K 上所有变体的扰动都升到 0.05 左右，其中稀疏化本身就是 0.053，FP8 的贡献不可分辨。

## 质量验证（贪心生成，needle 检索）

| 配置 | 64K | 128K | 256K | 备注 |
|---|---|---|---|---|
| dense | needle ✓ | ✓ | ✓ | 参考 |
| sparse τ=0.95（按 head 保留比例） | ✓，0.50 | ✓，0.48 | ✓，0.44 | 首 token 一致，分叉在 130～363 token 后 |
| sparse τ=0.90 | ✓，0.37 | ✓，0.35 | ✓，0.32 | |
| sparse τ=0.85 | ✓，0.29 | ✓，0.27 | ✓，0.24 | |
| sparse τ=0.95 + e4m3 舍入 Q/K/V（按张量 / 按 token，BF16 kernel 仿真） | ✓ | ✓ | ✓ | 分叉点与纯稀疏相同 |
| sparse τ=0.95 + FP8 kernel（m128） | ✓ | ✓ | ✓ | 首 token 一致，分叉 66/159/141 token |

两条长请求同时在飞（混合批次）时，输出与单请求逐 token 一致。needle 只是弱检验：贪心分叉
是近似计算的正常表现；上线前应补 RULER/NIAH 类长文本评测。真实激活上的 e4m3 探针：K/V 量化后
注意力输出相对 L2 误差 1.1%～2.4%，Q/K/V/P 全 FP8 约 1.2%～2.5%（`validation/fp8-numerics.json`）。

## TTFT 基线

GPU0（无干扰时段，search split，每 case warmup + 2 次，等权平均 9 组 16K～64K）：
stock 11781 ms，gate/SwiGLU 融合 11406（-3.2%），稀疏 FlashInfer 并集路径 11816（+0.3%，胶水开销，已退役）。

GPU7，64K 三个并发档：

| case | stock | fused | sparse-sm120 | sparse-sm120 对 stock |
|---|---:|---:|---:|---:|
| 64K c1 | 11940 | 11607 | 10920 | -8.5% |
| 64K c2 | 18546 | 17984 | 17701 | -4.6% |
| 64K c4 | 31766 | 30797 | 30712 | -3.3% |

并发 2/4 的收益低于并发 1 是因为当时的整批回退规则；按请求拆分的胶水已就绪，尚未重新测量。
GPU7，64K/128K/256K 长文档任务（`runs/qwen38-ultralong-gpu7-20261019`，search split，warmup + 2 次，
首 token 检查全部通过）：

| case | stock | fused | sparse-sm120 | sparse-sm120fp8 (cp.async) | sparse-sm120fp8 (TMA) = A | fp8gemm-sm120 = B | fp8gemm-sm120fp8 = C | A vs stock | B vs stock | C vs stock | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| in65536-out65536-c1 | 12063 | 11715 | 10540 | 10144 | 9994 | 7215 | 6677 | +17.1% | +40.2% | +44.6% | ok |
| in65536-out65536-c2 | 18698 | 18137 | 16425 | 15986 | 15780 | 11168 | 10481 | +15.6% | +40.3% | +43.9% | ok |
| in131072-out131072-c1 | 30171 | 29430 | 24014 | 22628 | 21973 | 17329 | 15203 | +27.2% | +42.6% | +49.6% | ok |
| in131072-out131072-c2 | 45771 | 44682 | 36148 | 34365 | 33467 | 25768 | 23144 | +26.9% | +43.7% | +49.4% | ok |
| in260096-out2048-c1 | 81662 | 80373 | 57597 | 52025 | 49446 | 44330 | 36156 | +39.5% | +45.7% | +55.7% | ok |
| in260096-out2048-c2 | 172197 | 121635 | 82446 | 76414 | 73331 | 62222 | 52799 | +57.4% | +63.9% | +69.3% | A、C 各有一次重复首 token 不同 |
| equal-weight mean | 60094 | 50996 | 37861 | 35260 | 33998 | 28006 | 24077 | +43.4% | +53.4% | +59.9% | |

（`runs/qwen38-ultralong-gpu7-20261019/summary-fp8gemm.md`，GPU7，search split，warmup + 2 次。A/B/C 即
上文误差实验的三个变体。）

**并发 2 下"首 token 不同"的真正原因是胶水层的缓存 bug，已修。** 块均值缓存和 FP8 尺度缓存都以
(层, 首个物理页) 标识请求；并发时调度器可能把一条新请求的首个 chunk 切得很短（不满足 min_q/min_kv），
该 chunk 走 dense 路径、不触碰缓存，而它的首页是从上一条请求回收来的，后面的 chunk 便沿用了上一条请求的
块均值（选错块）和 FP8 尺度。单请求时首个 chunk 总是完整的 8192 行，所以从不复现。修复
（`sm120_attn/sparse_prefill.py`）：缓存项记录上一 chunk 的结束位置，不连续即视为陈旧，块均值从本请求的完整 K 历史
重建，尺度重新取；`test_glue_v2.py stale` 覆盖该场景。修复后在 GPU0 复测 task C（`summary-cachefix-gpu0.md`）：
六个 case 的首 token 检查全部通过（修复前 256K 并发 2 每轮必挂一次），TTFT 在 ±0.2% 内不变。

256K 并发 2 的 stock 一格受 KV cache 容量影响而非 kernel：stock 的容量为 515,756 token，放不下两条 260,096
的序列（需 520,192），第二个请求要等第一个请求 decode 完 2048 个 token；fused/sparse 的容量 520,410
恰好放得下。其余格子都是同一 GPU、同一配置下的 kernel/调度差异。

### A/B/C 结论

- TTFT：FP8 GEMM 的收益大于 FP8 attention。64K 时 attention 只占 TTFT 约 22%，B（FP8 GEMM）一项就把 64K c1
  从 10.54 s（BF16 稀疏）压到 7.22 s；全 FP8（C）达到 6.68 s（对 stock 1.81×）、128K 15.20 s（1.98×）、
  256K 36.16 s（2.26×）。
- 误差：FP8 attention 几乎不增加误差（与 BF16 稀疏同量级），FP8 GEMM 的每 token logprob 扰动约两倍，
  全 FP8 ≈ 只 FP8 GEMM。三者 NLL 差都在 0.01 nat 内，needle 全中。
- 因此推荐 C（全 FP8）作为部署配置；如果对输出一致性更敏感，A（FP8 attention + BF16 GEMM）是误差最小的
  加速选项，B 不值得单独选。

### Kernel 管线空闲的去向（2026-10-19 夜，GPU0）

TMA 版的 tensor 管线在活跃周期内 84%，但按 elapsed 只有 76.7%，比值 0.91 就是 1536 个 CTA 在 188 个 SM 上
8.17 波的尾部损失。按 FLOP 归一化的阶梯实验（24 头 × m_block 个 CTA，64K 键）：1128/1488/2256 个 CTA
（整波）≈700 TFLOPS，1536 个（8192 行，vLLM 的 chunk）668 TFLOPS，差约 5%。所以这 5% 的解法是 prefill
chunk 取 12032 或 6016 行而不是改 kernel。三种源码层面的尝试（warp 组乒乓调度、显式 ldmatrix 预取、
最大值不变时跳过 O 缩放）都逐位一致但不提速：`mma.sync` 单 warp 的发射气泡本来就由同一子分区的另一个 warp
填补，调度上再做文章没有空间；mbarrier 等待的采样是生产者 warp 在空转，不是消费者缺数据。
详见 `docs/kernels.md`。

端到端验证（GPU0，task C，`summary-chunk-gpu0.md`）：prefill chunk 12032 对 8192，并发 1 的 64K/128K/256K
分别快 1.9%/1.9%/2.6%，并发 2 为 −2.7%/+1.2%/+2.5%（64K 并发 2 时两条请求各分到 6016 行，调度顺序变化
抵消了收益），等权平均 +1.8%。部署时单请求长 prompt 为主可取 12032。

## 下一步（按预期收益）

1. **FP8 kernel 的落地**：把 KV cache 本身改为 FP8（避免每层每 chunk 的 gather + 量化 + 转置）；
   decode 路径也要能读 FP8 cache；FP8 尺度可改为按层静态校准（省掉每请求的 amax 同步）。
   MXF8 块缩放已由仿真否定（见上）。
2. **τ 与质量门**：τ=0.85 的 256K 保留 24%，需要更广的质量评测后决定。
3. **分页 K/V 直读**（去掉每层每 chunk 的 gather）与选块融合进 KV 写入路径。
4. GEMM 周边访存型小 kernel 融合（norm、GDN conv/post、门控），约 2～3%。
