# 用 flydsl_block 改写的 GEMM 教程 kernel

gfx942-flydsl-tutorials 仓库 `kernels/gemm` 里的 v0 / v0_preshuffle / v1 / v2，用 block 级 API（`blk.load` / `blk.store` / `blk.dot` / `blk.Shared`）改写。接口和原版相同：`build_*(BM, BN, OCCUPANCY, M, N, K, ...)` 返回 launcher，开关也相同（`PRESHUFFLE` / `ILP` / `K32`，`SCHED` / `K32`）。v1 去掉了 `SWIZZLE` 开关，swizzle 统一由 `blk.Shared` 实现。

每个读、写、乘都是一次整块 tile 的 block op，直接写在调度里该出现的位置；原版里的 `gr` / `lw` / `lr` / `dot` 这类 helper 不再需要。只保留两种 helper：调度 hint（`interleave`）和 v2 中被循环与两个 peel 步共用的 K-step（`kstep`）。

| 文件 | 原版 | 改写要点 |
|---|---|---|
| [v0.py](v0.py) | 1 wave/CTA，global 直接读成 MFMA 操作数 | `Mma(tm, "A"/"B"/"C")` + K 置换 `(4,4,2):(1,8,4)`：每个 lane 8 个连续 K，一条 16 B load 喂两条 16x16x16；acc 是循环变量 |
| [v0_preshuffle.py](v0_preshuffle.py) | v0 + 寄存器双缓冲 + 输入预重排 | 预重排后的 A/B 用嵌套视图 `((16, R/16), (8, 4, NK))` 描述，kernel 里仍按逻辑 (m, k) tile 读。ILP 版每个 K-step 一次 A load、一次 B load、一次 `dot`；无 ILP 版按 16 行块读 A（见下） |
| [v1.py](v1.py) | 4 wave，global → 寄存器 → LDS → MFMA | staging 用 `Blocked((BM, 64), (1, 8), (8, 8), (4, 1))`；LDS 是 `Shared(vec=8, max_phase=8)`（`chunk ^= row % 8`）；M/N 置换让每个 wave 拿连续的 WM×WN；LDS → MFMA 按 32 K 两半做（见下） |
| [v2.py](v2.py) | v1 + 软件流水（4 个 region，GR/LW/LR 交错） | 半块按角色命名（`gA_top`、`sB_left` …），每个 region 是一次 `dot` 加三个 block op，读起来就是原版的调度表 |
| [v3.py](v3.py) | 无（新增） | v2 K32 + LDS DMA（`blk.copy`，global 直接进 LDS，没有寄存器中转和 `ds_write`）+ 两级 LDS（128 KB）+ XCD 感知的 tile 顺序，结构同 gfx950-gluon-tutorials 的 a16w16 v8/v9。输出与 v2_k32 逐位相同 |
| [compare.py](compare.py) | — | 与原版逐个对比：输出逐位相同，再交替计时 |
| [bench.py](bench.py) | — | 没有原版的 kernel（v3）互相对比：先与 torch 比对，再用 CUDA graph（20 次背靠背 launch）交替计时 |

## 运行

```bash
python examples/compare.py                      # 全部
python examples/compare.py v1 v2_k32            # 指定
TUTORIAL_GEMM=/path/to/kernels/gemm python examples/compare.py
python examples/bench.py                        # v2_k32 vs v3, 4096x4096x8192
MN=8192 K=8192 python examples/bench.py v2_k32 v3 v3_grid2d
```

形状按 CU 数取（256 CU：8192³），编译参数与原版 bench.py 相同（`opt_level=2`、`amdgpu-mfma-vgpr-form=False`）。每个 case 先检查输出与原版逐位相同（两者的累加顺序相同），再交替计时 3 轮：热身 40 次，取 30 次的中位数，输入在 4 份拷贝间轮换。

## 结果

gfx950（256 CU），FlyDSL 0.3.4.1，8192×8192×8192 bf16 → fp32：

| case | 原版 TFLOPS | 改写 TFLOPS | 改写 / 原版（时间） | 输出逐位相同 |
|---|---|---|---|---|
| v0_64x64 | 282.6 | 282.7 | 0.999 | ✓ |
| v0_128x128 | 519.4 | 521.6 | 0.996 | ✓ |
| v0_ilp | 509.8 | 522.5 | 0.976 | ✓ |
| v0_preshuffle_noilp | 648.4 | 656.0 | 0.988 | ✓ |
| v0_preshuffle | 754.2 | 755.7 | 0.998 | ✓ |
| v0_preshuffle_k32 | 1027.7 | 1027.4 | 1.000 | ✓ |
| v1 | 578.1 | 584.9 | 0.988 | ✓ |
| v2_nosched | 680.0 | 690.3 | 0.985 | ✓ |
| v2 | 741.8 | 744.1 | 0.997 | ✓ |
| v2_k32 | 1012.5 | 1029.7 | 0.983 | ✓ |

- FlyDSL 0.5.0 上抽测 v0_128x128、v0_preshuffle_noilp、v0_preshuffle、v1、v2、v2_k32：比值 0.981～1.009，输出逐位相同。
- gfx942 只能交叉编译（本机是 gfx950）：除 K32 外的 8 个 case 都能编译，MFMA / global load / store 条数与原版相同，地址计算的 VALU 更少，没有 spill。原版 v2 在 gfx942 上 spill 100～132 B。
- v1 的 LDS 访问与手写 swizzle 一致：循环内 32 条 `ds_read_b128`（其中 26 条带立即数偏移）、16 条 `ds_write_b128`，没有 XOR，2 个 barrier。

## 为了同速做的调度处理

block API 只决定数据怎么放，不决定指令顺序。下面几处顺序由 kernel 自己用 `rocdl.sched_barrier(0)` 或 `traversal_order` 固定，每处都在代码里注释了原因：

- **MFMA 顺序**：原版是 for i: for j: for k，对应 `traversal_order=KNM`（首字母是最内层）。整块 `dot` 加上这个顺序就够了：v0_preshuffle 的 ILP 版和 v2 都不需要再拆 tile。
- **v0、v0_preshuffle（无 ILP）**：load 和 MFMA 之间放 `sched_barrier(0)`。否则 post-RA 调度把部分 load 挪到 MFMA 中间，紧接着的 MFMA 要等这条 load，128x128 慢 38%。
- **v0_preshuffle 无 ILP 要拆 A**：没有预取时，每条 MFMA 都要一块 A 和一块 B，整块 load 时第一条 MFMA 要等第 9 条 load。按使用顺序 a[0]、b、a[1..] 发出（load 组之间用 `sched_barrier(0)` 固定）才同速；整块写法慢 2%～6%。
- **v1 按 32 K 两半做 LDS → MFMA**：64 K 一次 load + `dot` 时寄存器用满（512 VGPR / 256 AGPR），循环里约 650 条 `v_accvgpr` 拷贝，慢 9.5%。
- **v2 的 prologue**：第 1 个 tile 的 load 要按循环里的顺序发（B_left、A_top、A_bot、B_right）。waitcnt pass 会合并循环入口和循环体里在途的 load；入口顺序不同，循环里每个 `vmcnt` 都会变严。
- **v2 K32 的 epilogue**：C 的 store 放在最后一批 MFMA 之后。混在 MFMA 里时 store 会成串发出，写回变慢；16x16x16 不需要这样处理。

## v3：LDS DMA + 两级 LDS

对照 gfx950-gluon-tutorials 的 a16w16（v9：256x256x64、4 wave、`buffer_load_to_lds`、两级 LDS、XCD 重映射）。同一台 MI350X（功耗上限 1000 W），4096×4096×8192 bf16，CUDA graph 背靠背 launch，rocprofv3 kernel trace 取 1000 次里后 500 次的平均。每个 K-step 的周期（与频率无关）由 GRBM_GUI_ACTIVE 在 K=4096 / 16384 两点差分得到；loop MFMA 效率由 ATT 得到（一个 CU）：

| kernel | 时间 | TFLOPS | 每个 K-step 周期（理想 2048） | loop MFMA 效率（ATT） |
|---|---|---|---|---|
| v2_k32 | 268.0 us | 1026 | 2758 | 75.9% |
| **v3** | **229.3 us** | **1199** | **2120** | **98.3%** |
| a16w16 v9（Gluon，`llir+amdgcnas`，bf16 输出） | 222.3 us | 1237 | 2120 | 98.4% |
| hipBLASLt（`torch.matmul`，bf16 输出） | 235.3 us | 1168 | — | — |

- hot loop 与 v9 持平：整卡每个 K-step 都是 2120 周期；ATT 每次迭代（2 个 K-step）4144 vs 4131 周期，背靠背运行中途取样 4170 vs 4157。MFMA / `ds_read` / DMA / barrier 条数相同，v3 共 403 条指令（27 条 SALU），v9 423 条（42 条 SALU）。
- 时间还差 3.1%，不在 hot loop 的指令上：
  - loop 外：v3 输出 fp32，写回字节是 v9 的两倍；改成 bf16 输出是 225.9 us。
  - 功耗：背靠背时两者都顶着 1000 W，周期相同、每个 K-step 的时间却差 2.1%（1704 vs 1669 ns）；kernel 之间留空闲（不撞功耗上限）时只差 0.7%（1334 vs 1325 ns）。
- 其他形状（`bench.py`，v2_k32 → v3 TFLOPS）：8192³ 1034 → 1186，4096²×4096 963 → 1132，4096²×16384 1044 → 1215。FlyDSL 0.5.0 上 4096²×8192 是 1014 → 1182。
- a16w16 自带的 `run_perf_table.py --rocprof` 给 v9 报 1555 TFLOPS：Triton 每次 launch 要 188 us CPU，kernel 之间空 126 us，GPU 只忙 58%，功耗低、频率高。比较 kernel 要用同样的 launch 方式。

### hot loop 怎么做到的

16x16x32 MFMA 每 16 周期一条，自身发射占 4 周期，两条 MFMA 之间只剩 12 周期给别的指令。ATT 实测：`ds_read` 8、DMA 约 10、SALU 4、`s_waitcnt` 4。一个窗口放得下 1 个 `ds_read` + 1 条 SALU，或 1 个 DMA，或 3 条 SALU；超了 MFMA 就空转。下面每一条都是在让窗口不超预算（都在代码里注释了）：

- **每个 region 先写 LR，再写 DOT**。反过来写时，LR 的目标寄存器会复用本 region 最后一次读的操作数（region 1 的 `b_left` 等）：每条 `ds_read` 都要等它覆盖的那条 MFMA，DMA 被挤到 region 末尾。改顺序后 1116 → 1147 TFLOPS。
- **每个 region 用自己的 sched group id，结尾放 `sched_barrier(0)`**，hint 不会跨 region 配对。region 开头放 4 条 MFMA，比 2 条快 0.8%。
- **一个 copy 只写一次 M0**（`blk.copy` 内部）：每个 wave 填一段连续的行，4 条 DMA 共用 M0，每条的 LDS 步长放进指令的 12 bit 立即数偏移。原来每条 DMA 都和下一条的 M0 写挤在一个窗口里，每次迭代多 50 周期，现在多 10 周期。
- **每个 K-step 每个矩阵只建一个 buffer descriptor**（`halves`）：上下两半共用，半块的偏移并进循环不变的每线程偏移，LLVM 每次迭代只推进一次 A/B 基址。原来每半块一个 descriptor，约 7 条 SALU，落进 LR 的窗口，每次迭代多 62 周期。循环里的 SALU 从 89 条降到 27 条。
- **循环最后一个 region 允许 SALU 越过它的 fence**：循环计数和 A/B 基址更新（7 条 SALU）分散到 region 里，不再全部堆在跳转前的一个窗口（36 → 12 周期）。
- **tile 顺序**：连续的 tile 放在同一个 XCD，每次 `GROUP_M=4` 行 tile（a16w16 v9 的做法），4096² 快 2.5%，8192² 不变。
- 试过但没有变快的：DMA 的 K 偏移走 `soffset`（DMA 发射从 4 周期变成 8 周期）；DMA 在 4 条 MFMA 里的位置前后挪一条（慢 1.5%）；每次迭代做 4 个 K-step（寄存器分配崩掉，循环里 448 条 `v_accvgpr`）。
- M0、descriptor、循环末尾这三处合起来：ATT 每次迭代 4276 → 4144 周期，整卡每个 K-step 2156 → 2120 周期，时间 230.6 → 229.3 us。时间降得比周期少，推测是顶着功耗上限时，停顿少了频率也略降。
