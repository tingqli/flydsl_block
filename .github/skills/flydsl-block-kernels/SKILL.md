---
name: flydsl-block-kernels
description: 'Use when writing, porting or refactoring a FlyDSL kernel with flydsl_block (blk.load / blk.store / blk.copy / blk.dot / blk.Shared, Mma layouts), e.g. rewriting a thread-level kernel at block level, adding an LDS DMA multi-stage pipeline, or when a block-level kernel is slower than its original or a reference.'
---

# 用 flydsl_block 写 kernel

一个 block op 就是一个**整块** tile 的操作：一次 `blk.load` 发出这个 tile 的全部 load，一次 `blk.dot` 发出全部 MFMA。kernel 就是按调度表排好的一串整块 op。

## 步骤

1. **先写整块版本。** 每个 global / LDS 读写和每个 MMA 都写成一次整块调用，内联在调度里它出现的位置。tile 和 LDS slot 按角色命名（`gA_top`、`sB_left`、`a_top`），代码读起来就是调度表。thread 级原版里 `gr` / `lw` / `lr` / `dot` 这类 helper 各包了几十行线程级代码；到 block 版每个只剩一次调用，直接写这次调用。
   完成标准：kernel 里只剩这两类 helper：
   - 调度 hint 序列（`rocdl.sched_mfma` / `sched_vmem` / `sched_dsrd` / `sched_dswr`），例如 `interleave(...)`；
   - 被循环和 peel 多次实例化、只差静态开关的流水 stage 体，例如 v2 的 `kstep(..., last)`。
2. **用调度控制顺序。** MFMA 顺序用 `blk.dot(..., traversal_order=)`，首字母是最内层：原版 `for i: for j: for k` 对应 `KNM`。load 相对 MFMA 的位置用 `rocdl.sched_barrier(0)` 或 sched_group hint 固定。
3. **和原版对比。** 先看输出是否逐位相同，再交替计时（`examples/compare.py`）。慢了就看 ISA：VGPR / AGPR、scratch、循环里的 `v_accvgpr` 拷贝，以及每条 `s_waitcnt` 等的是哪条 load。
4. **只在实测需要时拆 tile。** 只有整块版本更慢、且 ISA 说明了原因时，才把 tile 拆成行块或 K 段。每次只改一个变量再测，否则收益会归到错的改动上。
   完成标准：每处拆分旁边都有注释，写明不拆慢多少、为什么慢。

## LDS DMA 和多级 LDS（`blk.copy`）

- `blk.copy(shared, view, warps=)` 是异步的，每次调用一个 group。读这块 tile 之前：`rocdl.wait_asyncmark(n)`（n = 它之后发出、允许还在途的 copy 数）+ `gpu.barrier()`（其他 wave 搬的部分）。`gpu.barrier` 会先等本 wave 的 LDS 读（`lgkmcnt(0)`），所以 barrier 之后可以覆盖任何 wave 在 barrier 前读过的 slot。先列出每个 slot 的写、读在哪个 region，再算每处 n。
- copy 的源：普通指针的 tile 视图（库在 tile 起点建 buffer descriptor，约 7 条 SALU），或同一个 buffer tile 视图的几个切片（共用一个 descriptor，切片的偏移进每线程偏移）。几个 copy 读同一个大 tile 时用后者，例如 v3 的 `halves`：每个 K-step 每个矩阵一个 descriptor。
- **一个 region 里先写 LR（LDS → 寄存器），再写 DOT。** 反过来时 RA 让 LR 的目标寄存器复用本 region 最后一次读的操作数，每条 `ds_read` 要等它覆盖的 MFMA，后面的 DMA 被挤到 region 末尾（v3 慢 2.7%）。判断：某些 region 的 `ds_read` 每隔 4 条 MFMA 才出一条，且目标寄存器就是前面 MFMA 的源。
- 多个 region 的 sched hint：每个 region 用自己的 `sched_group_barrier` sync id，结尾放 `sched_barrier(0)`。循环最后一个 region 的 fence 放行 SALU（`sched_barrier(0x4)`），否则循环计数和基址更新全堆在跳转前的一个窗口里。
- **按 MFMA 窗口排指令。** 16x16x32 MFMA 每 16 周期一条，自身发射 4 周期，两条 MFMA 之间只有 12 周期：放得下 1 个 `ds_read`（8）+ 1 条 SALU（4），或 1 个 DMA（约 10），或 3 条 SALU。用 ATT 逐个窗口看（两条 MFMA 之间超过 16 周期就是 MFMA 空转），把超出的窗口里的指令挪走或删掉。

## 测量

- 在功耗上限下，时间会随频率变：比 kernel 效率用与频率无关的周期（GRBM_GUI_ACTIVE 在两个 K 上差分得到每个 K-step 的周期，或 ATT）。比时间就用同样的 launch 方式（CUDA graph 背靠背）：launch 之间有空闲时功耗低、频率高，结果偏高（Triton 每次 launch 188 us，a16w16 v9 在自带脚本里 1555 TFLOPS，背靠背 1237）。
- 先看 ATT 的窗口再动手：同样是减 SALU，DMA 的 K 偏移走 `soffset`（80 → 37 条）让每条 DMA 发射从 4 周期变成 8 周期，反而慢；共用 descriptor、一个 copy 只写一次 M0（89 → 27 条）才有效。
- 保存编译结果：CUDA graph 引用的 kernel 在编译对象被释放后重放会触发 memory access fault。

## 本仓库实测（gfx950，8192³）

| kernel | 整块写法 | 结论 |
|---|---|---|
| v0（加 `sched_barrier(0)`）、v0_preshuffle 的 ILP 版（`KNM`）、v2（`KNM`） | 与原版同速 | 不拆。v0_preshuffle 曾把 A 拆成 16 行块、按 a0、b、a1.. 发 load，同时改了 MFMA 顺序；单独测下来起作用的是 `KNM`，拆分是多余的 |
| v0_preshuffle 无 ILP | 慢 2%～6% | 没有预取时每条 MFMA 都要一块 A 和一块 B，整块 load 让第一条 MFMA 等第 9 条 load；按 a[0]、b、a[1..] 的使用顺序拆开发出才同速 |
| v1 | 慢 9.5% | 64 K 一次 load + `dot` 让寄存器用满（512 VGPR / 256 AGPR，循环里约 650 条 `v_accvgpr` 拷贝）；按 32 K 两半做才同速 |
| v3（新增，LDS DMA + 两级 LDS） | 4096²×8192 1199 TFLOPS（v2_k32 1026，a16w16 v9 1237） | hot loop 与 v9 持平：整卡每个 K-step 2120 周期，ATT loop MFMA 效率 98.3%（v9 98.4%）；输出与 v2_k32 逐位相同 |
