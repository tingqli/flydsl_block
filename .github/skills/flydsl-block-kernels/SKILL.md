---
name: flydsl-block-kernels
description: 'Use when writing, porting or refactoring a FlyDSL kernel with flydsl_block (blk.load / blk.store / blk.dot / blk.Shared, Mma layouts), e.g. rewriting a thread-level kernel at block level, or when a block-level kernel is slower than its original.'
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

## 本仓库实测（gfx950，8192³）

| kernel | 整块写法 | 结论 |
|---|---|---|
| v0（加 `sched_barrier(0)`）、v0_preshuffle 的 ILP 版（`KNM`）、v2（`KNM`） | 与原版同速 | 不拆。v0_preshuffle 曾把 A 拆成 16 行块、按 a0、b、a1.. 发 load，同时改了 MFMA 顺序；单独测下来起作用的是 `KNM`，拆分是多余的 |
| v0_preshuffle 无 ILP | 慢 2%～6% | 没有预取时每条 MFMA 都要一块 A 和一块 B，整块 load 让第一条 MFMA 等第 9 条 load；按 a[0]、b、a[1..] 的使用顺序拆开发出才同速 |
| v1 | 慢 9.5% | 64 K 一次 load + `dot` 让寄存器用满（512 VGPR / 256 AGPR，循环里约 650 条 `v_accvgpr` 拷贝）；按 32 K 两半做才同速 |
