# flydsl_block

FlyDSL 上的 block 级 fragment 基本实现，对应设计文档 `pyhip/docs/flydsl_block_fragment_design.md` 的 A+B 方案：

- **A（对外）**：`Blocked` / `Mma` / `TV` 描述 layout，`BlockFragment` 提供 block 语义的读写和运算；
- **B（对内）**：所有 layout 统一转成 F2 `LinearLayout`（每个 reg/lane/warp bit 对应一个坐标增量），向量宽度、reduce、broadcast、convert 全部在它上面规划。

纯 Python 实现，不改 FlyDSL，不引入 MLIR dialect。block 语义只在 trace 期存在，IR 里只剩现有的线程级操作（tiled copy、`fx.gemm`、Vector 运算、`gpu.shuffle`、LDS）。

## 结构

| 文件 | 内容 |
|---|---|
| `flydsl_block/linear.py` | `LinearLayout`、`blocked()`、`from_tv()`，以及 reduce / convert / copy 的规划。不依赖 FlyDSL |
| `flydsl_block/fragment.py` | `BlockFragment` 和 `load/store/dot/wrap`，按规划发出 FlyDSL 操作 |
| `flydsl_block/viz.py` | `LinearLayout.to_svg()` 用的 SVG 渲染，纯 Python |

## 用法

```python
import flydsl.expr as fx
import flydsl_block as blk

# row softmax：64x128 的 tile，256 线程
L = blk.Blocked((64, 128), size_per_thread=(1, 8), threads_per_warp=(4, 16), warps=(4, 1))
x = blk.load(gX, L, bounds=(rows, None)).to(fx.Float32)   # gX: flat_divide 得到的 tile 视图
p = (x - x.max(axis=1)[:, None]).exp()
blk.store(gY, (p / p.sum(axis=1)[:, None]).to(fx.BFloat16), bounds=(rows, None))

# GEMM：MMA layout + 运行期 K 循环（BlockFragment 可以作为 loop-carried 值）
tm = fx.make_tiled_mma(fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                       fx.make_layout((2, 2, 1), (1, 2, 0)))
sa, sb, sc = blk.Mma(tm, "A", (64, 32)), blk.Mma(tm, "B", (64, 32)), blk.Mma(tm, "C", (64, 64))
acc = blk.zeros(sc, fx.Float32)
for k in range(K // 32):
    acc = blk.dot(blk.load(gA[None, None, k], sa), blk.load(gB[None, None, k], sb), acc)
blk.store(gC, acc)

# 和线程级 fragment 互通
acc = blk.wrap(thr_mma.make_fragment_C(gC), sc)   # 线程级 -> block
frag = acc.frag                                    # block -> rmem fx.Tensor（MMA C 的形状），可直接给 fx.gemm / fx.copy
```

## 可视化

`LinearLayout.to_svg(path)` 生成 SVG 图片，不依赖第三方库，VS Code 和浏览器可以直接打开：

```python
blk.Blocked((32, 64), (1, 8), (8, 8), (4, 1)).to_svg("blocked.svg")
blk.Mma(tm, "C", (64, 64)).linear.to_svg("mma_c.svg")
layout.to_svg("grid.svg", style="grid", highlight_thread=5)
```

有两种画法，用 `style=` 选，默认 `"auto"`：能分层就用 lens，否则用 grid。

**lens（分层放大）**：从左到右三张小图，行是 dim 0，列是 dim 1。
- block 图：每个格子是一个 warp tile，标 warp 号 `w<n>`；有寄存器 repeat 时加 `r+<offset>`。单 warp 且没有副本时省略这张图。
- warp tile 图：每个格子是一个 thread tile，标 lane 号 `L<n>`，颜色和 lane 颜色对照表一致。线程号 = lane + warp_size * warp。
- thread tile 图：每个格子是一个元素，标寄存器 `r<n>`。只有一个元素时省略。
- 每张图右上角的彩色框会在右边下一张图里放大，用同色虚线连接。所有 warp tile（以及所有 thread tile）的排布相同，放大哪一格都一样。
- `xN` 表示 N 个 warp / lane 持有同一份数据（副本）。
- 适用条件：每个维度的坐标 bit 从低到高依次来自 reg、lane、warp 或 repeat reg，并且每个非零基只对应一个坐标 bit。MFMA 的 A/B/C、Blocked 都满足；同一维度里 reg bit 和 lane bit 交错（例如 lane、reg、lane 的顺序），或者带 XOR swizzle 的 layout 不满足，会自动退回 grid。显式传 `style="lens"` 而 layout 不满足条件时会报错。

**grid（逐元素）**：每个格子是 tile 的一个元素。
- 同一线程持有的格子合成一块：线程号 `T<t>` 只写一次，每个格子标出寄存器 `r<r>`；`xN` 表示该元素有 N 份副本。
- 颜色表示 lane：色相取 lane%16，深浅取 lane/16。粗线是 warp 边界，黑框是 `highlight_thread` 指定线程持有的元素。
- tile 大于 64x64 时自动缩小格子并省略文字。

两种画法共用的下半部分：
- reg / lane / warp 每个 bit 对应的坐标增量，以及哪些线程 bit 是副本。
- lane 颜色对照表。
- 对照表右侧的空白处（图较窄时放在对照表下方）按 LinearLayout 论文的样式画出 F2 矩阵 `A`（`w = A v`）：
  - 列是输入 bit，按 Reg | Lane | Warp 分组；
  - 行是坐标 bit，每个维度一组，最后一维（最快变化的维度）在前，组内低位在前；
  - 1 加粗，0 用灰色；全 0 的列就是副本。
  - 传 `matrix=False` 可以关掉。

`python -m flydsl_block.viz /tmp/layouts` 会生成几个示例：Blocked（2D 和 1D）、MFMA C、2x2 warp 的 tiled MMA C/A、交换 reg/lane bit 后的合并写 layout，以及一个退回 grid 的交错 layout。

## API

| 类别 | API |
|---|---|
| layout | `Blocked(shape, size_per_thread, threads_per_warp, warps, order)`、`Mma(tiled_mma, "A"/"B"/"C", tile)`、`TV(fly_tv_layout, tile)`、`LinearLayout(shape, reg, lane, warp)` |
| 读写 | `load(view, layout, bounds=, other=, max_vec=)`、`store(view, x, bounds=)`；`view` 可以是 global（buffer）或 LDS 的 tile 视图 |
| 运算 | `+ - * /`、`exp/exp2/maximum/minimum/to`、`sum/max/min/reduce(axis)`、`x[:, None]` 广播、`x.T` |
| layout 转换 | `x.convert(layout, explain=True)`：寄存器改名 / warp 内 reg↔lane 交换 + lane 置换 / 经 LDS |
| MMA | `dot(a, b, acc)`：`acc` 需为 `Mma(tm, "C", ...)` layout，a/b 不匹配时自动 convert |
| 互通 | `x.frag`、`x.vec`、`wrap(frag, layout)`、`x.coords()` |

## 测试

```bash
cd /data2/tingqli/flydsl_block
PYTHONPATH=/tmp/flydsl050 python -m pytest tests      # FlyDSL 0.5.0
python -m pytest tests                                 # /opt/venv 里的 FlyDSL 0.3.4.1
```

在 gfx950 上两个版本都是 103 passed：

- `test_linear.py`（CPU）：用 FlyDSL 实测的 TV layout 校验 `from_tv`；TV round trip；用寄存器堆模拟器逐条执行 convert / reduce 规划，覆盖 40 组随机 layout。
- `test_fragment_gpu.py`：带边界的 load/store（Blocked 和 `TV`）、Mma A/C、5 种 convert、3 种 layout 的 reduce（两个轴，sum/max）、softmax、运行期 K 循环的 GEMM、`wrap` 与 `.frag` 互通。结果与 torch 对比。
- `test_isa.py`：检查最终 ISA。
  - bf16 copy：2 条 `buffer_load_dwordx4` 加 2 条 `buffer_store_dwordx4`。
  - MMA C 经 warp 内 convert 后：每对寄存器 1 条 `ds_swizzle_b32`，不占 LDS、没有 barrier，store 变成 4 条 `buffer_store_dwordx4`。
  - 经 LDS 的 convert：占用 16 KB LDS，2 个 barrier。

## 限制 / 后续

- tile 各维必须是 2 的幂。K=192 这类需要“线性 bit + 外层重复”的扩展，尚未实现。
- warp 内 convert 只支持每个 bit 恰好对应一个坐标 bit、且没有复制的 layout；16-bit 浮点先升到 f32 再交换。其余情况走 LDS。
- lane 间搬运用 `gpu.shuffle`，LLVM 生成的是 `ds_swizzle`/`ds_bpermute`，还没有用 DPP / `v_permlane*_swap` 做 target override。
- LDS 路径没有 swizzle（不处理 bank conflict）；每次 convert / 跨 warp reduce 都单独分配一块 LDS，并插 2~3 个 barrier。
- 带 `bounds` 的 copy 每个 atom 单独发一次（一个 `scf.if`）。向量化维上的 bound 必须是向量宽度的倍数，否则传 `max_vec=1`。向量宽度只看连续性，默认地址已对齐。
- 16-bit 浮点的 reduce 结果是 f32。
- 已知 FlyDSL 问题（0.5.0 和 0.3.4.1 都有）：2x2 warp 的 tiled MMA 中，用**静态**线程号调用 `thr_slice(t).partition_B` 得到的坐标，与 `tv_layout_B_tiled` 及运行期结果不一致（M/N 两个方向的 warp 对调）。因此 `Mma()` 改用 `tv_layout_*_tiled` 构造 layout。
