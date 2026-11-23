"""v1 (kernels/gemm/v1.py) written with flydsl_block.

4 waves in a 2x2 grid; each CTA computes a BM x BN tile of C, each wave a
contiguous WM x WN slice.  Per K-step (BK = 64) the A / B tiles go
global -> staging registers (coalesced: 8 threads x 16 B per 128 B row) ->
LDS -> MFMA operand registers.

LDS rows are 64 bf16 = 128 B = 32 banks, stored in 16 B chunks with
chunk ^= row % 8 (blk.Shared(vec=8, max_phase=8)): the 8 lanes of a
ds_read_b128 phase (8 rows, same chunk) and of a ds_write_b128 phase (1 row,
8 chunks) both hit all 32 banks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

import flydsl_block as blk


def build_v1(BM, BN, OCCUPANCY, M, N, K):
    BK = 64
    WM, WN = BM // 2, BN // 2
    NI, NJ = WM // 16, WN // 16
    grid_size = (M // BM, N // BN, 1)
    block_size = (256, 1, 1)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, BM * BK, 16]
        b: fx.Array[fx.BFloat16, BN * BK, 16]

    @flyc.kernel(known_block_size=block_size)
    def gemm_v1(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        # MFMA 16x16x16 on 2x2 waves.  Permutations: wave wm owns rows [wm * WM, wm * WM + WM)
        # (16-row blocks i at +16 i), the same for N; K as in v0 (8 contiguous K per lane).
        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        perm = fx.make_tile(
            fx.make_layout((16, 2, NI), (1, WM, 16)),
            fx.make_layout((16, 2, NJ), (1, WN, 16)),
            fx.make_layout((4, 4, 2), (1, 8, 4)),
        )
        tm = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)), perm)
        la, lb, lc = blk.Mma(tm, "A", (BM, 32)), blk.Mma(tm, "B", (BN, 32)), blk.Mma(tm, "C", (BM, BN))
        # global -> staging: 8 threads per row (16 B each), 32 rows per pass of 256 threads
        sta = blk.Blocked((BM, BK), (1, 8), (8, 8), (4, 1))
        stb = blk.Blocked((BN, BK), (1, 8), (8, 8), (4, 1))

        bm, bn = fx.block_idx.x, fx.block_idx.y
        gA = fx.flat_divide(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))), (BM, BK))
        gB = fx.flat_divide(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))), (BN, BK))
        gC = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (BM, BN))
        gA, gB, gC = gA[None, None, bm, None], gB[None, None, bn, None], gC[None, None, bm, bn]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        sA = blk.Shared(lds.a, (BM, BK), vec=8, max_phase=8)
        sB = blk.Shared(lds.b, (BN, BK), vec=8, max_phase=8)

        acc = blk.zeros(lc, fx.Float32)
        for k in range(fx.Int32(0), fx.Int32(K // BK), fx.Int32(1)):
            # 1. global -> staging registers
            a_st, b_st = blk.load(gA[None, None, k], sta), blk.load(gB[None, None, k], stb)
            # 2. staging -> LDS, after every wave finished reading the previous tile
            gpu.barrier()
            blk.store(sA, a_st)
            blk.store(sB, b_st)
            gpu.barrier()
            # 3. LDS -> operand registers and MFMAs, 32 K at a time (for i: for j: for k).  The
            #    halves bound the live operand registers: one 64-K load + dot fills all 512 VGPR /
            #    256 AGPR with ~650 accvgpr copies in the loop and is 9.5% slower.
            for s in range_constexpr(BK // 32):
                a = blk.load(sA.slice(32 * s, 32, dim=1), la)
                b = blk.load(sB.slice(32 * s, 32, dim=1), lb)
                acc = blk.dot(a, b, acc, traversal_order=fx.GemmTraversalOrder.KNM)
        blk.store(gC, acc)

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v1(A, B, C, value_attrs={"rocdl.waves_per_eu": OCCUPANCY}).launch(
            grid=grid_size, block=block_size, stream=stream
        )

    return launcher
