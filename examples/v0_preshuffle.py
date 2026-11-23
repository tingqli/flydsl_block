"""v0_preshuffle (kernels/gemm/v0_preshuffle.py) written with flydsl_block.

v0 128x128 + ILP (register double buffer) + preshuffled inputs.

preshuffle(): A[M, K] / B[N, K] are rearranged once on the host so that each
16-row x 32-K MFMA block is stored in lane order as 1 KiB (lane l = row + 16 * kg
at byte 16 * l): a 16 B load per lane then reads 1 KiB contiguous.  The kernel
only changes the global view; loads, MMA and store are the same block ops.

ILP: two K-steps per iteration with two operand buffers; the next K-step's
loads are issued before the current K-step's MFMAs, and rocdl.sched_* spread
them over the first MFMAs (otherwise LLVM moves them next to their use).

Switches: PRESHUFFLE=False reads the row-major inputs (ILP only); ILP=False
uses the v0 loop (preshuffle only).  K32=True (gfx950 only): one
v_mfma_f32_16x16x32_bf16 per 16 B operand instead of two 16x16x16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

import flydsl_block as blk


def preshuffle(X):
    """[R, K] (K contiguous) -> [R/16, K/32, 4 kg, 16 row, 8 elem]:
    X[16 * rb + row, 32 * kb + 8 * kg + e] -> Y[rb, kb, kg, row, e]"""
    R, K = X.shape
    return X.view(R // 16, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()


def build_v0_preshuffle(BM, BN, OCCUPANCY, M, N, K, PRESHUFFLE=True, ILP=True, K32=False):
    BK = 32
    NK = K // BK
    assert NK % 2 == 0
    NI, NJ = BM // 16, BN // 16
    NMFMA = NI * NJ * (1 if K32 else 2)  # MFMAs per K-step
    grid_size = (M // BM, N // BN, 1)
    block_size = (64, 1, 1)
    KNM = fx.GemmTraversalOrder.KNM  # for i: for j: MFMAs over K, as in the original

    @flyc.kernel(known_block_size=block_size)
    def gemm_v0_preshuffle(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        if const_expr(K32):
            # 16x16x32: each lane's 8 contiguous K are one MFMA operand
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            tm = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        else:
            # 16x16x16 + K permutation: 8 contiguous K per lane feed two MFMAs (as in v0)
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            perm = fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4)))
            tm = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)), perm)
        la, lb, lc = blk.Mma(tm, "A", (BM, BK)), blk.Mma(tm, "B", (BN, BK)), blk.Mma(tm, "C", (BM, BN))

        # A / B as (m, k) tiles; preshuffle() only changes where (m, k) lives:
        # (16 rb + row, 32 kb + 8 kg + e) at e + 8 row + 128 kg + 512 kb + 512 NK rb
        if const_expr(PRESHUFFLE):
            mem_a = fx.make_layout(((16, M // 16), (8, 4, NK)), ((8, 512 * NK), (1, 128, 512)))
            mem_b = fx.make_layout(((16, N // 16), (8, 4, NK)), ((8, 512 * NK), (1, 128, 512)))
        else:
            mem_a, mem_b = fx.make_layout((M, K), (K, 1)), fx.make_layout((N, K), (K, 1))
        bm, bn = fx.block_idx.x, fx.block_idx.y
        gA = fx.flat_divide(fx.make_view(fx.get_iter(A), mem_a), (BM, BK))[None, None, bm, None]
        gB = fx.flat_divide(fx.make_view(fx.get_iter(B), mem_b), (BN, BK))[None, None, bn, None]
        gC = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (BM, BN))[None, None, bm, bn]

        def interleave():
            """the NI + NJ prefetch loads go between the first MFMAs of the K-step"""
            for _ in range_constexpr(NI + NJ):
                rocdl.sched_mfma(1)
                rocdl.sched_vmem(1)
            rocdl.sched_mfma(NMFMA - NI - NJ)

        if const_expr(ILP):
            # two K-steps per iteration, operand buffers (a, b) / (a1, b1) alternate.  The last
            # iteration prefetches K-step 0 again (unused) instead of peeling a K-step.
            acc = blk.zeros(lc, fx.Float32)
            a, b = blk.load(gA[None, None, 0], la), blk.load(gB[None, None, 0], lb)
            for k in range(fx.Int32(0), fx.Int32(NK // 2), fx.Int32(1)):
                kt, kn = k * 2, (k * 2 + 2) % NK
                a1, b1 = blk.load(gA[None, None, kt + 1], la), blk.load(gB[None, None, kt + 1], lb)  # prefetch kt+1
                acc = blk.dot(a, b, acc, traversal_order=KNM)  # ... while computing kt
                interleave()
                a, b = blk.load(gA[None, None, kn], la), blk.load(gB[None, None, kn], lb)  # prefetch kt+2
                acc = blk.dot(a1, b1, acc, traversal_order=KNM)  # ... while computing kt+1
                interleave()
            blk.store(gC, acc)
        else:
            # No prefetch: every MFMA needs an A and a B piece, so A (and the accumulator) are
            # kept as 16-row blocks and the loads go out in use order a[0], b, a[1..]: the first
            # MFMAs wait for 2 loads instead of 9 (whole-tile loads are 2-6% slower here).
            la16, lc16 = blk.Mma(tm, "A", (16, BK)), blk.Mma(tm, "C", (16, BN))
            gA16 = fx.flat_divide(fx.make_view(fx.get_iter(A), mem_a), (16, BK))
            gC16 = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (16, BN))
            acc = [blk.zeros(lc16, fx.Float32) for _ in range_constexpr(NI)]
            for k in range(fx.Int32(0), fx.Int32(NK), fx.Int32(1)):
                a0 = blk.load(gA16[None, None, bm * NI, k], la16)
                rocdl.sched_barrier(0)  # keep the issue order of the load groups
                b = blk.load(gB[None, None, k], lb)
                rocdl.sched_barrier(0)
                a = [a0] + [blk.load(gA16[None, None, bm * NI + i, k], la16) for i in range_constexpr(1, NI)]
                rocdl.sched_barrier(0)  # all loads ahead of the MFMAs (see v0.py)
                acc = [blk.dot(a[i], b, acc[i], traversal_order=KNM) for i in range_constexpr(NI)]
            for i in range_constexpr(NI):
                blk.store(gC16[None, None, bm * NI + i, bn], acc[i])

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v0_preshuffle(A, B, C, value_attrs={"rocdl.waves_per_eu": OCCUPANCY}).launch(
            grid=grid_size, block=block_size, stream=stream
        )

    return launcher
