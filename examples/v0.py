"""v0 (kernels/gemm/v0.py) written with flydsl_block.

One wave per CTA computes a BM x BN tile of C.  Each K-step (32) loads the A
and B tiles straight from global memory in the MFMA operand layout and runs
the MFMAs; the accumulator is a block fragment carried by the K loop.

The tiled MMA permutes K so that each lane holds 8 contiguous K (one 16 B
load per 16-row block): elements 0..3 feed one 16x16x16 MFMA and 4..7 the
next.  A and B use the same permutation, so the reduction is unchanged.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

import flydsl_block as blk


def build_v0(BM, BN, OCCUPANCY, M, N, K):
    BK = 32
    grid_size = (M // BM, N // BN, 1)
    block_size = (64, 1, 1)

    @flyc.kernel(known_block_size=block_size)
    def gemm_v0(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        perm = fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4)))
        tm = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)), perm)
        la, lb, lc = blk.Mma(tm, "A", (BM, BK)), blk.Mma(tm, "B", (BN, BK)), blk.Mma(tm, "C", (BM, BN))

        bm, bn = fx.block_idx.x, fx.block_idx.y
        gA = fx.flat_divide(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))), (BM, BK))
        gB = fx.flat_divide(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))), (BN, BK))
        gC = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (BM, BN))
        gA, gB, gC = gA[None, None, bm, None], gB[None, None, bn, None], gC[None, None, bm, bn]

        acc = blk.zeros(lc, fx.Float32)
        for k in range(fx.Int32(0), fx.Int32(K // BK), fx.Int32(1)):
            a = blk.load(gA[None, None, k], la)
            b = blk.load(gB[None, None, k], lb)
            # keep all loads ahead of the MFMAs: otherwise the post-RA scheduler sinks some
            # loads between MFMAs and the next MFMA waits for each of them
            fx.rocdl.sched_barrier(0)
            acc = blk.dot(a, b, acc)
        blk.store(gC, acc)

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v0(A, B, C, value_attrs={"rocdl.waves_per_eu": OCCUPANCY}).launch(
            grid=grid_size, block=block_size, stream=stream
        )

    return launcher
