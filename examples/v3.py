"""v3: v2 with LDS DMA and two LDS stages (the structure of gfx950-gluon-tutorials a16w16 v8/v9).

CTA tile 256x256x64, 4 waves, K32 MFMA (v_mfma_f32_16x16x32_bf16).  As in v2, A
is split into A_top / A_bot and B into B_left / B_right, a region is one
128x128 quadrant of C (32 MFMAs per wave) and LR reads the operand of the next
region.  Global tiles go straight to LDS (``blk.copy``: buffer_load ... lds, no
registers, no ds_write), two K-steps ahead, into the stage the K-step just read:

    region 0: LR A_bot(s)   | DMA B_left(s)  <- k+2 | DOT C_tl
    region 1: LR B_right(s) | DMA A_top(s)   <- k+2 | DOT C_bl
    region 2: LR B_left(o)  | DMA A_bot(s)   <- k+2 | DOT C_tr
    region 3: LR A_top(o)   | DMA B_right(s) <- k+2 | DOT C_br

s = k % 2 is the stage of K-step k, o = 1 - s.  Each copy is one async group.
A region starts with wait_asyncmark(5) + barrier: the half its LR reads was
copied 6 groups ago (now landed in every wave), and every wave has finished
reading the slot its DMA overwrites (gpu.barrier waits for this wave's LDS reads).
Tiles are ordered as in a16w16 v9: consecutive tiles on one XCD, GROUP_M tile
rows at a time (L2 reuse, 2.5% faster than the plain 2-D grid at 4096x4096).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

import flydsl_block as blk


def build_v3(BM, BN, OCCUPANCY, M, N, K, SCHED=True, GROUP_M=4):
    """GROUP_M=0: the plain 2-D grid (bm = block x, bn = block y)."""
    assert (BM, BN) == (256, 256)
    BK = 64
    NK = K // BK
    assert NK >= 4 and NK % 2 == 0
    H = 128  # half-tile rows
    WT = 64  # per-wave slice of a quadrant
    NI = WT // 16
    WARPS = 4
    NUM_XCDS = 8  # workgroup i runs on XCD i % 8 (MI300 / MI350)
    tiles_m, tiles_n = M // BM, N // BN
    if GROUP_M:
        assert (tiles_m * tiles_n) % NUM_XCDS == 0 and tiles_m % GROUP_M == 0
        grid_size = (tiles_m * tiles_n, 1, 1)
    else:
        grid_size = (tiles_m, tiles_n, 1)
    block_size = (WARPS * 64, 1, 1)
    KNM = fx.GemmTraversalOrder.KNM
    SALU = 0x4  # sched_barrier / sched_group_barrier mask

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, 2 * BM * BK, 16]  # stage s: A_top rows 256 s .., A_bot 256 s + 128 ..
        b: fx.Array[fx.BFloat16, 2 * BN * BK, 16]

    @flyc.kernel(known_block_size=block_size)
    def gemm_v3(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        wave_rows = fx.make_layout((16, 2, NI), (1, WT, 16))
        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        tm = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)), fx.make_tile(wave_rows, wave_rows, None))
        la, lb, lc = blk.Mma(tm, "A", (H, BK)), blk.Mma(tm, "B", (H, BK)), blk.Mma(tm, "C", (H, H))

        bm, bn = fx.block_idx.x, fx.block_idx.y
        if const_expr(GROUP_M):
            pid = fx.Int32(fx.block_idx.x)
            per_xcd = tiles_m * tiles_n // NUM_XCDS
            pid = pid % NUM_XCDS * per_xcd + pid // NUM_XCDS  # consecutive pids on one XCD
            group = GROUP_M * tiles_n
            bm = pid // group * GROUP_M + pid % group % GROUP_M
            bn = pid % group // GROUP_M
        gA = fx.flat_divide(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))), (BM, BK))
        gB = fx.flat_divide(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))), (BN, BK))
        gA, gB = gA[None, None, bm, None], gB[None, None, bn, None]  # this CTA's tiles, one per K-step
        gC = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (H, H))

        def halves(g, kn):
            """K-step kn's 256x64 tile as two 128-row halves on one buffer descriptor.  A half's
            offset becomes part of the loop-invariant per-thread offsets, so a K-step builds one
            descriptor per matrix (a descriptor per half costs ~7 SALU, which spill into the LR
            windows)."""
            t = fx.flat_divide(fx.rocdl.make_buffer_tensor(g[None, None, kn]), (H, BK))
            return t[None, None, 0, 0], t[None, None, 1, 0]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        sA = blk.Shared(lds.a, (2 * BM, BK), vec=8, max_phase=8)
        sB = blk.Shared(lds.b, (2 * BN, BK), vec=8, max_phase=8)
        sA_top = [sA.slice(s * BM, H) for s in range_constexpr(2)]
        sA_bot = [sA.slice(s * BM + H, H) for s in range_constexpr(2)]
        sB_left = [sB.slice(s * BN, H) for s in range_constexpr(2)]
        sB_right = [sB.slice(s * BN + H, H) for s in range_constexpr(2)]

        def interleave(n_dsrd, n_vmem, rid):
            """One region's order: 4 MFMAs (the barrier and the DMA address math fall
            between them), the LR reads 1:1 with MFMAs, then each DMA between 2 + 2 MFMAs;
            nothing crosses into the next region.  ``rid`` keeps the hint groups of
            different regions apart.  2 leading MFMAs instead of 4: 0.8% slower.
            The loop's last region (rid 7) lets SALU through: the loop counter and A/B
            base updates then spread over its windows instead of piling up in front of
            the branch (7 SALU in one window: +36 cycles per iteration)."""
            if const_expr(SCHED):
                mfma, dsrd, vmem = rocdl.mask_mfma, rocdl.mask_dsrd, rocdl.mask_vmem_rd
                rocdl.sched_group_barrier(mfma, 4, rid)
                for _ in range_constexpr(n_dsrd):
                    rocdl.sched_group_barrier(dsrd, 1, rid)
                    rocdl.sched_group_barrier(mfma, 1, rid)
                for _ in range_constexpr(n_vmem):
                    rocdl.sched_group_barrier(mfma, 2, rid)
                    rocdl.sched_group_barrier(vmem, 1, rid)
                    rocdl.sched_group_barrier(mfma, 2, rid)
                rocdl.sched_group_barrier(mfma, 28 - n_dsrd - 4 * n_vmem, rid)
            rocdl.sched_barrier(SALU if rid == 7 else 0)

        def sync(n):
            rocdl.wait_asyncmark(n)
            gpu.barrier()

        # ---- prologue: K-steps 0 and 1 -> stages 0 and 1, prime a_top / b_left
        for s in range_constexpr(2):
            (a_top_g, a_bot_g), (b_left_g, b_right_g) = halves(gA, s), halves(gB, s)
            blk.copy(sB_left[s], b_left_g, warps=WARPS)
            blk.copy(sA_top[s], a_top_g, warps=WARPS)
            blk.copy(sA_bot[s], a_bot_g, warps=WARPS)
            blk.copy(sB_right[s], b_right_g, warps=WARPS)
        sync(6)
        b_left, a_top = blk.load(sB_left[0], lb), blk.load(sA_top[0], la)

        def kstep(acc, a_top, b_left, s, kn, last):
            """K-step on stage s; DMA K-step kn into stage s.  last=0: steady state;
            1: no DMA; 2: no DMA, no LR of the next K-step.  Each region issues its LR
            before its DOT: the LR registers must not reuse the operand the DOT reads last
            (b_left in region 1, a_bot / b_right in region 3), or every ds_read waits for
            the MFMA it overwrites and the DMAs end up after the last MFMA."""
            o = 1 - s
            waits = [[5, 5, 5, 5], [5, 4, 3, 2], [1, 0, None, None]][last]
            # region 0: LR A_bot(s) | DMA B_left(s) | DOT C_tl
            sync(waits[0])
            a_bot = blk.load(sA_bot[s], la)
            if const_expr(last == 0):
                gb = halves(gB, kn)
                blk.copy(sB_left[s], gb[0], warps=WARPS)
            acc[0] = blk.dot(a_top, b_left, acc[0], traversal_order=KNM)
            interleave(8, 4 if last == 0 else 0, 4 * s)
            # region 1: LR B_right(s) | DMA A_top(s) | DOT C_bl
            sync(waits[1])
            b_right = blk.load(sB_right[s], lb)
            if const_expr(last == 0):
                ga = halves(gA, kn)
                blk.copy(sA_top[s], ga[0], warps=WARPS)
            acc[1] = blk.dot(a_bot, b_left, acc[1], traversal_order=KNM)
            interleave(8, 4 if last == 0 else 0, 4 * s + 1)
            # region 2: LR B_left(o) | DMA A_bot(s) | DOT C_tr
            b_left_next = b_left
            if const_expr(last < 2):
                sync(waits[2])
                b_left_next = blk.load(sB_left[o], lb)
            if const_expr(last == 0):
                blk.copy(sA_bot[s], ga[1], warps=WARPS)
            acc[2] = blk.dot(a_top, b_right, acc[2], traversal_order=KNM)
            interleave(8 if last < 2 else 0, 4 if last == 0 else 0, 4 * s + 2)
            # region 3: LR A_top(o) | DMA B_right(s) | DOT C_br
            a_top_next = a_top
            if const_expr(last < 2):
                sync(waits[3])
                a_top_next = blk.load(sA_top[o], la)
            if const_expr(last == 0):
                blk.copy(sB_right[s], gb[1], warps=WARPS)
            acc[3] = blk.dot(a_bot, b_right, acc[3], traversal_order=KNM)
            interleave(8 if last < 2 else 0, 4 if last == 0 else 0, 4 * s + 3)
            return acc, a_top_next, b_left_next

        # ---- main loop: K-steps 0 .. NK-3, two per iteration (static stages)
        acc = [blk.zeros(lc, fx.Float32) for _ in range_constexpr(4)]
        for k in range(fx.Int32(0), fx.Int32(NK - 2), fx.Int32(2)):
            acc, a_top, b_left = kstep(acc, a_top, b_left, 0, k + 2, 0)
            acc, a_top, b_left = kstep(acc, a_top, b_left, 1, k + 3, 0)
        # ---- peel: K-steps NK-2 (stage 0) and NK-1 (stage 1)
        acc, a_top, b_left = kstep(acc, a_top, b_left, 0, None, 1)
        acc, a_top, b_left = kstep(acc, a_top, b_left, 1, None, 2)

        rocdl.sched_barrier(0)
        for q in range_constexpr(4):
            blk.store(gC[None, None, bm * 2 + q % 2, bn * 2 + q // 2], acc[q])

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v3(A, B, C, value_attrs={"rocdl.waves_per_eu": OCCUPANCY}).launch(
            grid=grid_size, block=block_size, stream=stream
        )

    return launcher
