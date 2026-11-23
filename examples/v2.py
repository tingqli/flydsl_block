"""v2 (kernels/gemm/v2.py) written with flydsl_block: v1 + software pipeline.

CTA tile 256x256x64, 4 waves (1 per SIMD).  A is split along M into A_top /
A_bot and B along N into B_left / B_right (four 16 KB LDS slots = 64 KB);
each (A half, B half) pair is one 128x128 quadrant of C, and each wave owns a
64x64 slice of every quadrant (2x2 waves inside a quadrant).

One K-step = 4 regions, each = 64 MFMAs on one quadrant plus three memory ops:
    region 0: DOT C_tl | LR A_bot(k)    | LW B_left(k+1)  | GR B_left(k+2)
    region 1: DOT C_bl | LR B_right(k)  | LW A_top(k+1)   | GR A_top(k+2)
    region 2: DOT C_tr | LR B_left(k+1) | LW A_bot(k+1)   | GR A_bot(k+2)
    region 3: DOT C_br | LR A_top(k+1)  | LW B_right(k+1) | GR B_right(k+2)
GR = global -> registers (2 K-steps ahead), LW = registers -> LDS (1 ahead),
LR = LDS -> registers (one region ahead of its DOT).  A slot is rewritten one
region after its last read and read one region after its rewrite, so a
single LDS stage suffices; two barriers per K-step (after regions 1 and 3)
order every cross-wave read/write pair.

K32=True (gfx950 only): one v_mfma_f32_16x16x32_bf16 per 16 B operand
instead of two 16x16x16, so a region has 32 MFMAs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

import flydsl_block as blk


def build_v2(BM, BN, OCCUPANCY, M, N, K, SCHED=True, K32=False):
    assert (BM, BN) == (256, 256)
    BK = 64
    NK = K // BK
    assert NK >= 3
    H = 128  # half-tile rows
    WT = 64  # per-wave slice of a quadrant
    NI = WT // 16
    grid_size = (M // BM, N // BN, 1)
    block_size = (256, 1, 1)
    KNM = fx.GemmTraversalOrder.KNM

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, BM * BK, 16]  # A_top rows 0..127, A_bot 128..255
        b: fx.Array[fx.BFloat16, BN * BK, 16]  # B_left rows 0..127, B_right 128..255

    @flyc.kernel(known_block_size=block_size)
    def gemm_v2(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        # one 128x128 quadrant on 2x2 waves; wave (wm, wn) owns rows [64 wm, 64 wm + 64) and
        # cols [64 wn, 64 wn + 64) (16-row blocks at +16 i).  K: 8 contiguous K per lane feed
        # two 16x16x16 MFMAs (as in v0), or one 16x16x32 MFMA (K32).
        wave_rows = fx.make_layout((16, 2, NI), (1, WT, 16))
        if const_expr(K32):
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            perm = fx.make_tile(wave_rows, wave_rows, None)
        else:
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            perm = fx.make_tile(wave_rows, wave_rows, fx.make_layout((4, 4, 2), (1, 8, 4)))
        tm = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)), perm)
        la, lb, lc = blk.Mma(tm, "A", (H, BK)), blk.Mma(tm, "B", (H, BK)), blk.Mma(tm, "C", (H, H))
        stage = blk.Blocked((H, BK), (1, 8), (8, 8), (4, 1))  # 8 threads x 16 B per row, 4 passes

        # half-tiles: global (H, BK, NK) views and LDS slots
        bm, bn = fx.block_idx.x, fx.block_idx.y
        gA = fx.flat_divide(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))), (H, BK))
        gB = fx.flat_divide(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))), (H, BK))
        gC = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (H, H))
        gA_top, gA_bot = gA[None, None, bm * 2, None], gA[None, None, bm * 2 + 1, None]
        gB_left, gB_right = gB[None, None, bn * 2, None], gB[None, None, bn * 2 + 1, None]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        sA = blk.Shared(lds.a, (BM, BK), vec=8, max_phase=8)
        sB = blk.Shared(lds.b, (BN, BK), vec=8, max_phase=8)
        sA_top, sA_bot = sA.slice(0, H), sA.slice(H, H)
        sB_left, sB_right = sB.slice(0, H), sB.slice(H, H)

        def interleave(n_dsrd, n_dswr, n_vmem):
            """Ask the scheduler to spread one region's memory ops through its
            64 MFMAs: 2 MFMAs (32 cycles) per LDS access (CDNA3 LDS = 128 B/clk,
            shared by 4 SIMDs), 1 MFMA per global load."""
            if const_expr(SCHED):
                for _ in range_constexpr(n_dsrd):
                    rocdl.sched_mfma(2)
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(n_dswr):
                    rocdl.sched_mfma(2)
                    rocdl.sched_dswr(1)
                for _ in range_constexpr(n_vmem):
                    rocdl.sched_mfma(1)
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma((32 if K32 else 64) - 2 * n_dsrd - 2 * n_dswr - n_vmem)

        # ---- prologue: tile 0 -> LDS, tile 1 -> registers, prime a_top / b_left.
        # The tile-1 loads must be issued in the loop's order (B_left, A_top, A_bot, B_right):
        # the waitcnt pass merges the loads in flight at the loop entry with the loop's own,
        # so a different order here makes every vmcnt wait in the loop stricter.
        blk.store(sB_left, blk.load(gB_left[None, None, 0], stage))
        blk.store(sA_top, blk.load(gA_top[None, None, 0], stage))
        blk.store(sA_bot, blk.load(gA_bot[None, None, 0], stage))
        blk.store(sB_right, blk.load(gB_right[None, None, 0], stage))
        rocdl.sched_barrier(0)
        g_bl = blk.load(gB_left[None, None, 1], stage)
        gpu.barrier()  # tile 0 visible
        g_at = blk.load(gA_top[None, None, 1], stage)
        rocdl.sched_barrier(0)
        g_ab = blk.load(gA_bot[None, None, 1], stage)
        rocdl.sched_barrier(0)
        g_br = blk.load(gB_right[None, None, 1], stage)
        b_left, a_top = blk.load(sB_left, lb), blk.load(sA_top, la)
        gpu.barrier()  # prologue reads done before LW B_left

        def kstep(acc, a_top, b_left, g, kn, last):
            """One K-step (kn = K-step of the GR).  last=0: steady state; 1: no GR; 2: no GR/LW, no next LR."""
            g_bl, g_at, g_ab, g_br = g
            # region 0: DOT C_tl | LR A_bot(k) | LW B_left(k+1) | GR B_left(k+2)
            acc[0] = blk.dot(a_top, b_left, acc[0], traversal_order=KNM)
            a_bot = blk.load(sA_bot, la)
            if const_expr(last < 2):
                blk.store(sB_left, g_bl)
            if const_expr(last < 1):
                g_bl = blk.load(gB_left[None, None, kn], stage)
            interleave(8, 4 if last < 2 else 0, 4 if last < 1 else 0)
            # region 1: DOT C_bl | LR B_right(k) | LW A_top(k+1) | GR A_top(k+2)
            acc[1] = blk.dot(a_bot, b_left, acc[1], traversal_order=KNM)
            b_right = blk.load(sB_right, lb)
            if const_expr(last < 2):
                blk.store(sA_top, g_at)
            if const_expr(last < 1):
                g_at = blk.load(gA_top[None, None, kn], stage)
            interleave(8, 4 if last < 2 else 0, 4 if last < 1 else 0)
            if const_expr(last < 2):
                gpu.barrier()  # B_left/A_top(k+1) written; A_bot/B_right(k) read
            # region 2: DOT C_tr | LR B_left(k+1) | LW A_bot(k+1) | GR A_bot(k+2)
            acc[2] = blk.dot(a_top, b_right, acc[2], traversal_order=KNM)
            if const_expr(last < 2):
                b_left = blk.load(sB_left, lb)
                blk.store(sA_bot, g_ab)
            if const_expr(last < 1):
                g_ab = blk.load(gA_bot[None, None, kn], stage)
            interleave(8 if last < 2 else 0, 4 if last < 2 else 0, 4 if last < 1 else 0)
            # region 3: DOT C_br | LR A_top(k+1) | LW B_right(k+1) | GR B_right(k+2)
            acc[3] = blk.dot(a_bot, b_right, acc[3], traversal_order=KNM)
            if const_expr(last < 2):
                a_top = blk.load(sA_top, la)
                blk.store(sB_right, g_br)
            if const_expr(last < 1):
                g_br = blk.load(gB_right[None, None, kn], stage)
            interleave(8 if last < 2 else 0, 4 if last < 2 else 0, 4 if last < 1 else 0)
            if const_expr(last < 2):
                gpu.barrier()  # A_bot/B_right(k+1) written; B_left/A_top(k+1) read
            return acc, a_top, b_left, [g_bl, g_at, g_ab, g_br]

        # ---- main loop: K-steps 0 .. NK-3
        acc = [blk.zeros(lc, fx.Float32) for _ in range_constexpr(4)]
        g = [g_bl, g_at, g_ab, g_br]
        for k in range(fx.Int32(0), fx.Int32(NK - 2), fx.Int32(1)):
            acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, k + 2, 0)

        # ---- peel: K-step NK-2 (LW, no GR) and NK-1 (drain)
        acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, None, 1)
        acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, None, 2)

        # ---- epilogue: quadrant q = (qm, qn): tl, bl, tr, br.  K32: keep the stores after the
        # drain MFMAs (mixed into them, the stores leave in bursts and the C write-out is
        # slower: 1.5% for the whole GEMM); with 16x16x16 the mixed schedule is faster.
        if const_expr(K32):
            rocdl.sched_barrier(0)
        for q in range_constexpr(4):
            blk.store(gC[None, None, bm * 2 + q % 2, bn * 2 + q // 2], acc[q])

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v2(A, B, C, value_attrs={"rocdl.waves_per_eu": OCCUPANCY}).launch(
            grid=grid_size, block=block_size, stream=stream
        )

    return launcher
