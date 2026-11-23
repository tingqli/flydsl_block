"""GPU tests: every kernel round-trips data through BlockFragments and checks against torch."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch

import flydsl_block as blk
from flydsl_block import LinearLayout

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

THREADS = 256


def make_tiled_mma():
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    return fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)))


def tile2d(t, bm, bn, i=0, j=0):
    return fx.flat_divide(fx.rocdl.make_buffer_tensor(t, max_size=False), (bm, bn))[None, None, i, j]


def roundtrip_kernel(bm, bn, src_fn, dst_fn=None, bounded=False):
    """load with src_fn() layout, optionally convert to dst_fn(), store back."""

    @flyc.kernel
    def kern(X: fx.Tensor, Y: fx.Tensor, M: fx.Int32):
        bid = fx.block_idx.x
        rows = M - fx.Int32(bid) * bm
        bounds = (rows, None) if bounded else None
        x = blk.load(tile2d(X, bm, bn, bid), src_fn(), bounds=bounds)
        x = x.convert(dst_fn()) if dst_fn is not None else x
        blk.store(tile2d(Y, bm, bn, bid), x, bounds=bounds)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kern(X, Y, M).launch(grid=((M + bm - 1) // bm, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def run_roundtrip(launch, m, n, dtype):
    x = torch.randn(m, n, device="cuda").to(dtype)
    y = torch.zeros_like(x)
    launch(x, y, m, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return x, y


BLOCKED = {
    "bf16_spt1x8": (lambda: blk.Blocked((64, 64), (1, 8), (8, 8), (4, 1)), torch.bfloat16),
    "f32_spt2x4_2x2warps": (lambda: blk.Blocked((64, 64), (2, 4), (4, 16), (2, 2)), torch.float32),
    "f32_m_fastest": (lambda: blk.Blocked((64, 64), (4, 1), (16, 4), (1, 4), order=(0, 1)), torch.float32),
    # any FlyDSL TV layout: 32x8 threads, 2x8 values each -> 64x64 tile
    "f32_tv_from_make_layout_tv": (
        lambda: blk.TV(fx.make_layout_tv(fx.make_layout((32, 8), (8, 1)), fx.make_layout((2, 8), (8, 1)))[1], (64, 64)),
        torch.float32,
    ),
}


@pytest.mark.parametrize("name", list(BLOCKED))
def test_load_store_blocked_bounds(name):
    fn, dtype = BLOCKED[name]
    x, y = run_roundtrip(roundtrip_kernel(64, 64, fn, bounded=True), 100, 64, dtype)
    assert torch.equal(x, y)


@pytest.mark.parametrize("op,shape,dtype", [("C", (64, 64), torch.float32), ("A", (64, 32), torch.bfloat16)])
def test_load_store_mma(op, shape, dtype):
    fn = lambda: blk.Mma(make_tiled_mma(), op, shape)  # noqa: E731
    x, y = run_roundtrip(roundtrip_kernel(*shape, fn), 128, shape[1], dtype)
    assert torch.equal(x, y)


def _c64():
    return blk.Mma(make_tiled_mma(), "C", (64, 64)).linear


def _coalesced(c):
    """Swap C's two m register bits with the two lowest n lane bits: 4 contiguous n per lane."""
    return LinearLayout(c.shape, c.lane[:2] + c.reg[2:], c.reg[:2] + c.lane[2:], c.warp)


CONVERTS = {
    "regs": (
        lambda: blk.Blocked((64, 64), (2, 4), (8, 8), (4, 1)),
        lambda: blk.Blocked((64, 64), (2, 4), (8, 8), (4, 1)).reorder_regs([3, 0, 2, 1]),
        torch.float32,
    ),
    "warp_swaps": (_c64, lambda: _coalesced(_c64()), torch.float32),
    "warp_lane_perm_bf16": (
        lambda: blk.Blocked((64, 64), (1, 8), (8, 8), (4, 1)),
        lambda: (lambda L: LinearLayout(L.shape, L.reg, L.lane[::-1], L.warp))(
            blk.Blocked((64, 64), (1, 8), (8, 8), (4, 1))
        ),
        torch.bfloat16,
    ),
    "lds_blocked_to_mma": (lambda: blk.Blocked((64, 64), (1, 4), (8, 8), (4, 1)), _c64, torch.float32),
    "lds_mma_to_blocked_bf16": (_c64, lambda: blk.Blocked((64, 64), (1, 8), (8, 8), (4, 1)), torch.bfloat16),
}


@pytest.mark.parametrize("name", list(CONVERTS))
def test_convert(name):
    src, dst, dtype = CONVERTS[name]
    expected = {"regs": "regs", "warp_swaps": "warp", "warp_lane_perm_bf16": "warp"}.get(name, "lds")
    x, y = run_roundtrip(roundtrip_kernel(64, 64, src, dst), 128, 64, dtype)
    assert torch.equal(x, y)
    assert blk.plan_convert(*_plan_pair(name)).kind == expected


def _plan_pair(name):
    # the plan kind is pure Python; rebuild the layouts without a kernel when possible
    from flydsl_block.linear import LinearLayout as LL

    c = LL.from_tv(((16, 8, 2), ((4, 1), (1, 1))), ((32, 4, 512), ((1, 0), (0, 0))), (32, 32)).with_repeats((64, 64))
    if name == "warp_swaps":
        return c, _coalesced(c)
    if name == "lds_blocked_to_mma":
        return blk.blocked((64, 64), (1, 4), (8, 8), (4, 1)), c
    if name == "lds_mma_to_blocked_bf16":
        return c, blk.blocked((64, 64), (1, 8), (8, 8), (4, 1))
    src, dst, _ = CONVERTS[name]
    return src(), dst()


REDUCE_LAYOUTS = {
    "blocked_2x2warps": lambda: blk.Blocked((64, 64), (1, 4), (8, 8), (2, 2)),
    "blocked_replicated_warp": lambda: blk.Blocked((64, 64), (1, 4), (4, 16), (2, 2)),
    "mma_c": _c64,
}


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("name", list(REDUCE_LAYOUTS))
def test_reduce(name, axis, op):
    fn = REDUCE_LAYOUTS[name]

    @flyc.kernel
    def kern(X: fx.Tensor, Out: fx.Tensor):
        x = blk.load(tile2d(X, 64, 64), fn())
        r = getattr(x, op)(axis)
        blk.store(fx.flat_divide(fx.rocdl.make_buffer_tensor(Out, max_size=False), (64,))[None, 0], r)

    @flyc.jit
    def launch(X: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kern(X, Out).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream)

    x = torch.randn(64, 64, device="cuda")
    o = torch.zeros(64, device="cuda")
    launch(x, o, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = x.sum(axis) if op == "sum" else x.amax(axis)
    torch.testing.assert_close(o, ref, rtol=1e-5, atol=1e-5)


def test_softmax():
    bm, bn = 64, 128

    @flyc.kernel
    def kern(X: fx.Tensor, Y: fx.Tensor, M: fx.Int32):
        bid = fx.block_idx.x
        L = blk.Blocked((bm, bn), (1, 8), (4, 16), (4, 1))
        bounds = (M - fx.Int32(bid) * bm, None)
        x = blk.load(tile2d(X, bm, bn, bid), L, bounds=bounds).to(fx.Float32)
        p = (x - x.max(axis=1)[:, None]).exp()
        y = p / p.sum(axis=1)[:, None]
        blk.store(tile2d(Y, bm, bn, bid), y.to(fx.BFloat16), bounds=bounds)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kern(X, Y, M).launch(grid=((M + bm - 1) // bm, 1, 1), block=(THREADS, 1, 1), stream=stream)

    x = (torch.randn(200, bn, device="cuda") * 3).to(torch.bfloat16)
    y = torch.zeros_like(x)
    launch(x, y, 200, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.testing.assert_close(y.float(), torch.softmax(x.float(), dim=1), rtol=2e-2, atol=2e-3)


def test_gemm_runtime_k_loop():
    bm, bn, bk = 64, 64, 32

    @flyc.kernel
    def kern(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, K: fx.Int32):
        tm = make_tiled_mma()
        sa, sb, sc = blk.Mma(tm, "A", (bm, bk)), blk.Mma(tm, "B", (bn, bk)), blk.Mma(tm, "C", (bm, bn))
        bi, bj = fx.block_idx.x, fx.block_idx.y
        gA = fx.flat_divide(fx.rocdl.make_buffer_tensor(A, max_size=False), (bm, bk))[None, None, bi, None]
        gB = fx.flat_divide(fx.rocdl.make_buffer_tensor(B, max_size=False), (bn, bk))[None, None, bj, None]
        acc = blk.zeros(sc, fx.Float32)
        for kk in range(K // bk):
            acc = blk.dot(blk.load(gA[None, None, kk], sa), blk.load(gB[None, None, kk], sb), acc)
        blk.store(tile2d(C, bm, bn, bi, bj), acc)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, K: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kern(A, B, C, K).launch(grid=(2, 2, 1), block=(THREADS, 1, 1), stream=stream)

    a = torch.randn(128, 256, device="cuda").to(torch.bfloat16)
    b = torch.randn(128, 256, device="cuda").to(torch.bfloat16)
    c = torch.zeros(128, 128, device="cuda")
    launch(a, b, c, 256, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a.float() @ b.float().T, rtol=1e-3, atol=1e-3)


def test_interop_wrap_and_frag():
    """Thread-level code -> wrap -> block ops -> .frag -> thread-level code."""
    bm, bn, bk = 64, 64, 32

    @flyc.kernel
    def kern(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, R: fx.Tensor):
        tid = fx.thread_idx.x
        tm = make_tiled_mma()
        thr_mma = tm.thr_slice(tid)
        gA, gB, gC = tile2d(A, bm, bk), tile2d(B, bn, bk), tile2d(C, bm, bn)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        tca = fx.make_tiled_copy_A(atom, tm).get_slice(tid)
        tcb = fx.make_tiled_copy_B(atom, tm).get_slice(tid)
        fa, fb = thr_mma.make_fragment_A(gA), thr_mma.make_fragment_B(gB)
        fx.copy(atom, tca.partition_S(gA), tca.retile(fa))
        fx.copy(atom, tcb.partition_S(gB), tcb.retile(fb))
        fc = thr_mma.make_fragment_C(gC)
        fc.fill(0.0)
        fx.gemm(tm, fc, fa, fb, fc)
        acc = blk.wrap(fc, blk.Mma(tm, "C", (bm, bn)))  # thread fragment -> block
        rmax = acc.max(axis=1)
        blk.store(fx.flat_divide(fx.rocdl.make_buffer_tensor(R, max_size=False), (bm,))[None, 0], rmax)
        out = (acc * 2.0).frag  # block -> thread fragment (MMA C shape)
        atom_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcc = fx.make_tiled_copy_C(atom_c, tm).get_slice(tid)
        fx.copy(atom_c, tcc.retile(out), tcc.partition_S(gC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, R: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kern(A, B, C, R).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream)

    a = torch.randn(bm, bk, device="cuda").to(torch.bfloat16)
    b = torch.randn(bn, bk, device="cuda").to(torch.bfloat16)
    c = torch.zeros(bm, bn, device="cuda")
    r = torch.zeros(bm, device="cuda")
    launch(a, b, c, r, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    torch.testing.assert_close(c, ref * 2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(r, ref.amax(1), rtol=1e-3, atol=1e-3)
