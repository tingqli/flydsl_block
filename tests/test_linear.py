"""CPU tests for the F2 planning core (no GPU, no FlyDSL tracing).

The layout facts come from FlyDSL 0.5.0 trace-time probes (see docs); the
convert/reduce plans are checked with a small register-file simulator that
mirrors the GPU executor in ``fragment.py`` step by step.
"""

import random

import pytest

from flydsl_block.linear import (
    LinearLayout,
    _eval_modes,
    _flatten_modes,
    blocked,
    copy_index,
    copy_plan,
    lane_perm_source,
    plan_convert,
    plan_reduce,
)

# Printed by FlyDSL 0.5.0 for MFMA(16,16,16,bf16) and make_tiled_mma(atom, (2,2,1):(1,2,0)).
MFMA_C = (((16, 4), (4, 1)), ((16, 4), (1, 16)))
TILED_C = (((16, 8, 2), ((4, 1), (1, 1))), ((32, 4, 512), ((1, 0), (0, 0))))
TILED_A = (((16, 4, 2, 2), (4, (1, 1))), ((1, 128, 16, 0), (32, (0, 0))))


def test_from_tv_mfma_c():
    L = LinearLayout.from_tv(*MFMA_C, (16, 16))
    assert L.reg == ((1, 0), (2, 0))
    assert L.lane == ((0, 1), (0, 2), (0, 4), (0, 8), (4, 0), (8, 0))
    assert L.warp == ()


def test_from_tv_tiled():
    c = LinearLayout.from_tv(*TILED_C, (32, 32))
    assert c.warp == ((16, 0), (0, 16))
    a = LinearLayout.from_tv(*TILED_A, (32, 16))
    assert a.reg == ((0, 1), (0, 2))
    assert a.warp == ((16, 0), (0, 0))  # replicated over the N-direction warp
    assert a.replicated_thread_mask() == 1 << 7


def test_from_tv_rejects_nonlinear():
    with pytest.raises(NotImplementedError):  # 3 values per thread
        LinearLayout.from_tv((64, 3), (3, 1), (192,))


def test_blocked_matches_make_layout_tv():
    # make_layout_tv((8,8):(8,1), (1,8):(1,1)) -> ((8,8),8):((64,1),8) over (8,64)
    L = blocked((8, 64), (1, 8), (8, 8), (1, 1))
    ref = LinearLayout.from_tv(((8, 8), 8), ((64, 1), 8), (8, 64))
    assert L == ref


def test_blocked_repeat_and_replicate():
    L = blocked((64, 64), (1, 8), (8, 8), (4, 1))
    assert L.num_regs == 16 and L.warp == ((8, 0), (16, 0))
    assert L.reg[-1] == (32, 0)  # repeat along m
    small = blocked((16, 64), (1, 8), (8, 8), (4, 1))
    assert small.warp == ((8, 0), (0, 0))  # coverage 32 > 16 rows: last warp bit replicates


def _random_layout(rng, shape=(32, 32), nreg=4, nlane=6):
    atoms = [(d, b) for d, s in enumerate(shape) for b in range(s.bit_length() - 1)]
    rng.shuffle(atoms)

    def basis(a):
        e = [0] * len(shape)
        e[a[0]] = 1 << a[1]
        return tuple(e)

    bases = [basis(a) for a in atoms]
    return LinearLayout(shape, bases[:nreg], bases[nreg : nreg + nlane], bases[nreg + nlane :])


@pytest.mark.parametrize("seed", range(20))
def test_tv_modes_roundtrip(seed):
    L = _random_layout(random.Random(seed))
    order = list(range(len(L.reg)))
    random.Random(seed + 100).shuffle(order)
    shape, stride = L.tv_modes(order)
    thr, val = _flatten_modes(shape[0], stride[0]), _flatten_modes(shape[1], stride[1])
    for t in range(L.num_threads):
        for c in range(L.num_regs):
            r = sum(((c >> k) & 1) << i for k, i in enumerate(order))
            assert _eval_modes(thr, t) + _eval_modes(val, c) == L.colex(L(r, t))


def test_copy_plan():
    L = blocked((64, 64), (1, 8), (8, 8), (4, 1))
    assert copy_plan(L, 1, 16)[0] == 8  # bf16 x8 = 128 bit
    assert copy_plan(L, 1, 32)[0] == 4  # f32 x4
    assert copy_plan(L, 0, 32)[0] == 1  # column-major memory: nothing contiguous
    c = LinearLayout.from_tv(*TILED_C, (32, 32))
    assert copy_plan(c, 1, 32)[0] == 1 and copy_plan(c, 0, 32)[0] == 4
    vec, order = copy_plan(LinearLayout((8, 8), ((1, 0), (0, 2), (0, 1)), (), ()), 1, 32)
    assert vec == 4 and order == [2, 1, 0]
    assert [copy_index(order, r) for r in range(8)] == [0, 4, 2, 6, 1, 5, 3, 7]


# ------------------------------------------------------------------ simulator
def simulate_convert(src: LinearLayout, plan):
    """Run a plan on a register file holding each element's coordinate."""
    T, R, W = src.num_threads, src.num_regs, src.warp_size
    regs = [[src(r, t) for r in range(R)] for t in range(T)]
    for step in plan.steps:
        if step[0] == "swap":
            _, i, j = step
            new = [row[:] for row in regs]
            for t in range(T):
                hi = (t >> j) & 1
                p = t ^ (1 << j)
                phi = (p >> j) & 1
                for r in range(R):
                    if (r >> i) & 1:
                        continue
                    r1 = r | (1 << i)
                    x = regs[p][r] if phi else regs[p][r1]  # partner sends hi ? a : b
                    new[t][r] = x if hi else regs[t][r]
                    new[t][r1] = regs[t][r1] if hi else x
            regs = new
        elif step[0] == "lane_perm":
            regs = [regs[(t - t % W) + lane_perm_source(step[1], t % W)][:] for t in range(T)]
        else:
            regs = [[row[m] for m in step[1]] for row in regs]
    return regs


@pytest.mark.parametrize("seed", range(40))
def test_convert_warp_local(seed):
    rng = random.Random(seed)
    src = _random_layout(rng)
    pool = list(src.reg + src.lane)
    rng.shuffle(pool)
    dst = LinearLayout(src.shape, pool[: len(src.reg)], pool[len(src.reg) :], src.warp)
    plan = plan_convert(src, dst)
    assert plan.kind in ("warp", "regs", "identity")
    regs = simulate_convert(src, plan)
    for t in range(dst.num_threads):
        for r in range(dst.num_regs):
            assert regs[t][r] == dst(r, t)


def test_convert_kinds():
    c = LinearLayout.from_tv(*TILED_C, (32, 32))
    # swap reg bits (m+1, m+2) with lane bits (n+1, n+2): each lane gets 4 contiguous n
    coal = LinearLayout(c.shape, ((0, 1), (0, 2)), ((1, 0), (2, 0)) + c.lane[2:], c.warp)
    plan = plan_convert(c, coal)
    assert plan.kind == "warp" and [s[0] for s in plan.steps] == ["swap", "swap"]
    assert copy_plan(coal, 1, 32)[0] == 4
    assert plan_convert(c, c.reorder_regs([1, 0])).kind == "regs"
    assert plan_convert(c, blocked((32, 32), (1, 4), (8, 8), (4, 1))).kind == "lds"


def simulate_reduce(layout: LinearLayout, axis: int, data):
    plan = plan_reduce(layout, axis)
    T, nl = layout.num_threads, len(layout.lane)
    vals = []
    for t in range(T):
        v = [data[layout(r, t)] for r in range(layout.num_regs)]
        cur = list(range(len(layout.reg)))
        for bit in plan.reg_drop + plan.reg_reduce:
            p = cur.index(bit)
            lo = [i for i in range(1 << len(cur)) if not (i >> p) & 1]
            if bit in plan.reg_drop:
                v = [v[i] for i in lo]
            else:
                v = [v[i] + v[i | (1 << p)] for i in lo]
            cur.pop(p)
        vals.append(v)
    for j in plan.lane_bits:
        vals = [[a + b for a, b in zip(vals[t], vals[t ^ (1 << j)])] for t in range(T)]
    if plan.warp_bits:
        out = []
        for t in range(T):
            acc = [0] * len(vals[t])
            for w in range(1 << len(plan.warp_bits)):
                p = t
                for k, b in enumerate(plan.warp_bits):
                    p = (p & ~(1 << (nl + b))) | (((w >> k) & 1) << (nl + b))
                acc = [a + x for a, x in zip(acc, vals[p])]
            out.append(acc)
        vals = out
    return plan.out, vals


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "layout",
    [
        blocked((64, 64), (1, 4), (4, 16), (2, 2)),
        blocked((64, 64), (2, 2), (8, 8), (1, 4), order=(0, 1)),
        LinearLayout.from_tv(*TILED_C, (32, 32)).with_repeats((64, 64)),
    ],
)
def test_reduce_plan(layout, axis):
    rng = random.Random(1)
    data = {(m, n): rng.randint(-50, 50) for m in range(layout.shape[0]) for n in range(layout.shape[1])}
    out, vals = simulate_reduce(layout, axis, data)
    for t in range(layout.num_threads):
        for q in range(out.num_regs):
            (o,) = out(q, t)
            if axis == 1:
                ref = sum(data[(o, n)] for n in range(layout.shape[1]))
            else:
                ref = sum(data[(m, o)] for m in range(layout.shape[0]))
            assert vals[t][q] == ref


def test_slice_reg_map():
    L = blocked((64, 64), (2, 4), (8, 8), (2, 2))
    for axis in (0, 1):
        s, mp = L.sliced(axis), L.slice_reg_map(axis)
        for t in range(L.num_threads):
            for r in range(L.num_regs):
                full = L(r, t)
                assert s(mp[r], t) == tuple(c for d, c in enumerate(full) if d != axis)
