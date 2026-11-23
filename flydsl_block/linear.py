"""F2-linear distribution layouts (scheme B), pure Python.

A ``LinearLayout`` maps (register, lane, warp) bits of a workgroup to a coordinate
of a block tile.  Every input bit owns one *basis* (a coordinate vector); the
coordinate of an element is the XOR of the bases of all set input bits.  A zero
basis means replication (the bit does not change the coordinate).

This module has no FlyDSL dependency: all planning (vector width, reduce,
broadcast, convert) is done here at trace time; ``fragment.py`` only emits IR
from the plans.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Sequence, Tuple

Basis = Tuple[int, ...]


def is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def log2(x: int) -> int:
    if not is_pow2(x):
        raise ValueError(f"{x} is not a power of two")
    return x.bit_length() - 1


def _xor(a: Basis, b: Basis) -> Basis:
    return tuple(x ^ y for x, y in zip(a, b))


def _atoms(b: Basis) -> List[Tuple[int, int]]:
    """(dim, bit) pairs set in a basis."""
    return [(d, bit) for d, c in enumerate(b) for bit in range(c.bit_length()) if (c >> bit) & 1]


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _flatten_modes(shape, stride) -> List[Tuple[int, int]]:
    """Flatten a (nested) CuTe shape/stride pair into colex-ordered (size, stride) leaves."""
    if isinstance(shape, int):
        return [(shape, stride)]
    out = []
    for s, d in zip(shape, stride):
        out.extend(_flatten_modes(s, d))
    return out


def _eval_modes(leaves, i: int) -> int:
    idx = 0
    for size, stride in leaves:
        idx += (i % size) * stride
        i //= size
    return idx


@dataclass(frozen=True)
class LinearLayout:
    shape: Tuple[int, ...]
    reg: Tuple[Basis, ...]
    lane: Tuple[Basis, ...]
    warp: Tuple[Basis, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        for name in ("reg", "lane", "warp"):
            object.__setattr__(self, name, tuple(tuple(int(c) for c in b) for b in getattr(self, name)))
        for s in self.shape:
            if not is_pow2(s):
                raise ValueError(f"tile shape {self.shape} must be powers of two")
        for b in self.reg + self.lane + self.warp:
            if len(b) != len(self.shape):
                raise ValueError(f"basis {b} does not match rank {len(self.shape)}")
            for c, s in zip(b, self.shape):
                if c < 0 or c >= s:
                    raise ValueError(f"basis {b} is outside tile {self.shape}")

    # Zero-length DSL protocol: a layout is trace-time metadata, so it can be
    # referenced inside FlyDSL runtime loops without becoming loop state.
    def __extract_to_ir_values__(self):
        return []

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        return exemplar

    # ------------------------------------------------------------------ basics
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def num_regs(self) -> int:
        return 1 << len(self.reg)

    @property
    def warp_size(self) -> int:
        return 1 << len(self.lane)

    @property
    def num_warps(self) -> int:
        return 1 << len(self.warp)

    @property
    def num_threads(self) -> int:
        return self.warp_size * self.num_warps

    @property
    def thread_bases(self) -> Tuple[Basis, ...]:
        return self.lane + self.warp

    @property
    def size(self) -> int:
        return _prod(self.shape)

    def zero(self) -> Basis:
        return (0,) * self.rank

    def reg_coord(self, r: int) -> Basis:
        c = self.zero()
        for i, b in enumerate(self.reg):
            if (r >> i) & 1:
                c = _xor(c, b)
        return c

    def thread_coord(self, t: int) -> Basis:
        c = self.zero()
        for j, b in enumerate(self.thread_bases):
            if (t >> j) & 1:
                c = _xor(c, b)
        return c

    def __call__(self, r: int, t: int) -> Basis:
        return _xor(self.reg_coord(r), self.thread_coord(t))

    def colex(self, coord: Sequence[int]) -> int:
        idx, mul = 0, 1
        for c, s in zip(coord, self.shape):
            idx += c * mul
            mul *= s
        return idx

    def replicated_thread_mask(self) -> int:
        """Thread-id bits whose basis is zero (threads that hold copies)."""
        return sum(1 << j for j, b in enumerate(self.thread_bases) if not any(b))

    def is_disjoint(self) -> bool:
        """Every (dim, bit) is used by at most one basis, so XOR == addition."""
        seen = set()
        for b in self.reg + self.lane + self.warp:
            for a in _atoms(b):
                if a in seen:
                    return False
                seen.add(a)
        return True

    def is_surjective(self) -> bool:
        need = {(d, bit) for d, s in enumerate(self.shape) for bit in range(log2(s))}
        have = {a for b in self.reg + self.lane + self.warp for a in _atoms(b)}
        return need <= have

    # --------------------------------------------------------------- algebra
    def with_regs(self, reg) -> "LinearLayout":
        return LinearLayout(self.shape, tuple(reg), self.lane, self.warp)

    def reorder_regs(self, order: Sequence[int]) -> "LinearLayout":
        if sorted(order) != list(range(len(self.reg))):
            raise ValueError(f"{order} is not a permutation of the register bits")
        return self.with_regs(self.reg[i] for i in order)

    def transposed(self, perm: Sequence[int]) -> "LinearLayout":
        """Permute tile dims: new dim i is old dim perm[i]. Moves no data."""

        def p(b):
            return tuple(b[i] for i in perm)

        return LinearLayout(p(self.shape), tuple(map(p, self.reg)), tuple(map(p, self.lane)), tuple(map(p, self.warp)))

    def with_repeats(self, shape: Sequence[int]) -> "LinearLayout":
        """Cover a larger tile by repeating this layout in registers (dim 0 first)."""
        shape = tuple(shape)
        reg = list(self.reg)
        for d, (small, big) in enumerate(zip(self.shape, shape)):
            if big % small:
                raise ValueError(f"tile {shape} is not a multiple of {self.shape}")
            for i in range(log2(big // small)):
                e = [0] * self.rank
                e[d] = small << i
                reg.append(tuple(e))
        return LinearLayout(shape, tuple(reg), self.lane, self.warp)

    def _split_axis(self, b: Basis, axis: int):
        others = tuple(c for d, c in enumerate(b) if d != axis)
        if b[axis] and any(others):
            raise NotImplementedError(f"basis {b} mixes the reduced axis with other dims")
        return others

    def sliced(self, axis: int) -> "LinearLayout":
        """Layout of a reduction along ``axis`` (Triton's SliceLayout).

        Registers along ``axis`` (or zero) are dropped, lane/warp bits along
        ``axis`` become zero bases, i.e. the result is replicated over them.
        """
        reg = [o for o in (self._split_axis(b, axis) for b in self.reg) if any(o)]
        lane = [self._split_axis(b, axis) for b in self.lane]
        warp = [self._split_axis(b, axis) for b in self.warp]
        shape = tuple(s for d, s in enumerate(self.shape) if d != axis)
        return LinearLayout(shape, tuple(reg), tuple(lane), tuple(warp))

    def slice_reg_map(self, axis: int) -> List[int]:
        """For each register of self, its register index in ``self.sliced(axis)``."""
        keep = [i for i, b in enumerate(self.reg) if any(self._split_axis(b, axis))]
        return [sum(((r >> i) & 1) << k for k, i in enumerate(keep)) for r in range(self.num_regs)]

    def contiguous_reg_bits(self, dim: int) -> List[int]:
        """Register bits whose bases are dim+1, dim+2, dim+4, ... (in that order)."""
        bits, want = [], 1
        while True:
            hit = None
            for i, b in enumerate(self.reg):
                if b[dim] == want and all(c == 0 for d, c in enumerate(b) if d != dim):
                    hit = i
                    break
            if hit is None:
                return bits
            bits.append(hit)
            want <<= 1

    # ------------------------------------------------- CuTe (TV layout) bridge
    def tv_modes(self, reg_order: Optional[Sequence[int]] = None):
        """CuTe ``((thr), (val)) : ((thr strides), (val strides))`` over the colex tile index.

        Value mode enumerates registers in ``reg_order``.  Requires a disjoint
        layout (then XOR == addition and the map is a CuTe layout).
        """
        if not self.is_disjoint():
            raise NotImplementedError("XOR-mixed layouts have no CuTe TV form")
        order = list(range(len(self.reg))) if reg_order is None else list(reg_order)
        mul = [_prod(self.shape[:d]) for d in range(self.rank)]

        def lin(b):
            return sum(c * m for c, m in zip(b, mul))

        thr = _merge_bits([lin(b) for b in self.thread_bases])
        val = _merge_bits([lin(self.reg[i]) for i in order])
        return (thr[0], val[0]), (thr[1], val[1])

    @staticmethod
    def from_tv(tv_shape, tv_stride, tile: Sequence[int], warp_size: int = 64) -> "LinearLayout":
        """Read a CuTe TV layout ((thr),(val)) -> colex tile index; verify it is linear."""
        tile = tuple(tile)
        thr = _flatten_modes(tv_shape[0], tv_stride[0])
        val = _flatten_modes(tv_shape[1], tv_stride[1])
        nthr, nval = _prod(s for s, _ in thr), _prod(s for s, _ in val)
        if not is_pow2(nthr) or not is_pow2(nval):
            raise NotImplementedError(f"TV layout with {nthr} threads x {nval} values is not a power of two")
        if nthr % warp_size:
            raise NotImplementedError(f"{nthr} threads is not a multiple of the warp size {warp_size}")

        def coord(idx):
            c = []
            for s in tile:
                c.append(idx % s)
                idx //= s
            if idx:
                raise ValueError("TV layout reaches outside the tile")
            return tuple(c)

        tb = [coord(_eval_modes(thr, 1 << j)) for j in range(log2(nthr))]
        vb = [coord(_eval_modes(val, 1 << i)) for i in range(log2(nval))]
        nl = log2(warp_size)
        out = LinearLayout(tile, tuple(vb), tuple(tb[:nl]), tuple(tb[nl:]))
        tv = [_eval_modes(thr, t) for t in range(nthr)]
        vv = [_eval_modes(val, v) for v in range(nval)]
        for t in range(nthr):
            tc = out.thread_coord(t)
            for v in range(nval):
                if coord(tv[t] + vv[v]) != _xor(tc, out.reg_coord(v)):
                    raise NotImplementedError(f"TV layout is not F2-linear at thread {t}, value {v}")
        return out

    # ----------------------------------------------------------------- misc
    def describe(self) -> str:
        def fmt(bs):
            return " ".join(str(b) for b in bs) or "-"

        return f"shape={self.shape} reg[{fmt(self.reg)}] lane[{fmt(self.lane)}] warp[{fmt(self.warp)}]"

    def owner_table(self) -> List[List[str]]:
        """2D tiles only: 'T<thread>.r<reg>' of the first owner of each element."""
        if self.rank != 2:
            raise NotImplementedError("owner_table supports 2D tiles")
        table = [["" for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        for t in range(self.num_threads):
            for r in range(self.num_regs):
                m, n = self(r, t)
                if not table[m][n]:
                    table[m][n] = f"T{t}.r{r}"
        return table

    def to_svg(self, path: Optional[str] = None, **kwargs) -> str:
        """Picture of the tile: warp / lane / register owners, lane colours and the bases.
        Writes ``path`` if given; returns the SVG text.

        kwargs: ``style`` ("auto" default: "lens" = warp tiles -> lens into a warp tile ->
        lens into a thread tile, when the layout has that hierarchy, else "grid" = one cell
        per element), ``title``, ``highlight_thread`` (grid; default 0, None = off),
        ``cell=(w, h)`` pixels (grid), ``matrix`` (default True: also draw the F2 matrix
        ``A`` of ``w = A v``).
        """
        from .viz import layout_svg

        svg = layout_svg(self, **kwargs)
        if path is not None:
            with open(path, "w") as f:
                f.write(svg)
        return svg


def _merge_bits(strides: List[int]):
    """Merge size-2 modes with doubling (or all-zero) strides into CuTe modes."""
    modes: List[List[int]] = []
    for s in strides:
        if modes and ((s == 0 and modes[-1][1] == 0) or (s != 0 and modes[-1][1] * modes[-1][0] == s)):
            modes[-1][0] *= 2
        else:
            modes.append([2, s])
    if not modes:
        return 1, 0
    if len(modes) == 1:
        return modes[0][0], modes[0][1]
    return tuple(m[0] for m in modes), tuple(m[1] for m in modes)


def blocked(shape, size_per_thread, threads_per_warp, warps_per_block, order=None) -> LinearLayout:
    """Triton-style BlockedLayout.

    ``order`` lists dims from fastest to slowest (default: last dim fastest).
    Registers, lanes and warps are each laid out in ``order``; a tile larger
    than one block coverage repeats in registers, a smaller one replicates.
    """
    shape = tuple(shape)
    rank = len(shape)
    order = tuple(range(rank - 1, -1, -1)) if order is None else tuple(order)
    if sorted(order) != list(range(rank)):
        raise ValueError(f"order {order} is not a permutation of dims")

    def e(d, v):
        b = [0] * rank
        b[d] = v if v < shape[d] else 0
        return tuple(b)

    reg, lane, warp = [], [], []
    for d in order:
        reg += [e(d, 1 << i) for i in range(log2(size_per_thread[d]))]
    for d in order:
        lane += [e(d, size_per_thread[d] << i) for i in range(log2(threads_per_warp[d]))]
    for d in order:
        warp += [e(d, size_per_thread[d] * threads_per_warp[d] << i) for i in range(log2(warps_per_block[d]))]
    for d in order:
        cover = size_per_thread[d] * threads_per_warp[d] * warps_per_block[d]
        if shape[d] > cover:
            reg += [e(d, cover << i) for i in range(log2(shape[d] // cover))]
    return LinearLayout(shape, tuple(reg), tuple(lane), tuple(warp))


# ----------------------------------------------------------------- planning
@dataclass
class ReducePlan:
    reg_reduce: List[int]
    reg_drop: List[int]
    lane_bits: List[int]
    warp_bits: List[int]
    out: LinearLayout


def plan_reduce(layout: LinearLayout, axis: int) -> ReducePlan:
    reg_reduce, reg_drop = [], []
    for i, b in enumerate(layout.reg):
        others = layout._split_axis(b, axis)
        if not any(others):
            (reg_reduce if b[axis] else reg_drop).append(i)
    lane_bits = [j for j, b in enumerate(layout.lane) if b[axis]]
    warp_bits = [k for k, b in enumerate(layout.warp) if b[axis]]
    return ReducePlan(reg_reduce, reg_drop, lane_bits, warp_bits, layout.sliced(axis))


@dataclass
class ConvertPlan:
    kind: str  # identity | regs | warp | lds
    steps: list = field(default_factory=list)

    def describe(self) -> str:
        if self.kind in ("identity", "lds"):
            return self.kind
        parts = []
        for s in self.steps:
            if s[0] == "swap":
                parts.append(f"swap(reg{s[1]}<->lane{s[2]})")
            elif s[0] == "lane_perm":
                parts.append(f"lane_perm{tuple(s[1])}")
            elif s[0] == "regs":
                parts.append("reg_rename")
        return f"{self.kind}: " + ", ".join(parts)


def _single_atoms(bases):
    out = []
    for b in bases:
        a = _atoms(b)
        if len(a) != 1:
            return None
        out.append(a[0])
    return out


def _reg_mask(src: LinearLayout, dst: LinearLayout) -> Optional[List[int]]:
    """dst register r -> src register with the same register coordinate."""
    where = {}
    for r in range(src.num_regs):
        where.setdefault(src.reg_coord(r), r)
    mask = [where.get(dst.reg_coord(r)) for r in range(dst.num_regs)]
    return None if None in mask else mask


def plan_convert(src: LinearLayout, dst: LinearLayout) -> ConvertPlan:
    if src.shape != dst.shape:
        raise ValueError(f"cannot convert between tiles {src.shape} and {dst.shape}")
    if src == dst:
        return ConvertPlan("identity")
    if src.thread_bases == dst.thread_bases:
        mask = _reg_mask(src, dst)
        if mask is not None:
            return ConvertPlan("regs", [("regs", mask)])
    if src.warp == dst.warp and src.lane and len(src.lane) == len(dst.lane) and len(src.reg) == len(dst.reg):
        steps = _plan_warp_local(src, dst)
        if steps is not None:
            return ConvertPlan("warp", steps)
    return ConvertPlan("lds")


def _plan_warp_local(src: LinearLayout, dst: LinearLayout):
    """Register<->lane bit swaps, then one lane permutation, then a register rename.

    Needs permutation-type layouts inside the warp: one atom per reg/lane basis,
    no replication, and the same set of atoms on both sides.
    """
    s_reg, s_lane = _single_atoms(src.reg), _single_atoms(src.lane)
    d_reg, d_lane = _single_atoms(dst.reg), _single_atoms(dst.lane)
    if None in (s_reg, s_lane, d_reg, d_lane):
        return None
    if sorted(s_reg + s_lane) != sorted(d_reg + d_lane) or len(set(s_reg + s_lane)) != len(s_reg + s_lane):
        return None
    cur_reg, cur_lane = list(s_reg), list(s_lane)
    want_lane = set(d_lane)
    steps = []
    for j in range(len(cur_lane)):
        if cur_lane[j] not in want_lane:
            i = next(i for i, a in enumerate(cur_reg) if a in want_lane)
            steps.append(("swap", i, j))
            cur_reg[i], cur_lane[j] = cur_lane[j], cur_reg[i]
    if cur_lane != d_lane:
        steps.append(("lane_perm", [cur_lane.index(a) for a in d_lane]))
        cur_lane = list(d_lane)
    if cur_reg != d_reg:
        pos = [cur_reg.index(a) for a in d_reg]
        n = len(d_reg)
        steps.append(("regs", [sum(((r >> b) & 1) << pos[b] for b in range(n)) for r in range(1 << n)]))
    return steps


def lane_perm_source(perm: Sequence[int], lane: int) -> int:
    """Source lane for ``lane`` in a lane_perm step (dst lane bit j <- src lane bit perm[j])."""
    return sum(((lane >> j) & 1) << p for j, p in enumerate(perm))


def copy_plan(layout: LinearLayout, unit_dim: Optional[int], elem_bits: int, max_vec: Optional[int] = None):
    """Vector width (elements) and register order for a tile copy.

    The contiguous register bits along the memory's unit-stride dim go first,
    so one copy atom covers ``vec`` adjacent elements.
    """
    bits = layout.contiguous_reg_bits(unit_dim) if unit_dim is not None else []
    cap = max(1, 128 // elem_bits)
    if max_vec is not None:
        cap = min(cap, max_vec)
    while bits and (1 << len(bits)) > cap:
        bits.pop()
    order = bits + [i for i in range(len(layout.reg)) if i not in bits]
    return 1 << len(bits), order


def copy_index(order: Sequence[int], r: int) -> int:
    """Copy-order position of layout register ``r``."""
    return sum(((r >> i) & 1) << k for k, i in enumerate(order))


def all_coords(layout: LinearLayout):
    """(thread, reg, coord) for every element owner (testing helper)."""
    for t, r in product(range(layout.num_threads), range(layout.num_regs)):
        yield t, r, layout(r, t)
