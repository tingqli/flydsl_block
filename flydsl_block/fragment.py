"""Block-level fragments for FlyDSL (scheme A on top of the scheme-B LinearLayout).

A ``BlockFragment`` is a trace-time object: a flat ``fx.Vector`` holding this
thread's registers plus the ``LinearLayout`` that says which tile coordinate
each (register, thread) owns.  All planning happens in ``linear.py``; this
module only emits existing FlyDSL operations (tiled copies, ``fx.gemm``,
Vector ops, ``gpu.shuffle``, LDS), so nothing survives in the IR except
thread-level code.
"""

import operator
from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

from .linear import (
    LinearLayout,
    blocked,
    copy_index,
    copy_plan,
    log2,
    plan_convert,
    plan_reduce,
)

__all__ = [
    "BlockFragment",
    "Blocked",
    "TV",
    "Mma",
    "MmaSpec",
    "Shared",
    "load",
    "store",
    "copy",
    "dot",
    "wrap",
    "zeros",
    "full",
]


def _warp_size() -> int:
    get = getattr(fx.gpu, "num_warp_threads", None)
    return get() if get is not None else 64


def _flat(t):
    if isinstance(t, (tuple, list)):
        out = []
        for x in t:
            out.extend(_flat(x))
        return out
    return [t]


def _as_tuple(t):
    return tuple(t) if isinstance(t, (tuple, list)) else (t,)


def _size(t):
    n = 1
    for x in _flat(t):
        n *= x
    return n


# --------------------------------------------------------------- layouts (A)
def Blocked(shape, size_per_thread, threads_per_warp, warps=None, order=None) -> LinearLayout:
    """Triton/Gluon-style BlockedLayout; ``order`` lists dims fastest first."""
    warps = tuple(warps) if warps is not None else (1,) * len(shape)
    return blocked(shape, size_per_thread, threads_per_warp, warps, order)


def TV(tv_layout, tile, warp_size=None) -> LinearLayout:
    """Any static FlyDSL TV layout ((thr),(val)) -> colex tile index."""
    return LinearLayout.from_tv(
        tv_layout.shape.to_py_value(), tv_layout.stride.to_py_value(), tile, warp_size or _warp_size()
    )


@dataclass(frozen=True, eq=False)
class MmaSpec:
    """An MMA operand layout: the LinearLayout plus the TiledMma that emits it."""

    tiled_mma: object
    operand: str
    tile: tuple
    linear: LinearLayout
    atom_bits: int = 0  # leading register bits that one MMA instruction holds

    def __extract_to_ir_values__(self):
        return []

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        return exemplar

    def partition_shape(self):
        op = getattr(fx.MmaOperand, self.operand)
        return fx.tiled_mma_partition_shape(op, self.tiled_mma, self.tile)


_MMA_CACHE = {}


def Mma(tiled_mma, operand: str, tile) -> MmaSpec:
    """Operand layout of ``tiled_mma`` over ``tile`` ((M,K) for A, (N,K) for B, (M,N) for C).

    Built from ``tiled_mma.tv_layout_{A,B,C}_tiled`` (one ``tile_size_mnk`` tile, values
    ``(V, (repeats along dim 0, dim 1))``) plus the repeats of that tile over ``tile``.
    Registers follow ``make_fragment_X``: ``(V, dim-0 repeats, dim-1 repeats)``, colex,
    where in each dim the repeats inside one tiled tile (from a permutation) come first.
    """
    operand = operand.upper()
    tile = tuple(int(x) for x in tile)
    key = (str(tiled_mma.type), operand, tile)
    hit = _MMA_CACHE.get(key)
    if hit is None:
        tv = {"A": tiled_mma.tv_layout_A_tiled, "B": tiled_mma.tv_layout_B_tiled, "C": tiled_mma.tv_layout_C_tiled}[
            operand
        ]
        m, n, k = tiled_mma.tile_size_mnk.to_py_value()
        sub = {"A": (m, k), "B": (n, k), "C": (m, n)}[operand]
        tv_shape = tv.shape.to_py_value()
        lin = LinearLayout.from_tv(tv_shape, tv.stride.to_py_value(), sub, _warp_size()).with_repeats(tile)
        vals = tv_shape[1]
        if isinstance(vals, tuple) and len(vals) == 2 and isinstance(vals[1], tuple) and len(vals[1]) == 2:
            nv, n0, n1 = log2(_size(vals[0])), log2(_size(vals[1][0])), log2(_size(vals[1][1]))
        else:
            nv, n0, n1 = log2(_size(vals)), 0, 0
        o0, o1 = log2(tile[0] // sub[0]), log2(tile[1] // sub[1])
        # lin registers: V, inner dim 0, inner dim 1, outer dim 0, outer dim 1
        inner0 = list(range(nv, nv + n0))
        inner1 = list(range(nv + n0, nv + n0 + n1))
        outer0 = list(range(nv + n0 + n1, nv + n0 + n1 + o0))
        outer1 = list(range(nv + n0 + n1 + o0, nv + n0 + n1 + o0 + o1))
        lin = lin.reorder_regs(list(range(nv)) + inner0 + outer0 + inner1 + outer1)
        fshape = fx.tiled_mma_partition_shape(getattr(fx.MmaOperand, operand), tiled_mma, tile).to_py_value()
        want = (1 << nv, 1 << (n0 + o0), 1 << (n1 + o1))
        if len(fshape) != 3 or tuple(_size(s) for s in fshape) != want:
            raise NotImplementedError(f"MMA {operand} fragment {fshape} does not have the expected modes {want}")
        hit = _MMA_CACHE[key] = (lin, nv)
    return MmaSpec(tiled_mma, operand, tile, *hit)


def _layout_of(layout):
    return layout.linear if isinstance(layout, MmaSpec) else layout


# --------------------------------------------------------- runtime helpers
def _tid():
    return fx.Int32(fx.thread_idx.x)


def _thread_coord(layout: LinearLayout, tid):
    """Runtime tile coordinate of this thread's register 0 (XOR of thread bases)."""
    coord = [0] * layout.rank
    for j, b in enumerate(layout.thread_bases):
        if not any(b):
            continue
        bit = (tid >> j) & 1
        for d, c in enumerate(b):
            if c:
                coord[d] = coord[d] ^ (bit * c)
    return [c if not isinstance(c, int) else fx.Int32(c) for c in coord]


def _op(op, a, b):
    if op == "add":
        return a + b
    if op == "max":
        return fx.arith.max(a, b)
    if op == "min":
        return fx.arith.min(a, b)
    raise ValueError(f"unknown reduce op {op!r}")


def _copy_atom(view, bits, dtype):
    if "buffer_desc" in str(view.address_space):
        op = getattr(fx.rocdl, f"BufferCopy{bits}b")()
    else:
        op = fx.UniversalCopy(bits)
    return fx.make_copy_atom(op, dtype)


def _atom_conds(layout: LinearLayout, order, nvec: int, bounds, owner_only: bool):
    """One predicate per copy atom: its first element is in bounds (and this thread owns it)."""
    tid = _tid()
    owner = None
    mask = layout.replicated_thread_mask()
    if owner_only and mask:
        owner = (tid & mask) == 0
    base = _thread_coord(layout, tid) if bounds is not None else None
    conds = []
    for a in range(layout.num_regs // nvec):
        c = a * nvec  # copy-order register of the atom's first element
        r = sum(((c >> k) & 1) << i for k, i in enumerate(order))
        cond = owner
        if bounds is not None:
            rc = layout.reg_coord(r)
            for d, b in enumerate(bounds):
                if b is None:
                    continue
                ok = (base[d] ^ rc[d] if rc[d] else base[d]) < b
                cond = ok if cond is None else (cond & ok)
        conds.append(fx.Boolean(True) if cond is None else cond)
    return conds


def _mem_offset(coord, modes, strides):
    """Element offset of a tile coordinate in a view whose dims may be nested modes
    (None if it depends on a dynamic stride)."""
    off = 0
    for c, m, st in zip(coord, modes, strides):
        for s, t in zip(_flat(m), _flat(st)):
            if c % s:
                if not isinstance(t, int):
                    return None
                off += (c % s) * t
            c //= s
    return off


def _unflatten(i, shape):
    if isinstance(shape, int):
        return i
    out = []
    for s in shape:
        size = 1
        for x in _flat(s):
            size *= x
        out.append(_unflatten(i % size, s))
        i //= size
    return tuple(out)


def _atom_slices(part, n_atoms):
    """Partition coordinates selecting each copy atom: ((None, rest), 0, ...)."""
    pshape = part.shape.to_py_value()
    tail = (0,) * (len(pshape) - 1)
    if not isinstance(pshape[0], tuple):
        if n_atoms != 1:
            raise ValueError(f"unexpected partition shape {pshape}")
        return [(None,) + tail]
    return [((None, _unflatten(a, pshape[0][1])),) + tail for a in range(n_atoms)]


def _tile_copy(
    view, layout: LinearLayout, *, vec=None, bounds=None, other=0, max_vec=None, owner_only=False, keep_bits=0
):
    """Copy a whole tile between ``view`` (global/LDS) and registers.

    Each view dim may be a nested mode (e.g. ``((16, 8), (8, 4))``); vectors run
    along the dim whose first sub-mode has stride 1, at most that sub-mode long.
    Copy order: the vector, then register bits ``< keep_bits`` (one MMA
    instruction's values), then the rest by memory offset.
    load: ``vec is None`` -> returns the registers in ``layout`` order.
    store: writes ``vec`` (``layout`` order).
    """
    shape = layout.shape
    modes = _as_tuple(view.shape.to_py_value())
    strides = _as_tuple(view.layout.stride.to_py_value())
    if tuple(_size(m) for m in modes) != shape:
        raise ValueError(f"view shape {modes} does not match the layout tile {shape}")
    unit_dim, run = None, None
    for d, (m, st) in enumerate(zip(modes, strides)):
        first = next(((s, t) for s, t in zip(_flat(m), _flat(st)) if s > 1), None)
        if first is not None and isinstance(first[1], int) and first[1] == 1:
            unit_dim, run = d, first[0]
            break
    if run is not None and run < shape[unit_dim]:
        max_vec = run if max_vec is None else min(max_vec, run)
    dtype = view.element_type
    nvec, order = copy_plan(layout, unit_dim, dtype.width, max_vec)
    # after the vector: the rest of one MMA instruction's values, then the other register
    # bits along the smallest memory strides first, so consecutive copies touch adjacent memory
    nb = log2(nvec)
    atom = [i for i in order[nb:] if i < keep_bits]
    rest = [i for i in order[nb:] if i >= keep_bits]
    offsets = [_mem_offset(layout.reg[i], modes, strides) for i in rest]
    if None not in offsets:
        rest = [i for _, i in sorted(zip(offsets, rest), key=lambda t: t[0])]
    order = order[:nb] + atom + rest
    (ts, vs), (tst, vst) = layout.tv_modes(order)
    tv = fx.make_layout((ts, vs), (tst, vst))
    atom = _copy_atom(view, nvec * dtype.width, dtype)
    thr = fx.make_tiled_copy(atom, tv, fx.make_tile(*shape)).get_slice(fx.thread_idx.x)
    nreg = layout.num_regs
    to_copy = [copy_index(order, r) for r in range(nreg)]  # layout reg -> copy position
    need_pred = bounds is not None or (owner_only and layout.replicated_thread_mask())
    conds = _atom_conds(layout, order, nvec, bounds, owner_only) if need_pred else None

    def do_copy(src, dst):
        if conds is None:
            fx.copy(atom, src, dst)
            return
        # one call per atom with a single predicate: works for any nesting of the value mode
        for a, sl in enumerate(_atom_slices(src, nreg // nvec)):
            p = fx.make_rmem_tensor(1, fx.Boolean)
            p[0] = conds[a]
            fx.copy(atom, src[sl], dst[sl], pred=p)

    if vec is None:
        part = thr.partition_S(view)
        frag = fx.make_fragment_like(part)
        if conds is not None:
            frag.fill(other)
        do_copy(part, frag)
        v = fx.Vector(frag.load())
        return v if to_copy == list(range(nreg)) else v.shuffle(v, to_copy)
    part = thr.partition_D(view)
    frag = fx.make_fragment_like(part)
    inv = [0] * nreg
    for r, c in enumerate(to_copy):
        inv[c] = r
    frag.store(vec if inv == list(range(nreg)) else vec.shuffle(vec, inv))
    do_copy(frag, part)
    return None


def _lds_view(shape, dtype, unit_dim):
    n = 1
    for s in shape:
        n *= s
    arr = fx.SharedAllocator().allocate(fx.Array[dtype, n]).peek()
    order = [unit_dim] + [d for d in range(len(shape)) if d != unit_dim]
    stride, mul = [0] * len(shape), 1
    for d in order:
        stride[d] = mul
        mul *= shape[d]
    return arr.view(fx.make_layout(tuple(shape), tuple(stride)))


# ------------------------------------------------------------- shared tiles
class Shared:
    """A 2D (rows, cols) LDS tile, cols contiguous, with Triton-style XOR swizzle.

    Columns are grouped in chunks of ``vec`` elements; the chunk index is XORed
    with ``(row // per_phase) % max_phase``::

        offset(r, c) = r * cols + ((c // vec) ^ ((r // per_phase) % max_phase)) * vec + c % vec

    ``vec=8, max_phase=8`` for bf16 rows of 128 B: the 8 lanes of one ds_read_b128
    phase (8 rows, same chunk) and of one ds_write_b128 phase (1 row, 8 chunks)
    hit all 32 banks.  ``max_phase=1`` = no swizzle.  ``memory`` is an LDS array
    (``SharedAllocator`` field) with at least ``rows * cols`` elements.
    Build it at the top of the kernel; ``slice`` gives sub-tiles that keep the
    swizzle of the whole tile.
    """

    def __init__(self, memory, shape, vec=1, per_phase=1, max_phase=1, *, _origin=(0, 0), _views=None, _stride=None):
        rows, cols = self.shape = tuple(shape)
        for name, x in (("vec", vec), ("per_phase", per_phase), ("max_phase", max_phase)):
            if x < 1 or x & (x - 1):
                raise ValueError(f"{name}={x} must be a power of two")
        self.vec, self.per_phase, self.max_phase = vec, per_phase, max_phase
        self.origin = _origin
        self.row_stride = cols if _stride is None else _stride
        if _views is None:
            if cols % vec:
                raise ValueError(f"cols {cols} must be a multiple of vec {vec}")
            # one view per vector width w <= vec: (w, vec // w, cols // vec, rows)
            _views, w = {}, vec
            while w >= 1:
                _views[w] = memory.view(fx.make_ordered_layout((w, vec // w, cols // vec, rows), 0))
                w //= 2
        self._views = _views
        self.element_type = _views[vec].element_type

    # DSL protocol: no run-time state of its own (the views are built before any loop)
    def __extract_to_ir_values__(self):
        return []

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        return exemplar

    def slice(self, start: int, length: int, dim: int = 0) -> "Shared":
        """Static sub-tile ``[start, start + length)`` along ``dim``."""
        shape = list(self.shape)
        if start % length or start + length > shape[dim]:
            raise ValueError(f"slice [{start}, {start + length}) of dim {dim} in {self.shape}")
        shape[dim] = length
        origin = list(self.origin)
        origin[dim] += start
        return Shared(
            None,
            shape,
            self.vec,
            self.per_phase,
            self.max_phase,
            _origin=tuple(origin),
            _views=self._views,
            _stride=self.row_stride,
        )

    def _chunk(self, nvec, r, c):
        """nvec-element view at tile coordinate (r, c) (Int32, origin included)."""
        chunk = c >> log2(self.vec)
        if self.max_phase > 1:
            chunk = chunk ^ ((r >> log2(self.per_phase)) & (self.max_phase - 1))
        sub = 0 if nvec == self.vec else (c & (self.vec - 1)) >> log2(nvec)
        return self._views[nvec][None, sub, chunk, r]


def _dma_copy(dst: Shared, src, warps: int):
    """Global / buffer tile view -> Shared tile by LDS DMA, see ``copy``."""
    rows, cols = dst.shape
    if cols != dst.row_stride or dst.origin[1]:
        raise NotImplementedError("DMA needs whole rows of the Shared tile (no column slice)")
    if tuple(_size(m) for m in _as_tuple(src.shape.to_py_value())) != dst.shape:
        raise ValueError(f"view shape {src.shape.to_py_value()} does not match the Shared tile {dst.shape}")
    dtype = dst.element_type
    # elements per lane: one chunk inside a swizzle chunk; 16 B chunks only on gfx950 (else 4 B)
    wide = get_rocm_arch().startswith("gfx95")
    nvec = min(dst.vec, (128 if wide else 32) // dtype.width)
    bits = nvec * dtype.width
    ws = _warp_size()
    cpr = cols // nvec  # chunks per row
    lane_rows = ws // cpr  # rows one instruction fills
    if bits not in (32, 128) or warps & (warps - 1) or ws % cpr or rows % (warps * lane_rows):
        raise NotImplementedError(f"DMA of a {dst.shape} tile with {bits}-bit chunks over {warps} warps")
    wave_rows = rows // warps  # each wave fills its own block of rows
    step = ws * bits // 8  # LDS bytes per instruction
    group = 4096 // step  # instructions sharing one M0: the rest of the LDS step is the 12-bit offset
    lv, lp, pm = log2(dst.vec), log2(dst.per_phase), dst.max_phase - 1
    lsub = log2(dst.vec // nvec)

    def phase(r):
        return (r >> lp) & pm

    # this thread's chunk in the first instruction: (row, physical chunk); the global vector to
    # fetch is the one the swizzle puts there (XOR is its own inverse)
    tid = _tid()
    lane, wave = tid & (ws - 1), tid >> log2(ws)
    row_t = wave * wave_rows + (lane >> log2(cpr))
    thread_rows = (lane_rows - 1) | ((warps - 1) * wave_rows)  # row bits that come from the thread
    pc = lane & (cpr - 1)
    chunk_t, sub_t = pc >> lsub, pc & ((1 << lsub) - 1)
    # LDS: a wave-uniform base (M0); the hardware adds the instruction offset and lane * chunk
    m0 = fx.Int32(fx.rocdl.readfirstlane(fx.Int32.ir_type, wave * (wave_rows * cols)))
    lds = fx.add_offset(fx.get_iter(dst._views[dst.vec]), m0)
    # global: a buffer descriptor based at the tile, so only the base changes from tile to tile
    # (SGPRs) and the per-thread offsets stay the same; a buffer view is used as it is
    gsrc = fx.get_iter(src)
    if "buffer_desc" not in str(src.address_space):
        gsrc = fx.rocdl.make_buffer_ptr(gsrc)
    atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS(bits), dtype)
    for j in range(wave_rows // lane_rows):
        r0 = dst.origin[0] + j * lane_rows  # parent row added to this thread's row
        if r0 & thread_rows == 0:  # disjoint row bits: split the phase
            chunk = chunk_t ^ phase(row_t) ^ phase(r0) if pm else chunk_t
        else:
            chunk = chunk_t ^ phase(row_t + r0)
        col = (chunk << lv) + (sub_t * nvec if lsub else 0)
        # the instruction offset moves both addresses: take it off the global one
        imm = j % group * step
        off = fx.get_scalar(fx.crd2idx((row_t + j * lane_rows, col), src.layout))
        s = fx.make_view(fx.add_offset(gsrc, off - imm * 8 // dtype.width), fx.make_layout(nvec, 1))
        d = fx.make_view(fx.add_offset(lds, (r0 - j % group * lane_rows) * cols), fx.make_layout(nvec, 1))
        fx.copy_atom_call(atom.set_value("imm_offset", fx.Int32(imm)) if imm else atom, s, d)
    fx.rocdl.asyncmark()


def _concat(parts):
    """Concatenate equal-length vectors (a power-of-two count) with shuffles."""
    while len(parts) > 1:
        n = parts[0].numel
        parts = [parts[i].shuffle(parts[i + 1], list(range(2 * n))) for i in range(0, len(parts), 2)]
    return parts[0]


def _shared_copy(tile: Shared, layout: LinearLayout, vec=None, max_vec=None):
    """Register <-> swizzled LDS copy, one ds_read/ds_write per vector of contiguous columns.

    When a register's constant (row, col) offset has no bits in common with the
    thread's, the address is the thread's base pointer (one per distinct XORed
    chunk) plus a constant element offset, i.e. an immediate offset of the
    ds_read / ds_write, as in hand-written code.  Other registers use the full
    swizzle formula.
    """
    if layout.shape != tile.shape:
        raise ValueError(f"shared tile {tile.shape} does not match the layout tile {layout.shape}")
    if vec is not None and layout.replicated_thread_mask():
        raise NotImplementedError("storing a replicated layout to a Shared tile")
    cap = tile.vec if max_vec is None else min(tile.vec, max_vec)
    nvec, order = copy_plan(layout, 1, tile.element_type.width, cap)
    nreg = layout.num_regs
    to_copy = [copy_index(order, r) for r in range(nreg)]  # layout reg -> copy position
    identity = to_copy == list(range(nreg))
    if vec is not None and not identity:
        inv = [0] * nreg
        for r, c in enumerate(to_copy):
            inv[c] = r
        vec = vec.shuffle(vec, inv)  # copy order

    tr, tc = _thread_coord(layout, _tid())
    bits_r = bits_c = 0
    for b in layout.thread_bases:
        bits_r, bits_c = bits_r | b[0], bits_c | b[1]
    lv, lp, lw, pm = log2(tile.vec), log2(tile.per_phase), log2(nvec), tile.max_phase - 1

    def phase(r):
        return (r >> lp) & pm

    # the thread's own swizzled chunk (and sub-chunk), and which chunk bits it can set
    base_chunk = (tc >> lv) ^ phase(tr) if pm else tc >> lv
    base_bits = (bits_c >> lv) | (phase(bits_r) if pm else 0)
    base_sub = (tc & (tile.vec - 1)) >> lw if nvec < tile.vec else 0
    bases = {}  # XORed constant chunk -> the thread's base pointer

    def base_ptr(kx):
        if kx not in bases:
            bases[kx] = fx.get_iter(tile._views[nvec][None, base_sub, base_chunk ^ kx if kx else base_chunk, tr])
        return bases[kx]

    parts = []
    for c0 in range(0, nreg, nvec):
        r = sum(((c0 >> k) & 1) << i for k, i in enumerate(order))  # the atom's first register
        rc = layout.reg_coord(r)
        cr, cc = rc[0] + tile.origin[0], rc[1] + tile.origin[1]
        if cr & bits_r or cc & bits_c:  # constant and thread bits overlap: full formula
            chunk = tile._chunk(nvec, tr ^ cr, tc ^ cc)
        else:
            k = (cc >> lv) ^ (phase(cr) if pm else 0)
            kx = k & base_bits  # XORed into the thread's chunk; the rest is a plain offset
            off = cr * tile.row_stride + (k & ~base_bits) * tile.vec + ((cc & (tile.vec - 1)) >> lw) * nvec
            ptr = base_ptr(kx)
            chunk = fx.make_view(fx.add_offset(ptr, off) if off else ptr, fx.make_layout(nvec, 1))
        if vec is None:
            parts.append(fx.Vector(chunk.load()))
        else:
            chunk.store(vec.shuffle(vec, list(range(c0, c0 + nvec))))
    if vec is not None:
        return None
    v = _concat(parts)
    return v if identity else v.shuffle(v, to_copy)


# ------------------------------------------------------------ the fragment
class BlockFragment:
    """A block tile distributed over the workgroup by a LinearLayout (value semantics)."""

    def __init__(self, vec, layout, mma=None):
        if isinstance(layout, MmaSpec):
            mma, layout = layout, layout.linear
        vec = fx.Vector(vec)
        if vec.numel != layout.num_regs:
            raise ValueError(f"{vec.numel} registers do not match layout with {layout.num_regs}")
        self.vec = vec
        self.layout = layout
        self.mma = mma

    # DSL protocol: the registers are the only run-time state, carried as 128-bit pieces
    # (like hand-written per-tile vectors) rather than one wide vector.
    def __extract_to_ir_values__(self):
        n = max(1, 128 // self.vec.dtype.width)
        total = self.vec.numel
        if total <= n or total % n:
            return [self.vec]
        return [self.vec.shuffle(self.vec, list(range(i, i + n))) for i in range(0, total, n)]

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        parts = [fx.Vector(v) for v in values]
        return cls(_concat(parts) if len(parts) > 1 else parts[0], exemplar.layout, exemplar.mma)

    def __repr__(self):
        return f"BlockFragment({self.dtype.__name__}, {self.layout.describe()})"

    @property
    def dtype(self):
        return self.vec.dtype

    @property
    def shape(self):
        return self.layout.shape

    # ---------------------------------------------- thread-level interop
    @property
    def frag(self):
        """This thread's registers as an rmem ``fx.Tensor`` (MMA fragment shape for Mma layouts)."""
        shape = self.mma.partition_shape() if self.mma is not None else self.layout.num_regs
        t = fx.make_rmem_tensor(shape, self.dtype)
        t.store(self.vec)
        return t

    def coords(self):
        """Run-time tile coordinate of each register of this thread."""
        base = _thread_coord(self.layout, _tid())
        out = []
        for r in range(self.layout.num_regs):
            rc = self.layout.reg_coord(r)
            out.append(tuple(base[d] ^ rc[d] if rc[d] else base[d] for d in range(self.layout.rank)))
        return out

    # ----------------------------------------------------- elementwise
    def _like(self, vec):
        return BlockFragment(vec, self.layout, self.mma)

    def _binary(self, other, fn, reverse=False):
        if isinstance(other, _Expanded):
            other = other.broadcast_like(self)
        if isinstance(other, BlockFragment):
            if other.layout != self.layout:
                raise ValueError(f"layout mismatch:\n  {self.layout.describe()}\n  {other.layout.describe()}")
            other = other.vec
        return self._like(fn(other, self.vec) if reverse else fn(self.vec, other))

    def __add__(self, o):
        return self._binary(o, operator.add)

    def __radd__(self, o):
        return self._binary(o, operator.add, True)

    def __sub__(self, o):
        return self._binary(o, operator.sub)

    def __rsub__(self, o):
        return self._binary(o, operator.sub, True)

    def __mul__(self, o):
        return self._binary(o, operator.mul)

    def __rmul__(self, o):
        return self._binary(o, operator.mul, True)

    def __truediv__(self, o):
        return self._binary(o, operator.truediv)

    def __rtruediv__(self, o):
        return self._binary(o, operator.truediv, True)

    def __neg__(self):
        return self._like(-self.vec)

    def maximum(self, o):
        return self._binary(o, fx.arith.max)

    def minimum(self, o):
        return self._binary(o, fx.arith.min)

    def to(self, dtype):
        return self._like(self.vec.to(dtype))

    def exp(self):
        return self._like(fx.exp(self.vec))

    def exp2(self):
        return self._like(fx.exp2(self.vec))

    # ----------------------------------------------- shape-only changes
    @property
    def T(self):
        if self.layout.rank != 2:
            raise ValueError("T needs a 2D tile")
        return BlockFragment(self.vec, self.layout.transposed((1, 0)))

    def __getitem__(self, idx):
        """``x[:, None]`` / ``x[None, :]``: broadcast along the new axis (resolved by the other operand)."""
        idx = idx if isinstance(idx, tuple) else (idx,)
        axes = [i for i, s in enumerate(idx) if s is None]
        full_slices = all(s is None or s == slice(None) for s in idx)
        if len(axes) != 1 or len(idx) != self.layout.rank + 1 or not full_slices:
            raise NotImplementedError("only x[:, None]-style broadcasting is supported")
        return _Expanded(self, axes[0])

    # ----------------------------------------------------------- reduce
    def reduce(self, axis: int, op: str = "add"):
        plan = plan_reduce(self.layout, axis)
        v = self.vec
        if v.dtype.width < 32 and v.dtype.is_float:
            v = v.to(fx.Float32)
        dtype = v.dtype
        cur = list(range(len(self.layout.reg)))
        for bit in plan.reg_drop + plan.reg_reduce:
            p = cur.index(bit)
            lo = [i for i in range(1 << len(cur)) if not (i >> p) & 1]
            v_lo = v.shuffle(v, lo)
            v = v_lo if bit in plan.reg_drop else _op(op, v_lo, v.shuffle(v, [i | (1 << p) for i in lo]))
            cur.pop(p)
        if plan.lane_bits:
            w = self.layout.warp_size
            elems = [v[i] for i in range(v.numel)]
            for j in plan.lane_bits:
                elems = [_op(op, e, fx.gpu.shuffle_xor(e, 1 << j, w)) for e in elems]
            v = fx.Vector.from_elements(elems, dtype)
        if plan.warp_bits:
            v = self._warp_reduce(v, plan, op)
        return BlockFragment(v, plan.out)

    def _warp_reduce(self, v, plan, op):
        out, nl = plan.out, len(self.layout.lane)
        tid = _tid()
        nw = 1 << len(plan.warp_bits)
        wr = 0
        for k, b in enumerate(plan.warp_bits):
            wr = wr + (((tid >> (nl + b)) & 1) << k)
        base = _thread_coord(out, tid)
        base_idx = 0
        mul = 1
        for d in range(out.rank):
            base_idx = base_idx + base[d] * mul
            mul *= out.shape[d]
        n_out = out.size
        arr = fx.SharedAllocator().allocate(fx.Array[v.dtype, n_out * nw]).peek()
        fx.gpu.barrier()
        idx = [base_idx + out.colex(out.reg_coord(q)) for q in range(out.num_regs)]
        for q in range(out.num_regs):
            arr[idx[q] + wr * n_out] = v[q]
        fx.gpu.barrier()
        vals = []
        for q in range(out.num_regs):
            acc = arr[idx[q]]
            for w in range(1, nw):
                acc = _op(op, acc, arr[idx[q] + w * n_out])
            vals.append(acc)
        fx.gpu.barrier()
        return fx.Vector.from_elements(vals, v.dtype)

    def sum(self, axis: int):
        return self.reduce(axis, "add")

    def max(self, axis: int):
        return self.reduce(axis, "max")

    def min(self, axis: int):
        return self.reduce(axis, "min")

    # ---------------------------------------------------------- convert
    def convert(self, layout, explain: bool = False):
        mma = layout if isinstance(layout, MmaSpec) else None
        dst = _layout_of(layout)
        plan = plan_convert(self.layout, dst)
        if explain:
            print(f"[flydsl_block] convert {plan.describe()}")
        if plan.kind == "identity":
            return BlockFragment(self.vec, dst, mma)
        if plan.kind == "regs":
            return BlockFragment(self.vec.shuffle(self.vec, plan.steps[0][1]), dst, mma)
        if plan.kind == "warp" and (self.dtype.width == 32 or self.dtype.is_float):
            v = self.vec if self.dtype.width == 32 else self.vec.to(fx.Float32)
            v = _run_warp_steps(v, plan.steps, self.layout.warp_size)
            return BlockFragment(v if v.dtype is self.dtype else v.to(self.dtype), dst, mma)
        return BlockFragment(_convert_lds(self, dst), dst, mma)


class _Expanded:
    """``x[:, None]``: a fragment waiting to be broadcast along ``axis``."""

    def __init__(self, frag: BlockFragment, axis: int):
        self.frag, self.axis = frag, axis

    def broadcast_like(self, parent: BlockFragment) -> BlockFragment:
        want = parent.layout.sliced(self.axis)
        child = self.frag if self.frag.layout == want else self.frag.convert(want)
        mask = parent.layout.slice_reg_map(self.axis)
        return BlockFragment(child.vec.shuffle(child.vec, mask), parent.layout, parent.mma)

    def _swap(self, other, fn):
        if not isinstance(other, BlockFragment):
            raise TypeError("broadcast needs a BlockFragment on the other side")
        return other._binary(self, fn, reverse=True)

    def __add__(self, o):
        return self._swap(o, operator.add)

    def __sub__(self, o):
        return self._swap(o, operator.sub)

    def __mul__(self, o):
        return self._swap(o, operator.mul)

    def __truediv__(self, o):
        return self._swap(o, operator.truediv)


def _run_warp_steps(vec, steps, warp_size):
    """Execute a warp-local convert plan on 32-bit registers (no LDS)."""
    dtype = vec.dtype
    elems = [vec[i] for i in range(vec.numel)]
    lane = _tid() & (warp_size - 1)
    for step in steps:
        if step[0] == "swap":
            _, i, j = step
            hi = ((lane >> j) & 1) == 1
            for r in range(len(elems)):
                if (r >> i) & 1:
                    continue
                r1 = r | (1 << i)
                a, b = elems[r], elems[r1]
                x = fx.gpu.shuffle_xor(hi.select(a, b), 1 << j, warp_size)
                elems[r] = hi.select(x, a)
                elems[r1] = hi.select(b, x)
        elif step[0] == "lane_perm":
            src = 0
            for j, p in enumerate(step[1]):
                src = src + (((lane >> j) & 1) << p)
            elems = [fx.gpu.shuffle_idx(e, src, warp_size) for e in elems]
        else:
            elems = [elems[m] for m in step[1]]
    return fx.Vector.from_elements(elems, dtype)


def _convert_lds(x: BlockFragment, dst: LinearLayout):
    """General convert through LDS: write with the source layout, read with the target."""
    shape, bits = dst.shape, x.dtype.width
    best = None
    for unit_dim in sorted(set([0, len(shape) - 1])):
        vs, _ = copy_plan(x.layout, unit_dim, bits)
        vd, _ = copy_plan(dst, unit_dim, bits)
        score = (min(vs, vd), vs + vd)
        if best is None or score > best[0]:
            best = (score, unit_dim)
    view = _lds_view(shape, x.dtype, best[1])
    _tile_copy(view, x.layout, vec=x.vec)
    fx.gpu.barrier()
    v = _tile_copy(view, dst)
    fx.gpu.barrier()
    return v


# --------------------------------------------------------------- free API
def load(view, layout, *, bounds=None, other=0, max_vec=None) -> BlockFragment:
    """Load a whole tile view (global or LDS) or a ``Shared`` tile into a BlockFragment.

    ``bounds`` (per dim, None = full) masks atoms whose first element is out of
    range; they read ``other``.  Bounds along the vectorized dim must be a
    multiple of the vector width (or pass ``max_vec=1``).
    """
    mma = layout if isinstance(layout, MmaSpec) else None
    lin = _layout_of(layout)
    if isinstance(view, Shared):
        if bounds is not None:
            raise NotImplementedError("bounds on a Shared tile")
        return BlockFragment(_shared_copy(view, lin, max_vec=max_vec), lin, mma)
    keep = mma.atom_bits if mma is not None else 0
    v = _tile_copy(view, lin, bounds=bounds, other=other, max_vec=max_vec, keep_bits=keep)
    return BlockFragment(v, lin, mma)


def store(view, x: BlockFragment, *, bounds=None, max_vec=None):
    """Store a BlockFragment to a tile view or a ``Shared`` tile; replicated copies are written once."""
    vec = x.vec if x.dtype is view.element_type else x.vec.to(view.element_type)
    if isinstance(view, Shared):
        if bounds is not None:
            raise NotImplementedError("bounds on a Shared tile")
        _shared_copy(view, x.layout, vec=vec, max_vec=max_vec)
        return
    keep = x.mma.atom_bits if x.mma is not None else 0
    _tile_copy(view, x.layout, vec=vec, bounds=bounds, max_vec=max_vec, owner_only=True, keep_bits=keep)


def copy(dst: Shared, src, *, warps: int):
    """Copy a whole global (or buffer) tile view into a ``Shared`` tile with LDS DMA
    (``buffer_load ... lds``, gfx950), without registers.

    Each instruction writes 64 consecutive 16 B chunks of LDS, lane i at chunk i
    (``warps`` waves cover consecutive rows, round after round), so each thread
    fetches the global vector the swizzle places at its chunk; the 8 lanes of a
    128 B row still read that whole row.  Asynchronous: each call is one group.
    Before reading the tile, ``rocdl.wait_asyncmark(n)`` (at most ``n`` later
    copies still in flight) and ``gpu.barrier()`` (the other waves' chunks).
    """
    _dma_copy(dst, src, warps)


def wrap(frag, layout) -> BlockFragment:
    """Wrap an existing thread-level fragment (e.g. ``thr_mma.make_fragment_C``)."""
    return BlockFragment(fx.Vector(frag.load()), layout)


def full(layout, value, dtype) -> BlockFragment:
    return BlockFragment(fx.Vector.filled((_layout_of(layout).num_regs,), value, dtype), layout)


def zeros(layout, dtype) -> BlockFragment:
    return full(layout, 0.0 if dtype.is_float else 0, dtype)


def dot(a: BlockFragment, b: BlockFragment, acc: BlockFragment, *, traversal_order=None) -> BlockFragment:
    """acc + a @ b.T with ``acc`` in an ``Mma(tm, "C", ...)`` layout; a is (M,K), b is (N,K).

    ``traversal_order`` (``fx.GemmTraversalOrder``, first letter = innermost loop)
    sets the MFMA issue order, e.g. ``KNM`` = for m: for n: for k.
    """
    if acc.mma is None or acc.mma.operand != "C":
        raise ValueError("dot needs acc created with an Mma(tiled_mma, 'C', tile) layout")
    tm = acc.mma.tiled_mma
    (bm, bn), bk = acc.shape, a.shape[1]
    spec_a, spec_b = Mma(tm, "A", (bm, bk)), Mma(tm, "B", (bn, bk))
    if a.layout != spec_a.linear:
        a = a.convert(spec_a)
    if b.layout != spec_b.linear:
        b = b.convert(spec_b)
    fa = fx.make_rmem_tensor(spec_a.partition_shape(), a.dtype)
    fa.store(a.vec)
    fb = fx.make_rmem_tensor(spec_b.partition_shape(), b.dtype)
    fb.store(b.vec)
    fc = fx.make_rmem_tensor(acc.mma.partition_shape(), acc.dtype)
    fc.store(acc.vec)
    fx.gemm(tm, fc, fa, fb, fc, traversal_order=traversal_order)
    return BlockFragment(fc.load(), acc.layout, acc.mma)
