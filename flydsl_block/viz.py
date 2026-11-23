"""SVG pictures of a LinearLayout: who owns each element of the tile.

Pure Python, no dependencies.  Two drawing styles:

* ``lens`` (chosen automatically when the layout allows it): a small picture
  of the warp tiles of the block, a zoom ("annotation lens") into warp 0's
  tile showing its thread tiles, and a second zoom into thread 0's tile
  showing its registers.  Possible when, in every dim, the coordinate bits
  from low to high come from registers, then lanes, then warps / repeat
  registers.
* ``grid``: one cell per tile element with its thread ``T<t>`` and register
  ``r<r>``; cells of one thread merge into a block, lane colours, thick warp
  boundaries.

Both list the bases (reg / lane / warp bit -> coordinate step), a lane colour
key and the F2 matrix ``A`` of ``w = A v`` drawn like the LinearLayout paper.

    python -m flydsl_block.viz /tmp/layouts     # writes a few example SVGs
"""

import colorsys
import os
import sys
from html import escape

from .linear import _atoms

_SHADES = (0.86, 0.77, 0.68, 0.59)
_WARP_FILLS = ("#cfe2f3", "#fde2c4", "#d5ecd4", "#f4cccc", "#e1d5f0", "#fff2cc", "#d0e0e3", "#ead1dc")
_LENS = ("#e8590c", "#7048e8")
_STYLE = (
    "<style>text{font-family:DejaVu Sans Mono,Menlo,Consolas,monospace;font-size:11px}"
    ".c{text-anchor:middle;font-size:9px}.r{text-anchor:middle;font-size:8px;fill:#333}"
    ".ax{font-size:9px;fill:#666;paint-order:stroke;stroke:#fff;stroke-width:3px}.k{fill:#444}"
    ".lc{text-anchor:middle;font-size:11px}.ls{text-anchor:middle;font-size:9px;fill:#333}"
    ".pt{font-size:12px;font-weight:bold}"
    ".mh{font-size:12px;text-anchor:middle}.ml{font-size:13px;font-style:italic}.ma{font-size:15px}"
    ".m1{font-size:12px;text-anchor:middle;font-weight:bold}.m0{font-size:12px;text-anchor:middle;fill:#9a9a9a}"
    "</style>"
)


def lane_color(lane: int) -> str:
    r, g, b = colorsys.hls_to_rgb((lane % 16) / 16.0, _SHADES[(lane // 16) % 4], 0.62)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def owners(layout):
    """Element coordinate -> (first owner thread, its register, number of owners)."""
    regs = [layout.reg_coord(r) for r in range(layout.num_regs)]
    out = {}
    for t in range(layout.num_threads):
        tc = layout.thread_coord(t)
        for r, rc in enumerate(regs):
            c = tuple(a ^ b for a, b in zip(tc, rc))
            if c in out:
                out[c][2] += 1
            else:
                out[c] = [t, r, 1]
    return out


def _wrap(items, width_px, char_px=6.7):
    """Greedy line wrap of legend items to the picture width."""
    lines, cur = [], ""
    limit = max(20, int(width_px / char_px))
    for it in items:
        cand = f"{cur}  {it}" if cur else it
        if len(cand) > limit and cur:
            lines.append(cur)
            cur = "      " + it
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


# ------------------------------------------------------------------ hierarchy
class Hierarchy:
    """thread tile (registers) inside warp tile (lanes) inside the block (warps, repeats)."""

    def __init__(self, thread_tile, warp_tile, src, lane_copies, warp_copies):
        self.thread_tile, self.warp_tile = thread_tile, warp_tile
        self.src = src  # (dim, bit) -> ("reg" | "lane" | "warp", index)
        self.lane_copies, self.warp_copies = lane_copies, warp_copies
        self.outer_regs = any(k == "reg" and (1 << b) >= thread_tile[d] for (d, b), (k, _) in src.items())

    def locate(self, coord):
        """(register, lane, warp) holding ``coord``; the lowest copy for replicated bits."""
        idx = {"reg": 0, "lane": 0, "warp": 0}
        for d, c in enumerate(coord):
            for b in range(c.bit_length()):
                if (c >> b) & 1:
                    kind, i = self.src[(d, b)]
                    idx[kind] |= 1 << i
        return idx["reg"], idx["lane"], idx["warp"]


def hierarchy(layout):
    """Return a ``Hierarchy`` when every dim's coordinate bits, low to high, are
    ``reg* lane* (warp | reg)*`` and each nonzero basis is one coordinate bit used
    once (zero lane / warp bases = copies are fine); otherwise None.
    """
    if layout.rank not in (1, 2):
        return None
    src, zero = {}, {"lane": 0, "warp": 0}
    for kind, bases in (("reg", layout.reg), ("lane", layout.lane), ("warp", layout.warp)):
        for i, b in enumerate(bases):
            atoms = _atoms(b)
            if not atoms:
                if kind == "reg":
                    return None  # duplicated registers
                zero[kind] += 1
                continue
            if len(atoms) != 1 or atoms[0] in src:
                return None
            src[atoms[0]] = (kind, i)
    thread_tile, warp_tile = [], []
    for d, s in enumerate(layout.shape):
        kinds = [src.get((d, b), (None,))[0] for b in range(s.bit_length() - 1)]
        if None in kinds:
            return None
        k = 0
        while k < len(kinds) and kinds[k] == "reg":
            k += 1
        t = k
        while k < len(kinds) and kinds[k] == "lane":
            k += 1
        if "lane" in kinds[k:]:
            return None
        thread_tile.append(1 << t)
        warp_tile.append(1 << k)
    return Hierarchy(tuple(thread_tile), tuple(warp_tile), src, 1 << zero["lane"], 1 << zero["warp"])


# ---------------------------------------------------------------- matrix
def _matrix_panel(layout):
    """The F2 matrix ``A`` with ``w = A v``, drawn like the LinearLayout paper.

    Columns are the input bits grouped Reg | Lane | Warp, rows the coordinate
    bits: the last (fastest) dim first, low bit first, one labelled group per
    dim.  Returns ``(width, height, render(x, y) -> [svg elements])``.
    """
    names = {1: ("i",), 2: ("m", "n")}.get(layout.rank) or tuple(f"d{d}" for d in range(layout.rank))
    dim_rows = [(d, layout.shape[d].bit_length() - 1) for d in reversed(range(layout.rank))]
    dim_rows = [(d, n) for d, n in dim_rows if n]
    rows = [(d, b) for d, n in dim_rows for b in range(n)]
    groups = [(h, bs) for h, bs in (("Reg", layout.reg), ("Lane", layout.lane), ("Warp", layout.warp)) if bs]
    if not rows or not groups:
        return None
    cw, ch, gap = 15, 17, 10
    x_lbr = 40  # left bracket, after "A ="
    x_lab = x_lbr + 9  # row-group labels sit inside the bracket
    offs, x = [], x_lab + 16 + gap / 2
    for _, bs in groups:
        offs.append(x)
        x += len(bs) * cw + gap
    x_end = x - gap
    x_rbr = x_end + 8
    y_line, y_rows = 20, 24
    y_bot = y_rows + len(rows) * ch + 2
    caption = ["w = A v (mod 2)", "v: reg | lane | warp bits", "w: coordinate bits, low bit first"]
    width = max(x_rbr + 6, 6.7 * max(len(c) for c in caption))
    height = y_bot + 10 + 14 * len(caption)

    def render(x0, y0):
        out = []
        for (h, bs), ox in zip(groups, offs):
            out.append(f'<text class="mh" x="{x0 + ox + len(bs) * cw / 2}" y="{y0 + 14}">{h}</text>')
        d = [f"M{x0 + x_lab} {y0 + y_line}H{x0 + x_end}"]
        d += [f"M{x0 + ox - gap / 2} {y0 + y_line}V{y0 + y_bot}" for ox in offs]
        k = 0
        for i, (dim, n) in enumerate(dim_rows):
            y = y0 + y_rows + k * ch
            out.append(f'<text class="ml" x="{x0 + x_lab}" y="{y + ch / 2 + 4}">{names[dim]}</text>')
            if i:
                d.append(f"M{x0 + x_lab} {y}H{x0 + x_end}")
            k += n
        for r, (dim, b) in enumerate(rows):
            y = y0 + y_rows + r * ch + ch / 2 + 4
            for (_, bs), ox in zip(groups, offs):
                for c, basis in enumerate(bs):
                    bit = (basis[dim] >> b) & 1
                    out.append(f'<text class="m{bit}" x="{x0 + ox + c * cw + cw / 2}" y="{y}">{bit}</text>')
        top, bot = y0 + 2, y0 + y_bot + 2
        d.append(f"M{x0 + x_lbr + 6} {top}H{x0 + x_lbr}V{bot}H{x0 + x_lbr + 6}")
        d.append(f"M{x0 + x_rbr - 6} {top}H{x0 + x_rbr}V{bot}H{x0 + x_rbr - 6}")
        out.append(f'<path d="{" ".join(d)}" stroke="#222" stroke-width="1.2" fill="none"/>')
        out.append(f'<text class="ma" x="{x0}" y="{(top + bot) / 2 + 5}">A =</text>')
        for k, line in enumerate(caption):
            out.append(f'<text class="k" x="{x0}" y="{bot + 16 + 14 * k}">{escape(line)}</text>')
        return out

    return width, height, render


# ---------------------------------------------------------------- grid body
def _grid_body(layout, highlight_thread, cell):
    """One cell per element.  Cells of one thread merge into one block (borders
    only between different threads); each block shows its thread once and every
    cell its register.  A rank-1 tile is drawn as rows of up to 32 elements, each
    row labelled by its first index.  Returns (width, height, key lines, render(top)).
    """
    row_step = 1
    if layout.rank == 1:
        cols = min(layout.shape[0], 32)
        rows, row_step = layout.shape[0] // cols, cols

        def rc(c):
            return divmod(c[0], cols)
    else:
        rows, cols = layout.shape

        def rc(c):
            return c

    if cell is None:
        cw = 34 if cols <= 64 else max(6, 2176 // cols)
        ch = 22 if rows <= 64 else max(6, 1408 // rows)
    else:
        cw, ch = cell
    show_text = cw >= 26 and ch >= 18
    W = layout.warp_size
    grid = [[None] * cols for _ in range(rows)]
    for c, o in owners(layout).items():
        i, j = rc(c)
        grid[i][j] = o
    mine = set()
    if highlight_thread is not None:
        mine = {rc(layout(r, highlight_thread)) for r in range(layout.num_regs)}

    def thread(i, j):
        o = grid[i][j] if 0 <= i < rows and 0 <= j < cols else None
        return None if o is None else o[0]

    # one thread label per connected block of a thread: on its lowest register
    heads, seen = set(), set()
    for i in range(rows):
        for j in range(cols):
            if (i, j) in seen or grid[i][j] is None:
                continue
            t, stack, block = grid[i][j][0], [(i, j)], []
            seen.add((i, j))
            while stack:
                a, b = stack.pop()
                block.append((a, b))
                for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if (na, nb) not in seen and thread(na, nb) == t:
                        seen.add((na, nb))
                        stack.append((na, nb))
            heads.add(min(block, key=lambda p: grid[p[0]][p[1]][1]))

    left, pad = 44, 14
    key = [
        "block = one thread: T<thread> once, r<register> per element [xN = N copies]",
        "colour = lane (hue = lane%16, shade = lane/16)",
        "thick line = warp boundary",
    ]
    if highlight_thread is not None:
        key.append(f"black outline = thread {highlight_thread}")

    def render(top):
        s = []
        step_c = 1 if cw >= 20 else 8
        for j in range(0, cols, step_c):
            s.append(f'<text class="ax" x="{left + j * cw + cw / 2}" y="{top - 6}" text-anchor="middle">{j}</text>')
        step_r = 1 if ch >= 12 else 8
        for i in range(0, rows, step_r):
            y = top + i * ch + ch / 2 + 3
            s.append(f'<text class="ax" x="{left - 5}" y="{y}" text-anchor="end">{i * row_step}</text>')
        if layout.rank == 2:
            s.append(f'<text class="ax" x="{pad}" y="{top - 6}">m\\n</text>')

        for i in range(rows):
            for j in range(cols):
                x, y, o = left + j * cw, top + i * ch, grid[i][j]
                fill = "#dddddd" if o is None else lane_color(o[0] % W)
                s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}"/>')
                if not show_text or o is None:
                    continue
                t, r, k = o
                reg = f"r{r}" + (f"x{k}" if k > 1 else "")
                if (i, j) in heads:
                    s.append(f'<text class="c" x="{x + cw / 2}" y="{y + ch / 2 - 1}">T{t}</text>')
                    s.append(f'<text class="r" x="{x + cw / 2}" y="{y + ch / 2 + 8}">{reg}</text>')
                else:
                    s.append(f'<text class="r" x="{x + cw / 2}" y="{y + ch / 2 + 3}">{reg}</text>')

        def edges(differs):
            """Path of cell edges where differs(a, b) holds (b may be outside the grid)."""
            d = []
            for i in range(rows):
                for j in range(cols):
                    x, y = left + j * cw, top + i * ch
                    if differs((i, j), (i, j + 1)):
                        d.append(f"M{x + cw} {y}v{ch}")
                    if differs((i, j), (i + 1, j)):
                        d.append(f"M{x} {y + ch}h{cw}")
                    if j == 0 and differs((i, j), (i, -1)):
                        d.append(f"M{x} {y}v{ch}")
                    if i == 0 and differs((i, j), (-1, j)):
                        d.append(f"M{x} {y}h{cw}")
            return " ".join(d)

        def warp(p):
            t = thread(*p)
            return None if t is None else t // W

        d = edges(lambda a, b: thread(*a) != thread(*b))
        s.append(f'<path d="{d}" stroke="#777" stroke-width="0.8" fill="none"/>')
        s.append(f'<path d="{edges(lambda a, b: warp(a) != warp(b))}" stroke="#111" stroke-width="2.6" fill="none"/>')
        if mine:
            d = edges(lambda a, b: (a in mine) != (b in mine))
            s.append(f'<path d="{d}" stroke="black" stroke-width="2.2" fill="none"/>')
        return s

    return left + cols * cw, rows * ch, key, render


# ---------------------------------------------------------------- lens body
def _lens_body(layout, h):
    """Block of warp tiles -> lens into one warp tile (lanes) -> lens into one
    thread tile (registers).  Returns (width, height, key lines, render(top)).

    The lens magnifies the top-right cell of a panel into the next panel on its
    right, so the lens lines only cross the gap between the panels.
    """
    rank, T, Wt, W = layout.rank, h.thread_tile, h.warp_tile, layout.warp_size
    wrap = 16  # long rank-1 strips wrap after 16 cells

    def sz(v):
        return "x".join(map(str, v))

    def tiles(n, what):
        return f"{n} {what}{'s' if n > 1 else ''}"

    def counts(outer, inner):  # panel (rows, cols): outer tile split into inner tiles
        n = [a // b for a, b in zip(outer, inner)]
        return (1, n[0]) if rank == 1 else tuple(n)

    def origin(i, j, ext):  # first element of panel cell (i, j)
        return (j * ext[0],) if rank == 1 else (i * ext[0], j * ext[1])

    def phys(i, j, C):
        return divmod(j, wrap) if rank == 1 and C > wrap else (i, j)

    panels = []  # (title lines, (rows, cols), cell extent in elements, (i, j) -> (fill, label, small label))
    R, C = counts(layout.shape, Wt)
    if R * C > 1 or h.warp_copies > 1:
        cells = {}
        for i in range(R):
            for j in range(C):
                reg, _, warp = h.locate(origin(i, j, Wt))
                small = ([f"r+{reg}"] if h.outer_regs else []) + ([f"x{h.warp_copies}"] if h.warp_copies > 1 else [])
                cells[i, j] = (_WARP_FILLS[warp % len(_WARP_FILLS)], f"w{warp}", " ".join(small))
        note = "cell: warp, + register offset" if h.outer_regs else "cell: warp"
        panels.append(
            ((f"block {sz(layout.shape)}", tiles(R * C, "warp tile") + f" of {sz(Wt)}", note), (R, C), Wt, cells)
        )
    R, C = counts(Wt, T)
    small = f"x{h.lane_copies}" if h.lane_copies > 1 else ""
    cells = {}
    for i in range(R):
        for j in range(C):
            lane = h.locate(origin(i, j, T))[1]
            cells[i, j] = (lane_color(lane), f"L{lane}", small)
    panels.append(
        ((f"warp tile {sz(Wt)}", tiles(R * C, "thread tile") + f" of {sz(T)}", "cell: lane"), (R, C), T, cells)
    )
    if _prod(T) > 1:
        unit = (1,) * rank
        R, C = counts(T, unit)
        cells = {(i, j): ("#eef2f7", f"r{h.locate(origin(i, j, unit))[0]}", "") for i in range(R) for j in range(C)}
        panels.append(((f"thread tile {sz(T)}", f"{_prod(T)} elements", "cell: register"), (R, C), unit, cells))

    pad, lm, gap, title_h = 14, 30, 70, 58
    geo, x = [], pad + lm
    for titles, (R, C), ext, _ in panels:
        eh, ew = (1, ext[0]) if rank == 1 else ext
        pr, pc = (-(-C // wrap), wrap) if rank == 1 and C > wrap else (R, C)
        k = 240 / max(pr * eh, pc * ew)  # px per element: longer side ~240 px
        k = min(k, (56 if eh == ew == 1 else 160) / max(eh, ew))
        cw, ch = round(max(k * ew, 34), 1), round(max(k * eh, 26), 1)
        geo.append((x, cw, ch, pr, pc))
        x = round(x + max(pc * cw, 7.4 * max(map(len, titles))) + gap + lm, 1)
    width = x - gap - lm
    height = title_h + max(pr * ch for _, _, ch, pr, _ in geo)

    key = ["lens: the framed top-right cell of a panel is magnified in the next panel (same colour)"]
    key.append(f"thread = lane + {W} * warp")
    if h.outer_regs:
        key.append("register = r+offset of the warp tile + r in the thread tile")
    if h.warp_copies > 1 or h.lane_copies > 1:
        key.append("xN = held by N warps / lanes (copies)")

    def render(top):
        out, y0 = [], top + title_h
        for n in range(len(panels) - 1):  # lens lines first, everything else on top
            px, cw, ch, _, pc = geo[n]
            qx, _, qch, qpr, _ = geo[n + 1]
            d = f"M{px + pc * cw} {y0}L{qx} {y0}M{px + pc * cw} {y0 + ch}L{qx} {y0 + qpr * qch}"
            out.append(f'<path d="{d}" stroke="{_LENS[n % 2]}" stroke-width="1.3" stroke-dasharray="5 3" fill="none"/>')
        for n, ((t1, t2, t3), (R, C), ext, cells) in enumerate(panels):
            px, cw, ch, pr, pc = geo[n]
            out.append(f'<text class="pt" x="{px}" y="{top + 12}">{escape(t1)}</text>')
            out.append(f'<text class="k" x="{px}" y="{top + 27}">{escape(t2)}</text>')
            out.append(f'<text class="k" x="{px}" y="{top + 41}">{escape(t3)}</text>')
            for (i, j), (fill, label, small) in cells.items():
                pi, pj = phys(i, j, C)
                x, y = px + pj * cw, y0 + pi * ch
                out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="white"/>')
                if small:
                    out.append(f'<text class="lc" x="{x + cw / 2}" y="{y + ch / 2 - 1}">{label}</text>')
                    out.append(f'<text class="ls" x="{x + cw / 2}" y="{y + ch / 2 + 10}">{escape(small)}</text>')
                else:
                    out.append(f'<text class="lc" x="{x + cw / 2}" y="{y + ch / 2 + 4}">{label}</text>')
            for j in range(pc):  # first element of each column / row
                x = px + j * cw + cw / 2
                out.append(f'<text class="ax" x="{x}" y="{y0 - 4}" text-anchor="middle">{j * ext[-1]}</text>')
            row_step = ext[0] if rank == 2 else wrap * ext[0] * (pr > 1)
            for i in range(pr if row_step else 0):
                y = y0 + i * ch + ch / 2 + 3
                out.append(f'<text class="ax" x="{px - 4}" y="{y}" text-anchor="end">{i * row_step}</text>')
            color, sw = (_LENS[(n - 1) % 2], 2.4) if n else ("#333", 1.2)
            out.append(
                f'<rect x="{px}" y="{y0}" width="{pc * cw}" height="{pr * ch}" fill="none" '
                f'stroke="{color}" stroke-width="{sw}"/>'
            )
            if n + 1 < len(panels):
                out.append(
                    f'<rect x="{px + (pc - 1) * cw}" y="{y0}" width="{cw}" height="{ch}" fill="none" '
                    f'stroke="{_LENS[n % 2]}" stroke-width="3"/>'
                )
        return out

    return width, height, key, render


# ---------------------------------------------------------------- picture
def layout_svg(layout, *, title=None, highlight_thread=0, cell=None, matrix=True, style="auto") -> str:
    """Render a rank-1 or rank-2 LinearLayout as an SVG string (rows = dim 0, cols = dim 1).

    ``style``: "lens" = warp tiles of the block, a lens into one warp tile (lanes)
    and a lens into one thread tile (registers); "grid" = one cell per element
    (``highlight_thread`` and ``cell=(w, h)`` apply here); "auto" (default) =
    lens when ``hierarchy(layout)`` exists, else grid.  ``matrix`` adds the
    paper-style F2 matrix next to the lane colour key.
    """
    if layout.rank not in (1, 2):
        raise NotImplementedError("layout_svg draws rank-1 and rank-2 tiles")
    if style not in ("auto", "grid", "lens"):
        raise ValueError(f"unknown style {style!r}")
    h = None if style == "grid" else hierarchy(layout)
    if style == "lens" and h is None:
        raise ValueError("layout has no reg | lane | warp hierarchy in every dim; use style='grid'")
    body_w, body_h, key, render_body = _lens_body(layout, h) if h else _grid_body(layout, highlight_thread, cell)

    pad, W = 14, layout.warp_size
    mat = _matrix_panel(layout) if matrix else None
    width = int(max(body_w + pad, 760, 2 * pad + (mat[0] if mat else 0)))
    if title is None:
        title = (
            f"LinearLayout {'x'.join(map(str, layout.shape))}: {layout.num_threads} threads "
            f"({layout.num_warps} warps x {W}), {layout.num_regs} regs/thread"
        )
    key_lines = _wrap(key, width - 2 * pad)

    def fmt(name, bases):
        return [f"{name}{i}{tuple(b)}" for i, b in enumerate(bases)]

    legend = []
    for head, name, bases in (("reg ", "r", layout.reg), ("lane", "l", layout.lane), ("warp", "w", layout.warp)):
        legend += _wrap([head] + (fmt(name, bases) or ["-"]), width - 2 * pad)
    rep = [j for j, b in enumerate(layout.thread_bases) if not any(b)]
    if rep:
        legend.append(f"zero thread bases (copies): thread-id bits {rep}")
    legend.append("element coordinate = XOR of the bases of all set reg / lane / warp bits")

    top = 56 + 14 * len(key_lines) + 8
    sw_w, sw_h = 26, 16  # lane key: 16 hues per row, one row per shade
    ly = top + body_h + 26
    key_y = ly + 16 * len(legend) + 14
    key_w, key_h = 16 * sw_w, 12 + sw_h * ((W + 15) // 16)
    foot = key_h
    if mat:  # right of the lane key when it fits, else below it
        if pad + key_w + 56 + mat[0] <= width - pad:
            mx, my = pad + key_w + 56, key_y - 14
            foot = max(key_h, mat[1] - 14)
        else:
            mx, my = pad, key_y + key_h + 16
            foot = key_h + 16 + mat[1]
    height = int(key_y + foot + pad)

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _STYLE,
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{pad}" y="26" style="font-size:16px;font-weight:bold">{escape(title)}</text>',
    ]
    for k, line in enumerate(key_lines):
        s.append(f'<text class="k" x="{pad}" y="{48 + 14 * k}" xml:space="preserve">{escape(line)}</text>')
    s += render_body(top)
    for k, line in enumerate(legend):
        s.append(f'<text x="{pad}" y="{ly + 16 * k}" xml:space="preserve">{escape(line)}</text>')
    s.append(f'<text class="k" x="{pad}" y="{key_y}">lane colour key</text>')
    for lane in range(W):
        x, y = pad + (lane % 16) * sw_w, key_y + 8 + (lane // 16) * sw_h
        s.append(f'<rect x="{x}" y="{y}" width="{sw_w}" height="{sw_h}" fill="{lane_color(lane)}" stroke="white"/>')
        s.append(f'<text class="c" x="{x + sw_w / 2}" y="{y + sw_h - 4}">{lane}</text>')
    if mat:
        s += mat[2](mx, my)
    s.append("</svg>")
    return "\n".join(s)


def _examples():
    from .linear import LinearLayout, blocked

    mfma_c = LinearLayout.from_tv(((16, 4), (4, 1)), ((16, 4), (1, 16)), (16, 16))
    tiled_c = LinearLayout.from_tv(((16, 8, 2), ((4, 1), (1, 1))), ((32, 4, 512), ((1, 0), (0, 0))), (32, 32))
    tiled_a = LinearLayout.from_tv(((16, 4, 2, 2), (4, (1, 1))), ((1, 128, 16, 0), (32, (0, 0))), (32, 16))
    coalesced = LinearLayout(
        tiled_c.shape, tiled_c.lane[:2] + tiled_c.reg[2:], tiled_c.reg[:2] + tiled_c.lane[2:], tiled_c.warp
    )
    # lane bit 0 and register bit 0 interleaved along n: no reg | lane | warp hierarchy -> grid
    interleaved = LinearLayout((16, 16), ((0, 2),), ((0, 1), (0, 4), (0, 8), (1, 0), (2, 0), (4, 0)), ((8, 0),))
    return {
        "blocked_32x64_spt1x8_tpw8x8_w4x1": blocked((32, 64), (1, 8), (8, 8), (4, 1)),
        "mfma16x16x16_c": mfma_c,
        "tiled_mma_c_64x64_2x2warps": tiled_c.with_repeats((64, 64)),
        "tiled_mma_a_32x16_2x2warps": tiled_a,
        "mma_c_coalesced_32x32": coalesced,
        "blocked_1024_spt4_w2": blocked((1024,), (4,), (64,), (2,)),
        "interleaved_grid_fallback": interleaved,
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "layout_svg"
    os.makedirs(out, exist_ok=True)
    for name, layout in _examples().items():
        path = os.path.join(out, name + ".svg")
        layout.to_svg(path, title=name)
        print(path)


if __name__ == "__main__":
    main()
