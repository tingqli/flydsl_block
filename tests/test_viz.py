"""CPU tests for the SVG layout pictures."""

import xml.etree.ElementTree as ET

import pytest

from flydsl_block.linear import LinearLayout, all_coords, blocked
from flydsl_block.viz import hierarchy, owners

NS = "{http://www.w3.org/2000/svg}"


def _parse(svg):
    root = ET.fromstring(svg)
    texts = [t.text for t in root.iter(NS + "text")]
    return root, texts


def test_svg_blocked(tmp_path):
    L = blocked((32, 64), (1, 8), (8, 8), (4, 1))
    path = tmp_path / "b.svg"
    svg = L.to_svg(str(path), style="grid")
    assert path.read_text() == svg
    root, texts = _parse(svg)
    cells = [r for r in root.iter(NS + "rect") if r.get("width") == "34"]
    assert len(cells) == 32 * 64
    assert texts.count("T0") == 1 and "T255" in texts  # one thread label per block
    assert sum(t == "r7" for t in texts) == 256  # every register of every thread


def test_svg_replication_and_rank1():
    a = LinearLayout.from_tv(((16, 4, 2, 2), (4, (1, 1))), ((1, 128, 16, 0), (32, (0, 0))), (32, 16))
    _, texts = _parse(a.to_svg(style="grid"))
    assert "r0x2" in texts and any("copies" in (t or "") for t in texts)
    assert owners(a)[(0, 0)] == [0, 0, 2]
    v = LinearLayout((256,), ((1,), (2,)), ((4,), (8,), (16,), (32,), (64,), (128,)), ())
    _, texts = _parse(v.to_svg(highlight_thread=None, style="grid"))
    assert "T63" in texts


def test_svg_large_tile_drops_text():
    _, texts = _parse(blocked((128, 128), (4, 4), (8, 8), (2, 2)).to_svg(style="grid"))
    assert "T0" not in texts


def test_svg_rank3_rejected():
    with pytest.raises(NotImplementedError):
        LinearLayout((2, 2, 64), (), ((0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 8), (0, 0, 16), (0, 0, 32))).to_svg()


def _matrix_entries(svg):
    return [t for t in ET.fromstring(svg).iter(NS + "text") if t.get("class") in ("m0", "m1")]


def test_matrix_panel():
    L = blocked((32, 64), (1, 8), (8, 8), (4, 1))
    svg = L.to_svg()
    ents = _matrix_entries(svg)
    assert len(ents) == 11 * 11 and sum(e.text == "1" for e in ents) == 11  # a bit permutation
    _, texts = _parse(svg)
    assert {"Reg", "Lane", "Warp", "A ="} <= set(texts)
    assert not _matrix_entries(L.to_svg(matrix=False))


def test_matrix_values_match_bases():
    a = LinearLayout.from_tv(((16, 4, 2, 2), (4, (1, 1))), ((1, 128, 16, 0), (32, (0, 0))), (32, 16))
    cols = a.reg + a.lane + a.warp
    rows = [(1, b) for b in range(4)] + [(0, b) for b in range(5)]  # last dim first, low bit first
    expect = [str((basis[d] >> b) & 1) for d, b in rows for basis in cols]
    assert [e.text for e in _matrix_entries(a.to_svg())] == expect


def test_matrix_moves_below_key_when_narrow():
    v = LinearLayout((16,), ((1,), (2,)), ((4,), (8,)) + ((0,),) * 4, ((0,),) * 6)
    root = ET.fromstring(v.to_svg(style="grid"))
    y = {t.text: float(t.get("y")) for t in root.iter(NS + "text") if t.text in ("lane colour key", "A =")}
    assert y["A ="] > y["lane colour key"] + 60


MFMA_C = LinearLayout.from_tv(((16, 4), (4, 1)), ((16, 4), (1, 16)), (16, 16))
TILED_C = LinearLayout.from_tv(((16, 8, 2), ((4, 1), (1, 1))), ((32, 4, 512), ((1, 0), (0, 0))), (32, 32))
TILED_A = LinearLayout.from_tv(((16, 4, 2, 2), (4, (1, 1))), ((1, 128, 16, 0), (32, (0, 0))), (32, 16))
# lane bit 0 and register bit 0 interleaved along n
INTERLEAVED = LinearLayout((16, 16), ((0, 2),), ((0, 1), (0, 4), (0, 8), (1, 0), (2, 0), (4, 0)), ((8, 0),))


@pytest.mark.parametrize(
    "layout, thread_tile, warp_tile, outer_regs, copies",
    [
        (blocked((32, 64), (1, 8), (8, 8), (4, 1)), (1, 8), (8, 64), False, (1, 1)),
        (blocked((128, 128), (4, 4), (8, 8), (2, 2)), (4, 4), (32, 32), True, (1, 1)),
        (blocked((1024,), (4,), (64,), (2,)), (4,), (256,), True, (1, 1)),
        (MFMA_C, (4, 1), (16, 16), False, (1, 1)),
        (TILED_C.with_repeats((64, 64)), (4, 1), (16, 16), True, (1, 1)),
        (TILED_A, (1, 4), (16, 16), False, (1, 2)),
    ],
)
def test_hierarchy(layout, thread_tile, warp_tile, outer_regs, copies):
    h = hierarchy(layout)
    assert (h.thread_tile, h.warp_tile, h.outer_regs) == (thread_tile, warp_tile, outer_regs)
    assert (h.lane_copies, h.warp_copies) == copies
    for _, _, c in all_coords(layout):  # locate is the inverse of the layout
        reg, lane, warp = h.locate(c)
        assert layout(reg, lane + layout.warp_size * warp) == c


def test_no_hierarchy():
    assert hierarchy(INTERLEAVED) is None
    xor = LinearLayout((16, 16), ((0, 1), (0, 2)), ((0, 4), (0, 8), (1, 0), (2, 0), (4, 1), (8, 0)))
    assert hierarchy(xor) is None  # a basis touching two coordinate bits
    dup = LinearLayout((16, 16), ((0, 0), (0, 1)), ((0, 2), (0, 4), (0, 8), (1, 0), (2, 0), (4, 0)), ((8, 0),))
    assert hierarchy(dup) is None  # duplicated registers


def test_svg_lens():
    _, texts = _parse(TILED_C.with_repeats((64, 64)).to_svg())
    assert {"w0", "w3", "r+12", "L0", "L63", "r3"} <= set(texts)
    assert "T0" not in texts  # no per-element grid
    _, texts = _parse(MFMA_C.to_svg())  # one warp: no block panel
    assert "w0" not in texts and {"L63", "r3"} <= set(texts)
    _, texts = _parse(TILED_A.to_svg())
    assert {"w0", "w1"} <= set(texts) and texts.count("x2") == 2


def test_svg_style_fallback():
    _, texts = _parse(INTERLEAVED.to_svg())
    assert "T0" in texts and "L0" not in texts  # auto -> grid
    with pytest.raises(ValueError):
        INTERLEAVED.to_svg(style="lens")
    with pytest.raises(ValueError):
        MFMA_C.to_svg(style="zoom")
