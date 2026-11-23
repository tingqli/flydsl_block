"""Kernels whose ISA is checked by tests/test_isa.py (run in a subprocess with IR dumps on).

Usage: python isa_kernels.py <case>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402

import flydsl_block as blk  # noqa: E402
from flydsl_block import LinearLayout  # noqa: E402


def tile(t, i=0):
    return fx.flat_divide(fx.rocdl.make_buffer_tensor(t, max_size=False), (64, 64))[None, None, i, 0]


def c64():
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tm = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    return blk.Mma(tm, "C", (64, 64)).linear


def coalesced(c):
    return LinearLayout(c.shape, c.lane[:2] + c.reg[2:], c.reg[:2] + c.lane[2:], c.warp)


CASES = {
    # bf16, 8 contiguous per thread -> 128-bit buffer loads/stores
    "copy_bf16": (torch.bfloat16, lambda: blk.Blocked((64, 64), (1, 8), (8, 8), (4, 1)), None),
    # MMA C -> 4 contiguous n per lane via two reg<->lane swaps: no LDS, 128-bit f32 stores
    "convert_warp": (torch.float32, c64, lambda: coalesced(c64())),
    # different warp bases -> LDS round trip
    "convert_lds": (torch.float32, lambda: blk.Blocked((64, 64), (1, 4), (8, 8), (4, 1)), c64),
}


def build(case):
    dtype, src, dst = CASES[case]

    @flyc.kernel
    def kern(X: fx.Tensor, Y: fx.Tensor):
        x = blk.load(tile(X), src())
        x = x.convert(dst()) if dst is not None else x
        blk.store(tile(Y), x)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kern(X, Y).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    x = torch.randn(64, 64, device="cuda").to(dtype)
    y = torch.zeros_like(x)
    launch(x, y, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(x, y), case


if __name__ == "__main__":
    build(sys.argv[1])
