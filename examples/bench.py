"""Time flydsl_block GEMMs against each other (kernels without a tutorial original, e.g. v3).

    python examples/bench.py                      # v2_k32 v3 at 4096x4096x8192
    MN=8192 K=8192 python examples/bench.py v2_k32 v3

Each output is checked against torch, then the kernels are timed in alternating rounds,
each as a CUDA graph of 20 back-to-back launches (no CPU launch gaps; inputs rotate
over 4 copies).
"""

import math
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import v2
import v3

HINTS = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}  # as compare.py
M = N = int(os.environ.get("MN", 4096))
K = int(os.environ.get("K", 8192))
CASES = {
    "v2_k32": lambda: v2.build_v2(256, 256, 1, M, N, K, K32=True),
    "v3": lambda: v3.build_v3(256, 256, 1, M, N, K),
    "v3_grid2d": lambda: v3.build_v3(256, 256, 1, M, N, K, GROUP_M=0),
}


def main(names):
    torch.manual_seed(0)
    inputs = [
        (
            torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K),
            torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K),
        )
        for _ in range(4)
    ]
    ref = inputs[0][0].float() @ inputs[0][1].float().T
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    print(f"gemm {M}x{N}x{K} bf16 -> fp32, {torch.cuda.get_device_properties(0).gcnArchName}")
    graphs, compiled = {}, {}  # a graph replays the compiled module's code: keep it alive
    for name in names:
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        fn = compiled[name] = flyc.compile[HINTS](CASES[name](), *inputs[0], C, stream)
        C.zero_()
        fn(*inputs[0], C, stream)
        torch.cuda.synchronize()
        print(f"{name:<12} correct={torch.allclose(C, ref, atol=1e-5, rtol=1e-5)}")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(20):
                fn(*inputs[i % 4], C, fx.Stream(torch.cuda.current_stream().cuda_stream))
        graphs[name] = g
    us = {name: [] for name in graphs}
    for _ in range(3):
        for name, g in graphs.items():
            for _ in range(3):  # warm-up: 60 launches
                g.replay()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            us[name].append(s.elapsed_time(e) * 1e3 / 200)
    for name, t in us.items():
        best = min(t)
        print(f"{name:<12} {best:8.1f} us {2 * M * N * K / best / 1e6:7.1f} TFLOPS")


if __name__ == "__main__":
    main(sys.argv[1:] or ["v2_k32", "v3"])
