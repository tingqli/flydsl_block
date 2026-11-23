"""Compare the flydsl_block rewrites with the original tutorial kernels.

    python examples/compare.py                 # all cases
    python examples/compare.py v0_128x128      # selected cases

For each case: the output must be bit-identical to the original kernel's
(same MFMA accumulation order), then both are timed in alternating rounds
(40 warm-up launches, median of 30; inputs rotate over 4 copies).
TUTORIAL_GEMM points at gfx942-flydsl-tutorials/kernels/gemm.
"""

import importlib.util
import math
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TUTORIAL = os.environ.get("TUTORIAL_GEMM", "/data2/tingqli/gfx942-flydsl-tutorials/kernels/gemm")
HINTS = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}  # same as the tutorial bench.py

sm_count = torch.cuda.get_device_properties(0).multi_processor_count
M, N, K = {304: (8192, 9728, 8192), 256: (8192, 8192, 8192), 80: (4096, 5120, 8192)}[sm_count]

CASES = {  # name -> (module, build(module), inputs preshuffled on the host)
    "v0_64x64": ("v0", lambda m: m.build_v0(64, 64, 1, M, N, K), False),
    "v0_128x128": ("v0", lambda m: m.build_v0(128, 128, 1, M, N, K), False),
    "v0_ilp": ("v0_preshuffle", lambda m: m.build_v0_preshuffle(128, 128, 1, M, N, K, PRESHUFFLE=False), False),
    "v0_preshuffle_noilp": ("v0_preshuffle", lambda m: m.build_v0_preshuffle(128, 128, 1, M, N, K, ILP=False), True),
    "v0_preshuffle": ("v0_preshuffle", lambda m: m.build_v0_preshuffle(128, 128, 1, M, N, K), True),
    "v0_preshuffle_k32": ("v0_preshuffle", lambda m: m.build_v0_preshuffle(128, 128, 1, M, N, K, K32=True), True),
    "v1": ("v1", lambda m: m.build_v1(256, 256, 1, M, N, K), False),
    "v2_nosched": ("v2", lambda m: m.build_v2(256, 256, 1, M, N, K, SCHED=False), False),
    "v2": ("v2", lambda m: m.build_v2(256, 256, 1, M, N, K), False),
    "v2_k32": ("v2", lambda m: m.build_v2(256, 256, 1, M, N, K, K32=True), False),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timeit(fn, inputs, warmup=40, iters=30):
    for i in range(warmup):
        fn(*inputs[i % len(inputs)])
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*inputs[i % len(inputs)])
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main(names):
    torch.manual_seed(0)
    inputs = [
        (
            torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K),
            torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K),
        )
        for _ in range(4)
    ]
    A, B = inputs[0]
    ref = A.float() @ B.float().T
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    flop = 2 * M * N * K
    print(f"gemm {M}x{N}x{K} bf16 -> fp32, {torch.cuda.get_device_properties(0).gcnArchName}")
    for name in names:
        mod, build, shuffled = CASES[name]
        variants = {
            "orig": load_module(os.path.join(TUTORIAL, mod + ".py"), "tutorial_" + mod),
            "block": load_module(os.path.join(HERE, mod + ".py"), "block_" + mod),
        }
        prep = variants["block"].preshuffle if shuffled else (lambda x: x)
        ins = [(prep(a), prep(b)) for a, b in inputs]
        run, out = {}, {}
        for tag, m in variants.items():
            compiled = flyc.compile[HINTS](build(m), *ins[0], C, stream)
            C.zero_()
            compiled(*ins[0], C, stream)
            out[tag] = C.clone()
            run[tag] = lambda a, b, compiled=compiled: compiled(a, b, C, stream)
        same = torch.equal(out["orig"], out["block"])
        ok = torch.allclose(out["block"], ref, atol=1e-5, rtol=1e-5)
        us = {tag: [] for tag in run}
        for _ in range(3):
            for tag, fn in run.items():
                us[tag].append(timeit(fn, ins))
        t = {tag: sorted(v)[1] for tag, v in us.items()}
        print(
            f"{name:<16} orig {t['orig']:8.1f} us {flop / t['orig'] / 1e6:6.1f} TFLOPS | "
            f"block {t['block']:8.1f} us {flop / t['block'] / 1e6:6.1f} TFLOPS | "
            f"block/orig {t['block'] / t['orig']:.3f}  same_bits={same} correct={ok}"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
