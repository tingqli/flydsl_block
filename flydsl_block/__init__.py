"""Block-level fragments for FlyDSL: scheme A (Blocked / Mma / TV layouts, block API)
built on scheme B (F2 LinearLayout planning).

    import flydsl_block as blk
    L = blk.Blocked((64, 64), size_per_thread=(1, 8), threads_per_warp=(8, 8), warps=(4, 1))
    x = blk.load(g_tile, L, bounds=(rows, 64))
    y = (x - x.max(axis=1)[:, None]).exp()
    blk.store(out_tile, y / y.sum(axis=1)[:, None])
"""

from .linear import ConvertPlan, LinearLayout, ReducePlan, blocked, plan_convert, plan_reduce

__all__ = [
    "LinearLayout",
    "ConvertPlan",
    "ReducePlan",
    "blocked",
    "plan_convert",
    "plan_reduce",
]

try:  # the FlyDSL layer is optional so the planning core stays importable on CPU-only hosts
    from .fragment import (  # noqa: F401
        TV,
        Blocked,
        BlockFragment,
        Mma,
        MmaSpec,
        Shared,
        dot,
        full,
        load,
        store,
        wrap,
        zeros,
    )

    __all__ += [
        "BlockFragment",
        "Blocked",
        "TV",
        "Mma",
        "MmaSpec",
        "Shared",
        "load",
        "store",
        "dot",
        "wrap",
        "zeros",
        "full",
    ]
except ImportError:  # pragma: no cover
    pass
