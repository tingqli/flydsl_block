"""ISA checks: vector widths and which convert path is taken (no LDS for warp-local converts)."""

import collections
import glob
import os
import re
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))


def isa_stats(case, tmp_path):
    env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    subprocess.run([sys.executable, os.path.join(HERE, "isa_kernels.py"), case], env=env, check=True)
    (path,) = glob.glob(str(tmp_path / "*" / "*final_isa.s"))
    text = open(path).read()
    ops = collections.Counter(re.findall(r"^\s+((?:buffer|ds|global|scratch)_\w+|s_barrier)\b", text, re.M))
    lds = int(re.search(r"\.amdhsa_group_segment_fixed_size (\d+)", text).group(1))
    scratch = int(re.search(r"\.amdhsa_private_segment_fixed_size (\d+)", text).group(1))
    return ops, lds, scratch


def test_copy_bf16_is_128bit(tmp_path):
    ops, lds, scratch = isa_stats("copy_bf16", tmp_path)
    assert ops["buffer_load_dwordx4"] == 2 and ops["buffer_store_dwordx4"] == 2
    assert lds == 0 and scratch == 0 and not any(k.startswith("ds_") for k in ops)


def test_convert_warp_has_no_lds(tmp_path):
    ops, lds, scratch = isa_stats("convert_warp", tmp_path)
    assert ops["buffer_store_dwordx4"] == 4  # coalesced after two reg<->lane swaps
    assert lds == 0 and scratch == 0 and ops["s_barrier"] == 0
    assert not any(k.startswith(("ds_read", "ds_write")) for k in ops)


def test_convert_lds_path(tmp_path):
    ops, lds, scratch = isa_stats("convert_lds", tmp_path)
    assert lds == 64 * 64 * 4 and ops["s_barrier"] == 2 and scratch == 0
    assert ops["ds_read_b128"] > 0
