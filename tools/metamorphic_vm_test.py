#!/usr/bin/env python3
"""
metamorphic_vm_test.py — Property-Based & Metamorphic 4-Way VM Verifier

Tests algebraic, set-theoretic, graph-theoretic, and aliasing invariants by
generating randomized pairs of semantically equivalent .impas programs (P1, P2)
over randomized small & multi-word (>64-node) graphs and numeric vectors.

For each pair, verifies BOTH:
  1. 4-way polyglot consensus on P1 and P2 independently (C++ ON == OFF == Java == OCaml)
  2. Metamorphic equivalence: observable(P1) == observable(P2) (catches shared bugs!)
"""

import os
import sys
import time
import random
import struct
import tempfile
import argparse
import multiprocessing as mp
from collections import Counter
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
SPEC_DIR = REPO_DIR / "spec"
ROOT_DIR = REPO_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SPEC_DIR / "tools"))

from run_vm_asm_suite import Instruction, parse_impas_file
from four_way_validation import (
    JavaBatchClient, OcamlBatchClient, _setup_cpp_lib, run_cpp_parsed, compare_state,
)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _make_graph(rng: random.Random) -> tuple[int, list[str]]:
    """Generate small (4..12) or multi-word (68..132) directed graphs."""
    if rng.random() < 0.35:
        n = rng.choice([68, 72, 130])
    else:
        n = rng.randint(4, 10)
    lines = []
    for u in range(n):
        if n > 64 and u in (0, 1, 63, 64, 65, n - 2):
            nbrs = sorted({(u + 1) % n, (u + 63) % n, (u + 64) % n})
        elif rng.random() < 0.65:
            k = rng.randint(1, min(3, n))
            nbrs = sorted({rng.randint(0, n - 1) for _ in range(k)})
        else:
            nbrs = []
        comma = "," if u < n - 1 else ""
        lines.append(f"    {u}: [{', '.join(map(str, nbrs))}]{comma}")
    return n, lines


def _make_floats(rng: random.Random, n: int) -> list[str]:
    pool = ["0.0", "-0.0", "1.0", "-1.0", "2.5", "-3.5", "4.0", "0.5", "10.0", "-0.25"]
    return [rng.choice(pool) for _ in range(n)]


def _wrap_prog(title: str, graph_lines: list[str], extra_data: list[str], instrs: list[str]) -> str:
    lines = [
        f"; METAMORPHIC TEST: {title}",
        ".fuel 5000",
        "",
        ".data",
        ".csr_inline G = {",
        *graph_lines,
        "}",
        *extra_data,
        "",
        ".text",
    ]
    for pc, ins in enumerate(instrs):
        lines.append(f"0x{pc:02X}: {ins}")
    return "\n".join(lines) + "\n"


def _rand_set(rng: random.Random, n: int) -> list[int]:
    k = rng.randint(0, min(6, n))
    vals = {rng.randint(0, n - 1) for _ in range(k)}
    if n > 64 and rng.random() < 0.5:
        vals.add(rng.choice([63, 64, 65, n - 1]))
    return sorted(vals)


LAWS = (
    "set_union_comm",
    "set_intersect_comm",
    "set_union_assoc",
    "set_de_morgan",
    "mask_double_not",
    "set_diff_self",
    "roaring_vs_set",
    "walk_2hop_equiv",
    "lane_vs_single_walk",
    "ewise_add_comm",
    "ewise_mult_comm",
    "scale_identity_and_l1",
    "blend_self_identity",
    "aliasing_union_equiv",
    "stream_csr_vs_walk",
    "stream_csc_vs_walk",
    "stream_filter_false_empty",
    "kleene_idempotent",
    "coalesce_self_idempotent",
    "mxv_zero_vec_zero",
    "set_distributive",
    "set_absorption",
    "degree_matches_walk_card",
    "l1_norm_symmetry",
    "vxm_zero_vec_zero",
)


def generate_metamorphic_pair(seed: int, law: str | None = None) -> tuple[str, str, str, str]:
    """Returns (law_name, prog1_text, prog2_text, check_reg)."""
    rng = random.Random(seed)
    if law is None:
        law = rng.choice(LAWS)
    n, g_lines = _make_graph(rng)
    sA = _rand_set(rng, n)
    sB = _rand_set(rng, n)
    sC = _rand_set(rng, n)
    v_len = min(n, rng.choice([1, 4, 8, n]))
    fA = _make_floats(rng, v_len)
    fB = _make_floats(rng, v_len)
    data = [
        f".set SA = [{', '.join(map(str, sA))}]",
        f".set SB = [{', '.join(map(str, sB))}]",
        f".set SC = [{', '.join(map(str, sC))}]",
        f".array_float FA = [{', '.join(fA)}]",
        f".array_float FB = [{', '.join(fB)}]",
    ]

    if law == "set_union_comm":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_SET_UNION R3, R1, R2",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_SET_UNION R3, R2, R1",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "set_intersect_comm":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_SET_INTERSECT R3, R1, R2",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_SET_INTERSECT R3, R2, R1",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "set_union_assoc":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_LOAD_INLINE_SET R3, SC",
            "OP_SET_UNION R4, R1, R2",
            "OP_SET_UNION R5, R4, R3",
            "OP_SET_CARDINALITY R10, R5",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_LOAD_INLINE_SET R3, SC",
            "OP_SET_UNION R4, R2, R3",
            "OP_SET_UNION R5, R1, R4",
            "OP_SET_CARDINALITY R10, R5",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "set_de_morgan":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_MASK_OR R3, R1, R2",
            "OP_MASK_NOT R4, R3",
            "OP_SET_CARDINALITY R10, R4",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_MASK_NOT R3, R1",
            "OP_MASK_NOT R4, R2",
            "OP_MASK_AND R5, R3, R4",
            "OP_SET_CARDINALITY R10, R5",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "mask_double_not":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_MASK_NOT R2, R1",
            "OP_MASK_NOT R3, R2",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_SET_CARDINALITY R10, R1",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "set_diff_self":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_SET_DIFFERENCE R2, R1, R1",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_FRONTIER_DIFF R2, R1, R1",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "roaring_vs_set":
        op_pair = rng.choice([
            ("OP_ROARING_BITMAP_AND", "OP_SET_INTERSECT"),
            ("OP_ROARING_BITMAP_OR", "OP_SET_UNION"),
            ("OP_ROARING_BITMAP_AND_NOT", "OP_SET_DIFFERENCE"),
        ])
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            f"{op_pair[0]} R3, R1, R2",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            f"{op_pair[1]} R3, R1, R2",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "walk_2hop_equiv":
        seed_node = rng.randint(0, n - 1)
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R0, {seed_node}",
            "OP_CSR_WALK R1, R0, 0",
            "OP_CSR_WALK R2, R1, 0",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R0, {seed_node}",
            "OP_CSR_WALK_2HOP R2, 0, 0",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "lane_vs_single_walk":
        seed_node = rng.randint(0, n - 1)
        data_lane = data + [f".array_int SINGLE_SEED = [{seed_node}]"]
        p1 = _wrap_prog(f"{law} P1", g_lines, data_lane, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_INT_ARRAY R1, SINGLE_SEED",
            "OP_LANE_INIT_SEEDS R2, R1, 1",
            "OP_LANE_CSR_WALK R3, R2, 0",
            "OP_LANE_FILTER_QUORUM R4, R3, 1",
            "OP_SET_CARDINALITY R10, R4",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data_lane, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {seed_node}",
            "OP_CSR_WALK R4, R1, 0",
            "OP_SET_CARDINALITY R10, R4",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "ewise_add_comm":
        promote = rng.random() < 0.5
        if promote:
            prep = [
                "OP_INIT_MOCK_GRAPH R0, G",
                "OP_LOAD_INLINE_ARRAY R1, FA",
                "OP_LOAD_INLINE_ARRAY R2, FB",
                "OP_VEC_MATH_UNARY R3, R1, 1, 1",
                "OP_VEC_MATH_UNARY R4, R2, 1, 1",
            ]
            a, b = "R3", "R4"
        else:
            prep = [
                "OP_INIT_MOCK_GRAPH R0, G",
                "OP_LOAD_INLINE_ARRAY R1, FA",
                "OP_LOAD_INLINE_ARRAY R2, FB",
            ]
            a, b = "R1", "R2"
        p1 = _wrap_prog(f"{law} P1", g_lines, data, prep + [
            f"OP_EWISE_ADD R5, {a}, {b}",
            "OP_VECTOR_REDUCE_SUM R10, R5",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, prep + [
            f"OP_EWISE_ADD R5, {b}, {a}",
            "OP_VECTOR_REDUCE_SUM R10, R5",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "ewise_mult_comm":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_LOAD_INLINE_ARRAY R2, FB",
            "OP_EWISE_MULT R5, R1, R2",
            "OP_VECTOR_REDUCE_SUM R10, R5",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_LOAD_INLINE_ARRAY R2, FB",
            "OP_EWISE_MULT R5, R2, R1",
            "OP_VECTOR_REDUCE_SUM R10, R5",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "scale_identity_and_l1":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_LOAD_INLINE_ARRAY R3, FA",
            "OP_FLOAT_VECTOR_SCALE R3, 1.0",
            "OP_L1_NORM_DIFF R10, R1, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_L1_NORM_DIFF R10, R1, R1",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "blend_self_identity":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_ARRAY R2, FA",
            "OP_VEC_BLEND R3, R1, R2, R2",
            "OP_VECTOR_REDUCE_SUM R10, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R2, FA",
            "OP_VECTOR_REDUCE_SUM R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "stream_csr_vs_walk":
        seed_node = rng.randint(0, n - 1)
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {seed_node}",
            "OP_CSR_WALK_STREAM R2, R1, 0, 5",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
            "OP_STREAM_FUNC_BEGIN",
            "OP_STREAM_FUNC_END",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {seed_node}",
            "OP_CSR_WALK R2, R1, 0",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "stream_csc_vs_walk":
        # CSC stream gathers incoming edges into seed: summing 1.0 per incoming edge == in-degree == |OP_CSC_WALK(seed)|
        seed_node = rng.randint(0, n - 1)
        ones_data = data + [
            f".array_float ONES = [{', '.join(['1.0'] * n)}]",
            f".array_float ZEROS = [{', '.join(['0.0'] * n)}]",
        ]
        p1 = _wrap_prog(f"{law} P1", g_lines, ones_data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R5, ONES",
            "OP_INIT_MOCK_NODE_ATTR R5, R5, 0",
            "OP_LOAD_INLINE_ARRAY R4, ZEROS",
            f"OP_INIT_INPUT_NODE R1, {seed_node}",
            "OP_CSC_WALK_STREAM R2, R1, 0, 8",
            "OP_VECTOR_REDUCE_SUM R10, R4",
            "OP_HALT",
            "OP_STREAM_FUNC_BEGIN",
            "OP_STREAM_LOAD_SRC 0, 5",
            "OP_STREAM_REDUCE 0, 4, 0",
            "OP_STREAM_FUNC_END",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, ones_data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {seed_node}",
            "OP_CSC_WALK R2, R1, 0",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "stream_filter_false_empty":
        seed_node = rng.randint(0, n - 1)
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {seed_node}",
            "OP_CSR_WALK_STREAM R2, R1, 0, 5",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
            "OP_STREAM_FUNC_BEGIN",
            "OP_STREAM_LOAD_CONST 0, 0.0",
            "OP_STREAM_FILTER 0",
            "OP_STREAM_FUNC_END",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_INIT_INPUT_SET R2",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "kleene_idempotent":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_FIXPOINT_KLEENE_STAR R2, R1, 0",
            "OP_FIXPOINT_KLEENE_STAR R3, R2, 0",
            "OP_SET_CARDINALITY R10, R3",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_FIXPOINT_KLEENE_STAR R2, R1, 0",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "coalesce_self_idempotent":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_COALESCE R2, R1, R1",
            "OP_VECTOR_REDUCE_SUM R10, R2",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_VECTOR_REDUCE_SUM R10, R1",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "mxv_zero_vec_zero":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_FLOAT_VECTOR_SCALE R1, 0.0",
            "OP_MXV R2, R1, 0",
            "OP_VECTOR_REDUCE_SUM R10, R2",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_CONST_FLOAT R10, 0.0",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "vxm_zero_vec_zero":
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_FLOAT_VECTOR_SCALE R1, 0.0",
            "OP_VXM R2, R1, 0",
            "OP_VECTOR_REDUCE_SUM R10, R2",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_CONST_FLOAT R10, 0.0",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "set_distributive":
        # A ∩ (B ∪ C) == (A ∩ B) ∪ (A ∩ C)
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_LOAD_INLINE_SET R3, SC",
            "OP_SET_UNION R4, R2, R3",
            "OP_SET_INTERSECT R5, R1, R4",
            "OP_SET_CARDINALITY R10, R5",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_LOAD_INLINE_SET R3, SC",
            "OP_SET_INTERSECT R4, R1, R2",
            "OP_SET_INTERSECT R5, R1, R3",
            "OP_SET_UNION R6, R4, R5",
            "OP_SET_CARDINALITY R10, R6",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "set_absorption":
        # A ∪ (A ∩ B) == A
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_LOAD_INLINE_SET R2, SB",
            "OP_SET_INTERSECT R3, R1, R2",
            "OP_SET_UNION R4, R1, R3",
            "OP_SET_CARDINALITY R10, R4",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_SET R1, SA",
            "OP_SET_CARDINALITY R10, R1",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "degree_matches_walk_card":
        # Simple graph (sorted unique neighbors): OP_CSR_DEGREE(u) == |OP_CSR_WALK(u)|
        u = rng.randint(0, n - 1)
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {u}",
            "OP_CSR_DEGREE R10, R1, 0",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            f"OP_INIT_INPUT_NODE R1, {u}",
            "OP_CSR_WALK R2, R1, 0",
            "OP_SET_CARDINALITY R10, R2",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    if law == "l1_norm_symmetry":
        # L1(FA, FB) == L1(FB, FA)
        p1 = _wrap_prog(f"{law} P1", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_LOAD_INLINE_ARRAY R2, FB",
            "OP_L1_NORM_DIFF R10, R1, R2",
            "OP_HALT",
        ])
        p2 = _wrap_prog(f"{law} P2", g_lines, data, [
            "OP_INIT_MOCK_GRAPH R0, G",
            "OP_LOAD_INLINE_ARRAY R1, FA",
            "OP_LOAD_INLINE_ARRAY R2, FB",
            "OP_L1_NORM_DIFF R10, R2, R1",
            "OP_HALT",
        ])
        return law, p1, p2, "R10"

    # aliasing_union_equiv
    p1 = _wrap_prog(f"{law} P1", g_lines, data, [
        "OP_INIT_MOCK_GRAPH R0, G",
        "OP_LOAD_INLINE_SET R1, SA",
        "OP_LOAD_INLINE_SET R2, SB",
        "OP_SET_UNION R3, R1, R2",
        "OP_SET_CARDINALITY R10, R3",
        "OP_HALT",
    ])
    p2 = _wrap_prog(f"{law} P2", g_lines, data, [
        "OP_INIT_MOCK_GRAPH R0, G",
        "OP_LOAD_INLINE_SET R1, SA",
        "OP_LOAD_INLINE_SET R2, SB",
        "OP_MOV R3, R1",
        "OP_SET_UNION R3, R3, R2",
        "OP_SET_CARDINALITY R10, R3",
        "OP_HALT",
    ])
    return law, p1, p2, "R10"


_W = {}


def _init_worker(lib_on_path, lib_off_path, cp):
    _W["lib_on"] = _setup_cpp_lib(lib_on_path)
    _W["lib_off"] = _setup_cpp_lib(lib_off_path)
    _W["java"] = JavaBatchClient(cp)
    _W["ocaml"] = OcamlBatchClient()
    _W["tmp"] = tempfile.TemporaryDirectory()


def _eval_4way(prog_text: str):
    tp = Path(_W["tmp"].name)
    impas = tp / "p.impas"
    impb = tp / "p.impb"
    impd = tp / "p.impb_data"
    impas.write_text(prog_text)
    instrs, data_bytes, ex, _ = parse_impas_file(impas)
    c_instrs = (Instruction * len(instrs))(*instrs)
    impb.write_bytes(bytes(c_instrs))
    impd.write_bytes(data_bytes)
    c_on = run_cpp_parsed(_W["lib_on"], instrs, data_bytes, ex)
    c_off = run_cpp_parsed(_W["lib_off"], instrs, data_bytes, ex)
    jv = _W["java"].run(impas, impb, impd)
    oc = _W["ocaml"].run(impb, impd)
    diffs = compare_state(c_on, c_off, jv, oc)
    return diffs, c_on


def _run_chunk(task):
    start_i, count, base_seed, law_filter = task
    passed = 0
    failures = []
    law_counts = Counter()
    for off in range(count):
        seed = base_seed + start_i + off
        law, p1, p2, reg = generate_metamorphic_pair(seed, law_filter)
        law_counts[law] += 1
        try:
            d1, s1 = _eval_4way(p1)
            d2, s2 = _eval_4way(p2)
            if d1:
                failures.append((seed, law, "4WAY_P1", d1[:3], p1, p2))
                continue
            if d2:
                failures.append((seed, law, "4WAY_P2", d2[:3], p1, p2))
                continue
            if s1["status"] != "IMPULSE_VM_OK" or s2["status"] != "IMPULSE_VM_OK":
                failures.append((seed, law, "STATUS", [f"P1={s1['status']}, P2={s2['status']}"], p1, p2))
                continue
            v1 = s1["registers"][reg]
            v2 = s2["registers"][reg]
            t1 = s1["register_types"][reg]
            t2 = s2["register_types"][reg]
            # Normalize -0.0 and +0.0 for float/double scalar sums (e.g. -0.0 + 0.0)
            # Also allow float-vs-int exact integer equality when law compares float sum vs int cardinality
            eq = (v1 == v2 and t1 == t2)
            if not eq and t1 == t2 == 8:
                f1 = struct.unpack("<f", struct.pack("<I", v1 & 0xFFFFFFFF))[0]
                f2 = struct.unpack("<f", struct.pack("<I", v2 & 0xFFFFFFFF))[0]
                eq = (f1 == f2)
            elif not eq and t1 == t2 == 9:
                f1 = struct.unpack("<d", struct.pack("<Q", v1 & 0xFFFFFFFFFFFFFFFF))[0]
                f2 = struct.unpack("<d", struct.pack("<Q", v2 & 0xFFFFFFFFFFFFFFFF))[0]
                eq = (f1 == f2)
            elif not eq and law == "stream_csc_vs_walk" and t1 == 8 and t2 == 1:
                f1 = struct.unpack("<f", struct.pack("<I", v1 & 0xFFFFFFFF))[0]
                eq = (f1 == float(v2))
            if not eq:
                failures.append((seed, law, "METAMORPHIC", [f"{reg} mismatch: P1=({v1}, type={t1}) vs P2=({v2}, type={t2})"], p1, p2))
            else:
                passed += 1
        except Exception as e:
            failures.append((seed, law, "ERROR", [str(e)], p1, p2))
    return passed, failures, dict(law_counts)


def main():
    parser = argparse.ArgumentParser(description="ImpulseVM Metamorphic 4-Way Property Verifier")
    parser.add_argument("--iterations", type=int, default=500, help="Number of metamorphic pairs (default: 500)")
    parser.add_argument("--seed", type=int, default=42000, help="Starting seed")
    parser.add_argument("--law", type=str, default=None, choices=LAWS, help="Test a single metamorphic law")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--dump-dir", type=str, default=str(TOOLS_DIR / "fuzz_findings"))
    args = parser.parse_args()

    core_cpp = ROOT_DIR / "impulse-graph-core" / "impulse-cpp"
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib_on = os.environ.get("IMPULSE_LIB_PATH_ON", str(core_cpp / "build_on" / f"libimpulse_graph.{ext}"))
    lib_off = os.environ.get("IMPULSE_LIB_PATH_OFF", str(core_cpp / "build_off" / f"libimpulse_graph.{ext}"))
    java_dir = ROOT_DIR / "impulse-graph-java"
    maven_cp = (java_dir / "impulse-vm" / "cp.txt").read_text().strip()
    cp = f"impulse-vm/target/classes:impulse-vm/target/test-classes:impulse-api/target/classes:impulse-storage/target/classes:impulse-spec/target/classes:{maven_cp}"

    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    print(f"{BOLD}{BLUE} ImpulseVM Metamorphic Property Verifier ({len(LAWS)} Laws x 4 Engines) {RESET}")
    print(f"{BOLD}{BLUE}==============================================================={RESET}")

    t0 = time.monotonic()
    chunk_size = max(10, min(100, args.iterations // max(1, args.workers)))
    chunks = []
    i = 0
    while i < args.iterations:
        n = min(chunk_size, args.iterations - i)
        chunks.append((i, n, args.seed, args.law))
        i += n

    passed = 0
    failures = []
    law_totals = Counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(lib_on, lib_off, cp)) as pool:
        for p, fails, lc in pool.imap_unordered(_run_chunk, chunks):
            passed += p
            failures.extend(fails)
            law_totals.update(lc)

    elapsed = max(1e-6, time.monotonic() - t0)
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for idx, (seed, law, kind, msgs, p1, p2) in enumerate(failures[:15], 1):
        print(f"{RED}[FAIL #{idx}]{RESET} Seed={seed} Law={law} Kind={kind}: {'; '.join(msgs)}")
        (dump_dir / f"meta_{law}_seed_{seed}_p1.impas").write_text(p1)
        (dump_dir / f"meta_{law}_seed_{seed}_p2.impas").write_text(p2)

    print(f"\n{BOLD}{BLUE}==============================================================={RESET}")
    print(f"Metamorphic Verification Complete: {GREEN}{passed:,}/{args.iterations:,} Pairs Passed{RESET}, {RED}{len(failures)} Failures{RESET}")
    print(f"Elapsed: {elapsed:.2f}s ({args.iterations / elapsed:,.1f} pairs/s = {2 * args.iterations / elapsed:,.1f} 4-way runs/s across {args.workers} workers)")
    print("Laws:", ", ".join(f"{k}:{law_totals[k]}" for k in LAWS if law_totals[k] > 0))
    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
