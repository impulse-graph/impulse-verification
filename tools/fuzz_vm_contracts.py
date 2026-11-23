#!/usr/bin/env python3
"""
fuzz_vm_contracts.py — Multi-Profile Contract-Guided Differential VM Fuzzer
Generates diverse, deep, adversarial bytecode instruction sequences across 6
specialized profiles (boundary, aliasing, precision, multilane, controlflow,
negative) and verifies 100% bitwise state consensus (including ICTX v1 context
hash) across:
  - C++ (Optimizations ON)
  - C++ (Optimizations OFF)
  - Java 25 (FFM & Vector API, persistent --batch daemon)
  - OCaml (Normative Oracle, persistent --bin-batch daemon)

Supports multi-process parallelism (--workers), 1-minimal delta-debugging
shrinking (--shrink), and live opcode/status/depth telemetry.
"""

import os
import re
import sys
import time
import json
import ctypes
import random
import shutil
import atexit
import tempfile
import argparse
import subprocess
import multiprocessing as mp
from collections import Counter
from pathlib import Path

# Add tools and spec/tools directories to path
TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
SPEC_DIR = REPO_DIR / "spec"
ROOT_DIR = REPO_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SPEC_DIR / "tools"))

from run_vm_asm_suite import (
    Instruction, VmState, OPCODES, parse_impas_file, STATUS_NAMES
)
from four_way_validation import compare_state

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Register Type Tags matching impulse_vm.h
TYPE_NULL           = 0x00
TYPE_INT64          = 0x01
TYPE_NODE_ID        = 0x02
TYPE_RELATION_ID    = 0x03
TYPE_BITSET_HANDLE  = 0x04
TYPE_NODE_VECTOR    = 0x05
TYPE_STRING_HANDLE  = 0x06
TYPE_CSR_SPAN       = 0x07
TYPE_FLOAT          = 0x08
TYPE_DOUBLE         = 0x09
TYPE_STRING_VECTOR  = 0x0A
TYPE_ATTR_CATALOG   = 0x0B
TYPE_FLOAT_VECTOR   = 0x0C
TYPE_DOUBLE_VECTOR  = 0x0D
TYPE_UINT64_VECTOR  = 0x0E
TYPE_LANE_FRONTIER  = 0x10

PROFILES = ("mixed", "boundary", "aliasing", "precision", "multilane", "controlflow", "graph_algos", "stream", "deep_chain", "negative")

# Exact runtime type-contract overrides (eliminates spurious IMPULSE_VM_ERR_INVALID_REGISTER on positive profiles)
_B_NULL   = 1 << TYPE_NULL
_B_INT    = 1 << TYPE_INT64
_B_NODE   = 1 << TYPE_NODE_ID
_B_BS     = 1 << TYPE_BITSET_HANDLE
_B_NVEC   = (1 << TYPE_NODE_VECTOR) | (1 << TYPE_UINT64_VECTOR)
_B_FLT    = 1 << TYPE_FLOAT
_B_DBL    = 1 << TYPE_DOUBLE
_B_FVEC   = 1 << TYPE_FLOAT_VECTOR
_B_DVEC   = 1 << TYPE_DOUBLE_VECTOR
_B_FOD    = _B_FVEC | _B_DVEC
_B_LANE   = 1 << TYPE_LANE_FRONTIER
_B_SET    = _B_BS | _B_NODE | _B_NVEC

STRICT_TYPE_OVERRIDES = {
    "OP_SET_UNION":             (_B_SET, _B_SET, 0),
    "OP_SET_INTERSECT":         (_B_SET, _B_SET, 0),
    "OP_SET_DIFFERENCE":        (_B_SET, _B_SET, 0),
    "OP_SET_CARDINALITY":       (_B_SET, 0, 0),
    "OP_FRONTIER_DIFF":         (_B_BS, _B_BS, 0),
    "OP_MASK_AND":              (_B_BS, _B_BS, 0),
    "OP_MASK_OR":               (_B_BS, _B_BS, 0),
    "OP_MASK_NOT":              (_B_BS | _B_FOD, 0, 0),
    "OP_ROARING_BITMAP_AND":    (_B_BS, _B_BS, 0),
    "OP_ROARING_BITMAP_OR":     (_B_BS, _B_BS, 0),
    "OP_ROARING_BITMAP_AND_NOT":(_B_BS, _B_BS, 0),
    "OP_ASSERT_FINITE":         (_B_FOD | _B_FLT | _B_DBL, 0, 0),
    "OP_VEC_MATH_UNARY":        (_B_FOD | _B_FLT | _B_DBL, 0, 0),
    "OP_VEC_MATH_BINARY":       (_B_FOD | _B_FLT, _B_FOD | _B_FLT, 0),
    "OP_VEC_MATH_TERNARY":      (_B_FOD | _B_FLT, _B_FOD | _B_FLT, _B_FOD | _B_FLT),
    "OP_VEC_BLEND":             (_B_BS, _B_FOD, _B_FOD),
    "OP_VEC_CMP_EQ":            (_B_FOD, _B_FOD | _B_FLT | _B_DBL, 0),
    "OP_VEC_CMP_GT":            (_B_FOD, _B_FOD | _B_FLT | _B_DBL, 0),
    "OP_VEC_CMP_LT":            (_B_FOD, _B_FOD | _B_FLT | _B_DBL, 0),
    "OP_VEC_CMP_BETWEEN":       (_B_FOD, _B_FOD | _B_FLT | _B_DBL | _B_INT | _B_NODE, _B_FOD | _B_FLT | _B_DBL | _B_INT | _B_NODE),
    "OP_EWISE_ADD":             (_B_FOD, _B_FOD, 0),
    "OP_EWISE_MULT":            (_B_FOD, _B_FOD, 0),
    "OP_VECTOR_DIV":            (_B_FOD, _B_FOD, 0),
    "OP_FLOAT_VECTOR_SCALE":    (_B_FVEC, _B_FLT, 0),
    "OP_L1_NORM_DIFF":          (_B_FVEC, _B_FVEC, 0),
    "OP_COALESCE":              (_B_FVEC, _B_FVEC | _B_FLT, 0),
    "OP_EXTRACT_VALIDITY":      (_B_FOD | _B_NVEC, 0, 0),
    "OP_VECTOR_REDUCE_SUM":     (_B_FOD | _B_FLT | _B_DBL, 0, 0),
    "OP_REDUCE":                (_B_FOD | _B_NVEC, 0, 0),
    "OP_MXV":                   (_B_FOD, 0, 0),
    "OP_VXM":                   (_B_FOD, 0, 0),
    "OP_CSR_WALK":              (_B_SET, 0, 0),
    "OP_CSC_WALK":              (_B_SET, 0, 0),
    "OP_ADAPTIVE_WALK":         (_B_SET, 0, 0),
    "OP_LANE_CSR_WALK":         (_B_LANE, 0, 0),
    "OP_LANE_CSC_WALK":         (_B_LANE, 0, 0),
    "OP_LANE_PRUNE_MASK":       (_B_LANE, _B_BS | _B_NVEC | _B_INT, 0),
    "OP_LANE_FILTER_QUORUM":    (_B_LANE, 0, 0),
    "OP_LANE_REDUCE_AND":       (_B_LANE, 0, 0),
    "OP_LANE_REDUCE_ATTR":      (_B_LANE, _B_FOD, 0),
}

# Profile-weighted opcode preferences
PRECISION_OPS = {
    "OP_VEC_MATH_UNARY", "OP_VEC_MATH_BINARY", "OP_VEC_MATH_TERNARY",
    "OP_EWISE_ADD", "OP_EWISE_MULT", "OP_REDUCE", "OP_VECTOR_REDUCE_SUM",
    "OP_VECTOR_DIV", "OP_FLOAT_VECTOR_SCALE", "OP_COALESCE", "OP_EXTRACT_VALIDITY",
    "OP_VEC_CMP_EQ", "OP_VEC_CMP_GT", "OP_VEC_CMP_LT", "OP_VEC_CMP_BETWEEN",
    "OP_VEC_BLEND", "OP_L1_NORM_DIFF", "OP_MXV", "OP_VXM", "OP_ASSERT_FINITE",
}
MULTILANE_OPS = {
    "OP_LANE_INIT_SEEDS", "OP_LANE_INIT_GROUPS", "OP_LANE_CSR_WALK",
    "OP_LANE_CSC_WALK", "OP_LANE_PRUNE_MASK", "OP_LANE_FILTER_QUORUM",
    "OP_LANE_REDUCE_AND", "OP_LANE_REDUCE_ATTR", "OP_VECTOR_ANN_SEARCH",
    "OP_VECTOR_RERANK_DOT", "OP_CSR_WALK", "OP_SET_UNION", "OP_SET_INTERSECT",
}
ALIASING_OPS = {
    "OP_MOV", "OP_SWAP_REG", "OP_CLEAR_REG", "OP_FRONTIER_DIFF",
    "OP_SET_UNION", "OP_SET_INTERSECT", "OP_SET_DIFFERENCE",
    "OP_MASK_AND", "OP_MASK_OR", "OP_MASK_NOT",
    "OP_ROARING_BITMAP_AND", "OP_ROARING_BITMAP_OR", "OP_ROARING_BITMAP_AND_NOT",
    "OP_FLOAT_VECTOR_SCALE", "OP_VECTOR_DIV", "OP_VEC_BLEND", "OP_COALESCE",
    "OP_VEC_MATH_UNARY", "OP_VEC_MATH_BINARY", "OP_EWISE_ADD", "OP_EWISE_MULT",
}
BOUNDARY_OPS = {
    "OP_VECTOR_RERANK_DOT", "OP_VECTOR_ANN_SEARCH", "OP_COALESCE",
    "OP_EXTRACT_VALIDITY", "OP_ASSERT_FINITE", "OP_VECTOR_REDUCE_SUM",
    "OP_REDUCE", "OP_EWISE_ADD", "OP_EWISE_MULT", "OP_VECTOR_DIV",
    "OP_VEC_CMP_EQ", "OP_VEC_CMP_GT", "OP_VEC_CMP_LT", "OP_VEC_CMP_BETWEEN",
    "OP_VEC_BLEND", "OP_L1_NORM_DIFF", "OP_MXV", "OP_VXM",
    "OP_CSR_WALK", "OP_CSC_WALK", "OP_ADAPTIVE_WALK",
    "OP_SET_UNION", "OP_SET_INTERSECT", "OP_SET_DIFFERENCE", "OP_SET_CARDINALITY",
    "OP_MASK_AND", "OP_MASK_OR", "OP_MASK_NOT", "OP_FRONTIER_DIFF",
    "OP_LANE_INIT_SEEDS", "OP_LANE_CSR_WALK", "OP_LANE_FILTER_QUORUM",
}


def get_possible_types_from_mask(mask: int) -> list[int]:
    return [t for t in range(17) if (mask & (1 << t)) != 0]


def _make_large_csr_lines(num_nodes: int, rng: random.Random) -> list[str]:
    """Generate a deterministic multi-word graph (>64 nodes) crossing 64/128-bit bitset word boundaries."""
    out = []
    for u in range(num_nodes):
        if u in (0, 1, 62, 63, 64, 65, num_nodes - 2):
            nbrs = sorted({(u + 1) % num_nodes, (u + 63) % num_nodes, (u + 64) % num_nodes})
        elif rng.random() < 0.35:
            k = rng.randint(1, 3)
            nbrs = sorted({rng.randint(0, num_nodes - 1) for _ in range(k)})
        else:
            nbrs = []
        comma = "," if u < num_nodes - 1 else ""
        out.append(f"    {u}: [{', '.join(map(str, nbrs))}]{comma}")
    return out


class VmContractFuzzer:
    def __init__(self, contracts_path: Path):
        with open(contracts_path, "r") as f:
            raw_contracts = json.load(f)

        # Deduplicate by mnemonic and filter executable fuzzer contracts
        seen = set()
        self.contracts = []
        for c in raw_contracts:
            m = c["mnemonic"]
            if m in seen:
                continue
            if m.startswith("OP_STREAM_") or m in (
                "OP_HALT", "OP_TRAP", "OP_THROW", "OP_RET", "OP_LEAVE_FRAME",
                "OP_JMP", "OP_JZ", "OP_JNZ", "OP_LOOP_DECR", "OP_CALL",
                "OP_ENTER_FRAME", "OP_STABLE_CHECK",
            ):
                continue
            seen.add(m)
            self.contracts.append(c)

    def pick_profile(self, seed: int, requested_profile: str) -> str:
        if requested_profile != "all":
            return requested_profile
        # Deterministic weighted profile selection per seed across all 10 profiles
        r = random.Random(seed ^ 0x5A5A5A5A).random()
        if r < 0.14:
            return "boundary"
        elif r < 0.27:
            return "precision"
        elif r < 0.39:
            return "aliasing"
        elif r < 0.51:
            return "multilane"
        elif r < 0.61:
            return "controlflow"
        elif r < 0.74:
            return "graph_algos"
        elif r < 0.84:
            return "stream"
        elif r < 0.92:
            return "deep_chain"
        elif r < 0.97:
            return "negative"
        return "mixed"

    def _build_preamble(self, seed: int, profile: str, rng: random.Random):
        reg_state: dict[int, set[int]] = {i: {TYPE_NULL} for i in range(64)}
        fuel = 5000

        if profile == "boundary" and rng.random() < 0.55:
            # Multi-word graph crossing 64-bit / 128-bit bitset word boundaries
            n_nodes = rng.choice([72, 132])
            graph_lines = _make_large_csr_lines(n_nodes, rng)
            init_node = rng.choice([0, 63, 64, 65, n_nodes - 1])
            f_len = rng.choice([1, 4, 64, 65, n_nodes])
            f_vals = [rng.choice(["0.0", "-0.0", "1.5", "-2.5", "3.5", "-4.5", "100.0"]) for _ in range(f_len)]
            attr_float = "[" + ", ".join(f_vals) + "]"
            i_len = rng.choice([1, 4, 8])
            i_vals = [str(rng.choice([0, 1, 63, 64, 65, n_nodes - 1, 200])) for _ in range(i_len)]
            attr_int = "[" + ", ".join(i_vals) + "]"
        else:
            graph_pools = [
                ["    0: [1, 2],", "    1: [2, 3],", "    2: [3],", "    3: []"],
                ["    0: [1, 2],", "    1: [0, 3],", "    2: [3, 4],", "    3: [5],", "    4: [2],", "    5: []"],
                ["    0: [0, 1],", "    1: [],", "    2: [3, 4],", "    3: [2],", "    4: [5, 6],", "    5: [7],", "    6: [],", "    7: [4]"],
                ["    0: [],", "    1: [2],", "    2: [],", "    3: [1]"],
            ]
            float_pools = [
                "[1.5, 2.5, 3.5, 4.5]",
                "[-2.5, 0.0, 3.5, -4.5]",
                "[0.0, -0.0, 1.0, -1.0]",
                "[0.25, 100.0, -0.5, 42.5]",
                "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
                "[-0.0]",
                "[-1.0, -4.0, 0.0, 9.0]",
            ]
            int_pools = [
                "[10, 20, 30, 40]",
                "[0, 1, 2, 3]",
                "[0, 100, 200, 300]",
                "[1, 2, 3, 4, 5, 6]",
                "[0]",
            ]
            graph_lines = rng.choice(graph_pools)
            attr_float = rng.choice(float_pools)
            attr_int = rng.choice(int_pools)
            init_node = rng.choice([0, 1, 2])

        init_int = rng.choice([0, 1, 2, 42, 7])
        init_flt = rng.choice(["3.5", "-2.5", "0.0", "-0.0", "1.0"])
        if profile == "controlflow" and rng.random() < 0.30:
            fuel = rng.choice([15, 25, 40, 60])

        lines = [
            f"; CONTRACT-GUIDED FUZZ TEST (Seed: {seed}, Profile: {profile})",
            "; {EXPECT: STATUS = IMPULSE_VM_OK}",
            f".fuel {fuel}",
            "",
            ".data",
            ".csr_inline FUZZ_GRAPH = {",
            *graph_lines,
            "}",
            f".array_float ATTR_FLOAT = {attr_float}",
            f".array_int ATTR_INT = {attr_int}",
            "",
            ".text",
            "0x00: OP_INIT_MOCK_GRAPH         R0, FUZZ_GRAPH",
            f"0x01: OP_LOAD_CONST_INT          R1, {init_int}",
            f"0x02: OP_LOAD_CONST_FLOAT        R2, {init_flt}",
            f"0x03: OP_INIT_INPUT_NODE         R3, {init_node}",
            "0x04: OP_INIT_INPUT_SET          R4",
            "0x05: OP_LOAD_INLINE_ARRAY       R5, ATTR_FLOAT",
            "0x06: OP_LOAD_INLINE_INT_ARRAY   R6, ATTR_INT",
        ]
        reg_state[1] = {TYPE_INT64}
        reg_state[2] = {TYPE_FLOAT}
        reg_state[3] = {TYPE_NODE_ID}
        reg_state[4] = {TYPE_BITSET_HANDLE}
        reg_state[5] = {TYPE_FLOAT_VECTOR}
        reg_state[6] = {TYPE_NODE_VECTOR}
        pc = 7

        # Seed TYPE_DOUBLE_VECTOR (R7) on precision / boundary / 50% of other profiles
        if profile in ("precision", "boundary") or rng.random() < 0.5:
            lines.append(f"0x{pc:02X}: OP_VEC_MATH_UNARY          R7, R5, 1, 1")
            reg_state[7] = {TYPE_DOUBLE_VECTOR}
            pc += 1

        # Seed TYPE_DOUBLE scalar (R8) on precision / 35% of boundary
        if profile == "precision" or (profile == "boundary" and rng.random() < 0.35):
            lines.append(f"0x{pc:02X}: OP_VEC_MATH_UNARY          R8, R2, 1, 1")
            reg_state[8] = {TYPE_DOUBLE}
            pc += 1

        # Seed NaN-containing float vector (R9) on boundary/precision (SQRT of ATTR_FLOAT when negative present)
        if profile in ("boundary", "precision") and "-" in attr_float and rng.random() < 0.5:
            lines.append(f"0x{pc:02X}: OP_VEC_MATH_UNARY          R9, R5, 2")
            reg_state[9] = {TYPE_FLOAT_VECTOR}
            pc += 1

        return lines, reg_state, pc

    def _generate_controlflow_program(self, seed: int, rng: random.Random, max_instructions: int) -> str:
        lines, reg_state, pc = self._build_preamble(seed, "controlflow", rng)
        idiom = rng.choice(["loop_walk", "loop_math", "cond_branch", "call_ret", "frame_call_ret", "stable_loop"])

        if idiom == "loop_walk":
            iters = rng.randint(2, 5)
            lines.append(f"0x{pc:02X}: OP_LOAD_CONST_INT          R10, {iters}")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_CSR_WALK                R11, R3, 0")
            pc += 1
            loop_pc = pc
            lines.append(f"0x{pc:02X}: OP_CSR_WALK                R12, R11, 0")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_SET_UNION               R11, R11, R12")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_FRONTIER_DIFF           R13, R12, R4")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_LOOP_DECR               R10, {loop_pc}")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_SET_CARDINALITY         R14, R11")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_HALT")
            return "\n".join(lines) + "\n"

        if idiom == "loop_math":
            iters = rng.randint(2, 5)
            vec_r = 7 if 7 in reg_state and TYPE_DOUBLE_VECTOR in reg_state[7] else 5
            lines.append(f"0x{pc:02X}: OP_LOAD_CONST_INT          R10, {iters}")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_MOV                     R11, R{vec_r}")
            pc += 1
            loop_pc = pc
            lines.append(f"0x{pc:02X}: OP_EWISE_ADD               R11, R11, R5")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_MXV                     R12, R11, 0")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_LOOP_DECR               R10, {loop_pc}")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_VECTOR_REDUCE_SUM       R13, R12")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_HALT")
            return "\n".join(lines) + "\n"

        if idiom == "cond_branch":
            cmp_op = rng.choice(["OP_VEC_CMP_EQ", "OP_VEC_CMP_GT", "OP_VEC_CMP_LT"])
            jmp_op = rng.choice(["OP_JZ", "OP_JNZ"])
            lines.append(f"0x{pc:02X}: {cmp_op:<24} R11, R5, R2")
            pc += 1
            # Jump over 2 instructions if condition matches
            target_pc = pc + 3
            lines.append(f"0x{pc:02X}: {jmp_op:<24} {target_pc}")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_EWISE_MULT              R12, R5, R5")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_VECTOR_REDUCE_SUM       R13, R12")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_SET_CARDINALITY         R14, R11")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_HALT")
            return "\n".join(lines) + "\n"

        if idiom == "frame_call_ret":
            call_pc = pc
            halt_pc = pc + 2
            sub_pc = pc + 3
            lines.append(f"0x{call_pc:02X}: OP_CALL                    {sub_pc}")
            lines.append(f"0x{call_pc + 1:02X}: OP_VECTOR_REDUCE_SUM       R15, R12")
            lines.append(f"0x{halt_pc:02X}: OP_HALT")
            lines.append(f"0x{sub_pc:02X}: OP_ENTER_FRAME             0")
            lines.append(f"0x{sub_pc + 1:02X}: OP_EWISE_MULT              R12, R5, R5")
            lines.append(f"0x{sub_pc + 2:02X}: OP_LEAVE_FRAME             0")
            lines.append(f"0x{sub_pc + 3:02X}: OP_RET")
            return "\n".join(lines) + "\n"

        if idiom == "stable_loop":
            lines.append(f"0x{pc:02X}: OP_CSR_WALK                R11, R3, 0")
            pc += 1
            loop_pc = pc
            lines.append(f"0x{pc:02X}: OP_CSR_WALK                R12, R11, 0")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_FRONTIER_DIFF           R13, R12, R11")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_SET_UNION               R11, R11, R12")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_STABLE_CHECK            {loop_pc}")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_SET_CARDINALITY         R14, R11")
            pc += 1
            lines.append(f"0x{pc:02X}: OP_HALT")
            return "\n".join(lines) + "\n"

        # call_ret subroutine idiom
        call_pc = pc
        halt_pc = pc + 2
        sub_pc = pc + 3
        lines.append(f"0x{call_pc:02X}: OP_CALL                    {sub_pc}")
        lines.append(f"0x{call_pc + 1:02X}: OP_VECTOR_REDUCE_SUM       R15, R12")
        lines.append(f"0x{halt_pc:02X}: OP_HALT")
        lines.append(f"0x{sub_pc:02X}: OP_EWISE_ADD               R12, R5, R5")
        lines.append(f"0x{sub_pc + 1:02X}: OP_RET")
        return "\n".join(lines) + "\n"

    def _generate_graph_algos_program(self, seed: int, rng: random.Random) -> str:
        """Exercises high-level graph analytics & utility opcodes (2-hop, Kleene*, CC, K-Core, TC, Louvain, Motif, SpMV, Collect, SetCover)."""
        lines, reg_state, pc = self._build_preamble(seed, "graph_algos", rng)
        # Add an inline seed set to the .data block right before .text
        text_idx = lines.index(".text")
        seed_vals = sorted({rng.randint(0, 3) for _ in range(rng.randint(1, 3))})
        lines.insert(text_idx - 1, f".set SEED_SET = [{', '.join(map(str, seed_vals))}]")

        # Load inline set & set max DOP & NOP
        lines.append(f"0x{pc:02X}: OP_LOAD_INLINE_SET         R10, SEED_SET")
        pc += 1
        if rng.random() < 0.5:
            lines.append(f"0x{pc:02X}: OP_SET_MAX_DOP             {rng.choice([1, 2, 4])}")
            pc += 1
        if rng.random() < 0.4:
            lines.append(f"0x{pc:02X}: OP_NOP")
            pc += 1

        # Pick 5-7 analytics / structural / storage opcodes and chain them
        ops = rng.sample([
            "2hop", "kleene", "cc", "cc_hook", "island", "kcore", "tc",
            "louvain", "motif3", "iso", "brandes", "brandes_bwd", "delta_step",
            "coo_walk", "dense_walk", "walk_reduce_sum",
            "spmv", "collect", "collect_vmap", "indirect", "setcover", "lane_edge_mask",
            "direct_stores", "walk_reduces", "walk_filtered", "storage_gather",
            "keys_and_time",
        ], k=rng.randint(5, 7))

        for op in ops:
            if op == "2hop":
                lines.append(f"0x{pc:02X}: OP_CSR_WALK_2HOP           R11, 0, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_SET_CARDINALITY         R12, R11")
                pc += 1
            elif op == "kleene":
                src = rng.choice(["R3", "R10"])
                lines.append(f"0x{pc:02X}: OP_FIXPOINT_KLEENE_STAR    R13, {src}, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_SET_CARDINALITY         R14, R13")
                pc += 1
            elif op == "cc":
                lines.append(f"0x{pc:02X}: OP_CC_AFFOREST             R15, R0")
                pc += 1
            elif op == "island":
                lines.append(f"0x{pc:02X}: OP_ISLAND_DETECT           R16, R10")
                pc += 1
            elif op == "kcore":
                lines.append(f"0x{pc:02X}: OP_KCORE_DECOMPOSITION     R17, R10")
                pc += 1
            elif op == "tc":
                lines.append(f"0x{pc:02X}: OP_TC_SWEEP_BATCH          R18, R10, 0")
                pc += 1
            elif op == "louvain":
                lines.append(f"0x{pc:02X}: OP_LOUVAIN_MODULARITY      R19, R10")
                pc += 1
            elif op == "motif3":
                lines.append(f"0x{pc:02X}: OP_MOTIF_MATCH_3           R20, R10")
                pc += 1
            elif op == "spmv":
                lines.append(f"0x{pc:02X}: OP_SPARSE_MATVEC           R21, R5")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_VECTOR_REDUCE_SUM       R22, R21")
                pc += 1
            elif op == "collect":
                lines.append(f"0x{pc:02X}: OP_COLLECT_BITSET          R23, R10")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_COLLECT_ARRAY           R24, R5")
                pc += 1
            elif op == "indirect":
                lines.append(f"0x{pc:02X}: OP_LOAD_INDIRECT           R25, R5, R1, 1")
                pc += 1
            elif op == "setcover":
                lines.append(f"0x{pc:02X}: OP_LANE_INIT_SEEDS         R26, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_LANE_CSR_WALK           R27, R26, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_LANE_SET_COVER          R28, R27, R0, {rng.choice([2, 4])}, {rng.choice([0, 1])}")
                pc += 1
            elif op == "cc_hook":
                lines.append(f"0x{pc:02X}: OP_CC_HOOK_COMPRESS        R29, R10")
                pc += 1
            elif op == "iso":
                lines.append(f"0x{pc:02X}: OP_GRAPH_ISOMORPHISM       R30, R10")
                pc += 1
            elif op == "brandes":
                lines.append(f"0x{pc:02X}: OP_BRANDES_FORWARD         R31, R10, R0")
                pc += 1
            elif op == "delta_step":
                lines.append(f"0x{pc:02X}: OP_DELTA_STEP_RELAX        R32, R10, R5, 0")
                pc += 1
            elif op == "coo_walk":
                lines.append(f"0x{pc:02X}: OP_COO_WALK                R33, R10, 0")
                pc += 1
            elif op == "dense_walk":
                lines.append(f"0x{pc:02X}: OP_DENSE_WALK              R34, R10, 0")
                pc += 1
            elif op == "walk_reduce_sum":
                lines.append(f"0x{pc:02X}: OP_CSR_WALK_REDUCE_SUM     R35, R10, 0")
                pc += 1
            elif op == "lane_edge_mask":
                lines.append(f"0x{pc:02X}: OP_LANE_INIT_SEEDS         R36, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_LANE_SET_EDGE_MASK      R37, R36, R6, 0, 1")
                pc += 1
            elif op == "brandes_bwd":
                lines.append(f"0x{pc:02X}: OP_BRANDES_BACKWARD        R38, R10, R0")
                pc += 1
            elif op == "collect_vmap":
                lines.append(f"0x{pc:02X}: OP_COLLECT_VALUE_MAP       R39, R10, R5")
                pc += 1
            elif op == "direct_stores":
                lines.append(f"0x{pc:02X}: OP_CSR_WALK_DIRECT_STORE   R40, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_CSC_WALK_DIRECT_STORE   R41, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_COO_WALK_DIRECT_STORE   R42, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_DENSE_WALK_DIRECT_STORE R43, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_DENSE_WALK_BITMATRIX    R44, R10, 0")
                pc += 1
            elif op == "walk_reduces":
                lines.append(f"0x{pc:02X}: OP_CSR_WALK_REDUCE         R45, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_COO_WALK_REDUCE         R46, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_DENSE_WALK_REDUCE       R47, R10, 0")
                pc += 1
            elif op == "walk_filtered":
                lines.append(f"0x{pc:02X}: OP_CSR_WALK_FILTERED       R48, R10, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_COO_WALK_FILTERED       R49, R10, R10, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_CSR_WALK_PREDICATE      R50, R10, 0")
                pc += 1
            elif op == "storage_gather":
                lines.append(f"0x{pc:02X}: OP_INIT_MOCK_NODE_ATTR     R1, R5, 63")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_LOAD_COLUMN_VECTOR      R51, 1")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_GATHER_NODE_ATTR        R52, R10, 1")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_GATHER_EDGE_ATTR        R53, R10, 1")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_BRIN_ZONE_SKIP          R54, R10")
                pc += 1
            elif op == "keys_and_time":
                lines.append(f"0x{pc:02X}: OP_HAS_KEY_CATALOG         R55, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_MAP_KEYS_TO_DENSE       R55, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_MAP_DENSE_TO_KEYS       R56, R10")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_LOAD_CONST_STR_PREFIX   R57, 0x58595A")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_NODE_FILTER_STR_PREFIX  R57, R3, R57, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_VECTOR_STR_CONCAT       R56, R56, R56")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_VECTOR_TIME_VALID_AT    R58, R1, 1")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_READ_EDGE_WEIGHT        R59, R1, R3, 0")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_VECTOR_MUL_ATTR         R60, R10")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_CREATE_SCRATCH_INDEX    R61, R10")
                pc += 1
                lines.append(f"0x{pc:02X}: OP_DROP_SCRATCH_INDEX      R61")
                pc += 1

        lines.append(f"0x{pc:02X}: OP_HALT")
        return "\n".join(lines) + "\n"

    def _generate_stream_program(self, seed: int, rng: random.Random) -> str:
        """Synthesizes randomized fused edge-stream shaders (0xA0..0xBF) over CSR/CSC/COO walks."""
        graph_pools = [
            ["    0: [1, 2],", "    1: [2, 3],", "    2: [3],", "    3: []"],
            ["    0: [1, 2],", "    1: [0, 3],", "    2: [3, 4],", "    3: [5],", "    4: [2],", "    5: []"],
            ["    0: [0, 1],", "    1: [2],", "    2: [3],", "    3: [0]"],
        ]
        graph_lines = rng.choice(graph_pools)
        n_nodes = len(graph_lines)
        src_vals = [rng.choice(["1.0", "2.5", "-1.5", "0.0", "4.0", "10.0"]) for _ in range(n_nodes)]
        edge_vals = [rng.choice(["0.5", "1.0", "2.0", "-3.0", "0.0", "5.0"]) for _ in range(max(8, n_nodes * 2))]
        walk_op = rng.choice(["OP_CSR_WALK_STREAM", "OP_CSC_WALK_STREAM", "OP_COO_WALK_STREAM"])
        init_node = rng.randint(0, n_nodes - 1)

        # Build shader instructions first so we know exact shader_pc
        shader_ops = []
        shader_ops.append("OP_STREAM_FUNC_BEGIN")
        shader_ops.append("OP_STREAM_LOAD_SRC         0, 1")
        if rng.random() < 0.6:
            shader_ops.append("OP_STREAM_LOAD_EDGE        1, 2")
        else:
            shader_ops.append("OP_STREAM_LOAD_TGT         1, 1")
        if rng.random() < 0.5:
            shader_ops.append("OP_STREAM_LOAD_SRC_ID      3")
            shader_ops.append("OP_STREAM_LOAD_TGT_ID      4")
        else:
            shader_ops.append(f"OP_STREAM_LOAD_CONST       3, {rng.choice(['1.0', '2.0', '0.5', '-1.0'])}")

        # Arithmetic op on stream registers 0 and 1 -> reg 2
        math_op = rng.choice(["OP_STREAM_MATH_ADD", "OP_STREAM_MATH_SUB", "OP_STREAM_MATH_MUL", "OP_STREAM_MATH_DIV", "OP_STREAM_MATH_MOD"])
        shader_ops.append(f"{math_op:<26} 2, 0, 1")
        if rng.random() < 0.4:
            shader_ops.append(f"OP_STREAM_MATH_UNARY       2, 2, {rng.choice(['0x01', '0x1E', '0x1F'])}")

        # Optional comparison + logic (AND/OR/NOT) + select / filter
        if rng.random() < 0.70:
            cmp_op = rng.choice(["OP_STREAM_CMP_EQ", "OP_STREAM_CMP_NEQ", "OP_STREAM_CMP_GT", "OP_STREAM_CMP_LT"])
            shader_ops.append(f"{cmp_op:<26} 5, 2, 3")
            logic_choice = rng.choice(["not_select", "and_select", "or_select", "filter"])
            if logic_choice == "not_select":
                shader_ops.append("OP_STREAM_LOGIC_NOT        6, 5")
                shader_ops.append("OP_STREAM_SELECT           2, 5, 2, 0")
            elif logic_choice == "and_select":
                shader_ops.append("OP_STREAM_LOGIC_AND        6, 5, 5")
                shader_ops.append("OP_STREAM_SELECT           2, 6, 2, 0")
            elif logic_choice == "or_select":
                shader_ops.append("OP_STREAM_LOGIC_OR         6, 5, 3")
                shader_ops.append("OP_STREAM_SELECT           2, 6, 2, 0")
            else:
                shader_ops.append("OP_STREAM_FILTER           5")

        red_kind = rng.choice(["reduce", "reduce", "argmin", "argmax", "yield"])
        if red_kind == "reduce":
            reduce_mode = rng.choice([0, 1, 2])  # 0=SUM, 1=MIN, 2=MAX
            shader_ops.append(f"OP_STREAM_REDUCE           2, 4, {reduce_mode}")
        elif red_kind == "argmin":
            shader_ops.append("OP_STREAM_REDUCE_ARGMIN    2, 4, 8")
        elif red_kind == "argmax":
            shader_ops.append("OP_STREAM_REDUCE_ARGMAX    2, 4, 8")
        else:
            shader_ops.append("OP_STREAM_YIELD            2, 4")
        shader_ops.append("OP_STREAM_FUNC_END")

        lines = [
            f"; CONTRACT-GUIDED FUZZ TEST (Seed: {seed}, Profile: stream)",
            "; {EXPECT: STATUS = IMPULSE_VM_OK}",
            ".fuel 5000",
            "",
            ".data",
            ".csr_inline FUZZ_GRAPH = {",
            *graph_lines,
            "}",
            f".array_float ATTR_SOURCE = [{', '.join(src_vals)}]",
            f".array_float ATTR_EDGE = [{', '.join(edge_vals)}]",
            "",
            ".text",
            "0x00: OP_INIT_MOCK_GRAPH         R0, FUZZ_GRAPH",
            "0x01: OP_LOAD_INLINE_ARRAY       R1, ATTR_SOURCE",
            "0x02: OP_INIT_MOCK_NODE_ATTR     R1, R1, 0",
            "0x03: OP_LOAD_INLINE_ARRAY       R2, ATTR_EDGE",
            "0x04: OP_INIT_MOCK_EDGE_ATTR     R2, R2, 0",
            f"0x05: OP_INIT_INPUT_NODE         R3, {init_node}",
            "0x06: OP_CLEAR_REG               R4",
            "0x07: OP_CLEAR_REG               R5",
            f"0x08: {walk_op:<26} R6, R3, 0, 12",
            "0x09: OP_VECTOR_REDUCE_SUM       R5, R4",
            "0x0A: OP_SET_CARDINALITY         R7, R6",
            "0x0B: OP_HALT",
            "",
        ]
        pc = 12
        for sop in shader_ops:
            lines.append(f"0x{pc:02X}: {sop}")
            pc += 1
        return "\n".join(lines) + "\n"

    def generate_program(self, seed: int, max_instructions: int = 12, profile: str = "all", neg_rate: float = 0.0) -> tuple[str, str]:
        actual_profile = self.pick_profile(seed, profile)
        rng = random.Random(seed)

        if actual_profile == "controlflow":
            return self._generate_controlflow_program(seed, rng, max_instructions), actual_profile
        if actual_profile == "graph_algos":
            return self._generate_graph_algos_program(seed, rng), actual_profile
        if actual_profile == "stream":
            return self._generate_stream_program(seed, rng), actual_profile

        lines, reg_state, pc = self._build_preamble(seed, actual_profile, rng)
        aliased_regs: list[int] = []
        eff_neg_rate = 0.40 if actual_profile == "negative" else neg_rate

        preferred_set = None
        eff_max_instrs = max_instructions
        if actual_profile == "precision":
            preferred_set = PRECISION_OPS
        elif actual_profile == "multilane":
            preferred_set = MULTILANE_OPS
        elif actual_profile == "aliasing":
            preferred_set = ALIASING_OPS
        elif actual_profile == "boundary":
            preferred_set = BOUNDARY_OPS
        elif actual_profile == "deep_chain":
            preferred_set = BOUNDARY_OPS | PRECISION_OPS | MULTILANE_OPS | ALIASING_OPS
            eff_max_instrs = rng.randint(max(20, max_instructions), max(28, max_instructions + 14))

        for _ in range(eff_max_instrs):
            eligible = []
            for c in self.contracts:
                m = c["mnemonic"]
                if m in STRICT_TYPE_OVERRIDES:
                    allowed_src, allowed_aux1, allowed_aux2 = STRICT_TYPE_OVERRIDES[m]
                else:
                    allowed_src = c["allowed_src_types"]
                    allowed_aux1 = c["allowed_aux1_types"]
                    allowed_aux2 = c["allowed_aux2_types"]

                src_regs = [r for r, types in reg_state.items() if (allowed_src == 0) or any((allowed_src & (1 << t)) for t in types if t != TYPE_NULL)]
                if allowed_src != 0 and not src_regs:
                    continue

                aux1_regs = [r for r, types in reg_state.items() if (allowed_aux1 == 0) or c["allow_zero_skip_aux1"] or any((allowed_aux1 & (1 << t)) for t in types if t != TYPE_NULL)]
                if allowed_aux1 != 0 and not c["allow_zero_skip_aux1"] and not aux1_regs:
                    continue

                aux2_regs = [r for r, types in reg_state.items() if (allowed_aux2 == 0) or c["allow_zero_skip_aux2"] or any((allowed_aux2 & (1 << t)) for t in types if t != TYPE_NULL)]
                if allowed_aux2 != 0 and not c["allow_zero_skip_aux2"] and not aux2_regs:
                    continue

                eligible.append(c)

            if not eligible:
                break

            c = None
            if aliased_regs and rng.random() < (0.50 if actual_profile == "aliasing" else 0.25):
                alias_ops = [x for x in eligible if x["mnemonic"] in ALIASING_OPS]
                if alias_ops:
                    c = rng.choice(alias_ops)
            if c is None and preferred_set and rng.random() < 0.75:
                pref_ops = [x for x in eligible if x["mnemonic"] in preferred_set]
                if pref_ops:
                    c = rng.choice(pref_ops)
            if c is None:
                c = rng.choice(eligible)

            mnemonic = c["mnemonic"]
            if mnemonic in STRICT_TYPE_OVERRIDES:
                allowed_src, allowed_aux1, allowed_aux2 = STRICT_TYPE_OVERRIDES[mnemonic]
            else:
                allowed_src = c["allowed_src_types"]
                allowed_aux1 = c["allowed_aux1_types"]
                allowed_aux2 = c["allowed_aux2_types"]

            inject_neg = eff_neg_rate > 0.0 and rng.random() < eff_neg_rate

            # Select src register
            src_reg = 0
            if allowed_src != 0:
                valid_srcs = [r for r, types in reg_state.items() if any((allowed_src & (1 << t)) for t in types if t != TYPE_NULL)]
                if inject_neg:
                    invalid_srcs = [r for r, types in reg_state.items() if r <= 31 and not any((allowed_src & (1 << t)) for t in types if t != TYPE_NULL)]
                    src_reg = rng.choice(invalid_srcs) if invalid_srcs else (rng.choice(valid_srcs) if valid_srcs else 0)
                else:
                    alias_srcs = [r for r in aliased_regs if r in valid_srcs]
                    if alias_srcs and mnemonic in ALIASING_OPS:
                        src_reg = rng.choice(alias_srcs)
                    else:
                        src_reg = rng.choice(valid_srcs) if valid_srcs else 0

            # Select aux1 register
            aux1_reg = 0
            if allowed_aux1 != 0:
                valid_aux1 = [r for r, types in reg_state.items() if any((allowed_aux1 & (1 << t)) for t in types if t != TYPE_NULL)]
                if actual_profile == "aliasing" and src_reg in valid_aux1 and rng.random() < 0.30:
                    aux1_reg = src_reg
                elif c["allow_zero_skip_aux1"] and rng.random() < 0.15:
                    aux1_reg = 0
                elif valid_aux1:
                    aux1_reg = rng.choice(valid_aux1)

            # Select aux2 register
            aux2_reg = 0
            if allowed_aux2 != 0:
                valid_aux2 = [r for r, types in reg_state.items() if any((allowed_aux2 & (1 << t)) for t in types if t != TYPE_NULL)]
                if c["allow_zero_skip_aux2"] and rng.random() < 0.15:
                    aux2_reg = 0
                elif valid_aux2:
                    aux2_reg = rng.choice(valid_aux2)

            # Destination register selection (aggressive aliasing in aliasing profile, OOB in negative)
            if actual_profile == "negative" and rng.random() < 0.10:
                dst_reg = rng.choice([64, 99])
            elif actual_profile == "aliasing" and rng.random() < 0.50:
                cand_dst = [r for r in (src_reg, aux1_reg) if r >= 10]
                dst_reg = rng.choice(cand_dst) if cand_dst else rng.randint(10, 31)
            elif rng.random() < 0.15:
                active_regs = [r for r, types in reg_state.items() if r >= 10 and TYPE_NULL not in types]
                dst_reg = rng.choice(active_regs) if active_regs else rng.randint(10, 31)
            else:
                dst_reg = rng.randint(10, 31)

            instr_str = self._format_instruction(pc, c, dst_reg, src_reg, aux1_reg, aux2_reg, rng, actual_profile)
            if instr_str:
                lines.append(instr_str)
                if dst_reg < 64:
                    self._update_reg_state(reg_state, aliased_regs, c, mnemonic, dst_reg, src_reg, aux1_reg, aux2_reg, rng)
                pc += 1

        lines.append(f"0x{pc:02X}: OP_HALT")
        return "\n".join(lines) + "\n", actual_profile

    def _update_reg_state(self, reg_state, aliased_regs, c, mnemonic, dst_reg, src_reg, aux1_reg, aux2_reg, rng):
        if mnemonic == "OP_VEC_MATH_UNARY":
            st_src = reg_state.get(src_reg, {TYPE_NULL})
            if st_src == {TYPE_FLOAT}:
                reg_state[dst_reg] = {TYPE_FLOAT}
            elif st_src == {TYPE_DOUBLE}:
                reg_state[dst_reg] = {TYPE_DOUBLE}
            elif st_src == {TYPE_DOUBLE_VECTOR}:
                reg_state[dst_reg] = {TYPE_DOUBLE_VECTOR}
            else:
                reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        elif mnemonic == "OP_VEC_MATH_BINARY":
            s1, s2 = reg_state.get(src_reg, set()), reg_state.get(aux1_reg, set())
            if s1 <= {TYPE_FLOAT, TYPE_DOUBLE} and s2 <= {TYPE_FLOAT, TYPE_DOUBLE}:
                reg_state[dst_reg] = {TYPE_DOUBLE} if (TYPE_DOUBLE in s1 or TYPE_DOUBLE in s2) else {TYPE_FLOAT}
            elif TYPE_DOUBLE_VECTOR in s1 or TYPE_DOUBLE_VECTOR in s2 or TYPE_DOUBLE in s1 or TYPE_DOUBLE in s2:
                reg_state[dst_reg] = {TYPE_DOUBLE_VECTOR}
            else:
                reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        elif mnemonic == "OP_VEC_MATH_TERNARY":
            s1, s2, s3 = reg_state.get(src_reg, set()), reg_state.get(aux1_reg, set()), reg_state.get(aux2_reg, set())
            if TYPE_DOUBLE_VECTOR in (s1 | s2 | s3) or TYPE_DOUBLE in (s1 | s2 | s3):
                reg_state[dst_reg] = {TYPE_DOUBLE_VECTOR}
            else:
                reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        elif mnemonic in ("OP_EWISE_ADD", "OP_EWISE_MULT", "OP_VECTOR_DIV"):
            if TYPE_DOUBLE_VECTOR in reg_state.get(src_reg, set()) or TYPE_DOUBLE_VECTOR in reg_state.get(aux1_reg, set()):
                reg_state[dst_reg] = {TYPE_DOUBLE_VECTOR}
            else:
                reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        elif mnemonic in ("OP_MXV", "OP_VXM"):
            if TYPE_DOUBLE_VECTOR in reg_state.get(src_reg, set()):
                reg_state[dst_reg] = {TYPE_DOUBLE_VECTOR}
            else:
                reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        elif mnemonic in (
            "OP_CSR_WALK", "OP_CSC_WALK", "OP_ADAPTIVE_WALK",
            "OP_SET_UNION", "OP_SET_INTERSECT", "OP_SET_DIFFERENCE",
            "OP_INIT_INPUT_SET", "OP_EXTRACT_VALIDITY", "OP_VEC_CMP_BETWEEN",
            "OP_VEC_CMP_EQ", "OP_VEC_CMP_GT", "OP_VEC_CMP_LT",
            "OP_MASK_AND", "OP_MASK_OR", "OP_MASK_NOT",
            "OP_ROARING_BITMAP_AND", "OP_ROARING_BITMAP_OR", "OP_ROARING_BITMAP_AND_NOT",
            "OP_FRONTIER_DIFF", "OP_LANE_FILTER_QUORUM",
        ):
            reg_state[dst_reg] = {TYPE_BITSET_HANDLE}
        elif mnemonic in ("OP_SET_CARDINALITY", "OP_CSR_DEGREE", "OP_HAS_CSR", "OP_HAS_CSC", "OP_HAS_COO", "OP_LANE_REDUCE_AND"):
            reg_state[dst_reg] = {TYPE_INT64}
        elif mnemonic == "OP_L1_NORM_DIFF":
            reg_state[dst_reg] = {TYPE_FLOAT}
        elif mnemonic == "OP_VEC_BLEND":
            if TYPE_DOUBLE_VECTOR in reg_state.get(aux1_reg, set()) or TYPE_DOUBLE_VECTOR in reg_state.get(aux2_reg, set()):
                reg_state[dst_reg] = {TYPE_DOUBLE_VECTOR}
            else:
                reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        elif mnemonic == "OP_SWAP_REG":
            t_d = set(reg_state.get(dst_reg, {TYPE_NULL}))
            t_s = set(reg_state.get(src_reg, {TYPE_NULL}))
            reg_state[dst_reg] = t_s
            if src_reg < 64:
                reg_state[src_reg] = t_d
        elif mnemonic == "OP_CLEAR_REG":
            reg_state[dst_reg] = {TYPE_NULL}
        elif mnemonic == "OP_MOV":
            reg_state[dst_reg] = set(reg_state.get(src_reg, {TYPE_NULL}))
            if reg_state[dst_reg] & {TYPE_FLOAT_VECTOR, TYPE_DOUBLE_VECTOR, TYPE_BITSET_HANDLE, TYPE_NODE_VECTOR}:
                aliased_regs.append(dst_reg)
        elif mnemonic in ("OP_FLOAT_VECTOR_SCALE", "OP_ASSERT_FINITE", "OP_ASSERT", "OP_ALLOC_SCRATCH", "OP_ASSERT_SCRATCH_BYTES"):
            pass
        elif mnemonic in ("OP_LANE_INIT_SEEDS", "OP_LANE_INIT_GROUPS", "OP_LANE_CSR_WALK", "OP_LANE_CSC_WALK", "OP_LANE_PRUNE_MASK"):
            reg_state[dst_reg] = {TYPE_LANE_FRONTIER}
        elif mnemonic in ("OP_VECTOR_ANN_SEARCH", "OP_VECTOR_RERANK_DOT", "OP_LANE_REDUCE_ATTR"):
            reg_state[dst_reg] = {TYPE_FLOAT_VECTOR}
        else:
            out_types = get_possible_types_from_mask(c["output_dst_types"])
            if out_types and c["modifies_dst_state"]:
                reg_state[dst_reg] = {rng.choice(out_types)}

    def _format_instruction(self, pc: int, c: dict, dst: int, src: int, aux1: int, aux2: int, rng: random.Random, profile: str) -> str:
        m = c["mnemonic"]
        hex_pc = f"0x{pc:02X}"

        if m in ("OP_CSR_WALK", "OP_CSC_WALK", "OP_ADAPTIVE_WALK", "OP_CSR_DEGREE"):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, 0"
        elif m in ("OP_HAS_CSR", "OP_HAS_CSC", "OP_HAS_COO"):
            return f"{hex_pc}: {m:<24} R{dst}, 0"
        elif m in (
            "OP_SET_UNION", "OP_SET_INTERSECT", "OP_SET_DIFFERENCE",
            "OP_MASK_AND", "OP_MASK_OR",
            "OP_ROARING_BITMAP_AND", "OP_ROARING_BITMAP_OR", "OP_ROARING_BITMAP_AND_NOT",
            "OP_FRONTIER_DIFF",
        ):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}"
        elif m in ("OP_SET_CARDINALITY", "OP_MOV", "OP_SWAP_REG", "OP_VECTOR_REDUCE_SUM", "OP_EXTRACT_VALIDITY", "OP_MASK_NOT"):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}"
        elif m == "OP_CLEAR_REG":
            return f"{hex_pc}: {m:<24} R{dst}"
        elif m == "OP_ASSERT_FINITE":
            return f"{hex_pc}: {m:<24} R{src}"
        elif m in ("OP_VEC_CMP_EQ", "OP_VEC_CMP_GT", "OP_VEC_CMP_LT"):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}"
        elif m == "OP_VEC_CMP_BETWEEN":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, R{aux2}"
        elif m == "OP_VEC_BLEND":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, R{aux2}"
        elif m == "OP_L1_NORM_DIFF":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}"
        elif m == "OP_VEC_MATH_UNARY":
            # IEEE-754 bit-exact unary ops: ABS(1), SQRT(2), FLOOR(30), CEIL(31), TRUNC(32), ROUND(33), RELU(37), ISNAN(52), ISINF(53), ISFINITE(54)
            sub_op = rng.choice([1, 2, 30, 31, 32, 33, 37, 52, 53, 54])
            if profile == "precision" and rng.random() < 0.35:
                return f"{hex_pc}: {m:<24} R{dst}, R{src}, {sub_op}, 1"
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, {sub_op}"
        elif m == "OP_VEC_MATH_BINARY":
            # Binary math opcodes 0..3 (add, sub, mul, div)
            sub_op = rng.randint(0, 3)
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, {sub_op}"
        elif m == "OP_VEC_MATH_TERNARY":
            # Ternary math opcodes: 0 (FMA), 34 (CLAMP), 51 (LERP)
            sub_op = rng.choice([0, 34, 51])
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, R{aux2}, {sub_op}"
        elif m == "OP_FLOAT_VECTOR_SCALE":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, 2"
        elif m == "OP_LANE_INIT_SEEDS":
            return f"{hex_pc}: {m:<24} R{dst}, 0"
        elif m == "OP_LANE_INIT_GROUPS":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}"
        elif m in ("OP_LANE_CSR_WALK", "OP_LANE_CSC_WALK"):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, 0"
        elif m == "OP_LANE_PRUNE_MASK":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, 0"
        elif m == "OP_LANE_FILTER_QUORUM":
            k = rng.choice([0, 1, 2, 3]) if profile == "multilane" else 1
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, {k}"
        elif m == "OP_LANE_REDUCE_AND":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}"
        elif m == "OP_LANE_REDUCE_ATTR":
            red_op = rng.choice([0, 1, 2, 3, 6, 7])
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, {red_op}, 1"
        elif m == "OP_VECTOR_ANN_SEARCH":
            k = rng.choice([1, 2, 4]) if profile in ("multilane", "boundary") else 2
            metric = rng.choice([0, 1, 2, 3]) if profile == "multilane" else 0
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, {k}, 0, {metric}"
        elif m == "OP_VECTOR_RERANK_DOT":
            metric = rng.choice([0, 1, 2, 3]) if profile in ("multilane", "boundary") else 0
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}, 0, {metric}"
        elif m == "OP_LOAD_CONST_INT":
            val = rng.choice([0, 1, 2, 63, 64, rng.randint(1, 100)])
            return f"{hex_pc}: {m:<24} R{dst}, {val}"
        elif m == "OP_LOAD_CONST_FLOAT":
            val = rng.choice([-0.0, 0.0, -3.5, -1.0, round(rng.uniform(0.1, 10.0), 2)])
            return f"{hex_pc}: {m:<24} R{dst}, {val}"
        elif m == "OP_VECTOR_DIV":
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}"
        elif m in ("OP_EWISE_ADD", "OP_EWISE_MULT", "OP_COALESCE"):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, R{aux1}"
        elif m in ("OP_MXV", "OP_VXM"):
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, 0"
        elif m == "OP_REDUCE":
            red_op = rng.choice([0, 1, 2, 3, 6, 7])
            return f"{hex_pc}: {m:<24} R{dst}, R{src}, {red_op}"
        elif m in ("OP_ALLOC_SCRATCH", "OP_ASSERT_SCRATCH_BYTES"):
            return f"{hex_pc}: {m:<24} R{src}"
        elif m == "OP_ASSERT":
            return f"{hex_pc}: {m:<24} R{src}, 0"
        else:
            return None


def _resolve_java_cmd():
    java_bin = os.environ.get("JAVA_HOME", "")
    if java_bin and os.path.exists(os.path.join(java_bin, "bin", "java")):
        return os.path.join(java_bin, "bin", "java")
    if os.path.exists("/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home/bin/java"):
        return "/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home/bin/java"
    return "java"


def _configure_cpp_lib(path):
    lib = ctypes.CDLL(path)
    lib.impulse_vm_context_create.argtypes = [ctypes.c_void_p]
    lib.impulse_vm_context_create.restype = ctypes.c_void_p
    lib.impulse_vm_context_destroy.argtypes = [ctypes.c_void_p]
    lib.impulse_vm_context_destroy.restype = None
    lib.impulse_vm_context_bind_inline_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.impulse_vm_context_bind_inline_data.restype = None
    lib.impulse_vm_context_set_fuel.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.impulse_vm_context_set_fuel.restype = None
    lib.impulse_vm_execute.argtypes = [ctypes.POINTER(Instruction), ctypes.c_size_t, ctypes.POINTER(VmState), ctypes.c_uint64]
    lib.impulse_vm_execute.restype = ctypes.c_int
    lib.impulse_vm_state_hash_hex.argtypes = [ctypes.POINTER(VmState), ctypes.c_int32, ctypes.c_char_p]
    lib.impulse_vm_state_hash_hex.restype = ctypes.c_int
    return lib


def _run_cpp_parsed(lib, c_instrs, num_instrs, data_bytes, expectations):
    ctx = lib.impulse_vm_context_create(None)
    if "fuel" in expectations:
        lib.impulse_vm_context_set_fuel(ctx, expectations["fuel"])
    elif os.environ.get("IMPULSE_TEST_ENABLE_FUEL") == "1":
        lib.impulse_vm_context_set_fuel(ctx, 1000000)

    c_buf = None
    if data_bytes:
        c_buf = (ctypes.c_char * len(data_bytes)).from_buffer_copy(data_bytes)
        lib.impulse_vm_context_bind_inline_data(ctx, ctypes.cast(c_buf, ctypes.c_void_p), len(data_bytes))

    state = VmState()
    state.query_context = ctx
    status_code = lib.impulse_vm_execute(c_instrs, num_instrs, ctypes.byref(state), 0)
    status_name = STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")

    hex_buf = ctypes.create_string_buffer(65)
    lib.impulse_vm_state_hash_hex(ctypes.byref(state), status_code, hex_buf)
    context_hash = hex_buf.value.decode("utf-8")

    zf = bool(state.flags & (1 << 0))
    st = bool(state.flags & (1 << 4))
    regs = {f"R{r}": state.registers[r] for r in range(64)}
    reg_types = {f"R{r}": state.register_types[r] for r in range(64)}
    lib.impulse_vm_context_destroy(ctx)
    return {
        "status": status_name,
        "pc": state.pc,
        "flags": {"zf": zf, "st": st},
        "context_hash": context_hash,
        "registers": regs,
        "register_types": reg_types,
    }


class PersistentBatchClient:
    def __init__(self, cmd, cwd=None, name="worker"):
        self.cmd = cmd
        self.cwd = cwd
        self.name = name
        self.proc = None
        self._start()

    def _start(self):
        self.close()
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, line):
        for attempt in range(2):
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            try:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()
                resp = self.proc.stdout.readline()
                if not resp:
                    raise EOFError(f"{self.name} closed stdout unexpectedly")
                data = json.loads(resp)
                if "error" in data and "status" not in data:
                    raise RuntimeError(f"{self.name} error: {data['error']}")
                return data
            except Exception:
                if attempt == 0:
                    self._start()
                else:
                    raise

    def close(self):
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


_WORKER_STATE = {}


def _worker_cleanup():
    for key in ("java", "ocaml"):
        client = _WORKER_STATE.get(key)
        if client is not None:
            client.close()
    tmpdir = _WORKER_STATE.get("tmpdir")
    if tmpdir and os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)


def _worker_init(lib_on_path, lib_off_path, cp, java_dir, ocaml_bin, contracts_path):
    _WORKER_STATE["lib_on"] = _configure_cpp_lib(lib_on_path)
    _WORKER_STATE["lib_off"] = _configure_cpp_lib(lib_off_path)
    _WORKER_STATE["fuzzer"] = VmContractFuzzer(Path(contracts_path))
    _WORKER_STATE["tmpdir"] = tempfile.mkdtemp(prefix="impulse_fuzz_")

    java_cmd = [
        _resolve_java_cmd(),
        "--add-modules", "jdk.incubator.vector",
        "-cp", cp,
        "org.impulsegraph.vm.JavaVmJsonRunner",
        "--batch",
    ]
    _WORKER_STATE["java"] = PersistentBatchClient(java_cmd, cwd=java_dir, name="JavaBatch")
    _WORKER_STATE["ocaml"] = PersistentBatchClient([ocaml_bin, "--bin-batch"], name="OcamlBatch")
    atexit.register(_worker_cleanup)


def _eval_prog_text(prog_text):
    lib_on = _WORKER_STATE["lib_on"]
    lib_off = _WORKER_STATE["lib_off"]
    java_client = _WORKER_STATE["java"]
    ocaml_client = _WORKER_STATE["ocaml"]
    tmpdir = Path(_WORKER_STATE["tmpdir"])

    temp_impas = tmpdir / "prog.impas"
    impb_file = tmpdir / "prog.impb"
    impb_data_file = tmpdir / "prog.impb_data"

    temp_impas.write_text(prog_text)
    instrs, data_bytes, expectations, opcodes_used = parse_impas_file(temp_impas)
    c_instrs = (Instruction * len(instrs))(*instrs)
    impb_file.write_bytes(bytes(c_instrs))
    impb_data_file.write_bytes(data_bytes)

    c_on = _run_cpp_parsed(lib_on, c_instrs, len(instrs), data_bytes, expectations)
    c_off = _run_cpp_parsed(lib_off, c_instrs, len(instrs), data_bytes, expectations)
    j_state = java_client.request(f"{temp_impas}\t{impb_file}\t{impb_data_file}")
    o_state = ocaml_client.request(f"{impb_file}\t{impb_data_file}")
    diffs = compare_state(c_on, c_off, j_state, o_state, None)
    return diffs, c_on, opcodes_used


def _shrink_program(prog_text):
    """1-minimal delta-debugging minimizer for straight-line .impas divergences."""
    # Skip shrinking control-flow programs with hardcoded PC targets
    if any(op in prog_text for op in ("OP_JMP", "OP_JZ", "OP_JNZ", "OP_LOOP_DECR", "OP_CALL")):
        return prog_text

    all_lines = prog_text.splitlines()
    try:
        text_idx = all_lines.index(".text")
    except ValueError:
        return prog_text

    header = all_lines[:text_idx + 1]
    body = [l for l in all_lines[text_idx + 1:] if l.strip()]
    if len(body) <= 2:
        return prog_text

    # Keep OP_INIT_MOCK_GRAPH (0x00) and trailing OP_HALT
    def _rebuild(kept_body):
        renum = []
        for idx, line in enumerate(kept_body):
            m = re.match(r"^\s*0x[0-9A-Fa-f]+:\s*(.*)$", line)
            rest = m.group(1) if m else line.strip()
            renum.append(f"0x{idx:02X}: {rest}")
        return "\n".join(header + renum) + "\n"

    current = list(body)
    changed = True
    while changed and len(current) > 2:
        changed = False
        # Try removing each instruction from the end backwards (except 0x00 and OP_HALT)
        for i in range(len(current) - 2, 0, -1):
            candidate = current[:i] + current[i + 1:]
            cand_text = _rebuild(candidate)
            try:
                diffs, _, _ = _eval_prog_text(cand_text)
                if diffs:
                    current = candidate
                    changed = True
                    break
            except Exception:
                pass
    return _rebuild(current)


def _run_chunk(task):
    start_iter, count, base_seed, max_instrs, profile, neg_rate, do_shrink, collect_samples = task
    fuzzer = _WORKER_STATE["fuzzer"]

    passed = 0
    divergences = []
    errors = []
    samples = []
    status_counts = Counter()
    profile_counts = Counter()
    profile_divs = Counter()
    opcodes_seen = set()
    pc_sum = 0
    pc_max = 0

    for offset in range(count):
        iter_idx = start_iter + offset
        iter_seed = base_seed + iter_idx
        prog_text, actual_profile = fuzzer.generate_program(
            iter_seed, max_instructions=max_instrs, profile=profile, neg_rate=neg_rate
        )
        profile_counts[actual_profile] += 1
        try:
            diffs, c_on, opcodes_used = _eval_prog_text(prog_text)
            status_counts[c_on["status"]] += 1
            pc_sum += c_on["pc"]
            if c_on["pc"] > pc_max:
                pc_max = c_on["pc"]
            opcodes_seen.update(opcodes_used)

            if diffs:
                profile_divs[actual_profile] += 1
                min_text = _shrink_program(prog_text) if do_shrink else None
                divergences.append((iter_idx + 1, iter_seed, actual_profile, diffs[:4], prog_text, min_text))
            else:
                passed += 1
                if collect_samples:
                    samples.append((iter_idx + 1, iter_seed, actual_profile, c_on["context_hash"][:16], c_on["status"], c_on["pc"]))
        except Exception as e:
            errors.append((iter_idx + 1, iter_seed, actual_profile, str(e)))

    return (
        passed, divergences, errors, samples,
        dict(status_counts), dict(profile_counts), dict(profile_divs),
        list(opcodes_seen), pc_sum, pc_max,
    )


def main():
    default_workers = min(8, os.cpu_count() or 4)
    parser = argparse.ArgumentParser(description="ImpulseVM Multi-Profile Contract-Guided Differential Fuzzer")
    parser.add_argument("--iterations", type=int, default=25, help="Number of fuzzing iterations")
    parser.add_argument("--seed", type=int, default=1337, help="Starting random seed")
    parser.add_argument("--max-instrs", type=int, default=12, help="Max random instructions per program")
    parser.add_argument("--profile", type=str, default="all", choices=("all",) + PROFILES, help="Generation profile (default: all)")
    parser.add_argument("--neg-rate", type=float, default=0.0, help="Negative-type injection probability for positive profiles (default: 0.0; negative profile always uses 0.40)")
    parser.add_argument("--max-errors", type=int, default=0, help="Stop early after this many divergences/errors (0 = unlimited)")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Parallel worker processes (default: {default_workers})")
    parser.add_argument("--chunk-size", type=int, default=200, help="Iterations per worker chunk (default: 200)")
    parser.add_argument("--verbose", action="store_true", help="Print every [CONSENSUS] iteration")
    parser.add_argument("--shrink", action="store_true", help="Perform 1-minimal delta-debugging shrinking on discrepancies")
    parser.add_argument("--coverage-report", action="store_true", help="Print spec opcode coverage summary after run")
    parser.add_argument("--contracts", type=str, default=str(SPEC_DIR / "tools" / "contracts.json"), help="Path to contracts.json")
    parser.add_argument("--dump-dir", type=str, default=str(TOOLS_DIR / "fuzz_findings"), help="Directory to save finding artifacts")
    args = parser.parse_args()

    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    print(f"{BOLD}{BLUE} ImpulseVM Multi-Profile Differential Fuzzer (4-Way ICTX v1)   {RESET}")
    print(f"{BOLD}{BLUE} C++(ON & OFF) vs Java 25 vs OCaml | Profile={args.profile:<14} {RESET}")
    print(f"{BOLD}{BLUE}==============================================================={RESET}\n")

    ext = "dylib" if sys.platform == "darwin" else "so"
    core_cpp = ROOT_DIR / "impulse-graph-core" / "impulse-cpp"
    lib_on_path = os.environ.get("IMPULSE_LIB_PATH_ON", str(core_cpp / "build_on" / f"libimpulse_graph.{ext}"))
    lib_off_path = os.environ.get("IMPULSE_LIB_PATH_OFF", str(core_cpp / "build_off" / f"libimpulse_graph.{ext}"))
    if not os.path.exists(lib_on_path) or not os.path.exists(lib_off_path):
        print(f"{RED}Error: C++ libraries not found ({lib_on_path}, {lib_off_path}).{RESET}")
        sys.exit(1)

    java_dir = Path(os.environ.get("IMPULSE_JAVA_DIR", str(ROOT_DIR / "impulse-graph-java")))
    cp_file = java_dir / "impulse-vm" / "cp.txt"
    if not cp_file.exists():
        subprocess.run(["mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-pl", "impulse-vm"], cwd=java_dir, check=True)
    with open(cp_file, "r") as f:
        maven_cp = f.read().strip()
    cp = f"impulse-vm/target/classes:impulse-vm/target/test-classes:impulse-api/target/classes:impulse-storage/target/classes:impulse-spec/target/classes:{maven_cp}"

    ocaml_bin = str(ROOT_DIR / "impulse-ocaml" / "_build" / "install" / "default" / "bin" / "impulse-ocaml")
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(args.workers, args.iterations))
    chunk_size = max(1, min(args.chunk_size, max(1, args.iterations // workers)))
    verbose = args.verbose or args.iterations <= 50

    chunks = []
    for start in range(0, args.iterations, chunk_size):
        cnt = min(chunk_size, args.iterations - start)
        chunks.append((start, cnt, args.seed, args.max_instrs, args.profile, args.neg_rate, args.shrink, verbose))

    print(f"Running {args.iterations:,} iterations (seed={args.seed}, profile={args.profile}, max_instrs={args.max_instrs}, workers={workers}, shrink={args.shrink})...\n")

    passed_count = 0
    divergence_count = 0
    error_count = 0
    completed = 0
    status_totals = Counter()
    profile_totals = Counter()
    profile_div_totals = Counter()
    all_opcodes_seen = set()
    total_pc_sum = 0
    global_pc_max = 0
    t0 = time.monotonic()

    def _handle_result(res):
        nonlocal passed_count, divergence_count, error_count, completed, total_pc_sum, global_pc_max
        p, divs, errs, samples, st_c, pr_c, pr_d, ops_seen, pc_s, pc_m = res
        passed_count += p
        completed += p + len(divs) + len(errs)
        status_totals.update(st_c)
        profile_totals.update(pr_c)
        profile_div_totals.update(pr_d)
        all_opcodes_seen.update(ops_seen)
        total_pc_sum += pc_s
        if pc_m > global_pc_max:
            global_pc_max = pc_m

        if verbose:
            for it_num, s, prof, h, st, pc in samples:
                print(f"{GREEN}[CONSENSUS]{RESET} Iter {it_num:3d} (Seed {s}, {prof}): Hash={h}... Status={st} PC={pc}")

        for it_num, iter_seed, prof, diffs, prog_text, min_text in divs:
            divergence_count += 1
            print(f"{YELLOW}[DIVERGENCE #{divergence_count}]{RESET} Iteration {it_num} (Seed: {iter_seed}, Profile: {prof}):")
            for d in diffs:
                print(f"       -> {d}")
            repro_path = dump_dir / f"fuzz_repro_seed_{iter_seed}.impas"
            repro_path.write_text(prog_text)
            if min_text and min_text != prog_text:
                min_path = dump_dir / f"fuzz_repro_seed_{iter_seed}_min.impas"
                min_path.write_text(min_text)
                orig_n = sum(1 for l in prog_text.splitlines() if l.startswith("0x"))
                min_n = sum(1 for l in min_text.splitlines() if l.startswith("0x"))
                print(f"       Saved repro to: {repro_path.name} (shrunk {orig_n} -> {min_n} instrs: {min_path.name})")
            else:
                print(f"       Saved repro to: {repro_path.name}")

        for it_num, iter_seed, prof, err_msg in errs:
            error_count += 1
            print(f"{RED}[ERROR]{RESET} Iteration {it_num} (Seed: {iter_seed}, Profile: {prof}) failed: {err_msg}")

        if not verbose and (completed % max(1000, chunk_size) < (p + len(divs) + len(errs)) or completed == args.iterations):
            elapsed = max(1e-6, time.monotonic() - t0)
            rate = completed / elapsed
            rem_s = (args.iterations - completed) / rate if rate > 0 else 0.0
            print(
                f"{BLUE}[PROGRESS]{RESET} {completed:,}/{args.iterations:,} ({100.0 * completed / args.iterations:.1f}%) "
                f"| {rate:,.1f} iter/s | ETA {rem_s:.1f}s | consensus={passed_count:,} div={divergence_count} err={error_count}",
                flush=True,
            )

    if workers == 1:
        _worker_init(lib_on_path, lib_off_path, cp, str(java_dir), ocaml_bin, args.contracts)
        try:
            for ch in chunks:
                _handle_result(_run_chunk(ch))
                if args.max_errors > 0 and (divergence_count + error_count) >= args.max_errors:
                    print(f"\n{YELLOW}[EARLY STOP]{RESET} Reached --max-errors={args.max_errors} after {completed:,} iterations.")
                    break
        finally:
            _worker_cleanup()
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(lib_on_path, lib_off_path, cp, str(java_dir), ocaml_bin, args.contracts),
        ) as pool:
            for res in pool.imap_unordered(_run_chunk, chunks):
                _handle_result(res)
                if args.max_errors > 0 and (divergence_count + error_count) >= args.max_errors:
                    print(f"\n{YELLOW}[EARLY STOP]{RESET} Reached --max-errors={args.max_errors} after {completed:,} iterations.")
                    pool.terminate()
                    break

    total_elapsed = max(1e-6, time.monotonic() - t0)
    avg_pc = (total_pc_sum / max(1, completed - error_count))
    print(f"\n{BOLD}{BLUE}==============================================================={RESET}")
    print(f"Fuzzing Complete: {passed_count:,} Consensus Matches, {divergence_count} Divergences Captured, {error_count} Errors")
    print(f"Elapsed: {total_elapsed:.2f}s ({completed / total_elapsed:,.1f} iter/s across {workers} workers)")
    print(f"Coverage: {len(all_opcodes_seen)} unique opcodes executed | Avg PC={avg_pc:.1f} (Max={global_pc_max})")
    status_str = ", ".join(f"{k}={v:,} ({100.0 * v / max(1, completed):.1f}%)" for k, v in status_totals.most_common(6))
    print(f"Statuses: {status_str}")
    prof_str = ", ".join(
        f"{p}:{profile_totals[p]:,}(div={profile_div_totals[p]})"
        for p in PROFILES if profile_totals[p] > 0
    )
    print(f"Profiles: {prof_str}")
    if args.coverage_report:
        covered_names = sorted(str(c) for c in all_opcodes_seen)
        missing_names = sorted(k for k in OPCODES if k not in all_opcodes_seen)
        print(f"Opcode Coverage: {len(all_opcodes_seen)}/{len(OPCODES)} ({100.0 * len(all_opcodes_seen) / max(1, len(OPCODES)):.1f}%)")
        print(f"  Covered ({len(covered_names)}): {', '.join(covered_names[:20])}{' ...' if len(covered_names) > 20 else ''}")
        print(f"  Missing ({len(missing_names)}): {', '.join(missing_names[:25])}{' ...' if len(missing_names) > 25 else ''}")
    print(f"Reproducing artifacts saved in: {dump_dir}")
    print(f"{BOLD}{BLUE}==============================================================={RESET}\n")
    if divergence_count > 0 or error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
