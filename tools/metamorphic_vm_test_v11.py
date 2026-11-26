#!/usr/bin/env python3
"""
metamorphic_vm_test_v11.py — ImpulseVM v0.11.0 Property-Based & Metamorphic 4-Way Verifier

Ported from v0.10 metamorphic_vm_test.py onto the frozen v0.11 ISA (isa_v0_11.yaml).
Generates randomized pairs of semantically equivalent v0.11 .impas programs (P1, P2)
over randomized small & multi-word (>64-node) graphs and nullable float vectors.

For each pair, verifies BOTH:
  1. Cross-engine consensus on P1 and P2 independently (OCaml == Java 25 == C++ ON == C++ OFF)
  2. Metamorphic equivalence: canonical output (COLLECT_0 / F0 / R0) of P1 == P2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
ROOT_DIR = REPO_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_vm_contracts_v11 import (
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    discover_default_ocaml_bin,
    resolve_spec_dir,
)

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

LAWS = (
    "set_union_comm",
    "set_intersect_comm",
    "set_union_assoc",
    "set_de_morgan",
    "set_diff_self",
    "set_distributive",
    "set_absorption",
    "walk_fixpoint_star_vs_plus",
    "morsel_fwd_vs_walk",
    "morsel_rev_vs_walk",
    "lane_vs_single_walk",
    "vec_add_comm",
    "vec_mul_comm",
    "vec_scale_identity",
    "coalesce_self_idempotent",
    "spmv_zero_vec_zero",
    "kleene_and_or_idempotent",
    "l1_norm_neg_symmetry",
)


def _make_csr(rng: random.Random) -> tuple[int, list[int], list[int]]:
    n = rng.choice([68, 72]) if rng.random() < 0.25 else rng.randint(4, 10)
    offsets = [0]
    targets: list[int] = []
    for u in range(n):
        if n > 64 and u in (0, 1, 63, 64, 65, n - 2):
            nbrs = sorted({(u + 1) % n, (u + 63) % n, (u + 64) % n})
        elif rng.random() < 0.65:
            k = rng.randint(1, min(3, n))
            nbrs = sorted({rng.randint(0, n - 1) for _ in range(k)})
        else:
            nbrs = []
        targets.extend(nbrs)
        offsets.append(len(targets))
    return n, offsets, targets


def _rand_set(rng: random.Random, n: int) -> list[int]:
    if rng.random() < 0.12:
        return []
    k = rng.randint(1, min(n, 10))
    return sorted(rng.sample(range(n), k))


def _make_floats(rng: random.Random, n: int) -> list[str]:
    pool = ["0.0", "1.0", "-1.0", "2.5", "-3.5", "4.0", "0.5", "10.0", "-0.25"]
    return [rng.choice(pool) for _ in range(n)]


def _wrap(title: str, n: int, offsets: list[int], targets: list[int],
          extra_data: list[str], code: list[str]) -> str:
    w = "u16" if n < 65536 else "u32"
    lines = [
        f"; METAMORPHIC v0.11: {title}",
        ".data",
        f'  .domain 0 name="N" count={n} width={w}',
        '  .relation 0 name="E" src_dom=0 tgt_dom=0',
        f"  .csr_inline rel=0 offsets={offsets} targets={targets}",
    ]
    lines.extend(f"  {d}" for d in extra_data)
    lines.append(".code")
    lines.extend(f"  {c}" for c in code)
    lines.append("  HALT")
    return "\n".join(lines) + "\n"


def generate_pair(seed: int, law: str | None = None) -> tuple[str, str, str, str]:
    """Returns (law_name, prog1_src, prog2_src, check_kind: 'F'|'V'|'R')."""
    rng = random.Random(seed)
    if law is None:
        law = rng.choice(LAWS)
    n, offsets, targets = _make_csr(rng)
    sA = _rand_set(rng, n)
    sB = _rand_set(rng, n)
    sC = _rand_set(rng, n)
    vlen = min(n, rng.choice([1, 4, 8, n]))
    fA = _make_floats(rng, vlen)
    fB = _make_floats(rng, vlen)
    edge_w = ["1.0"] * len(targets)

    base_sets = [
        f".param 0 type=frontier dom=0 values={sA}",
        f".param 1 type=frontier dom=0 values={sB}",
        f".param 2 type=frontier dom=0 values={sC}",
    ]
    base_vecs = [
        f".param 0 type=f64_vec values=[{', '.join(fA)}]",
        f".param 1 type=f64_vec values=[{', '.join(fB)}]",
    ]

    if law == "set_union_comm":
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.union F0, F1, F2",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.union F0, F2, F1",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "set_intersect_comm":
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.intersect F0, F1, F2",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.intersect F0, F2, F1",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "set_union_assoc":
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "LOAD_PARAM.frontier F3, param=2, dom=0",
            "SET_OP.union F4, F1, F2",
            "SET_OP.union F0, F4, F3",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "LOAD_PARAM.frontier F3, param=2, dom=0",
            "SET_OP.union F4, F2, F3",
            "SET_OP.union F0, F1, F4",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "set_de_morgan":
        # U \ (A U B) == (U \ A) & (U \ B)
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "INIT_DOMAIN_SPAN F5, dom=0",
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.union F3, F1, F2",
            "SET_OP.diff F0, F5, F3",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "INIT_DOMAIN_SPAN F5, dom=0",
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.diff F3, F5, F1",
            "SET_OP.diff F4, F5, F2",
            "SET_OP.intersect F0, F3, F4",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "set_diff_self":
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "SET_OP.diff F0, F1, F1",
            "SET_CARDINALITY R0, F0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_CONST.u32 R0, 0",
        ])
        return law, p1, p2, "R0"

    if law == "set_distributive":
        # A & (B U C) == (A & B) U (A & C)
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "LOAD_PARAM.frontier F3, param=2, dom=0",
            "SET_OP.union F4, F2, F3",
            "SET_OP.intersect F0, F1, F4",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "LOAD_PARAM.frontier F3, param=2, dom=0",
            "SET_OP.intersect F4, F1, F2",
            "SET_OP.intersect F5, F1, F3",
            "SET_OP.union F0, F4, F5",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "set_absorption":
        # A U (A & B) == A
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "LOAD_PARAM.frontier F2, param=1, dom=0",
            "SET_OP.intersect F3, F1, F2",
            "SET_OP.union F0, F1, F3",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F0, param=0, dom=0",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "walk_fixpoint_star_vs_plus":
        # WALK.fwd_fixpoint_star(F1) == F1 U WALK.fwd_fixpoint_plus(F1)
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "WALK.fwd_fixpoint_star F0, F1, rel=0",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "WALK.fwd_fixpoint_plus F2, F1, rel=0",
            "SET_OP.union F0, F1, F2",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "morsel_fwd_vs_walk":
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "WALK.fwd F0, F1, rel=0",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "WALK_MORSEL.fwd F1, rel=0",
            "  MORSEL_COMMIT.yield_frontier F0, M1, dom=0",
            "MORSEL_END",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "morsel_rev_vs_walk":
        p1 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "WALK.rev F0, F1, rel=0",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_sets, [
            "LOAD_PARAM.frontier F1, param=0, dom=0",
            "WALK_MORSEL.rev F1, rel=0",
            "  MORSEL_COMMIT.yield_frontier F0, M0, dom=0",
            "MORSEL_END",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "lane_vs_single_walk":
        seed_node = rng.randint(0, n - 1)
        data = [f".param 0 type=u64_vec values=[{seed_node}]"]
        p1 = _wrap(law, n, offsets, targets, data, [
            f"FSEED.single_imm F1, node={seed_node}, dom=0",
            "WALK.fwd F0, F1, rel=0",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, data, [
            "LOAD_PARAM.vec V1, param=0, type=u64",
            "LANE_INIT_SEEDS.auto_width F1, V1, dom=0",
            "LANE_WALK.fwd F2, F1, rel=0",
            "LANE_FILTER_QUORUM.at_least F0, F2, k=1",
            "SET_CONVERT.f2v_ids V0, F0",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "vec_add_comm":
        p1 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_PARAM.vec V2, param=1, type=f64",
            "VEC_MATH_ARITH.add V0, V1, V2",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_PARAM.vec V2, param=1, type=f64",
            "VEC_MATH_ARITH.add V0, V2, V1",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "vec_mul_comm":
        p1 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_PARAM.vec V2, param=1, type=f64",
            "VEC_MATH_ARITH.mul V0, V1, V2",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_PARAM.vec V2, param=1, type=f64",
            "VEC_MATH_ARITH.mul V0, V2, V1",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "vec_scale_identity":
        p1 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_CONST.f64.ext R1, 1.0",
            "VEC_MATH_ARITH.mul.scalar V0, V1, R1",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V0, param=0, type=f64",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "coalesce_self_idempotent":
        p1 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "COALESCE V0, V1, V1",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V0, param=0, type=f64",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "spmv_zero_vec_zero":
        zeros = ["0.0"] * n
        data = [
            f'.edge_attr rel=0 attr=0 name="w" type=f64 values=[{", ".join(edge_w)}]',
            f".param 0 type=f64_vec values=[{', '.join(zeros)}]",
        ]
        p1 = _wrap(law, n, offsets, targets, data, [
            "INIT_DOMAIN_SPAN F1, dom=0",
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "SPMV.fwd_plus_times V0, F1, V1, rel=0, attr=0",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, data, [
            "LOAD_PARAM.vec V0, param=0, type=f64",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "kleene_and_or_idempotent":
        p1 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_CONST.f64.ext R1, 0.0",
            "VEC_CMP.gt.scalar V2, V1, R1",
            "VEC_3VL_OP.and V3, V2, V2",
            "VEC_3VL_OP.or V0, V3, V2",
            "COLLECT.vec V0, out=0",
        ])
        p2 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_CONST.f64.ext R1, 0.0",
            "VEC_CMP.gt.scalar V0, V1, R1",
            "COLLECT.vec V0, out=0",
        ])
        return law, p1, p2, "COLLECT_0"

    if law == "l1_norm_neg_symmetry":
        p1 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "VEC_REDUCE.l1_norm R0, V1",
        ])
        p2 = _wrap(law, n, offsets, targets, base_vecs, [
            "LOAD_PARAM.vec V1, param=0, type=f64",
            "LOAD_CONST.f64.ext R1, -1.0",
            "VEC_MATH_ARITH.mul.scalar V2, V1, R1",
            "VEC_REDUCE.l1_norm R0, V2",
        ])
        return law, p1, p2, "R0"

    raise ValueError(f"Unknown law: {law}")


def _extract_observable(resp: dict[str, Any], check_kind: str) -> Any:
    if check_kind == "COLLECT_0":
        cols = resp.get("collect") or resp.get("outputs") or {}
        if isinstance(cols, dict):
            return cols.get("0") or cols.get(0)
        if isinstance(cols, list) and cols:
            return cols[0]
        return None
    if check_kind == "R0":
        regs = resp.get("r") or resp.get("r_regs") or {}
        if isinstance(regs, dict):
            return regs.get("0") or regs.get(0)
        if isinstance(regs, list) and regs:
            return regs[0]
        return None
    return resp.get("logical_hash")


def main() -> int:
    parser = argparse.ArgumentParser(description="ImpulseVM v0.11.0 Metamorphic 4-Way Property Gate")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--engines", type=str, default="ocaml,java",
                        help="Comma-separated engines: ocaml,java,cpp_on,cpp_off")
    parser.add_argument("--cpp-on-bin", type=Path, default=None)
    parser.add_argument("--cpp-off-bin", type=Path, default=None)
    args = parser.parse_args()

    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_v11 = _import_spec_modules(spec_dir)

    asm = impas_v11.ImpAsmV11()
    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    clients: dict[str, BatchEngineClient] = {}
    for name in engine_names:
        if name == "ocaml":
            ocaml_bin = discover_default_ocaml_bin()
            clients[name] = BatchEngineClient([str(ocaml_bin), "--v11-fixture-batch"], "ocaml")
        elif name == "java":
            clients[name] = BatchEngineClient(discover_default_java_cmd(), "java")
        elif name == "cpp_on":
            b = args.cpp_on_bin or (ROOT_DIR / ".worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_on/v11_fixture_batch_runner")
            clients[name] = BatchEngineClient([str(b)], "cpp_on")
        elif name == "cpp_off":
            b = args.cpp_off_bin or (ROOT_DIR / ".worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_off/v11_fixture_batch_runner")
            clients[name] = BatchEngineClient([str(b)], "cpp_off")
        else:
            raise ValueError(f"Unknown engine: {name}")

    def _to_payload(src: str) -> dict[str, Any]:
        prog = asm.assemble(src)
        return {
            "data_directives": prog.data_directives,
            "const_pool": prog.const_pool,
            "words_hex": [f"0x{w:016X}" for w in prog.words],
        }

    t0 = time.monotonic()
    counts: Counter[str] = Counter()
    failures = 0
    try:
        for i in range(args.iterations):
            law = LAWS[i % len(LAWS)]
            law_name, p1_src, p2_src, check_kind = generate_pair(args.seed + i, law)
            asm1 = _to_payload(p1_src)
            asm2 = _to_payload(p2_src)

            res1: dict[str, dict[str, Any]] = {}
            res2: dict[str, dict[str, Any]] = {}
            for ename, cli in clients.items():
                res1[ename] = cli.run_fixture(asm1, profile="P0")
                res2[ename] = cli.run_fixture(asm2, profile="P0")

            # 1. Cross-engine consensus on P1 and P2 independently
            ref = engine_names[0]
            for ename in engine_names[1:]:
                if (res1[ename].get("status"), res1[ename].get("logical_hash")) != (
                    res1[ref].get("status"), res1[ref].get("logical_hash")
                ):
                    print(f"{RED}CONSENSUS FAIL{RESET} iter={i} law={law_name} P1: {ref}={res1[ref]} vs {ename}={res1[ename]}")
                    failures += 1
                    break
                if (res2[ename].get("status"), res2[ename].get("logical_hash")) != (
                    res2[ref].get("status"), res2[ref].get("logical_hash")
                ):
                    print(f"{RED}CONSENSUS FAIL{RESET} iter={i} law={law_name} P2: {ref}={res2[ref]} vs {ename}={res2[ename]}")
                    failures += 1
                    break
            else:
                # 2. Metamorphic equivalence P1 == P2 on reference engine
                if res1[ref].get("status") != "IMPULSE_VM_OK" or res2[ref].get("status") != "IMPULSE_VM_OK":
                    print(f"{RED}STATUS FAIL{RESET} iter={i} law={law_name}: P1={res1[ref].get('status')} P2={res2[ref].get('status')}")
                    failures += 1
                else:
                    obs1 = _extract_observable(res1[ref], check_kind)
                    obs2 = _extract_observable(res2[ref], check_kind)
                    if obs1 != obs2:
                        print(f"{RED}LAW FAIL{RESET} iter={i} law={law_name}: P1={obs1} != P2={obs2}")
                        failures += 1
                    else:
                        counts[law_name] += 1
            if failures >= 5:
                break
    finally:
        for cli in clients.values():
            cli.close()

    dt = time.monotonic() - t0
    if failures == 0:
        print(f"{GREEN}{BOLD}PASS{RESET}: {args.iterations}/{args.iterations} v0.11 metamorphic pairs "
              f"across {len(LAWS)} laws x {engine_names} in {dt:.2f}s")
        return 0
    print(f"{RED}{BOLD}FAIL{RESET}: {failures} failures in {dt:.2f}s")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
