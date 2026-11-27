#!/usr/bin/env python3
"""
test_topology_contracts.py — Negative Topology Fail-Fast Contract Gate.

Enforces engine_invariants.md Lifecycle 4:
  "VM has zero capability to compute CSC, COO, or CST. Period. Raw execution of a
   reverse/undirected op on a !hasCsc relation fails immediately."

Tests every CSR-only, CSC-only, and COO-only relation in storage_coverage_small.imps
(or any .imps snapshot) across Java and C++, asserting that missing topology ALWAYS
fails fast (never silently returns 0 or synthesizes CSC/CSR).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fuzz_snapshot_morsels import (
    _discover_default_lib,
    load_snapshot_schema,
)


def assemble_fixture(asm, impas: str) -> dict:
    prog = asm.assemble(impas)
    return {
        "words_hex": [f"0x{w:016x}" for w in prog.words],
        "data_directives": prog.data_directives,
        "const_pool": prog.const_pool,
    }
from fuzz_vm_contracts import (
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    resolve_spec_dir,
)


def default_cpp_on_cmd() -> str:
    root = TOOLS_DIR.parent.parent
    for cand in [
        root / ".worktrees" / "v11-m3" / "impulse-graph-core" / "impulse-cpp" / "build_on" / "vm_fixture_batch_runner",
        root / "impulse-graph-core" / "impulse-cpp" / "build_on" / "vm_fixture_batch_runner",
    ]:
        if cand.exists():
            return str(cand)
    return "vm_fixture_batch_runner"

# Acceptable fail-fast status codes for missing topology
MISSING_TOPO_STATUSES = {
    "IMPULSE_VM_ERR_INVALID_OPERAND",
    "IMPULSE_VM_ERR_MISSING_TOPOLOGY",
    "IMPULSE_VM_ERR_UNSUPPORTED_OP",
}


def build_cases(schema) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for rid, rel in sorted(schema.relations.items()):
        # 1. CSR-only relation (!has_csc): reverse ops MUST fail fast
        if rel.has_csr and not rel.has_csc:
            cases.append((
                f"rel={rid}({rel.name}) WALK.rev on !has_csc",
                f".data\n  .param 0 type=frontier dom={rel.tgt_dom} values=[0, 1, 2]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.tgt_dom}\n"
                f"  WALK.rev F1, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) WALK_MORSEL.rev on !has_csc",
                f".data\n  .param 0 type=frontier dom={rel.tgt_dom} values=[0, 1, 2]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.tgt_dom}\n"
                f"  WALK_MORSEL.rev F0, rel={rid}\n"
                f"    MORSEL_COMMIT.count_add_r R0\n"
                f"  MORSEL_END\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) DEGREE.sum_rev_r on !has_csc",
                f".data\n  .param 0 type=frontier dom={rel.tgt_dom} values=[0, 1, 2]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.tgt_dom}\n"
                f"  DEGREE.sum_rev_r R0, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) LANE_WALK.rev on !has_csc",
                f".data\n  .param 0 type=u64_vec values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.vec V0, param=0, type=u64\n"
                f"  LANE_INIT_SEEDS.auto_width F0, V0, dom={rel.tgt_dom}\n"
                f"  LANE_WALK.rev F1, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) SPMV.rev_plus_times on !has_csc",
                f".data\n  .param 0 type=frontier dom={rel.tgt_dom} values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.tgt_dom}\n"
                f"  LOAD_CONST.f32 R1, 1.0\n"
                f"  SPMV.rev_plus_times.scalar V1, F0, R1, rel={rid}, attr=none\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) LANE_STATE_WALK.rev_min_plus on !has_csc",
                f".data\n  .param 0 type=u64_vec values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.vec V0, param=0, type=u64\n"
                f"  LANE_INIT_SEEDS.auto_width F0, V0, dom={rel.tgt_dom}\n"
                f"  LANE_STATE_INIT.from_frontier_seeds V1, F0, dom={rel.tgt_dom}, k=2\n"
                f"  LANE_STATE_WALK.rev_min_plus V2, V1, rel={rid}, attr=none\n"
                f"  HALT\n",
            ))

        # 2. CSC-only or COO-only relation (!has_csr): forward CSR ops MUST fail fast
        if not rel.has_csr and (rel.has_csc or rel.has_coo):
            cases.append((
                f"rel={rid}({rel.name}) WALK.fwd on !has_csr",
                f".data\n  .param 0 type=frontier dom={rel.src_dom} values=[0, 1, 2]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}\n"
                f"  WALK.fwd F1, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) WALK_MORSEL.fwd on !has_csr",
                f".data\n  .param 0 type=frontier dom={rel.src_dom} values=[0, 1, 2]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}\n"
                f"  WALK_MORSEL.fwd F0, rel={rid}\n"
                f"    MORSEL_COMMIT.count_add_r R0\n"
                f"  MORSEL_END\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) DEGREE.sum_fwd_r on !has_csr",
                f".data\n  .param 0 type=frontier dom={rel.src_dom} values=[0, 1, 2]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}\n"
                f"  DEGREE.sum_fwd_r R0, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) LANE_WALK.fwd on !has_csr",
                f".data\n  .param 0 type=u64_vec values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.vec V0, param=0, type=u64\n"
                f"  LANE_INIT_SEEDS.auto_width F0, V0, dom={rel.src_dom}\n"
                f"  LANE_WALK.fwd F1, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) SPMV.fwd_plus_times on !has_csr",
                f".data\n  .param 0 type=frontier dom={rel.src_dom} values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}\n"
                f"  LOAD_CONST.f32 R1, 1.0\n"
                f"  SPMV.fwd_plus_times.scalar V1, F0, R1, rel={rid}, attr=none\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) LANE_STATE_WALK.fwd_min_plus on !has_csr",
                f".data\n  .param 0 type=u64_vec values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.vec V0, param=0, type=u64\n"
                f"  LANE_INIT_SEEDS.auto_width F0, V0, dom={rel.src_dom}\n"
                f"  LANE_STATE_INIT.from_frontier_seeds V1, F0, dom={rel.src_dom}, k=2\n"
                f"  LANE_STATE_WALK.fwd_min_plus V2, V1, rel={rid}, attr=none\n"
                f"  HALT\n",
            ))

        # 3. Non-COO relation (!has_coo): COO ops MUST fail fast
        if not rel.has_coo and (rel.has_csr or rel.has_csc):
            cases.append((
                f"rel={rid}({rel.name}) WALK.coo_fwd on !has_coo",
                f".data\n  .param 0 type=frontier dom={rel.src_dom} values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}\n"
                f"  WALK.coo_fwd F1, F0, rel={rid}\n"
                f"  HALT\n",
            ))
            cases.append((
                f"rel={rid}({rel.name}) WALK_MORSEL.coo_fwd on !has_coo",
                f".data\n  .param 0 type=frontier dom={rel.src_dom} values=[0, 1]\n.code\n"
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}\n"
                f"  WALK_MORSEL.coo_fwd F0, rel={rid}\n"
                f"    MORSEL_COMMIT.count_add_r R0\n"
                f"  MORSEL_END\n"
                f"  HALT\n",
            ))

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Negative missing-topology fail-fast contract gate")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    parser.add_argument("--cpp-on-cmd", type=str, default=default_cpp_on_cmd())
    args = parser.parse_args()

    snap_path = args.snapshot.resolve()
    lib_path = args.lib_path or _discover_default_lib()
    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_asm = _import_spec_modules(spec_dir)
    asm = impas_asm.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    schema = load_snapshot_schema(snap_path, lib_path)

    cases = build_cases(schema)
    clients = {
        "cpp_on": BatchEngineClient([args.cpp_on_cmd, "--snapshot", str(snap_path)], "cpp_on"),
        "java": BatchEngineClient(discover_default_java_cmd() + ["--snapshot", str(snap_path)], "java"),
    }

    failures: list[str] = []
    try:
        for label, impas in cases:
            fixture = assemble_fixture(asm, impas)
            for eng, cli in clients.items():
                res = cli.run_fixture(fixture, "P0", 100_000)
                st = str(res.get("status", ""))
                if st not in MISSING_TOPO_STATUSES:
                    failures.append(f"[{eng}] {label}: expected fail-fast missing-topology error, got {st!r}")
    finally:
        for cli in clients.values():
            cli.close()

    print(f"Negative Topology Contract Gate | snapshot={snap_path.name} | cases={len(cases)} x 2 engines | failures={len(failures)}")
    for f in failures:
        print(f"  FAIL: {f}")
    if failures:
        return 1
    print("PASS: All missing-topology operations failed fast with zero runtime synthesis!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
