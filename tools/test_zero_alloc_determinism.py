#!/usr/bin/env python3
"""
Suite #11: Zero-Allocation & Workspace-Reuse Determinism Gate.

Runs diverse programs (`walk_morsel`, `filter_morsel`, `ml_morsel`, `OP_WALK`,
`OP_DEGREE`, `OP_SPMV`) with `"repeat_runs": 2` on `cpp_on`, `cpp_off`, and `java`
across geometry profiles `P0..P3` and asserts:
  a) `repeat_hash_match == True` on EVERY engine (zero stale arena/register leakage).
  b) `exec_allocs == 0` AND `exec2_allocs == 0` on `cpp_on` and `cpp_off` (strict 0-malloc).
  c) Full 3-way output/hash parity across `java`, `cpp_on`, `cpp_off`.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_snapshot_morsels import (  # noqa: E402
    SnapshotSchema,
    _discover_default_lib,
    _valid_dirs_for_rel,
    compare_results,
    generate_random_gene,
    load_snapshot_schema,
    render_impas,
)
from fuzz_vm_contracts import (  # noqa: E402
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    resolve_spec_dir,
)


def render_walk_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    valid_rels = [r for r in schema.relations.values() if _valid_dirs_for_rel(r)]
    rel = rng.choice(valid_rels)
    dir_sub = rng.choice(_valid_dirs_for_rel(rel))
    is_rev = dir_sub in ("rev", "coo_rev")
    in_dom = rel.tgt_dom if is_rev else rel.src_dom
    n_in = max(1, schema.dom_cards[in_dom])
    seed_cnt = rng.choice([1, 4, 16, 64, min(256, n_in)])
    seeds = sorted({rng.randint(0, n_in - 1) for _ in range(seed_cnt)})
    lines = [
        ".data",
        f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
        f"  WALK.{dir_sub} F1, F0, rel={rel.rel_id}",
        "  SET_CARDINALITY R0, F1",
        "  COLLECT.scalar R0, out=0",
        "  HALT",
    ]
    return "op_walk", "\n".join(lines) + "\n"


def render_degree_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    valid_rels = [r for r in schema.relations.values() if r.has_csr or r.has_csc]
    rel = rng.choice(valid_rels)
    use_rev = rel.has_csc and (not rel.has_csr or rng.random() < 0.5)
    in_dom = rel.tgt_dom if use_rev else rel.src_dom
    n_in = max(1, schema.dom_cards[in_dom])
    mode = rng.choice(["sum", "vec"])
    if mode == "sum":
        seed_cnt = rng.choice([1, 4, 16, 64, min(256, n_in)])
        seeds = sorted({rng.randint(0, n_in - 1) for _ in range(seed_cnt)})
        sub = "sum_rev_r" if use_rev else "sum_fwd_r"
        lines = [
            ".data",
            f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
            ".code",
            f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
            f"  DEGREE.{sub} R0, F0, rel={rel.rel_id}",
            "  COLLECT.scalar R0, out=0",
            "  HALT",
        ]
    else:
        sub = "vec_rev_v" if use_rev else "vec_fwd_v"
        lines = [
            ".data",
            ".code",
            f"  DEGREE.{sub} V0, rel={rel.rel_id}",
            "  VEC_REDUCE.sum R0, V0",
            "  COLLECT.scalar R0, out=0",
            "  HALT",
        ]
    return "op_degree", "\n".join(lines) + "\n"


def render_spmv_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    valid_rels = [r for r in schema.relations.values() if r.has_csr or r.has_csc]
    rel = rng.choice(valid_rels)
    use_rev = rel.has_csc and (not rel.has_csr or rng.random() < 0.5)
    in_dom = rel.tgt_dom if use_rev else rel.src_dom
    n_in = max(1, schema.dom_cards[in_dom])
    seed_cnt = rng.choice([1, 4, 16, 64, min(128, n_in)])
    seeds = sorted({rng.randint(0, n_in - 1) for _ in range(seed_cnt)})

    float_attrs = [a for a in rel.edge_attrs if a.is_float]
    attr_spec = f"{rng.choice(float_attrs).attr_id}" if (float_attrs and rng.random() < 0.6) else "0xFFFF"
    spmv_sub = "rev_plus_times" if use_rev else "fwd_plus_times"
    lines = [
        ".data",
        f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
        "  .param 1 type=f64 value=1.5",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
        "  LOAD_PARAM.scalar R1, param=1",
        f"  VEC_INIT.dom_f64 V0, dom={in_dom}",
        "  VEC_MATH_ARITH.add.scalar V0, V0, R1",
        f"  SPMV.{spmv_sub} V1, F0, V0, rel={rel.rel_id}, attr={attr_spec}",
        "  VEC_REDUCE.sum R0, V1",
        "  COLLECT.scalar R0, out=0",
        "  HALT",
    ]
    return "op_spmv", "\n".join(lines) + "\n"


def generate_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    choice = rng.choice([
        "morsel", "morsel", "morsel",
        "op_walk", "op_degree", "op_spmv",
    ])
    if choice == "op_walk":
        return render_walk_program(rng, schema)
    if choice == "op_degree":
        return render_degree_program(rng, schema)
    if choice == "op_spmv":
        return render_spmv_program(rng, schema)
    gene = generate_random_gene(rng, schema)
    return gene.kind, render_impas(gene, schema)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite #11: Zero-Allocation & Workspace-Reuse Determinism Gate"
    )
    parser.add_argument("snapshot", type=Path, help="Path to v0.11 .imps snapshot")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    parser.add_argument(
        "--cpp-on-cmd",
        type=str,
        default="/Users/jesse/impulse/impulse-graph-core/impulse-cpp/build_on/vm_fixture_batch_runner",
    )
    parser.add_argument(
        "--cpp-off-cmd",
        type=str,
        default="/Users/jesse/impulse/impulse-graph-core/impulse-cpp/build_off/vm_fixture_batch_runner",
    )
    parser.add_argument("--profiles", type=str, default="P0,P1,P2,P3")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()

    snap_path = args.snapshot.resolve()
    lib_path = args.lib_path or _discover_default_lib()
    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_asm = _import_spec_modules(spec_dir)
    asm = impas_asm.ImpAsm(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    schema = load_snapshot_schema(snap_path, lib_path)

    geom_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    clients: dict[str, BatchEngineClient] = {
        "java": BatchEngineClient(discover_default_java_cmd() + ["--snapshot", str(snap_path)], "java"),
        "cpp_on": BatchEngineClient([args.cpp_on_cmd, "--snapshot", str(snap_path)], "cpp_on"),
        "cpp_off": BatchEngineClient([args.cpp_off_cmd, "--snapshot", str(snap_path)], "cpp_off"),
    }

    print(
        f"Suite #11 Zero-Alloc & Workspace-Reuse Gate | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}) | "
        f"engines={','.join(clients.keys())} | geom={','.join(geom_profiles)} | "
        f"iters={args.iterations} | seed={args.seed}"
    )

    rng = random.Random(args.seed)
    kind_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    alloc_violations = 0
    repeat_violations = 0
    parity_violations = 0
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            kind, impas = generate_program(rng, schema)
            prof = geom_profiles[i % len(geom_profiles)]
            prog = asm.assemble(impas)
            fixture: dict[str, Any] = {
                "words_hex": [f"0x{w:016x}" for w in prog.words],
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
                "repeat_runs": 2,
            }

            results: dict[str, dict[str, Any]] = {}
            crashed = False
            for name, cli in clients.items():
                try:
                    results[name] = cli.run_fixture(fixture, prof, 500_000)
                except Exception as exc:
                    errors += 1
                    crashed = True
                    print(f"[CRASH #{errors}] iter={i} prof={prof} kind={kind} engine={name}: {exc}")
                    print(impas)
                    break
            if crashed:
                if errors >= args.max_errors:
                    break
                continue

            kind_counts[kind] += 1
            status_counts[str(results["cpp_on"].get("status", "UNKNOWN"))] += 1

            iter_failed = False

            # (a) repeat_hash_match == True on EVERY engine
            for name, res in results.items():
                if res.get("repeat_hash_match") is not True:
                    repeat_violations += 1
                    iter_failed = True
                    print(
                        f"[REPEAT MISMATCH] iter={i} prof={prof} kind={kind} engine={name}: "
                        f"repeat_hash_match={res.get('repeat_hash_match')!r}"
                    )

            # (b) exec_allocs == 0 AND exec2_allocs == 0 on cpp_on and cpp_off
            for name in ("cpp_on", "cpp_off"):
                res = results[name]
                a1 = res.get("exec_allocs")
                a2 = res.get("exec2_allocs")
                if a1 != 0 or a2 != 0:
                    alloc_violations += 1
                    iter_failed = True
                    print(
                        f"[NON-ZERO ALLOC] iter={i} prof={prof} kind={kind} engine={name}: "
                        f"exec_allocs={a1!r}, exec2_allocs={a2!r}"
                    )

            # (c) Full 3-way output/hash parity
            diff = compare_results(results)
            if diff is not None:
                parity_violations += 1
                iter_failed = True
                print(f"[3-WAY DIFF] iter={i} prof={prof} kind={kind}: {diff}")

            if iter_failed:
                errors += 1
                print(impas)
                if errors >= args.max_errors:
                    break
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    done = sum(kind_counts.values())
    print(
        f"\nSuite #11 Completed {done}/{args.iterations} iters in {dt:.2f}s "
        f"({done / dt:.1f} iters/s) | repeat_violations={repeat_violations} | "
        f"alloc_violations={alloc_violations} | parity_violations={parity_violations} | errors={errors}"
    )
    print(f"Kinds: {dict(kind_counts)}")
    print(f"Statuses: {dict(status_counts)}")
    if errors > 0:
        return 1
    print("PASS: 0-malloc execution, workspace-reuse determinism, and 3-way parity verified!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
