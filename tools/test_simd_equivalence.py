#!/usr/bin/env python3
"""
Suite #12: Differential SIMD / Architecture Equivalence.

Spawns up to 6 engine execution modes against a real v0.11 `.imps` snapshot:
  - `cpp_simd`    (`build_on/vm_fixture_batch_runner --snapshot <snap>`)
  - `cpp_scalar`  (`build_on/vm_fixture_batch_runner --snapshot <snap> --scalar-simd`)
  - `cpp_off`     (`build_off/vm_fixture_batch_runner --snapshot <snap>`)
  - `java_interp` (`VmFixtureBatchMain --snapshot <snap>`)
  - `java_jit`    (`VmFixtureBatchMain --snapshot <snap> --jit`)
  - `java_c1`     (`java -XX:TieredStopAtLevel=1 ... VmFixtureBatchMain --snapshot <snap>`)

Runs SIMD-heavy programs (`walk_morsel` with `GATHER_EDGE_ATTR` + `VEC_CMP` + `scatter_*`,
`filter_morsel` with `GATHER_NODE_ATTR` + `VEC_CMP`, `ml_morsel`, plus `VEC_MATH_ARITH`
+ `VEC_REDUCE`) across `P0..P3` and asserts 100% identical results & ICTX v2 hashes
across all 6 modes.
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
    MorselGene,
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


def render_vec_arith_reduce_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    # Prefer f64 node attributes (base type_code == 7) so VEC_MATH_ARITH + f64 scalar always type-match
    doms_with_f64 = [
        d for d, attrs in schema.node_attrs.items()
        if any((a.type_code & 0x7F) == 7 for a in attrs) and schema.dom_cards.get(d, 0) > 0
    ]
    if doms_with_f64:
        dom_id = rng.choice(doms_with_f64)
        fattrs = [a for a in schema.node_attrs[dom_id] if (a.type_code & 0x7F) == 7]
        a1 = rng.choice(fattrs)
        a2 = rng.choice(fattrs)
    else:
        dom_id = next(iter(schema.dom_cards.keys()))
        a1 = a2 = None

    n_dom = max(1, schema.dom_cards[dom_id])
    seed_cnt = rng.choice([1, 8, 32, 128, min(512, n_dom)])
    seeds = sorted({rng.randint(0, n_dom - 1) for _ in range(seed_cnt)})
    scalar_val = rng.choice([0.5, 1.0, -2.5, 3.25, 10.0, -0.0])

    bin_op = rng.choice(["add", "sub", "mul", "min", "max"])
    unary_op = rng.choice(["abs", "neg", "floor", "ceil", "round", "trunc"])
    red_op = rng.choice(["sum", "min", "max", "l1_norm", "l2_norm_sq", "count_valid", "argmin", "argmax"])

    lines = [
        ".data",
        f"  .param 0 type=frontier dom={dom_id} values=[{', '.join(str(x) for x in seeds)}]",
        f"  .param 1 type=f64 value={scalar_val!r}",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={dom_id}",
        "  LOAD_PARAM.scalar R1, param=1",
    ]
    if a1 is not None and a2 is not None:
        lines.append(f"  GATHER_NODE_ATTR.frontier_compact V0, F0, dom={dom_id}, attr={a1.attr_id}")
        lines.append(f"  GATHER_NODE_ATTR.frontier_compact V1, F0, dom={dom_id}, attr={a2.attr_id}")
    else:
        lines.append(f"  VEC_INIT.dom_f64 V0, dom={dom_id}")
        lines.append(f"  VEC_INIT.dom_f64 V1, dom={dom_id}")

    lines.append(f"  VEC_MATH_ARITH.{bin_op} V2, V0, V1")
    lines.append(f"  VEC_MATH_ARITH.{bin_op}.scalar V2, V2, R1")
    if rng.random() < 0.5:
        lines.append(f"  VEC_MATH_ARITH.{unary_op} V2, V2")
    lines.append(f"  VEC_REDUCE.{red_op} R0, V2")
    lines.append("  COLLECT.scalar R0, out=0")
    lines.append("  HALT")
    return "vec_arith_reduce", "\n".join(lines) + "\n"


def generate_simd_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    roll = rng.random()
    if roll < 0.25:
        return render_vec_arith_reduce_program(rng, schema)

    gene: MorselGene = generate_random_gene(rng, schema)
    # Bias walk_morsel towards edge-attribute gather + VEC_CMP + scatter_* to stress SIMD kernels
    if gene.kind == "walk_morsel":
        rel = schema.relations[gene.rel_id]
        if rel.edge_attrs and gene.attr_id is None:
            ea = rng.choice(rel.edge_attrs)
            gene.attr_id = ea.attr_id
            gene.attr_is_float = ea.is_float
            gene.thresh = rng.choice([0.0, -0.0, 20.0, 40.0]) if ea.is_float else rng.choice([0, 20, 40])
            if ea.is_float and rng.random() < 0.65:
                gene.commit_mode = rng.choice(["scatter_add", "scatter_min", "scatter_max", "scatter_argmin", "scatter_argmax"])
    return gene.kind, render_impas(gene, schema)


def build_mode_commands(
    snap_path: Path, cpp_on_cmd: str, cpp_off_cmd: str
) -> dict[str, list[str]]:
    base_java = discover_default_java_cmd()
    java_c1 = [base_java[0], "-XX:TieredStopAtLevel=1"] + base_java[1:] + ["--snapshot", str(snap_path)]
    return {
        "cpp_simd": [cpp_on_cmd, "--snapshot", str(snap_path)],
        "cpp_scalar": [cpp_on_cmd, "--snapshot", str(snap_path), "--scalar-simd"],
        "cpp_off": [cpp_off_cmd, "--snapshot", str(snap_path)],
        "java_interp": base_java + ["--snapshot", str(snap_path)],
        "java_jit": base_java + ["--snapshot", str(snap_path), "--jit"],
        "java_c1": java_c1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite #12: Differential SIMD / Architecture Equivalence"
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
    parser.add_argument(
        "--modes",
        type=str,
        default="cpp_simd,cpp_scalar,cpp_off,java_interp,java_jit,java_c1",
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
    all_cmds = build_mode_commands(snap_path, args.cpp_on_cmd, args.cpp_off_cmd)
    req_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    clients: dict[str, BatchEngineClient] = {}
    for mode in req_modes:
        if mode not in all_cmds:
            raise ValueError(f"Unsupported mode {mode!r}; valid: {sorted(all_cmds.keys())}")
        clients[mode] = BatchEngineClient(all_cmds[mode], mode)

    print(
        f"Suite #12 Differential SIMD / Arch Equivalence | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}) | "
        f"modes={','.join(clients.keys())} | geom={','.join(geom_profiles)} | "
        f"iters={args.iterations} | seed={args.seed}"
    )

    rng = random.Random(args.seed)
    kind_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            kind, impas = generate_simd_program(rng, schema)
            prof = geom_profiles[i % len(geom_profiles)]
            prog = asm.assemble(impas)
            fixture: dict[str, Any] = {
                "words_hex": [f"0x{w:016x}" for w in prog.words],
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
            }

            results: dict[str, dict[str, Any]] = {}
            crashed = False
            for name, cli in clients.items():
                try:
                    results[name] = cli.run_fixture(fixture, prof, 500_000)
                except Exception as exc:
                    errors += 1
                    crashed = True
                    print(f"[CRASH #{errors}] iter={i} prof={prof} kind={kind} mode={name}: {exc}")
                    print(impas)
                    break
            if crashed:
                if errors >= args.max_errors:
                    break
                continue

            kind_counts[kind] += 1
            first_res = next(iter(results.values()))
            status_counts[str(first_res.get("status", "UNKNOWN"))] += 1

            diff = compare_results(results)
            if diff is not None:
                errors += 1
                print(f"[SIMD DIFF #{errors}] iter={i} prof={prof} kind={kind}: {diff}")
                print(impas)
                if errors >= args.max_errors:
                    break
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    done = sum(kind_counts.values())
    print(
        f"\nSuite #12 Completed {done}/{args.iterations} iters in {dt:.2f}s "
        f"({done / dt:.1f} iters/s) | modes={len(clients)} | errors={errors}"
    )
    print(f"Kinds: {dict(kind_counts)}")
    print(f"Statuses: {dict(status_counts)}")
    if errors > 0:
        return 1
    print("PASS: 100% SIMD / Architecture equivalence across all modes!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
