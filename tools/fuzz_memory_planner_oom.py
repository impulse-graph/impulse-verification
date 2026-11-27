#!/usr/bin/env python3
"""
Suite #6: Workspace Memory-Planner Tight-Bound & OOM Fault-Injection Fuzzer.

Generates diverse `walk_morsel`, `filter_morsel`, `ml_morsel`, and `OP_WALK` programs
against a real v0.11 `.imps` snapshot and verifies:
  1. `planned_scratch_bytes > 0` on both `cpp_on` and `java` and baseline parity.
  2. Tight bound:
     - Flags any `planned_scratch_bytes < arena_peak_bytes` as a planner under-estimate bug.
     - Runs `cpp_on` with exact `scratch_bytes = planned_scratch_bytes` and with
       `scratch_bytes = max(planned_scratch_bytes, arena_peak_bytes + 4096)`, asserting
       identical status, outputs, and ICTX v2 logical_hash.
  3. OOM fault injection:
     - Runs `cpp_on` with `scratch_bytes = 0`, `scratch_bytes = 1024`, and
       `scratch_bytes = max(0, arena_peak_bytes - 64)` (when `arena_peak_bytes > 4096`),
       asserting clean `IMPULSE_VM_ERR_BUFFER_OVERFLOW` and zero process crashes.
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

    use_mask = rng.random() < 0.35 and dir_sub in ("fwd", "rev")
    lines = [
        ".data",
        f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
    ]
    if use_mask:
        lines.append(f"  WALK.{dir_sub}.mask F1, F0, F0, rel={rel.rel_id}")
    else:
        lines.append(f"  WALK.{dir_sub} F1, F0, rel={rel.rel_id}")
    lines.extend([
        "  SET_CARDINALITY R0, F1",
        "  COLLECT.scalar R0, out=0",
        "  HALT",
    ])
    return "op_walk", "\n".join(lines) + "\n"


def generate_program(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    if rng.random() < 0.25:
        return render_walk_program(rng, schema)
    gene = generate_random_gene(rng, schema)
    return gene.kind, render_impas(gene, schema)


def assemble_fixture(asm: Any, impas: str) -> dict[str, Any]:
    prog = asm.assemble(impas)
    return {
        "words_hex": [f"0x{w:016x}" for w in prog.words],
        "data_directives": prog.data_directives,
        "const_pool": prog.const_pool,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite #6: Workspace Memory-Planner Tight-Bound & OOM Fault-Injection Fuzzer"
    )
    parser.add_argument("snapshot", type=Path, help="Path to v0.11 .imps snapshot")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    parser.add_argument(
        "--cpp-on-cmd",
        type=str,
        default="/Users/jesse/impulse/.worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_on/vm_fixture_batch_runner",
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
    asm = impas_asm.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    schema = load_snapshot_schema(snap_path, lib_path)

    geom_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    cpp_cli = BatchEngineClient([args.cpp_on_cmd, "--snapshot", str(snap_path)], "cpp_on")
    java_cli = BatchEngineClient(discover_default_java_cmd() + ["--snapshot", str(snap_path)], "java")

    print(
        f"Suite #6 Memory-Planner & OOM Fuzzer | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}) | "
        f"geom={','.join(geom_profiles)} | iters={args.iterations} | seed={args.seed}"
    )

    rng = random.Random(args.seed)
    kind_counts: Counter[str] = Counter()
    oom_checks = 0
    tight_checks = 0
    underestimates = 0
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            kind, impas = generate_program(rng, schema)
            prof = geom_profiles[i % len(geom_profiles)]
            base_fixture = assemble_fixture(asm, impas)
            kind_counts[kind] += 1

            # 1. Baseline 64MB run on cpp_on + java
            try:
                res_cpp = cpp_cli.run_fixture(base_fixture, prof, 500_000)
                res_java = java_cli.run_fixture(base_fixture, prof, 500_000)
            except Exception as exc:
                errors += 1
                print(f"[CRASH #{errors}] iter={i} prof={prof} kind={kind} baseline: {exc}")
                print(impas)
                if errors >= args.max_errors:
                    break
                continue

            diff = compare_results({"cpp_on": res_cpp, "java": res_java})
            if diff is not None:
                errors += 1
                print(f"[BASELINE DIFF #{errors}] iter={i} prof={prof} kind={kind}: {diff}")
                print(impas)
                if errors >= args.max_errors:
                    break
                continue

            planned_cpp = int(res_cpp.get("planned_scratch_bytes", 0))
            planned_java = int(res_java.get("planned_scratch_bytes", 0))
            peak_cpp = int(res_cpp.get("arena_peak_bytes", 0))

            if planned_cpp <= 0:
                errors += 1
                print(f"[PLANNER ZERO #{errors}] iter={i} prof={prof} kind={kind}: cpp planned_scratch_bytes={planned_cpp}")
                if errors >= args.max_errors:
                    break
                continue
            if planned_java <= 0:
                errors += 1
                print(f"[PLANNER ZERO #{errors}] iter={i} prof={prof} kind={kind}: java planned_scratch_bytes={planned_java}")
                if errors >= args.max_errors:
                    break
                continue

            # 2. Check planner under-estimate
            if planned_cpp < peak_cpp:
                underestimates += 1
                errors += 1
                print(
                    f"[PLANNER UNDERESTIMATE #{errors}] iter={i} prof={prof} kind={kind}: "
                    f"planned_scratch_bytes={planned_cpp} < arena_peak_bytes={peak_cpp} "
                    f"(deficit={peak_cpp - planned_cpp} B)"
                )
                print(impas)
                if errors >= args.max_errors:
                    break

            # 3. Tight-bound runs on cpp_on
            # 3a) Exact planned_scratch_bytes
            tight_exact = dict(base_fixture, scratch_bytes=planned_cpp)
            # 3b) Safe tight bound max(planned_scratch_bytes, arena_peak_bytes + 4096)
            tight_safe_bytes = max(planned_cpp, peak_cpp + 4096)
            tight_safe = dict(base_fixture, scratch_bytes=tight_safe_bytes)

            for label, payload, req_bytes in [
                ("exact_planned", tight_exact, planned_cpp),
                ("tight_padded", tight_safe, tight_safe_bytes),
            ]:
                tight_checks += 1
                try:
                    res_tight = cpp_cli.run_fixture(payload, prof, 500_000)
                except Exception as exc:
                    errors += 1
                    print(f"[TIGHT CRASH #{errors}] iter={i} prof={prof} kind={kind} ({label}={req_bytes}): {exc}")
                    break
                tdiff = compare_results({"baseline": res_cpp, label: res_tight})
                if tdiff is not None:
                    errors += 1
                    print(
                        f"[TIGHT FAIL #{errors}] iter={i} prof={prof} kind={kind} "
                        f"({label}={req_bytes}, planned={planned_cpp}, peak={peak_cpp}): {tdiff}"
                    )
                    if errors >= args.max_errors:
                        break
            if errors >= args.max_errors:
                break

            # 4. OOM fault injection on cpp_on
            oom_budgets: list[tuple[str, int, bool]] = [
                ("zero", 0, True),
                ("1024B", 1024, peak_cpp > 1024),
            ]
            if peak_cpp > 4096:
                oom_budgets.append(("peak_minus_64", max(0, peak_cpp - 64), True))

            for label, budget, must_oom in oom_budgets:
                oom_checks += 1
                oom_payload = dict(base_fixture, scratch_bytes=budget)
                try:
                    res_oom = cpp_cli.run_fixture(oom_payload, prof, 500_000)
                except Exception as exc:
                    errors += 1
                    print(
                        f"[OOM CRASH #{errors}] iter={i} prof={prof} kind={kind} "
                        f"({label}={budget}, peak={peak_cpp}): {exc}"
                    )
                    break
                st = str(res_oom.get("status", ""))
                if must_oom:
                    if st != "IMPULSE_VM_ERR_BUFFER_OVERFLOW":
                        errors += 1
                        print(
                            f"[OOM STATUS #{errors}] iter={i} prof={prof} kind={kind} "
                            f"({label}={budget}, peak={peak_cpp}): expected IMPULSE_VM_ERR_BUFFER_OVERFLOW, got {st!r}"
                        )
                        if errors >= args.max_errors:
                            break
                else:
                    # Program peak <= 1024: either clean BUFFER_OVERFLOW or exact baseline match is acceptable.
                    if st != "IMPULSE_VM_ERR_BUFFER_OVERFLOW":
                        odiff = compare_results({"baseline": res_cpp, label: res_oom})
                        if odiff is not None:
                            errors += 1
                            print(
                                f"[OOM CORRUPT #{errors}] iter={i} prof={prof} kind={kind} "
                                f"({label}={budget}, peak={peak_cpp}): {odiff}"
                            )
                            if errors >= args.max_errors:
                                break
            if errors >= args.max_errors:
                break
    finally:
        cpp_cli.close()
        java_cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    done = sum(kind_counts.values())
    print(
        f"\nSuite #6 Completed {done}/{args.iterations} iters in {dt:.2f}s "
        f"({done / dt:.1f} iters/s) | tight_checks={tight_checks} | oom_checks={oom_checks} | "
        f"underestimates={underestimates} | errors={errors}"
    )
    print(f"Kinds: {dict(kind_counts)}")
    if errors > 0:
        return 1
    print("PASS: Memory planner tight-bound & OOM fault-injection checks passed with 0 errors!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
