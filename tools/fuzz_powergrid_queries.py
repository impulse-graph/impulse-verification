#!/usr/bin/env python3
"""
Phase B Five-Way Differential Query Fuzzer for PowerGrid Spec v0.10.0 Snapshots.

Validates schema-aware generated queries across:
  1. SQLite Ground-Truth Differential Oracle (`powergrid_ground_truth.sqlite`)
  2. OCaml Normative Reference Engine (`main.exe --batch-jsonl`)
  3. C++20 Zero-Copy Kernel — Optimizations ON (`build_on/kleisli_json_runner`)
  4. C++20 Zero-Copy Kernel — Optimizations OFF (`build_off/kleisli_json_runner`)
  5. Java 25 FFM & Vector API Engine (`org.impulsegraph.vm.harness.KleisliJsonQueryRunner`)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
FIXTURE_ROOT = REPO_ROOT / "datasets" / "powergrid_fixture"
FUZZ_RUNS_DIR = FIXTURE_ROOT / "fuzz_runs"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from powergrid_fuzzer import (
    ALL_FAMILIES,
    FixtureSchema,
    QueryGenerator,
    SQLiteOracle,
    compare_engine_outputs,
    save_reproducer,
)


SUPPORTED_ENGINES: Tuple[str, ...] = ("sqlite", "ocaml", "cpp_on", "cpp_off", "java")

FAMILY_ALIASES: Dict[str, str] = {
    "f1": "f1_cel_filters",
    "f2": "f2_cross_domain_encodings",
    "f3": "f3_kleisli_multihop_setops",
    "f3_kleisli_multihop": "f3_kleisli_multihop_setops",
    "f4": "f4_edge_predicates",
    "f5": "f5_reachability_fixpoint",
    "f6": "f6_scalar_aggregations",
    "f7": "f7_multilane_quorum_cover",
    "f8": "f8_vector_rerank_topk",
}


def resolve_families(raw_family: str) -> List[str]:
    if not raw_family or raw_family.strip().lower() == "all":
        return list(ALL_FAMILIES)
    out: List[str] = []
    for part in raw_family.split(","):
        tok = part.strip()
        if not tok:
            continue
        resolved = FAMILY_ALIASES.get(tok, tok)
        if resolved not in ALL_FAMILIES:
            raise ValueError(f"Unknown family '{tok}'. Valid families: {ALL_FAMILIES}")
        if resolved not in out:
            out.append(resolved)
    return out


def _find_java_cmd_and_cp() -> Optional[Tuple[str, str, Path]]:
    java_dir = Path(os.environ.get("IMPULSE_JAVA_DIR", str(WORKSPACE_ROOT / "impulse-graph-java")))
    if not java_dir.is_dir():
        return None

    java_home = os.environ.get("JAVA_HOME", "")
    if java_home and (Path(java_home) / "bin" / "java").is_file():
        java_bin = str(Path(java_home) / "bin" / "java")
    elif Path("/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home/bin/java").is_file():
        java_bin = "/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home/bin/java"
    else:
        found = shutil.which("java")
        if not found:
            return None
        java_bin = found

    modules = [
        "impulse-vm",
        "impulse-compiler",
        "impulse-api",
        "impulse-storage",
        "impulse-spec",
        "impulse-builder",
        "impulse-codegen",
    ]
    runner_rel = Path("org/impulsegraph/vm/harness/KleisliJsonQueryRunner.class")
    runner_found = False
    cp_entries: List[str] = []
    for mod in modules:
        for sub in ("target/classes", "target/test-classes"):
            p = java_dir / mod / sub
            if p.is_dir():
                cp_entries.append(str(p))
                if (p / runner_rel).is_file():
                    runner_found = True

    if not runner_found:
        return None

    for cp_txt in (java_dir / "impulse-vm" / "cp.txt", java_dir / "impulse-compiler" / "cp.txt"):
        if cp_txt.is_file():
            extra = cp_txt.read_text(encoding="utf-8").strip()
            if extra:
                cp_entries.append(extra)

    return java_bin, ":".join(cp_entries), java_dir


def _find_ocaml_bin() -> Optional[Path]:
    candidates = [
        WORKSPACE_ROOT / "impulse-ocaml" / "_build" / "default" / "bin" / "main.exe",
        WORKSPACE_ROOT / "impulse-ocaml" / "_build" / "install" / "default" / "bin" / "impulse-ocaml",
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            # Verify `--batch-jsonl` is supported if auto-detecting
            help_res = subprocess.run([str(cand), "--help"], capture_output=True, text=True)
            combined = (help_res.stdout or "") + (help_res.stderr or "")
            if "batch-jsonl" in combined:
                return cand
    return None


def _find_cpp_runner(build_sub: str) -> Optional[Path]:
    cand = WORKSPACE_ROOT / "impulse-graph-core" / "impulse-cpp" / build_sub / "kleisli_json_runner"
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    return None


def resolve_engines(raw_engines: str, sqlite_only: bool) -> List[str]:
    if sqlite_only:
        return ["sqlite"]
    if raw_engines.strip().lower() == "auto":
        detected = ["sqlite"]
        if _find_ocaml_bin() is not None:
            detected.append("ocaml")
        if _find_cpp_runner("build_on") is not None:
            detected.append("cpp_on")
        if _find_cpp_runner("build_off") is not None:
            detected.append("cpp_off")
        if _find_java_cmd_and_cp() is not None:
            detected.append("java")
        return detected

    requested: List[str] = []
    for part in raw_engines.split(","):
        eng = part.strip().lower()
        if not eng:
            continue
        if eng not in SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported engine '{eng}'. Supported: {SUPPORTED_ENGINES}")
        if eng not in requested:
            requested.append(eng)
    return requested


def run_external_batch(
    engine: str,
    imps_path: Path,
    jsonl_path: Path,
    expected_count: int,
) -> Tuple[List[Dict[str, Any]], float]:
    """Executes an external batch runner (`ocaml`, `cpp_on`, `cpp_off`, `java`) once over `jsonl_path`."""
    cwd: Optional[Path] = None
    if engine == "ocaml":
        cands = [
            WORKSPACE_ROOT / "impulse-ocaml" / "_build" / "default" / "bin" / "main.exe",
            WORKSPACE_ROOT / "impulse-ocaml" / "_build" / "install" / "default" / "bin" / "impulse-ocaml",
        ]
        bin_path = next((c for c in cands if c.is_file() and os.access(c, os.X_OK)), None)
        if bin_path is None:
            raise FileNotFoundError("OCaml binary not found at impulse-ocaml/_build/default/bin/main.exe")
        cmd = [str(bin_path), "--batch-jsonl", str(jsonl_path), str(imps_path)]

    elif engine in ("cpp_on", "cpp_off"):
        sub = "build_on" if engine == "cpp_on" else "build_off"
        bin_path = _find_cpp_runner(sub)
        if bin_path is None:
            raise FileNotFoundError(
                f"C++ runner not found at impulse-graph-core/impulse-cpp/{sub}/kleisli_json_runner"
            )
        cmd = [str(bin_path), str(imps_path), str(jsonl_path)]

    elif engine == "java":
        info = _find_java_cmd_and_cp()
        if info is None:
            raise FileNotFoundError(
                "Java KleisliJsonQueryRunner class not found in impulse-graph-java target classes"
            )
        java_bin, cp, java_dir = info
        cwd = java_dir
        cmd = [
            java_bin,
            "--add-modules",
            "jdk.incubator.vector",
            "-cp",
            cp,
            "org.impulsegraph.vm.harness.KleisliJsonQueryRunner",
            str(imps_path),
            str(jsonl_path),
        ]
    else:
        raise ValueError(f"Unknown external engine '{engine}'")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Engine '{engine}' exited with code {proc.returncode}.\n"
            f"CMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr[:2000]}\nSTDOUT:\n{proc.stdout[:1000]}"
        )

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # Filter out any JVM incubator vector warning lines printed to stdout if any, keeping JSON object lines
    json_lines = [ln for ln in lines if ln.startswith("{") and ln.endswith("}")]
    if len(json_lines) != expected_count:
        raise RuntimeError(
            f"Engine '{engine}' emitted {len(json_lines)} JSON lines, expected {expected_count}.\n"
            f"STDERR:\n{proc.stderr[:1000]}\nFIRST LINES:\n" + "\n".join(lines[:10])
        )

    parsed = [json.loads(ln) for ln in json_lines]
    return parsed, elapsed


def write_jsonl(path: Path, queries: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            f.write(json.dumps(q, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                queries.append(json.loads(s))
    return queries


def run_case(
    case_name: str,
    engines: Sequence[str],
    count: int,
    seed: int,
    families: Sequence[str],
    *,
    input_jsonl: Optional[Path] = None,
    emit_jsonl: Optional[Path] = None,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-5,
) -> Tuple[int, int]:
    """Runs fuzz generation + multi-engine evaluation for one PowerGrid case. Returns (passed, failed)."""
    case_dir = FIXTURE_ROOT / case_name
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {case_dir}")

    FUZZ_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    schema = FixtureSchema(case_dir)
    oracle = SQLiteOracle(schema)

    try:
        if input_jsonl is not None:
            queries = read_jsonl(input_jsonl.resolve())
            batch_path = input_jsonl.resolve()
        else:
            gen = QueryGenerator(schema, seed=seed, oracle=oracle)
            queries = gen.generate_batch(count=count, families=families)
            batch_path = (
                emit_jsonl.resolve()
                if emit_jsonl is not None
                else (FUZZ_RUNS_DIR / f"batch_{case_name}_seed{seed}_n{len(queries)}.jsonl")
            )
            write_jsonl(batch_path, queries)

        n_queries = len(queries)
        print(f"\n=========================================================================")
        print(f"  PHASE B DIFFERENTIAL QUERY FUZZER — {case_name}")
        print(f"=========================================================================")
        print(f"  Snapshot (.imps) : {schema.imps_path}")
        print(f"  SQLite Ground DB : {schema.sqlite_path}")
        print(f"  Batch JSONL      : {batch_path} ({n_queries} queries)")
        print(f"  Active Engines   : {', '.join(engines)}")
        print(f"=========================================================================")

        engine_results: Dict[str, List[Dict[str, Any]]] = {}
        engine_times: Dict[str, float] = {}

        if "sqlite" in engines:
            t0 = time.perf_counter()
            engine_results["sqlite"] = [oracle.evaluate(q) for q in queries]
            engine_times["sqlite"] = time.perf_counter() - t0

        for eng in engines:
            if eng == "sqlite":
                continue
            outs, dt = run_external_batch(eng, schema.imps_path, batch_path, n_queries)
            engine_results[eng] = outs
            engine_times[eng] = dt

        passed = 0
        failed = 0
        fam_pass: Counter = Counter()
        fam_fail: Counter = Counter()
        fam_unsup: Counter = Counter()
        non_empty_results = 0

        for idx, q in enumerate(queries):
            fam = str(q.get("family", "unknown"))
            per_eng = {eng: engine_results[eng][idx] for eng in engines}

            ref_out = per_eng.get("sqlite") or next(iter(per_eng.values()))
            rtype = ref_out.get("result_type")
            if rtype == "nodes" and len(ref_out.get("nodes", [])) > 0:
                non_empty_results += 1
            elif rtype == "keys" and len(ref_out.get("keys", [])) > 0:
                non_empty_results += 1
            elif rtype == "count" and int(ref_out.get("count", 0)) > 0:
                non_empty_results += 1
            elif rtype == "scalar" and ref_out.get("scalar") not in (None, 0.0):
                non_empty_results += 1
            elif rtype == "lane_masks" and len(ref_out.get("lanes", [])) > 0:
                non_empty_results += 1
            elif rtype == "top_k" and len(ref_out.get("nodes", [])) > 0:
                non_empty_results += 1

            mismatches, unsup = compare_engine_outputs(per_eng, rel_tol=rel_tol, abs_tol=abs_tol)
            if unsup:
                fam_unsup[fam] += 1

            if mismatches:
                failed += 1
                fam_fail[fam] += 1
                sql_str, _, _, _ = oracle.translate_to_sql(q, use_udf=False)
                repro_dir = save_reproducer(FUZZ_RUNS_DIR, case_name, q, sql_str, per_eng, mismatches)
                print(f"  [FAIL] {q['id']} ({fam}):")
                for m in mismatches[:5]:
                    print(f"         -> {m}")
                print(f"         Reproducer saved: {repro_dir}")
            else:
                passed += 1
                fam_pass[fam] += 1

        print("\n  Engine Execution Timings:")
        for eng in engines:
            dt_ms = engine_times[eng] * 1000.0
            qps = n_queries / max(engine_times[eng], 1e-9)
            print(f"    - {eng:<10} : {dt_ms:8.2f} ms ({qps:8.1f} queries/s)")

        print("\n  Per-Family Summary:")
        seen_fams = [f for f in ALL_FAMILIES if (fam_pass[f] + fam_fail[f]) > 0]
        for f in seen_fams:
            tot_f = fam_pass[f] + fam_fail[f]
            u_note = f" (unsupported={fam_unsup[f]})" if fam_unsup[f] else ""
            print(f"    - {f:<30} : {fam_pass[f]:4d}/{tot_f:<4d} PASS{u_note}")

        print(
            f"\n  Result Summary ({case_name}): {passed}/{n_queries} PASSED, "
            f"{failed} FAILED, non-empty selectivity: {non_empty_results}/{n_queries} "
            f"({100.0 * non_empty_results / max(1, n_queries):.1f}%)"
        )
        return passed, failed
    finally:
        oracle.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B Schema-Aware Differential Query Fuzzer & Multi-Engine Oracle."
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="auto",
        help="Comma-separated engines from sqlite,ocaml,cpp_on,cpp_off,java or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--sqlite-only",
        action="store_true",
        help="Run standalone against the SQLite ground-truth differential oracle only",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="case1354pegase",
        choices=["case1354pegase", "case2869pegase", "both"],
        help="Target PowerGrid fixture case (default: case1354pegase)",
    )
    parser.add_argument("--count", type=int, default=500, help="Number of queries per case (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed (default: 42)")
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        help="Comma-separated list of families (f1..f8 or full names) or 'all' (default: all)",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional pre-generated JSONL query batch to execute instead of generating",
    )
    parser.add_argument(
        "--emit-jsonl",
        type=Path,
        default=None,
        help="Optional output path for the generated JSONL query batch",
    )
    parser.add_argument("--rel-tol", type=float, default=1e-4, help="Float relative tolerance (default: 1e-4)")
    parser.add_argument("--abs-tol", type=float, default=1e-5, help="Float absolute tolerance (default: 1e-5)")
    args = parser.parse_args()

    engines = resolve_engines(args.engines, args.sqlite_only)
    families = resolve_families(args.family)
    cases = ["case1354pegase", "case2869pegase"] if args.case == "both" else [args.case]

    if args.emit_jsonl is not None and len(cases) > 1:
        raise ValueError("--emit-jsonl can only be used with a single --case")

    total_passed = 0
    total_failed = 0
    for c in cases:
        p, f = run_case(
            case_name=c,
            engines=engines,
            count=args.count,
            seed=args.seed,
            families=families,
            input_jsonl=args.input_jsonl,
            emit_jsonl=args.emit_jsonl,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )
        total_passed += p
        total_failed += f

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
