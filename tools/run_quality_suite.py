#!/usr/bin/env python3
"""
run_quality_suite.py — Unified ImpulseVM SQLite-Grade Quality Gate

Runs all 4-way polyglot verification pillars in one command (~35s default):
  1. Spec Compliance Suite     (four_way_validation.py, 550+ vectors x 4 engines)
  2. Metamorphic Property Gate (metamorphic_vm_test.py, 25 algebraic/graph laws x 4 engines)
  3. Multi-Profile Fuzzer      (fuzz_vm_contracts.py, 10 profiles, 152+ opcodes x 4 engines)
  4. Optional Corpus Mutator   (mutate_corpus_fuzzer.py, --corpus-iters N, dbsqlfuzz-style)

Optional: --with-unit-tests also runs C++ ctest, Java mvn test, and OCaml dune runtest.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
ROOT_DIR = REPO_DIR.parent

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _run_stage(name: str, cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[bool, float]:
    print(f"\n{BOLD}{BLUE}>>> [{name}]{RESET}")
    t0 = time.monotonic()
    res = subprocess.run(cmd, cwd=cwd, env=env)
    dt = time.monotonic() - t0
    ok = (res.returncode == 0)
    badge = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL (exit={res.returncode}){RESET}"
    print(f"{BOLD}<<< [{name}] {badge} in {dt:.2f}s{RESET}")
    return ok, dt


def main():
    parser = argparse.ArgumentParser(description="ImpulseVM Unified 4-Way Quality Gate")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--meta-pairs", type=int, default=2000, help="Metamorphic pairs to run (default: 2000)")
    parser.add_argument("--fuzz-iters", type=int, default=5000, help="Multi-profile fuzz iterations (default: 5000)")
    parser.add_argument("--corpus-iters", type=int, default=0, help="Optional corpus mutation iterations (default: 0)")
    parser.add_argument("--seed", type=int, default=20261122, help="Base seed for metamorphic + fuzzer runs")
    parser.add_argument("--spec-dir", type=str, default=None, help="Override IMPULSE_SPEC_DIR")
    parser.add_argument("--with-unit-tests", action="store_true", help="Also run C++ ctest, Java mvn test, OCaml dune runtest")
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir) if args.spec_dir else (ROOT_DIR / "impulse-graph-spec")
    core_cpp = ROOT_DIR / ".worktrees" / "v11-m3" / "impulse-graph-core" / "impulse-cpp"
    if not core_cpp.exists():
        core_cpp = ROOT_DIR / "impulse-graph-core" / "impulse-cpp"
    ext = "dylib" if sys.platform == "darwin" else "so"
    env = os.environ.copy()
    env.setdefault("IMPULSE_LIB_PATH_ON", str(core_cpp / "build_on" / f"libimpulse_graph.{ext}"))
    env.setdefault("IMPULSE_LIB_PATH_OFF", str(core_cpp / "build_off" / f"libimpulse_graph.{ext}"))
    if args.spec_dir:
        env["IMPULSE_SPEC_DIR"] = args.spec_dir

    py = sys.executable
    stages = [
        (
            "0/4 Core Engine Invariant Arch-Lint",
            [py, "-u", str(TOOLS_DIR / "check_engine_invariants.py")],
            REPO_DIR,
        ),
        (
            "1/4 Spec Compliance Suite",
            [py, "-u", str(spec_dir / "tools" / "run_vm_asm_suite.py"), "--check-coverage"],
            REPO_DIR,
        ),
        (
            f"2/4 Metamorphic Property Gate ({args.meta_pairs:,} pairs)",
            [py, "-u", str(TOOLS_DIR / "metamorphic_vm_test.py"), "--iterations", str(args.meta_pairs),
             "--seed", str(args.seed)],
            REPO_DIR,
        ),
        (
            f"3/4 Multi-Profile Differential Fuzzer ({args.fuzz_iters:,} iters)",
            [py, "-u", str(TOOLS_DIR / "fuzz_vm_contracts.py"), "--iterations", str(args.fuzz_iters),
             "--seed", str(args.seed + 100000), "--profile", "all", "--workers", str(args.workers),
             "--shrink", "--coverage-report"],
            REPO_DIR,
        ),
    ]

    if args.corpus_iters > 0:
        corpus_cmd = [
            py, "-u", str(TOOLS_DIR / "fuzz_vm_contracts.py"),
            "--profile", "corpus",
            "--iterations", str(args.corpus_iters),
            "--seed", str(args.seed + 200000),
            "--workers", str(args.workers),
            "--shrink",
            "--coverage-report",
        ]
        if args.spec_dir:
            corpus_cmd.extend(["--spec-dir", args.spec_dir])
        stages.append((f"4/4 Corpus Mutation Fuzzer ({args.corpus_iters:,} iters)", corpus_cmd, REPO_DIR))

    if args.with_unit_tests:
        stages.extend([
            ("Unit: OCaml dune runtest", ["dune", "runtest"], ROOT_DIR / "impulse-ocaml"),
            ("Unit: C++ ctest (build_on)", ["ctest", "--output-on-failure"], core_cpp / "build_on"),
            ("Unit: Java mvn impulse-vm test", ["mvn", "-q", "-pl", "impulse-vm", "test"], ROOT_DIR / "impulse-graph-java"),
        ])

    t_all = time.monotonic()
    results = []
    for name, cmd, cwd in stages:
        ok, dt = _run_stage(name, cmd, cwd, env)
        results.append((name, ok, dt))

    total_dt = time.monotonic() - t_all
    all_ok = all(ok for _, ok, _ in results)
    print(f"\n{BOLD}{BLUE}==============================================================={RESET}")
    print(f"{BOLD} ImpulseVM Quality Gate Summary ({total_dt:.2f}s total){RESET}")
    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    for name, ok, dt in results:
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name:<48s} ({dt:.2f}s)")
    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
