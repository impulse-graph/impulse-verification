#!/usr/bin/env python3
"""
Test Suite: 50 Seeded .imps Snapshots Round-Trip Idempotency (Seeds 1001..1050)
For each seed:
  1. Generates seeded .imps snapshot via generate_random_imps.py
  2. Disassembles/Exports snapshot to tabular datasets + manifest (Export 1)
  3. Re-assembles/Builds new .imps snapshot from Export 1
  4. Validates re-assembled snapshot (--strict-alignment --deep-audit)
  5. Disassembles/Exports re-assembled snapshot (Export 2)
  6. Verifies Export 1 and Export 2 are 100% bitwise/content identical
"""

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
VERIF_DIR = TOOLS_DIR.parent
ROOT_DIR = VERIF_DIR.parent

sys.path.insert(0, str(TOOLS_DIR))
from generate_random_imps import (
    GeneratorConfig,
    find_impulse_cli,
    generate_one,
)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def compare_directories(dir1: Path, dir2: Path) -> list[str]:
    """Recursively compares two directories and returns list of discrepancies."""
    diffs = []
    
    files1 = {p.relative_to(dir1) for p in dir1.rglob("*") if p.is_file()}
    files2 = {p.relative_to(dir2) for p in dir2.rglob("*") if p.is_file()}

    missing_in_2 = files1 - files2
    missing_in_1 = files2 - files1

    for m in sorted(missing_in_2):
        diffs.append(f"Missing in Export 2: {m}")
    for m in sorted(missing_in_1):
        diffs.append(f"Missing in Export 1: {m}")

    common_files = files1 & files2
    for rel_p in sorted(common_files):
        p1 = dir1 / rel_p
        p2 = dir2 / rel_p
        if p1.name == "manifest.json":
            # Compare JSON semantically
            j1 = json.loads(p1.read_text(encoding="utf-8"))
            j2 = json.loads(p2.read_text(encoding="utf-8"))
            if j1 != j2:
                diffs.append(f"JSON discrepancy in {rel_p}")
        else:
            # Compare file bytes
            if not filecmp.cmp(p1, p2, shallow=False):
                diffs.append(f"Content discrepancy in {rel_p}")

    return diffs


def run_single_seed_roundtrip(
    seed: int,
    cli: Path,
    workspace: Path,
    config: GeneratorConfig,
    gen_version: int = 1,
    fmt: str = "csv",
) -> bool:
    seed_dir = workspace / f"seed_{seed}"
    shutil.rmtree(seed_dir, ignore_errors=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = seed_dir / "orig"
    export1_dir = seed_dir / "export_1"
    rebuilt_dir = seed_dir / "rebuilt"
    export2_dir = seed_dir / "export_2"

    orig_dir.mkdir(parents=True, exist_ok=True)
    rebuilt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate original snapshot
    schema, manifest_path, orig_imps = generate_one(
        seed=seed,
        gen_version=gen_version,
        config=config,
        out_dir=orig_dir,
        cli=cli,
    )
    assert orig_imps is not None and orig_imps.is_file()

    # 2. Export 1 (Disassemble snapshot to datasets + manifest.json)
    res_exp1 = subprocess.run(
        [str(cli), "snapshot", "export", str(orig_imps), "-o", str(export1_dir), "--format", fmt],
        capture_output=True,
        text=True,
    )
    if res_exp1.returncode != 0:
        print(f"  {RED}[FAIL]{RESET} Seed {seed}: Export 1 failed:\n{res_exp1.stderr}")
        return False

    manifest_export1 = export1_dir / "manifest.json"
    if not manifest_export1.is_file():
        print(f"  {RED}[FAIL]{RESET} Seed {seed}: Export 1 manifest.json missing")
        return False

    # 3. Build/Assemble rebuilt snapshot from Export 1
    rebuilt_imps = rebuilt_dir / f"rebuilt_{seed}.imps"
    res_build = subprocess.run(
        [str(cli), "build", "-m", str(manifest_export1), "-o", str(rebuilt_imps)],
        capture_output=True,
        text=True,
    )
    if res_build.returncode != 0:
        print(f"  {RED}[FAIL]{RESET} Seed {seed}: Re-build from Export 1 failed:\n{res_build.stderr}")
        return False

    # 4. Validate rebuilt snapshot with deep audit
    res_val = subprocess.run(
        [str(cli), "snapshot", "validate", "--strict-alignment", "--deep-audit", str(rebuilt_imps)],
        capture_output=True,
        text=True,
    )
    if res_val.returncode != 0:
        print(f"  {RED}[FAIL]{RESET} Seed {seed}: Validation on rebuilt snapshot failed:\n{res_val.stderr}")
        return False

    # 5. Export 2 (Disassemble rebuilt snapshot to datasets + manifest.json)
    res_exp2 = subprocess.run(
        [str(cli), "snapshot", "export", str(rebuilt_imps), "-o", str(export2_dir), "--format", fmt],
        capture_output=True,
        text=True,
    )
    if res_exp2.returncode != 0:
        print(f"  {RED}[FAIL]{RESET} Seed {seed}: Export 2 failed:\n{res_exp2.stderr}")
        return False

    # 6. Compare Export 1 and Export 2
    diffs = compare_directories(export1_dir, export2_dir)
    if diffs:
        print(f"  {RED}[FAIL]{RESET} Seed {seed}: Export 1 and Export 2 differ:")
        for d in diffs:
            print(f"    - {d}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Impulse Snapshot Round-Trip Idempotency Test Suite")
    parser.add_argument("--start-seed", type=int, default=1001, help="Starting seed (default: 1001)")
    parser.add_argument("--end-seed", type=int, default=1050, help="Ending seed (default: 1050)")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"], help="Dataset format (default: csv)")
    args = parser.parse_args()

    start_seed = args.start_seed
    end_seed = args.end_seed
    total_seeds = end_seed - start_seed + 1

    cli = find_impulse_cli()
    print(f"=========================================================================")
    print(f"    IMPULSE GRAPH ENGINE — {total_seeds}-SEED SNAPSHOT ROUND-TRIP TEST SUITE     ")
    print(f"=========================================================================")
    print(f"CLI Executable: {cli}")
    print(f"Seed Range:     {start_seed}..{end_seed} ({total_seeds} seeds)")
    print(f"Format:         {args.format}")
    print(f"=========================================================================\n")

    workspace = Path(tempfile.mkdtemp(prefix="impulse_roundtrip_"))
    config = GeneratorConfig()

    passed = 0
    failed = 0
    t0 = time.time()

    for seed in range(start_seed, end_seed + 1):
        idx = seed - start_seed + 1
        t_seed_start = time.time()
        ok = run_single_seed_roundtrip(seed, cli, workspace, config, fmt=args.format)
        elapsed = time.time() - t_seed_start
        if ok:
            passed += 1
            print(f"[{idx:03d}/{total_seeds:03d}] {GREEN}[PASS]{RESET} Seed {seed} (roundtrip identical in {elapsed:.2f}s)")
        else:
            failed += 1
            print(f"[{idx:03d}/{total_seeds:03d}] {RED}[FAIL]{RESET} Seed {seed} FAILED")

    total_time = time.time() - t0
    shutil.rmtree(workspace, ignore_errors=True)

    print(f"\n=========================================================================")
    print(f"Test Suite Summary: {passed}/{total_seeds} passed, {failed} failed in {total_time:.2f}s")
    print(f"=========================================================================")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
