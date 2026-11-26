#!/usr/bin/env python3
"""
check_engine_invariants.py — Static Architectural Invariant Linter for Impulse Graph Engine.

Enforces the non-negotiable engine invariants (AGENTS.md §3.19) across Java and C++ VM/adapter
sources so agents and contributors cannot introduce:
  1. Runtime topology synthesis (ensureCsc, buildRevAdj, computeInDegs full-edge transpose).
  2. Heap/arena copies of .imps CSR/CSC topology or full-column attribute/key materialization.
  3. Direct heap-array topology dereferences (rel.rowOffsets / rel.colTargets) in production VM ops.
  4. Missing workspace reset cleanup (strTable / multi-frame spills).

Exit code 0 = clean; non-zero = invariant violations found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent.parent


def resolve_roots(java_root: Path | None, cpp_root: Path | None) -> tuple[Path, Path]:
    if java_root is None:
        java_root = ROOT_DIR / "impulse-graph-java"
    if cpp_root is None:
        wt = ROOT_DIR / ".worktrees" / "v11-m3" / "impulse-graph-core" / "impulse-cpp"
        cpp_root = wt if wt.exists() else (ROOT_DIR / "impulse-graph-core" / "impulse-cpp")
    return java_root, cpp_root


def strip_comments(line: str) -> str:
    idx = line.find("//")
    if idx >= 0:
        return line[:idx]
    return line


def scan_file(path: Path, rules: list[tuple[str, re.Pattern[str], str]]) -> list[str]:
    violations: list[str] = []
    if not path.exists():
        return violations
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        code = strip_comments(raw)
        if not code.strip():
            continue
        for rule_id, pat, msg in rules:
            if pat.search(code):
                violations.append(f"[{rule_id}] {path}:{lineno}: {msg}\n    {raw.strip()}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Impulse Graph Engine static invariant linter")
    parser.add_argument("--java-root", type=Path, default=None)
    parser.add_argument("--cpp-root", type=Path, default=None)
    args = parser.parse_args()

    java_root, cpp_root = resolve_roots(args.java_root, args.cpp_root)
    violations: list[str] = []

    # 1. Java VM runtime rules (no topology synthesis, no heap-array topology derefs)
    java_vm_dir = java_root / "impulse-vm" / "src" / "main" / "java" / "org" / "impulsegraph" / "vm" / "v11"
    java_vm_rules = [
        (
            "NO_TOPOLOGY_SYNTHESIS",
            re.compile(r"\b(ensureCsc|buildRevAdj|computeInDegs)\b"),
            "VM runtime must NEVER synthesize CSC/COO/CST or transpose CSR on the fly; fail with MISSING_TOPOLOGY.",
        ),
        (
            "SINGLE_OFFHEAP_PATH",
            re.compile(r"\b(rel|view\.relations\[[^\]]+\])\.(rowOffsets|colTargets|cscRowOffsets|cscColSources)\b"),
            "Production VM ops must read off-heap MemorySegments via RelView accessors, never heap topology arrays.",
        ),
    ]
    if java_vm_dir.exists():
        for f in sorted(java_vm_dir.glob("*.java")):
            violations.extend(scan_file(f, java_vm_rules))

    # 2. Java Snapshot Adapter rules (zero-copy mmap-and-go, no full-column copies or ID fabrication)
    adapter_file = java_vm_dir / "ImpulseSnapshotAdapter.java"
    adapter_rules = [
        (
            "NO_HEAP_TOPOLOGY_COPY",
            re.compile(r"new\s+(int|long)\s*\[\s*(nRows|nEdges|cscNodes)\b"),
            "Snapshot adapter must NOT copy CSR/CSC arrays onto Java heap.",
        ),
        (
            "NO_FULL_ATTR_COPY",
            re.compile(r"\bdecodeAttribute\s*\("),
            "Snapshot adapter must NOT eagerly or lazily decode entire attribute columns onto Java heap.",
        ),
        (
            "NO_KEY_CATALOG_MATERIALIZATION",
            re.compile(r"new\s+String\s*\[\s*n\s*\]"),
            "Snapshot adapter must NOT materialize full-domain String[N] key catalogs onto Java heap.",
        ),
        (
            "NO_FALLBACK_REL_ID",
            re.compile(r"\bfallbackId\b"),
            "Snapshot adapter must NOT invent sequential fallback relation IDs; fail fast on invalid metadata.",
        ),
    ]
    violations.extend(scan_file(adapter_file, adapter_rules))

    # 3. Java V11SnapshotView rules (no heap topology arrays on RelView)
    view_file = java_vm_dir / "V11SnapshotView.java"
    view_rules = [
        (
            "SINGLE_OFFHEAP_RELVIEW",
            re.compile(r"\b(int\[\]\s+rowOffsets|long\[\]\s+colTargets|int\[\]\s+cscRowOffsets|int\[\]\s+cscColSources)\b"),
            "RelView must not hold heap-array topology fields; fixtures must pack into off-heap MemorySegments.",
        ),
    ]
    violations.extend(scan_file(view_file, view_rules))

    # 4. C++ VM dispatch rules (no CSR/CSC arena copy from mmap_ptr, no topology synthesis)
    cpp_v11_dir = cpp_root / "src" / "v11"
    cpp_dispatch = cpp_v11_dir / "vm_dispatch.cpp"
    cpp_rules = [
        (
            "NO_CPP_CSR_ARENA_COPY",
            re.compile(r"arena\.alloc_array<uint(32|64)_t>\(\s*(n_rows|n_edges)\b"),
            "build_snapshot_view_from_imps must NOT copy CSR/CSC arrays out of mmap_ptr into arena memory.",
        ),
    ]
    violations.extend(scan_file(cpp_dispatch, cpp_rules))

    # 5. Java V11Workspace.reset() strTable cleanup check
    ws_file = java_vm_dir / "ImpulseWorkspace.java"
    if ws_file.exists():
        ws_text = ws_file.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"public void reset\(\)\s*\{(.*?)\n    \}", ws_text, re.DOTALL)
        if m and "strTable.clear()" not in m.group(1):
            violations.append(
                f"[WORKSPACE_RESET_LEAK] {ws_file}: V11Workspace.reset() must clear strTable and reset nextStrHandle."
            )

    if violations:
        print(f"FAIL: {len(violations)} core engine invariant violation(s) detected:\n")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("PASS: 0 core engine invariant violations detected (Java & C++).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
