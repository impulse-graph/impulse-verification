#!/usr/bin/env python3
"""
verify.py — Unified Impulse Graph Verification Runner, .imps Coverage Graph & Static Call-Chain CLI.

Can be invoked from any sibling repository or active git worktree (auto-detects cwd repo
and supports `--override <repo>=<path>`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from verify_lib.catalog import load_catalog, resolve_workspace
from verify_lib.code_index import build_code_index
from verify_lib.imps_graph import VerificationGraph, build_verification_imps, imps_snapshot_path
from verify_lib.prereqs import collect_repo_git_states, probe_all_prereqs
from verify_lib.runner import (
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    compute_gamut_status,
    execute_suites,
    load_all_runs,
    print_run_summary,
)


VERIFICATION_DIR = Path(__file__).resolve().parent


def _parse_overrides(raw: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --override '{item}'; expected <repo>=<path>")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _open_graph_checked(cat, ws, *, auto_refresh: bool = False, quiet: bool = False) -> VerificationGraph:
    snap_path = imps_snapshot_path(ws)
    if not snap_path.is_file():
        if auto_refresh:
            build_code_index(ws)
            res = build_verification_imps(cat, ws)
            if not res.ok:
                raise SystemExit(f"Failed to build verification.imps: {res.error}")
        else:
            raise SystemExit(f"Snapshot not found at {snap_path}. Run 'verify.py index' or pass '--auto-refresh'.")
    vg = VerificationGraph.open(ws)
    fresh = vg.check_freshness(ws)
    if not fresh["fresh"]:
        if auto_refresh:
            vg.close()
            build_code_index(ws)
            res = build_verification_imps(cat, ws)
            if not res.ok:
                raise SystemExit(f"Failed to rebuild verification.imps: {res.error}")
            vg = VerificationGraph.open(ws)
        elif not quiet:
            repos_str = ", ".join(
                f"{r['repo']}({r['indexed_sha']}->{r['current_sha']}{'*' if r['current_dirty'] else ''})"
                for r in fresh["stale_repos"]
            )
            print(
                f"{YELLOW}[STALE INDEX]{RESET} verification.imps lags live HEAD/tree: {repos_str} "
                f"(run 'verify.py index' or pass '--auto-refresh')",
                file=sys.stderr,
            )
    return vg


def cmd_list(args: argparse.Namespace, cat, ws) -> int:
    pillars = [p.strip() for p in args.pillar.split(",")] if args.pillar else None
    repos = [r.strip() for r in args.repo.split(",")] if args.repo else None
    selected = cat.select_suites(
        gamut=args.gamut,
        pillars=pillars,
        repos=repos,
        intensity_override=args.intensity,
    )
    prereqs = probe_all_prereqs(cat, ws)
    if args.json:
        payload = [
            {
                "id": s.id,
                "title": s.title,
                "pillar": s.pillar,
                "repo": s.repo,
                "determinism": s.determinism,
                "cost": s.cost,
                "intensity": intensity,
                "ready": all(prereqs[r].ok for r in s.requires),
                "missing_prereqs": [r for r in s.requires if not prereqs[r].ok],
            }
            for s, intensity in selected
        ]
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{BOLD}Suites ({len(selected)} selected){RESET}")
    for s, intensity in selected:
        missing = [r for r in s.requires if not prereqs[r].ok]
        badge = f"{GREEN}READY{RESET}" if not missing else f"{YELLOW}MISSING({','.join(missing)}){RESET}"
        print(f"  {s.id:<34s} [{s.pillar:<11s}] [{s.repo:<12s}] ({intensity:<8s}, {s.cost:<6s}) {badge}")
    return 0


def cmd_doctor(args: argparse.Namespace, cat, ws) -> int:
    prereqs = probe_all_prereqs(cat, ws)
    git_states = collect_repo_git_states(ws)
    if args.json:
        print(
            json.dumps(
                {
                    "workspace_root": str(ws.root),
                    "repos": {k: vars(v) for k, v in git_states.items()},
                    "prereqs": {k: vars(v) for k, v in prereqs.items()},
                },
                indent=2,
            )
        )
        return 0

    print(f"{BOLD}Workspace Repos (root: {ws.root}){RESET}")
    for name, gs in git_states.items():
        dirty = " (dirty)" if gs.dirty else ""
        badge = f"{GREEN}OK{RESET}" if gs.exists else f"{RED}MISSING{RESET}"
        print(f"  [{badge}] {name:<14s} {gs.branch:<26s} {gs.short_sha}{dirty}")

    print(f"\n{BOLD}Prerequisites{RESET}")
    for name, pr in prereqs.items():
        if pr.ok:
            badge = f"{GREEN}OK{RESET}"
        elif pr.optional:
            badge = f"{YELLOW}OPT{RESET}"
        else:
            badge = f"{RED}MISS{RESET}"
        print(f"  [{badge}] {name:<18s} {pr.detail}")
        if not pr.ok and pr.hint:
            print(f"         hint: {pr.hint}")
    return 0


def cmd_run(args: argparse.Namespace, cat, ws) -> int:
    pillars = [p.strip() for p in args.pillar.split(",")] if args.pillar else None
    repos = [r.strip() for r in args.repo.split(",")] if args.repo else None
    ids = [i.strip() for i in args.id.split(",")] if args.id else None
    selected = cat.select_suites(
        gamut=args.gamut,
        pillars=pillars,
        repos=repos,
        ids=ids,
        intensity_override=args.intensity,
    )
    gamut_label = args.gamut or ("custom" if (ids or pillars or repos) else cat.default_gamut)
    record = execute_suites(
        cat,
        ws,
        selected,
        gamut_label=gamut_label,
        seed=args.seed if args.seed is not None else cat.default_seed,
        workers=args.workers,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
    )
    print_run_summary(record)

    if not args.dry_run and not args.no_sync:
        sync_res = build_verification_imps(cat, ws, validate=True)
        if sync_res.ok:
            print(
                f"{GREEN}[dogfood.imps_sync PASS]{RESET} {sync_res.imps_path} "
                f"(domains={sum(sync_res.domain_counts.values())} nodes, "
                f"edges={sum(sync_res.relation_counts.values())}, validated=strict+deep-audit)"
            )
        else:
            print(f"{YELLOW}[dogfood.imps_sync FAIL]{RESET} {sync_res.error}", file=sys.stderr)

    return 0 if record.fail_count == 0 else 1


def cmd_status(args: argparse.Namespace, cat, ws) -> int:
    runs = load_all_runs(ws)
    rows = compute_gamut_status(cat, ws, runs, gamut_filter=args.gamut)

    imps_fresh: dict[str, Any] | None = None
    if imps_snapshot_path(ws).is_file():
        try:
            with VerificationGraph.open(ws) as vg:
                imps_fresh = vg.check_freshness(ws)
        except Exception:
            imps_fresh = None

    if args.json:
        print(json.dumps({"gamuts": rows, "imps_index": imps_fresh}, indent=2))
        return 0

    print(f"{BOLD}Verification Status by Gamut{RESET}")
    for row in rows:
        st = row["state"]
        color = GREEN if st == "PASS" else (YELLOW if "STALE" in st or "SKIP" in st or st == "NEVER_RUN" else RED)
        counts = f"{row['pass_count']}P/{row['fail_count']}F/{row['skip_count']}S" if row["run_id"] else "-"
        ts = (row["timestamp"] or "")[:19]
        print(f"  {row['gamut']:<14s} [{color}{st:<12s}{RESET}] {counts:<12s} {ts}")
        if row["stale_repos"]:
            s_str = ", ".join(f"{r['repo']}({r['recorded']}->{r['current']})" for r in row["stale_repos"])
            print(f"                 stale: {s_str}")
        if row["dirty_at_run"]:
            print(f"                 dirty-at-run: {', '.join(row['dirty_at_run'])}")
    if imps_fresh is not None:
        badge = f"{GREEN}FRESH{RESET}" if imps_fresh["fresh"] else f"{YELLOW}STALE{RESET}"
        print(f"\nverification.imps: [{badge}] indexed_at={imps_fresh['indexed_at'][:19]}")
    return 0


def cmd_sync(args: argparse.Namespace, cat, ws) -> int:
    res = build_verification_imps(cat, ws, validate=not args.no_validate)
    if args.json:
        print(json.dumps(vars(res), default=str, indent=2))
    elif res.ok:
        print(f"{GREEN}PASS{RESET}: built {res.imps_path} (validated={res.validated})")
        print(f"  domains:   {res.domain_counts}")
        print(f"  relations: {sum(res.relation_counts.values())} total edges")
    else:
        print(f"{RED}FAIL{RESET}: {res.error}", file=sys.stderr)
    return 0 if res.ok else 1


def cmd_index(args: argparse.Namespace, cat, ws) -> int:
    code_data = build_code_index(ws, run_scip=args.run_scip)
    res = build_verification_imps(cat, ws, code_data=code_data, validate=not args.no_validate)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": res.ok,
                    "imps_path": str(res.imps_path),
                    "validated": res.validated,
                    "mode_by_repo": code_data["mode_by_repo"],
                    "scip_run_status": code_data["scip_run_status"],
                    "domain_counts": res.domain_counts,
                    "relation_counts": res.relation_counts,
                    "error": res.error,
                },
                indent=2,
            )
        )
    elif res.ok:
        print(f"{GREEN}PASS{RESET}: indexed & built {res.imps_path} (validated={res.validated})")
        print(f"  modes:     {code_data['mode_by_repo']}")
        print(
            f"  code:      {len(code_data['source_files'])} files, {len(code_data['symbols'])} symbols, "
            f"{len(code_data['cabi_symbols'])} C-ABI symbols, {len(code_data['calls'])} CALLS edges"
        )
        print(
            f"  bridges:   {len(code_data['emits_opcode'])} EMITS_OPCODE, "
            f"{len(code_data['implements_opcode'])} IMPLEMENTS_OPCODE, "
            f"{len(code_data['vec_exercises_opcode'])} VEC_EXERCISES_OPCODE, "
            f"{len(code_data['calls_cabi'])} CALLS_CABI"
        )
    else:
        print(f"{RED}FAIL{RESET}: {res.error}", file=sys.stderr)
    return 0 if res.ok else 1


def cmd_gaps(args: argparse.Namespace, cat, ws) -> int:
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_gaps(min_branch=args.min_branch, min_mcdc=args.min_mcdc, min_mutation=args.min_mutation)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"{BOLD}Coverage Blind Spots{RESET}")
    print(f"  Uncovered active opcodes:     {len(data['uncovered_opcodes'])}")
    for op in data["uncovered_opcodes"][:20]:
        print(f"    - {op['hex']} {op['opcode']}")
    print(f"  Active opcodes w/o fuzzer:    {data['no_fuzzer_total']}")
    print(f"  Low-coverage/mutation files:  {len(data['low_coverage_files'])}")
    for f in data["low_coverage_files"][:20]:
        print(f"    - {f['file']} [{f['tool']}] line={f['line_pct']}% branch={f['branch_pct']}% mcdc={f['mcdc_pct']}%")
    return 0


def cmd_diff(args: argparse.Namespace, cat, ws) -> int:
    runs = {r["run_id"]: r for r in load_all_runs(ws)}
    if args.run_a not in runs or args.run_b not in runs:
        raise SystemExit(f"Run ID not found. Available: {', '.join(list(runs.keys())[-10:])}")
    ra, rb = runs[args.run_a], runs[args.run_b]
    ea = {e["suite_id"]: e for e in ra.get("executions", [])}
    eb = {e["suite_id"]: e for e in rb.get("executions", [])}
    regressions = []
    slowdowns = []
    for sid in sorted(set(ea) | set(eb)):
        a, b = ea.get(sid), eb.get(sid)
        if a and b:
            if a["status"] == "PASS" and b["status"] == "FAIL":
                regressions.append({"suite_id": sid, "from": "PASS", "to": "FAIL"})
            if a["duration_ms"] > 500 and b["duration_ms"] > a["duration_ms"] * 1.2:
                slowdowns.append({
                    "suite_id": sid,
                    "from_ms": a["duration_ms"],
                    "to_ms": b["duration_ms"],
                    "delta_pct": round(100.0 * (b["duration_ms"] - a["duration_ms"]) / a["duration_ms"], 1),
                })
    out = {"run_a": args.run_a, "run_b": args.run_b, "regressions": regressions, "slowdowns": slowdowns}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{BOLD}Diff {args.run_a} -> {args.run_b}{RESET}")
        print(f"  PASS->FAIL regressions: {len(regressions)}")
        for r in regressions:
            print(f"    - {r['suite_id']}")
        print(f"  >20% slowdowns:         {len(slowdowns)}")
        for s in slowdowns:
            print(f"    - {s['suite_id']}: {s['from_ms']}ms -> {s['to_ms']}ms (+{s['delta_pct']}%)")
    return 0


def cmd_opcode(args: argparse.Namespace, cat, ws) -> int:
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_opcode(args.opcode)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0 if data.get("found") else 1
    if not data.get("found"):
        print(f"{RED}Opcode not found:{RESET} {args.opcode}")
        return 1
    print(f"{BOLD}{data['hex']} {data['opcode']}{RESET} (category={data['category']})")
    print(f"  Emitters ({len(data['emitters'])}):")
    for s in data["emitters"][:25]:
        print(f"    - [{s['repo']}/{s['language']}] {s['name']}  {s['file_path']}:{s['line_start']}-{s['line_end']}")
    print(f"  Handlers ({len(data['handlers'])}):")
    for s in data["handlers"][:25]:
        print(f"    - [{s['repo']}/{s['language']}] {s['name']}  {s['file_path']}:{s['line_start']}-{s['line_end']}")
    print(f"  Spec Vectors ({len(data['spec_vectors'])}):")
    for v in data["spec_vectors"][:20]:
        print(f"    - {v['path']} ({v['expect_status']})")
    print(f"  Suites ({len(data['suites'])}): {', '.join(data['suites']) or '(none recorded yet)'}")
    return 0


def cmd_call_chain(args: argparse.Namespace, cat, ws, direction: str) -> int:
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_call_chain(args.symbol, direction=direction, depth=args.depth)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0 if data.get("found") else 1
    if not data.get("found"):
        print(f"{RED}Symbol not found:{RESET} {args.symbol}")
        return 1
    print(f"{BOLD}{direction.capitalize()} of '{args.symbol}' (depth<={args.depth}, {len(data['nodes'])} nodes){RESET}")
    for n in data["nodes"][:40]:
        indent = "  " * (n["depth"] + 1)
        print(f"{indent}[d={n['depth']}] {n['name']} ({n['repo']}/{n['language']})  {n['file_path']}:{n['line_start']}-{n['line_end']}")
    return 0


def cmd_where(args: argparse.Namespace, cat, ws) -> int:
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_where(args.name)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    if data["cabi"]:
        print(f"{BOLD}C-ABI Declarations ({len(data['cabi'])}){RESET}")
        for c in data["cabi"]:
            print(f"  {c['symbol']}  {c['header']}:{c['line']}")
            for imp in c["implementations"][:10]:
                print(f"    impl: {imp['file_path']}:{imp['line_start']}-{imp['line_end']}")
            for call in c["call_sites"][:20]:
                print(f"    call: [{call['repo']}/{call['language']}] {call['name']}  {call['file_path']}:{call['line_start']}")
    if data["symbols"]:
        print(f"{BOLD}Symbols ({len(data['symbols'])}){RESET}")
        for s in data["symbols"][:25]:
            print(f"  [{s['repo']}/{s['language']}] {s['name']} ({s['kind']})  {s['file_path']}:{s['line_start']}-{s['line_end']}")
    if not data["cabi"] and not data["symbols"]:
        print(f"{YELLOW}No matches for '{args.name}'{RESET}")
    return 0


def cmd_impacted(args: argparse.Namespace, cat, ws) -> int:
    changed: list[str] = list(args.files or [])
    if args.diff or not changed:
        repo_key, repo_root = ws.detect_repo_for_cwd(Path.cwd())
        target_repos = [(repo_key, repo_root)] if (repo_key and repo_root) else list(ws.repos.items())
        for rk, rpath in target_repos:
            if not rpath or not rpath.is_dir():
                continue
            res = subprocess.run(
                ["git", "-C", str(rpath), "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    line = line.strip()
                    if line:
                        changed.append(f"{rk}/{line}")
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_impacted(changed, cat)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"{BOLD}Impact Analysis ({len(changed)} changed file(s)){RESET}")
    print(f"  Matched source files: {len(data['matched_source_files'])}")
    print(f"  Impacted symbols:     {data['impacted_symbol_count']}")
    print(f"  Impacted opcodes:     {', '.join(data['impacted_opcodes'][:20]) or '(none)'}")
    print(f"  Recommended suites ({len(data['recommended_suites'])}):")
    for sid in data["recommended_suites"]:
        print(f"    - {sid}")
    return 0


def cmd_audit_invariants(args: argparse.Namespace, cat, ws) -> int:
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_audit_invariants()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0 if not data["transitive_alloc_violations"] else 1
    print(f"{BOLD}Call-Graph Invariant Audit{RESET}")
    print(f"  Hot-path roots checked:      {data['hot_roots_checked']}")
    print(f"  Transitive alloc violations: {len(data['transitive_alloc_violations'])}")
    for v in data["transitive_alloc_violations"][:15]:
        print(f"    - {v['root']} -[{v['hops']} hops]-> {v['target']} ({v['location']})")
    print(f"  Asymmetric opcodes (3-eng):  {data['asymmetric_opcode_count']}")
    for op in data["asymmetric_opcodes_sample"][:15]:
        print(f"    - {op['hex']} {op['opcode']} missing={op['missing_in']}")
    return 0 if not data["transitive_alloc_violations"] else 1


def cmd_matrix(args: argparse.Namespace, cat, ws) -> int:
    with _open_graph_checked(cat, ws, auto_refresh=args.auto_refresh, quiet=args.json) as vg:
        data = vg.query_binding_matrix(args.pattern)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    langs = data["languages"]
    print(f"{BOLD}Cross-Language Parity Matrix for '{args.pattern}'{RESET}")
    if data["cabi_rows"]:
        ffi_langs = ["cpp", "rust", "python", "go", "node", "dotnet"]
        hdr = f"  {'C-ABI Symbol':<48s} " + " ".join(f"{l:<7s}" for l in ffi_langs)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for row in data["cabi_rows"]:
            cells = []
            for l in ffi_langs:
                hits = row["by_lang"].get(l, [])
                if hits:
                    cells.append(f"{GREEN}{'YES':<7s}{RESET}")
                else:
                    cells.append(f"{RED}{'MISS':<7s}{RESET}")
            print(f"  {row['cabi']:<48s} " + " ".join(cells))
    print(f"\n{BOLD}Matching Symbols by Language / Binding (incl. JVM DSLs){RESET}")
    for l in langs:
        hits = data["symbols_by_lang"].get(l, [])
        badge = f"{GREEN}{len(hits):2d} hit(s){RESET}" if hits else f"{YELLOW} 0 hits {RESET}"
        print(f"  {l:<10s} [{badge}]")
        for h in hits[:5]:
            print(f"             - {h['name']}  {h['location']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify.py",
        description="Unified Impulse Graph Verification Runner, .imps Coverage Graph & Static Call-Chain CLI",
    )
    p.add_argument("--override", action="append", default=[], help="Override sibling repo path (<repo>=<path>)")
    p.add_argument("--auto-refresh", action="store_true", help="Automatically rebuild verification.imps if stale")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="List cataloged suites, gamuts, and prereq status")
    sp.add_argument("--gamut", type=str, default=None)
    sp.add_argument("--pillar", type=str, default=None)
    sp.add_argument("--repo", type=str, default=None)
    sp.add_argument("--intensity", type=str, default=None)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("doctor", help="Check sibling repos, toolchains, and SCIP indexers")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("run", help="Run a gamut, pillar, repo, or specific suite ID(s)")
    sp.add_argument("--gamut", type=str, default=None)
    sp.add_argument("--pillar", type=str, default=None)
    sp.add_argument("--repo", type=str, default=None)
    sp.add_argument("--id", type=str, default=None)
    sp.add_argument("--intensity", type=str, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    sp.add_argument("--fail-fast", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--no-sync", action="store_true")

    sp = sub.add_parser("status", help="Show latest status & commit staleness per gamut")
    sp.add_argument("--gamut", type=str, default=None)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("sync", help="Rebuild & validate datasets/verification.imps from catalog + runs + coverage")
    sp.add_argument("--no-validate", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("index", help="Index static call chains (SCIP-first + fallback) and rebuild verification.imps")
    sp.add_argument("--run-scip", action="store_true", help="Invoke installed SCIP indexers before ingesting")
    sp.add_argument("--no-validate", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("gaps", help="Query opcode and file-level coverage/mutation blind spots")
    sp.add_argument("--min-branch", type=float, default=80.0)
    sp.add_argument("--min-mcdc", type=float, default=80.0)
    sp.add_argument("--min-mutation", type=float, default=70.0)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("diff", help="Compare two verification runs for regressions & slowdowns")
    sp.add_argument("run_a")
    sp.add_argument("run_b")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("opcode", help="Bidirectional cross-repo provenance for an opcode (name or 0xHEX)")
    sp.add_argument("opcode")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("callers", help="Upward static call chain (CSC) for a symbol")
    sp.add_argument("symbol")
    sp.add_argument("--depth", type=int, default=3)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("callees", help="Downward static call chain (CSR) for a symbol")
    sp.add_argument("symbol")
    sp.add_argument("--depth", type=int, default=3)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("where", help="Locate symbol or C-ABI declaration + cross-language call sites")
    sp.add_argument("name")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("impacted", help="Compute impacted symbols, opcodes, and minimal covering suites for a diff")
    sp.add_argument("--diff", action="store_true", help="Read changed files from git diff HEAD")
    sp.add_argument("--files", nargs="*", default=[])
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("audit-invariants", help="Run transitive call-graph invariant & 3-engine opcode symmetry audit")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("matrix", help="Cross-language FFI & JVM DSL parity matrix for a capability/symbol pattern")
    sp.add_argument("pattern", help="Substring/capability to check (e.g. get_domain_attribute, domain_attribute, decimal)")
    sp.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cat = load_catalog(VERIFICATION_DIR / "catalog.toml")
    ws = resolve_workspace(VERIFICATION_DIR, cat.repo_names, overrides=_parse_overrides(args.override))

    dispatch = {
        "list": lambda: cmd_list(args, cat, ws),
        "doctor": lambda: cmd_doctor(args, cat, ws),
        "run": lambda: cmd_run(args, cat, ws),
        "status": lambda: cmd_status(args, cat, ws),
        "sync": lambda: cmd_sync(args, cat, ws),
        "index": lambda: cmd_index(args, cat, ws),
        "gaps": lambda: cmd_gaps(args, cat, ws),
        "diff": lambda: cmd_diff(args, cat, ws),
        "opcode": lambda: cmd_opcode(args, cat, ws),
        "callers": lambda: cmd_call_chain(args, cat, ws, "callers"),
        "callees": lambda: cmd_call_chain(args, cat, ws, "callees"),
        "where": lambda: cmd_where(args, cat, ws),
        "impacted": lambda: cmd_impacted(args, cat, ws),
        "audit-invariants": lambda: cmd_audit_invariants(args, cat, ws),
        "matrix": lambda: cmd_matrix(args, cat, ws),
    }
    return dispatch[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
