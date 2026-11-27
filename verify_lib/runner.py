"""Suite execution engine, JSONL run ledger, and gamut status/staleness reporting."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import Catalog, SuiteSpec, WorkspacePaths, build_suite_command
from .prereqs import (
    PrereqResult,
    RepoGitState,
    build_execution_env,
    collect_repo_git_states,
    probe_all_prereqs,
)


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class ExecutionRecord:
    suite_id: str
    title: str
    pillar: str
    repo: str
    intensity: str
    status: str  # PASS | FAIL | SKIP
    exit_code: int
    duration_ms: int
    cmd: list[str]
    cwd: str
    seed: int
    skip_reason: str = ""
    opcodes_covered: list[str] = field(default_factory=list)
    coverage_files: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    gamut: str
    intensity: str
    seed: int
    workers: int
    host: str
    status: str  # PASS | FAIL | PARTIAL_SKIP
    duration_ms: int
    pass_count: int
    fail_count: int
    skip_count: int
    commits: dict[str, dict[str, Any]]
    executions: list[ExecutionRecord]


def graph_data_dir(ws: WorkspacePaths) -> Path:
    return ws.repos["verification"] / "datasets" / "verification_graph"


def runs_dir(ws: WorkspacePaths) -> Path:
    return graph_data_dir(ws) / "runs"


def save_run_record(ws: WorkspacePaths, record: RunRecord) -> Path:
    rdir = runs_dir(ws)
    rdir.mkdir(parents=True, exist_ok=True)
    out_path = rdir / f"{record.run_id}.json"
    out_path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    ledger_path = graph_data_dir(ws) / "runs.jsonl"
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return out_path


def load_all_runs(ws: WorkspacePaths) -> list[dict[str, Any]]:
    ledger_path = graph_data_dir(ws) / "runs.jsonl"
    if ledger_path.is_file():
        runs: list[dict[str, Any]] = []
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return runs
    rdir = runs_dir(ws)
    if not rdir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(rdir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def _extract_coverage_for_suite(
    suite: SuiteSpec,
    stdout_text: str,
    ws: WorkspacePaths,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not suite.coverage_parser:
        return [], []
    from . import coverage as cov_mod

    opcodes: list[str] = []
    files: list[dict[str, Any]] = []
    if suite.coverage_parser in ("opcode_suite", "opcode_fuzz"):
        opcodes = cov_mod.parse_opcode_coverage_output(stdout_text)
    elif suite.coverage_parser == "jacoco":
        files = cov_mod.collect_jacoco_reports(ws.repos["java"])
    elif suite.coverage_parser == "pitest":
        files = cov_mod.collect_pitest_reports(ws.repos["java"])
    elif suite.coverage_parser == "llvm_cov":
        files = cov_mod.collect_llvm_cov_reports(ws.repos["core"])
    return opcodes, files


def execute_suites(
    cat: Catalog,
    ws: WorkspacePaths,
    selected: list[tuple[SuiteSpec, str]],
    *,
    gamut_label: str,
    seed: int,
    workers: int,
    fail_fast: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> RunRecord:
    prereqs: dict[str, PrereqResult] = probe_all_prereqs(cat, ws)
    commits: dict[str, RepoGitState] = collect_repo_git_states(ws)
    env = build_execution_env(ws)

    now = datetime.now(timezone.utc)
    run_id = f"run_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    t_all = time.monotonic()
    executions: list[ExecutionRecord] = []

    for idx, (suite, intensity) in enumerate(selected, start=1):
        cmd, cwd = build_suite_command(suite, intensity, ws, seed=seed, workers=workers)
        missing = [r for r in suite.requires if not prereqs[r].ok]
        if missing:
            reason = f"missing prereq(s): {', '.join(missing)}"
            if not quiet:
                print(f"{YELLOW}[SKIP]{RESET} ({idx}/{len(selected)}) {suite.id} — {reason}")
            executions.append(
                ExecutionRecord(
                    suite_id=suite.id,
                    title=suite.title,
                    pillar=suite.pillar,
                    repo=suite.repo,
                    intensity=intensity,
                    status="SKIP",
                    exit_code=0,
                    duration_ms=0,
                    cmd=cmd,
                    cwd=str(cwd),
                    seed=seed,
                    skip_reason=reason,
                )
            )
            continue

        if dry_run:
            if not quiet:
                print(f"[DRY-RUN] ({idx}/{len(selected)}) {suite.id} (cwd={cwd})\n  $ {' '.join(cmd)}")
            executions.append(
                ExecutionRecord(
                    suite_id=suite.id,
                    title=suite.title,
                    pillar=suite.pillar,
                    repo=suite.repo,
                    intensity=intensity,
                    status="SKIP",
                    exit_code=0,
                    duration_ms=0,
                    cmd=cmd,
                    cwd=str(cwd),
                    seed=seed,
                    skip_reason="dry-run",
                )
            )
            continue

        if not quiet:
            print(f"\n{BOLD}{BLUE}>>> [{idx}/{len(selected)}] {suite.id} ({suite.pillar}, {intensity}){RESET}")
            print(f"    cwd: {cwd}")
            print(f"    cmd: {' '.join(cmd)}")

        t0 = time.monotonic()
        capture_output = suite.coverage_parser in ("opcode_suite", "opcode_fuzz")
        if capture_output:
            proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            if not quiet and proc.stdout:
                print(proc.stdout, end="")
            if not quiet and proc.stderr:
                print(proc.stderr, end="")
            combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        else:
            proc = subprocess.run(cmd, cwd=cwd, env=env)
            combined_out = ""
        dt_ms = int((time.monotonic() - t0) * 1000)
        ok = proc.returncode == 0
        status = "PASS" if ok else "FAIL"

        opcodes_cov: list[str] = []
        cov_files: list[dict[str, Any]] = []
        if ok and suite.coverage_parser:
            opcodes_cov, cov_files = _extract_coverage_for_suite(suite, combined_out, ws)

        if not quiet:
            badge = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL (exit={proc.returncode}){RESET}"
            print(f"{BOLD}<<< [{suite.id}] {badge} in {dt_ms / 1000.0:.2f}s{RESET}")

        executions.append(
            ExecutionRecord(
                suite_id=suite.id,
                title=suite.title,
                pillar=suite.pillar,
                repo=suite.repo,
                intensity=intensity,
                status=status,
                exit_code=proc.returncode,
                duration_ms=dt_ms,
                cmd=cmd,
                cwd=str(cwd),
                seed=seed,
                opcodes_covered=opcodes_cov,
                coverage_files=cov_files,
            )
        )
        if fail_fast and not ok:
            break

    total_ms = int((time.monotonic() - t_all) * 1000)
    pass_c = sum(1 for e in executions if e.status == "PASS")
    fail_c = sum(1 for e in executions if e.status == "FAIL")
    skip_c = sum(1 for e in executions if e.status == "SKIP")
    if fail_c > 0:
        overall = "FAIL"
    elif skip_c > 0 and pass_c == 0:
        overall = "SKIP"
    elif skip_c > 0:
        overall = "PARTIAL_SKIP"
    else:
        overall = "PASS"

    intensity_label = selected[0][1] if selected else "standard"
    record = RunRecord(
        run_id=run_id,
        timestamp=now.isoformat(),
        gamut=gamut_label,
        intensity=intensity_label,
        seed=seed,
        workers=workers,
        host=f"{platform.node()} ({platform.system()}/{platform.machine()})",
        status=overall,
        duration_ms=total_ms,
        pass_count=pass_c,
        fail_count=fail_c,
        skip_count=skip_c,
        commits={k: asdict(v) for k, v in commits.items()},
        executions=executions,
    )
    if not dry_run:
        save_run_record(ws, record)
    return record


def compute_gamut_status(
    cat: Catalog,
    ws: WorkspacePaths,
    runs: list[dict[str, Any]],
    *,
    gamut_filter: str | None = None,
) -> list[dict[str, Any]]:
    live = collect_repo_git_states(ws)
    latest_by_gamut: dict[str, dict[str, Any]] = {}
    for r in runs:
        g = r.get("gamut", "")
        if g:
            latest_by_gamut[g] = r

    gamut_names = [gamut_filter] if gamut_filter else list(cat.gamuts.keys())
    rows: list[dict[str, Any]] = []
    for gname in gamut_names:
        last = latest_by_gamut.get(gname)
        if not last:
            rows.append(
                {
                    "gamut": gname,
                    "state": "NEVER_RUN",
                    "run_id": None,
                    "timestamp": None,
                    "pass_count": 0,
                    "fail_count": 0,
                    "skip_count": 0,
                    "stale_repos": [],
                    "dirty_at_run": [],
                }
            )
            continue
        stale_repos: list[dict[str, str]] = []
        dirty_at_run: list[str] = []
        recorded_commits = last.get("commits", {})
        for rname, lstate in live.items():
            if not lstate.exists:
                continue
            rec = recorded_commits.get(rname, {})
            rec_sha = rec.get("head_sha", "")
            if rec.get("dirty"):
                dirty_at_run.append(rname)
            if rec_sha and rec_sha != lstate.head_sha:
                stale_repos.append(
                    {
                        "repo": rname,
                        "recorded": rec_sha[:10],
                        "current": lstate.short_sha,
                    }
                )
            elif lstate.dirty:
                stale_repos.append(
                    {
                        "repo": rname,
                        "recorded": rec_sha[:10] if rec_sha else "?",
                        "current": f"{lstate.short_sha} (dirty)",
                    }
                )

        base_status = last.get("status", "UNKNOWN")
        state = f"{base_status}_STALE" if stale_repos and base_status == "PASS" else base_status
        rows.append(
            {
                "gamut": gname,
                "state": state,
                "run_id": last.get("run_id"),
                "timestamp": last.get("timestamp"),
                "pass_count": last.get("pass_count", 0),
                "fail_count": last.get("fail_count", 0),
                "skip_count": last.get("skip_count", 0),
                "duration_ms": last.get("duration_ms", 0),
                "stale_repos": stale_repos,
                "dirty_at_run": dirty_at_run,
            }
        )
    return rows


def print_run_summary(record: RunRecord) -> None:
    print(f"\n{BOLD}{BLUE}========================================================================={RESET}")
    print(
        f"{BOLD} Verification Run {record.run_id} | gamut={record.gamut} | "
        f"{record.pass_count} PASS, {record.fail_count} FAIL, {record.skip_count} SKIP "
        f"({record.duration_ms / 1000.0:.2f}s){RESET}"
    )
    print(f"{BOLD}{BLUE}========================================================================={RESET}")
    for ex in record.executions:
        color = GREEN if ex.status == "PASS" else (YELLOW if ex.status == "SKIP" else RED)
        extra = f" — {ex.skip_reason}" if ex.skip_reason else f" ({ex.duration_ms / 1000.0:.2f}s)"
        print(f"  [{color}{ex.status:4s}{RESET}] {ex.suite_id:<34s} [{ex.pillar:<11s}]{extra}")
    print(f"{BOLD}{BLUE}========================================================================={RESET}")
