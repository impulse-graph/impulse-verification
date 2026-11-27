"""Dogfooded .imps Verification, Coverage & Static Call-Chain Graph Builder + Native Query Engine."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import Catalog, WorkspacePaths
from .coverage import (
    OpcodeEntry,
    collect_jacoco_reports,
    collect_llvm_cov_reports,
    collect_pitest_reports,
    load_spec_opcodes,
)
from .prereqs import collect_repo_git_states, find_impulse_cli, find_impulse_python_path
from .runner import graph_data_dir, load_all_runs


DOMAIN_ORDER = [
    "Gamut",          # 0
    "Pillar",         # 1
    "Repo",           # 2
    "Prereq",         # 3
    "Suite",          # 4
    "Opcode",         # 5
    "Run",            # 6
    "Commit",         # 7
    "Execution",      # 8
    "SourceFile",     # 9
    "CoverageEntry",  # 10
    "Symbol",         # 11
    "CAbiSymbol",     # 12
    "SpecVector",     # 13
]
DOMAIN_IDS = {name: idx for idx, name in enumerate(DOMAIN_ORDER)}


@dataclass
class BuildResult:
    ok: bool
    imps_path: Path
    validated: bool
    domain_counts: dict[str, int]
    relation_counts: dict[str, int]
    error: str = ""


def imps_snapshot_path(ws: WorkspacePaths) -> Path:
    return ws.repos["verification"] / "datasets" / "verification.imps"


def staging_dir(ws: WorkspacePaths) -> Path:
    return graph_data_dir(ws) / "staging"


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def _san_str(val: Any) -> str:
    return str(val if val is not None else "").replace("\r", " ").replace("\n", " ")


def build_verification_imps(
    cat: Catalog,
    ws: WorkspacePaths,
    *,
    code_data: dict[str, Any] | None = None,
    validate: bool = True,
) -> BuildResult:
    """Export catalog, run history, coverage, and static code index to CSV+manifest and run impulse-graph build."""
    out_imps = imps_snapshot_path(ws)
    cli = find_impulse_cli(ws)
    if cli is None:
        return BuildResult(
            ok=False,
            imps_path=out_imps,
            validated=False,
            domain_counts={},
            relation_counts={},
            error="impulse-graph CLI binary not found (build impulse-cli or set IMPULSE_CLI)",
        )

    sdir = staging_dir(ws)
    sdir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    git_states = collect_repo_git_states(ws)
    runs = load_all_runs(ws)
    opcodes: list[OpcodeEntry] = load_spec_opcodes(ws.repos["spec"])

    # Load cached code index if not passed explicitly
    if code_data is None:
        cache_path = graph_data_dir(ws) / "code_index.json"
        if cache_path.is_file():
            try:
                code_data = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                code_data = None

    # --- Node tables (key -> row dict) ---
    gamut_rows: list[list[Any]] = []
    for gname, g in cat.gamuts.items():
        gamut_rows.append([gname, _san_str(g.description), g.target_time_sec, g.intensity])

    pillar_rows: list[list[Any]] = [
        [pname, _san_str(p.description)] for pname, p in cat.pillars.items()
    ]

    repo_rows: list[list[Any]] = []
    for rname, dirname in cat.repo_names.items():
        gs = git_states.get(rname)
        repo_rows.append(
            [
                rname,
                dirname,
                gs.head_sha if gs else "missing",
                gs.branch if gs else "missing",
                1 if (gs and gs.dirty) else 0,
                now_iso,
            ]
        )

    prereq_rows: list[list[Any]] = [
        [pname, p.kind, _san_str(p.hint)] for pname, p in cat.prereqs.items()
    ]

    suite_rows: list[list[Any]] = [
        [sid, _san_str(s.title), s.determinism, s.cost] for sid, s in cat.suites.items()
    ]

    opcode_map: dict[str, OpcodeEntry] = {op.key: op for op in opcodes}
    opcode_by_name: dict[str, str] = {op.name: op.key for op in opcodes}
    opcode_by_hex: dict[str, str] = {op.hex_code.upper(): op.key for op in opcodes}

    # Ensure placeholder row if spec opcodes were unavailable (keeps domain non-empty)
    if not opcode_map:
        opcode_map["0x00:OP_NOP"] = OpcodeEntry("0x00:OP_NOP", "OP_NOP", "0x00", -1, "control", 1)
        opcode_by_name["OP_NOP"] = "0x00:OP_NOP"
        opcode_by_hex["0x00"] = "0x00:OP_NOP"

    opcode_rows: list[list[Any]] = [
        [op.key, op.name, op.hex_code, op.subop, op.category, op.active]
        for op in opcode_map.values()
    ]

    # Runs, Commits, Executions, CoverageEntries, SourceFiles
    run_rows: list[list[Any]] = []
    commit_rows_map: dict[str, list[Any]] = {}
    exec_rows: list[list[Any]] = []
    source_files_map: dict[str, list[Any]] = {}
    cov_rows: list[list[Any]] = []

    # Edge collectors (deduplicated)
    edges: dict[str, set[tuple[str, ...]]] = {
        "INCLUDES": set(),
        "IN_PILLAR": set(),
        "TARGETS_REPO": set(),
        "REQUIRES": set(),
        "EXERCISES_OPCODE": set(),
        "OF_GAMUT": set(),
        "AT_COMMIT": set(),
        "COMMIT_IN_REPO": set(),
        "HAS_EXECUTION": set(),
        "OF_SUITE": set(),
        "PRODUCED_COVERAGE": set(),
        "COVERS_FILE": set(),
        "FILE_IN_REPO": set(),
        "DEFINES_SYMBOL": set(),
        "CALLS": set(),
        "IMPLEMENTS_OR_EXTENDS": set(),
        "IMPLEMENTS_CABI": set(),
        "CALLS_CABI": set(),
        "EMITS_OPCODE": set(),
        "IMPLEMENTS_OPCODE": set(),
        "VEC_EXERCISES_OPCODE": set(),
        "COVERS_SYMBOL": set(),
    }

    for gname in cat.gamuts:
        for suite, intensity in cat.resolve_gamut_suites(gname):
            edges["INCLUDES"].add((gname, suite.id, intensity))

    for sid, s in cat.suites.items():
        edges["IN_PILLAR"].add((sid, s.pillar))
        edges["TARGETS_REPO"].add((sid, s.repo))
        for req in s.requires:
            edges["REQUIRES"].add((sid, req))

    for r in runs:
        rid = str(r.get("run_id", ""))
        if not rid:
            continue
        gname = str(r.get("gamut", ""))
        run_rows.append(
            [
                rid,
                _san_str(r.get("timestamp", "")),
                int(r.get("seed", 0)),
                str(r.get("status", "UNKNOWN")),
                int(r.get("duration_ms", 0)),
                int(r.get("pass_count", 0)),
                int(r.get("fail_count", 0)),
                int(r.get("skip_count", 0)),
            ]
        )
        if gname in cat.gamuts:
            edges["OF_GAMUT"].add((rid, gname))

        for rname, cinfo in r.get("commits", {}).items():
            if rname not in cat.repo_names:
                continue
            sha = str(cinfo.get("head_sha", "unknown"))
            branch = str(cinfo.get("branch", "unknown"))
            dirty = 1 if cinfo.get("dirty") else 0
            ckey = f"{rname}:{sha}:{dirty}"
            commit_rows_map[ckey] = [ckey, sha, branch, dirty]
            edges["AT_COMMIT"].add((rid, ckey))
            edges["COMMIT_IN_REPO"].add((ckey, rname))

        for ex in r.get("executions", []):
            sid = str(ex.get("suite_id", ""))
            if sid not in cat.suites:
                continue
            eid = f"{rid}:{sid}"
            exec_rows.append(
                [
                    eid,
                    str(ex.get("status", "UNKNOWN")),
                    int(ex.get("exit_code", 0)),
                    int(ex.get("duration_ms", 0)),
                    int(ex.get("seed", 0)),
                    _san_str(ex.get("skip_reason", "")),
                ]
            )
            edges["HAS_EXECUTION"].add((rid, eid))
            edges["OF_SUITE"].add((eid, sid))

            for tok in ex.get("opcodes_covered", []):
                op_key = opcode_by_name.get(tok) or opcode_by_hex.get(tok.upper())
                if op_key:
                    edges["EXERCISES_OPCODE"].add((sid, op_key))

            for cf in ex.get("coverage_files", []):
                repo_k = str(cf.get("repo", "core"))
                fpath = str(cf.get("file_path", ""))
                if not fpath or repo_k not in cat.repo_names:
                    continue
                fkey = f"{repo_k}/{fpath}"
                if fkey not in source_files_map:
                    source_files_map[fkey] = [fkey, _guess_lang(fpath), 0, ""]
                    edges["FILE_IN_REPO"].add((fkey, repo_k))
                tool = str(cf.get("tool", "unknown"))
                cov_id = f"{rid}:{tool}:{fkey}"
                cov_rows.append(
                    [
                        cov_id,
                        tool,
                        float(cf.get("line_pct", 0.0)),
                        float(cf.get("branch_pct", 0.0)),
                        float(cf.get("mcdc_pct", 0.0)),
                        int(cf.get("mutants_killed", 0)),
                        int(cf.get("mutants_total", 0)),
                    ]
                )
                edges["PRODUCED_COVERAGE"].add((eid, cov_id))
                edges["COVERS_FILE"].add((cov_id, fkey))

    # Also auto-ingest any existing JaCoCo / Pitest / llvm-cov reports present on disk
    disk_cov = (
        collect_jacoco_reports(ws.repos["java"])
        + collect_pitest_reports(ws.repos["java"])
        + collect_llvm_cov_reports(ws.repos["core"])
    )
    for cf in disk_cov:
        repo_k = str(cf.get("repo", "core"))
        fpath = str(cf.get("file_path", ""))
        if not fpath or repo_k not in cat.repo_names:
            continue
        fkey = f"{repo_k}/{fpath}"
        if fkey not in source_files_map:
            source_files_map[fkey] = [fkey, _guess_lang(fpath), 0, ""]
            edges["FILE_IN_REPO"].add((fkey, repo_k))
        tool = str(cf.get("tool", "disk"))
        cov_id = f"disk:{tool}:{fkey}"
        cov_rows.append(
            [
                cov_id,
                tool,
                float(cf.get("line_pct", 0.0)),
                float(cf.get("branch_pct", 0.0)),
                float(cf.get("mcdc_pct", 0.0)),
                int(cf.get("mutants_killed", 0)),
                int(cf.get("mutants_total", 0)),
            ]
        )
        edges["COVERS_FILE"].add((cov_id, fkey))

    # --- Phase 3 Static Code Index tables ---
    symbol_rows_map: dict[str, list[Any]] = {}
    cabi_rows_map: dict[str, list[Any]] = {}
    vec_rows_map: dict[str, list[Any]] = {}

    if code_data:
        for fobj in code_data.get("source_files", []):
            fkey = str(fobj["key"])
            repo_k = str(fobj["repo"])
            source_files_map[fkey] = [
                fkey,
                str(fobj.get("language", "unknown")),
                int(fobj.get("line_count", 0)),
                str(fobj.get("content_sha256", "")),
            ]
            if repo_k in cat.repo_names:
                edges["FILE_IN_REPO"].add((fkey, repo_k))

        for sobj in code_data.get("symbols", []):
            skey = str(sobj["key"])
            symbol_rows_map[skey] = [
                skey,
                _san_str(sobj.get("name", "")),
                str(sobj.get("kind", "function")),
                _san_str(sobj.get("file_path", "")),
                int(sobj.get("line_start", 0)),
                int(sobj.get("line_end", 0)),
                str(sobj.get("repo", "")),
                str(sobj.get("language", "")),
                1 if sobj.get("is_test") else 0,
                1 if sobj.get("is_alloc_or_stdio") else 0,
            ]
            fkey = str(sobj.get("file_key", ""))
            if fkey and fkey in source_files_map:
                edges["DEFINES_SYMBOL"].add((fkey, skey))

        for cobj in code_data.get("cabi_symbols", []):
            ckey = str(cobj["key"])
            cabi_rows_map[ckey] = [
                ckey,
                _san_str(cobj.get("header_file", "")),
                int(cobj.get("line_start", 0)),
                _san_str(cobj.get("signature", "")),
            ]

        for vobj in code_data.get("spec_vectors", []):
            vkey = str(vobj["key"])
            vec_rows_map[vkey] = [
                vkey,
                _san_str(vobj.get("category", "")),
                _san_str(vobj.get("expect_status", "")),
            ]

        for src, tgt in code_data.get("calls", []):
            if src in symbol_rows_map and tgt in symbol_rows_map:
                edges["CALLS"].add((src, tgt))
        for src, tgt in code_data.get("implements_or_extends", []):
            if src in symbol_rows_map and tgt in symbol_rows_map:
                edges["IMPLEMENTS_OR_EXTENDS"].add((src, tgt))
        for src, tgt in code_data.get("implements_cabi", []):
            if src in symbol_rows_map and tgt in cabi_rows_map:
                edges["IMPLEMENTS_CABI"].add((src, tgt))
        for src, tgt in code_data.get("calls_cabi", []):
            if src in symbol_rows_map and tgt in cabi_rows_map:
                edges["CALLS_CABI"].add((src, tgt))
        for src, op_tok in code_data.get("emits_opcode", []):
            op_key = opcode_by_name.get(op_tok) or opcode_by_hex.get(op_tok.upper()) or (op_tok if op_tok in opcode_map else None)
            if src in symbol_rows_map and op_key:
                edges["EMITS_OPCODE"].add((src, op_key))
        for src, op_tok in code_data.get("implements_opcode", []):
            op_key = opcode_by_name.get(op_tok) or opcode_by_hex.get(op_tok.upper()) or (op_tok if op_tok in opcode_map else None)
            if src in symbol_rows_map and op_key:
                edges["IMPLEMENTS_OPCODE"].add((src, op_key))
        for vkey, op_tok in code_data.get("vec_exercises_opcode", []):
            op_key = opcode_by_name.get(op_tok) or opcode_by_hex.get(op_tok.upper()) or (op_tok if op_tok in opcode_map else None)
            if vkey in vec_rows_map and op_key:
                edges["VEC_EXERCISES_OPCODE"].add((vkey, op_key))
        for sid, skey in code_data.get("covers_symbol", []):
            if sid in cat.suites and skey in symbol_rows_map:
                edges["COVERS_SYMBOL"].add((sid, skey))

    # Deduplicate cov_rows by id
    cov_rows_dedup = list({row[0]: row for row in cov_rows}.values())

    # Write domain CSVs
    domain_specs: list[tuple[str, list[str], list[list[Any]], list[dict[str, str]]]] = [
        ("Gamut", ["id", "description", "target_time_sec", "intensity"], gamut_rows, [
            {"name": "description", "column": "description", "type": "string"},
            {"name": "target_time_sec", "column": "target_time_sec", "type": "int32"},
            {"name": "intensity", "column": "intensity", "type": "string"},
        ]),
        ("Pillar", ["id", "description"], pillar_rows, [
            {"name": "description", "column": "description", "type": "string"},
        ]),
        ("Repo", ["id", "rel_path", "head_sha", "branch", "dirty", "indexed_at"], repo_rows, [
            {"name": "rel_path", "column": "rel_path", "type": "string"},
            {"name": "head_sha", "column": "head_sha", "type": "string"},
            {"name": "branch", "column": "branch", "type": "string"},
            {"name": "dirty", "column": "dirty", "type": "int32"},
            {"name": "indexed_at", "column": "indexed_at", "type": "string"},
        ]),
        ("Prereq", ["id", "kind", "hint"], prereq_rows, [
            {"name": "kind", "column": "kind", "type": "string"},
            {"name": "hint", "column": "hint", "type": "string"},
        ]),
        ("Suite", ["id", "title", "determinism", "cost"], suite_rows, [
            {"name": "title", "column": "title", "type": "string"},
            {"name": "determinism", "column": "determinism", "type": "string"},
            {"name": "cost", "column": "cost", "type": "string"},
        ]),
        ("Opcode", ["id", "name", "hex_code", "subop", "category", "active"], opcode_rows, [
            {"name": "name", "column": "name", "type": "string"},
            {"name": "hex_code", "column": "hex_code", "type": "string"},
            {"name": "subop", "column": "subop", "type": "int32"},
            {"name": "category", "column": "category", "type": "string"},
            {"name": "active", "column": "active", "type": "int32"},
        ]),
        ("Run", ["id", "timestamp", "seed", "status", "duration_ms", "pass_count", "fail_count", "skip_count"], run_rows, [
            {"name": "timestamp", "column": "timestamp", "type": "string"},
            {"name": "seed", "column": "seed", "type": "int64"},
            {"name": "status", "column": "status", "type": "string"},
            {"name": "duration_ms", "column": "duration_ms", "type": "int64"},
            {"name": "pass_count", "column": "pass_count", "type": "int32"},
            {"name": "fail_count", "column": "fail_count", "type": "int32"},
            {"name": "skip_count", "column": "skip_count", "type": "int32"},
        ]),
        ("Commit", ["id", "sha", "branch", "dirty"], list(commit_rows_map.values()), [
            {"name": "sha", "column": "sha", "type": "string"},
            {"name": "branch", "column": "branch", "type": "string"},
            {"name": "dirty", "column": "dirty", "type": "int32"},
        ]),
        ("Execution", ["id", "status", "exit_code", "duration_ms", "seed", "skip_reason"], exec_rows, [
            {"name": "status", "column": "status", "type": "string"},
            {"name": "exit_code", "column": "exit_code", "type": "int32"},
            {"name": "duration_ms", "column": "duration_ms", "type": "int64"},
            {"name": "seed", "column": "seed", "type": "int64"},
            {"name": "skip_reason", "column": "skip_reason", "type": "string"},
        ]),
        ("SourceFile", ["id", "language", "line_count", "content_sha256"], list(source_files_map.values()), [
            {"name": "language", "column": "language", "type": "string"},
            {"name": "line_count", "column": "line_count", "type": "int32"},
            {"name": "content_sha256", "column": "content_sha256", "type": "string"},
        ]),
        ("CoverageEntry", ["id", "tool", "line_pct", "branch_pct", "mcdc_pct", "mutants_killed", "mutants_total"], cov_rows_dedup, [
            {"name": "tool", "column": "tool", "type": "string"},
            {"name": "line_pct", "column": "line_pct", "type": "float32"},
            {"name": "branch_pct", "column": "branch_pct", "type": "float32"},
            {"name": "mcdc_pct", "column": "mcdc_pct", "type": "float32"},
            {"name": "mutants_killed", "column": "mutants_killed", "type": "int32"},
            {"name": "mutants_total", "column": "mutants_total", "type": "int32"},
        ]),
        ("Symbol", ["id", "name", "kind", "file_path", "line_start", "line_end", "repo", "language", "is_test", "is_alloc_or_stdio"], list(symbol_rows_map.values()), [
            {"name": "name", "column": "name", "type": "string"},
            {"name": "kind", "column": "kind", "type": "string"},
            {"name": "file_path", "column": "file_path", "type": "string"},
            {"name": "line_start", "column": "line_start", "type": "int32"},
            {"name": "line_end", "column": "line_end", "type": "int32"},
            {"name": "repo", "column": "repo", "type": "string"},
            {"name": "language", "column": "language", "type": "string"},
            {"name": "is_test", "column": "is_test", "type": "int32"},
            {"name": "is_alloc_or_stdio", "column": "is_alloc_or_stdio", "type": "int32"},
        ]),
        ("CAbiSymbol", ["id", "header_file", "line_start", "signature"], list(cabi_rows_map.values()), [
            {"name": "header_file", "column": "header_file", "type": "string"},
            {"name": "line_start", "column": "line_start", "type": "int32"},
            {"name": "signature", "column": "signature", "type": "string"},
        ]),
        ("SpecVector", ["id", "category", "expect_status"], list(vec_rows_map.values()), [
            {"name": "category", "column": "category", "type": "string"},
            {"name": "expect_status", "column": "expect_status", "type": "string"},
        ]),
    ]

    manifest_domains: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    domain_attrs_sidecar: dict[str, dict[str, dict[str, Any]]] = {}
    for dname, headers, rows, attrs in domain_specs:
        fname = f"nodes_{dname.lower()}.csv"
        _write_csv(sdir / fname, headers, rows)
        domain_counts[dname] = len(rows)
        attr_names = headers[1:]
        domain_attrs_sidecar[dname] = {
            str(row[0]): {attr_names[i]: row[i + 1] for i in range(len(attr_names))}
            for row in rows
        }
        manifest_domains.append(
            {
                "id": DOMAIN_IDS[dname],
                "name": dname,
                "key_type": "string",
                "file": fname,
                "format": "csv",
                "id_column": "id",
                "attributes": attrs,
            }
        )
    (sdir / "domain_attrs.json").write_text(json.dumps(domain_attrs_sidecar), encoding="utf-8")

    relation_specs: list[tuple[str, str, str, list[str], list[dict[str, str]]]] = [
        ("INCLUDES", "Gamut", "Suite", ["src", "tgt", "intensity"], [{"name": "intensity", "column": "intensity", "type": "string"}]),
        ("IN_PILLAR", "Suite", "Pillar", ["src", "tgt"], []),
        ("TARGETS_REPO", "Suite", "Repo", ["src", "tgt"], []),
        ("REQUIRES", "Suite", "Prereq", ["src", "tgt"], []),
        ("EXERCISES_OPCODE", "Suite", "Opcode", ["src", "tgt"], []),
        ("OF_GAMUT", "Run", "Gamut", ["src", "tgt"], []),
        ("AT_COMMIT", "Run", "Commit", ["src", "tgt"], []),
        ("COMMIT_IN_REPO", "Commit", "Repo", ["src", "tgt"], []),
        ("HAS_EXECUTION", "Run", "Execution", ["src", "tgt"], []),
        ("OF_SUITE", "Execution", "Suite", ["src", "tgt"], []),
        ("PRODUCED_COVERAGE", "Execution", "CoverageEntry", ["src", "tgt"], []),
        ("COVERS_FILE", "CoverageEntry", "SourceFile", ["src", "tgt"], []),
        ("FILE_IN_REPO", "SourceFile", "Repo", ["src", "tgt"], []),
        ("DEFINES_SYMBOL", "SourceFile", "Symbol", ["src", "tgt"], []),
        ("CALLS", "Symbol", "Symbol", ["src", "tgt"], []),
        ("IMPLEMENTS_OR_EXTENDS", "Symbol", "Symbol", ["src", "tgt"], []),
        ("IMPLEMENTS_CABI", "Symbol", "CAbiSymbol", ["src", "tgt"], []),
        ("CALLS_CABI", "Symbol", "CAbiSymbol", ["src", "tgt"], []),
        ("EMITS_OPCODE", "Symbol", "Opcode", ["src", "tgt"], []),
        ("IMPLEMENTS_OPCODE", "Symbol", "Opcode", ["src", "tgt"], []),
        ("VEC_EXERCISES_OPCODE", "SpecVector", "Opcode", ["src", "tgt"], []),
        ("COVERS_SYMBOL", "Suite", "Symbol", ["src", "tgt"], []),
    ]

    manifest_relations: list[dict[str, Any]] = []
    relation_counts: dict[str, int] = {}
    for rname, src_dom, tgt_dom, headers, attrs in relation_specs:
        fname = f"edges_{rname.lower()}.csv"
        rows = [list(t) for t in sorted(edges[rname])]
        _write_csv(sdir / fname, headers, rows)
        relation_counts[rname] = len(rows)
        entry: dict[str, Any] = {
            "name": rname,
            "src_domain": DOMAIN_IDS[src_dom],
            "tgt_domain": DOMAIN_IDS[tgt_dom],
            "encoding": "raw_uint32",
            "include_csc": True,
            "file": fname,
            "format": "csv",
            "src_column": "src",
            "tgt_column": "tgt",
        }
        if attrs:
            entry["attributes"] = attrs
        manifest_relations.append(entry)

    manifest = {
        "version": "0.10.0",
        "domains": manifest_domains,
        "relations": manifest_relations,
    }
    manifest_path = sdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    out_imps.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [str(cli), "build", "-m", str(manifest_path), "-o", str(out_imps)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return BuildResult(
            ok=False,
            imps_path=out_imps,
            validated=False,
            domain_counts=domain_counts,
            relation_counts=relation_counts,
            error=f"impulse-graph build failed (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}",
        )

    validated = False
    if validate:
        vres = subprocess.run(
            [str(cli), "snapshot", "validate", "--strict-alignment", "--deep-audit", str(out_imps)],
            capture_output=True,
            text=True,
        )
        if vres.returncode != 0:
            return BuildResult(
                ok=False,
                imps_path=out_imps,
                validated=False,
                domain_counts=domain_counts,
                relation_counts=relation_counts,
                error=f"impulse-graph snapshot validate failed: {vres.stderr.strip() or vres.stdout.strip()}",
            )
        validated = True

    return BuildResult(
        ok=True,
        imps_path=out_imps,
        validated=validated,
        domain_counts=domain_counts,
        relation_counts=relation_counts,
    )


def _guess_lang(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "cpp",
        ".hpp": "cpp",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".rs": "rust",
        ".ml": "ocaml",
        ".mli": "ocaml",
        ".py": "python",
        ".go": "go",
        ".cs": "csharp",
        ".ts": "ts",
        ".js": "js",
    }.get(ext, "unknown")


def _import_impulse_graph(ws: WorkspacePaths):
    py_dir = find_impulse_python_path(ws)
    if py_dir and str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))
    import impulse_graph  # type: ignore[import-not-found]
    return impulse_graph


class VerificationGraph:
    """Native zero-copy CSR/CSC query wrapper over datasets/verification.imps via impulse_graph.Snapshot."""

    def __init__(self, snap: Any, attrs_by_domain: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.snap = snap
        self.attrs = attrs_by_domain
        self.domains: dict[str, dict[str, Any]] = {}
        self.relations: dict[str, int] = {}
        for i in range(snap.domain_count()):
            d = snap.get_domain(i)
            self.domains[d["name"]] = d
        for i in range(snap.relation_count()):
            r = snap.get_relation(i)
            self.relations[r["name"]] = i

    @classmethod
    def open(cls, ws: WorkspacePaths) -> "VerificationGraph":
        path = imps_snapshot_path(ws)
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot not found at {path}. Run 'verify.py sync' or 'verify.py index'.")
        ig = _import_impulse_graph(ws)
        sidecar_path = staging_dir(ws) / "domain_attrs.json"
        attrs: dict[str, dict[str, dict[str, Any]]] = {}
        if sidecar_path.is_file():
            attrs = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return cls(ig.Snapshot(str(path)), attrs)

    def close(self) -> None:
        self.snap.close()

    def __enter__(self) -> "VerificationGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _dom_id(self, name: str) -> int:
        return int(self.domains[name]["domain_id"])

    def _dom_count(self, name: str) -> int:
        return int(self.domains[name]["node_count"])

    def _key(self, dom: str, nid: int) -> str:
        raw = self.snap.resolve_key(self._dom_id(dom), int(nid))
        return raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)

    def _node_attrs(self, dom: str, nid: int) -> dict[str, Any]:
        return self.attrs.get(dom, {}).get(self._key(dom, nid), {})

    def _neighbors_out(self, rel_name: str, nid: int) -> list[int]:
        ridx = self.relations[rel_name]
        offs = self.snap.get_row_offsets_array(ridx)
        cols = self.snap.get_col_indices_array(ridx)
        if nid < 0 or nid + 1 >= len(offs):
            return []
        return [int(x) for x in cols[int(offs[nid]) : int(offs[nid + 1])]]

    def _neighbors_in(self, rel_name: str, nid: int) -> list[int]:
        ridx = self.relations[rel_name]
        offs = self.snap.get_csc_row_offsets_array(ridx)
        cols = self.snap.get_csc_col_indices_array(ridx)
        if nid < 0 or nid + 1 >= len(offs):
            return []
        return [int(x) for x in cols[int(offs[nid]) : int(offs[nid + 1])]]

    def _symbol_info(self, nid: int) -> dict[str, Any]:
        key = self._key("Symbol", nid)
        a = self.attrs.get("Symbol", {}).get(key, {})
        return {
            "key": key,
            "name": str(a.get("name", "")),
            "kind": str(a.get("kind", "function")),
            "file_path": str(a.get("file_path", "")),
            "line_start": int(a.get("line_start", 0)),
            "line_end": int(a.get("line_end", 0)),
            "repo": str(a.get("repo", "")),
            "language": str(a.get("language", "")),
            "is_test": bool(int(a.get("is_test", 0))),
            "is_alloc_or_stdio": bool(int(a.get("is_alloc_or_stdio", 0))),
        }

    def check_freshness(self, ws: WorkspacePaths) -> dict[str, Any]:
        """Compare embedded Repo.{head_sha, dirty} against live git HEADs."""
        live = collect_repo_git_states(ws)
        stale: list[dict[str, Any]] = []
        indexed_at = ""
        for nid in range(self._dom_count("Repo")):
            rname = self._key("Repo", nid)
            a = self.attrs.get("Repo", {}).get(rname, {})
            rec_sha = str(a.get("head_sha", ""))
            rec_dirty = bool(int(a.get("dirty", 0)))
            indexed_at = str(a.get("indexed_at", "")) or indexed_at
            ls = live.get(rname)
            if not ls or not ls.exists:
                continue
            if rec_sha != ls.head_sha or rec_dirty != ls.dirty:
                stale.append(
                    {
                        "repo": rname,
                        "indexed_sha": rec_sha[:10],
                        "current_sha": ls.short_sha,
                        "indexed_dirty": rec_dirty,
                        "current_dirty": ls.dirty,
                    }
                )
        return {"fresh": len(stale) == 0, "indexed_at": indexed_at, "stale_repos": stale}

    def query_gaps(
        self,
        *,
        min_branch: float = 80.0,
        min_mcdc: float = 80.0,
        min_mutation: float = 70.0,
    ) -> dict[str, Any]:
        uncovered_opcodes: list[dict[str, Any]] = []
        no_fuzzer_opcodes: list[dict[str, Any]] = []
        for oid in range(self._dom_count("Opcode")):
            a = self._node_attrs("Opcode", oid)
            if not int(a.get("active", 1)):
                continue
            op_name = str(a.get("name", ""))
            op_hex = str(a.get("hex_code", ""))
            suites = [self._key("Suite", sid) for sid in self._neighbors_in("EXERCISES_OPCODE", oid)]
            vecs = [self._key("SpecVector", vid) for vid in self._neighbors_in("VEC_EXERCISES_OPCODE", oid)]
            if not suites and not vecs:
                uncovered_opcodes.append({"opcode": op_name, "hex": op_hex})
            elif not any(s.startswith("fuzz.") for s in suites):
                no_fuzzer_opcodes.append({"opcode": op_name, "hex": op_hex, "vector_count": len(vecs)})

        low_coverage: list[dict[str, Any]] = []
        for cid in range(self._dom_count("CoverageEntry")):
            a = self._node_attrs("CoverageEntry", cid)
            tool = str(a.get("tool", ""))
            fids = self._neighbors_out("COVERS_FILE", cid)
            fkey = self._key("SourceFile", fids[0]) if fids else self._key("CoverageEntry", cid)
            br = float(a.get("branch_pct", 0.0))
            mcdc = float(a.get("mcdc_pct", 0.0))
            mk = int(a.get("mutants_killed", 0))
            mt = int(a.get("mutants_total", 0))
            mut_pct = (100.0 * mk / mt) if mt > 0 else 100.0
            flagged = False
            if tool in ("jacoco", "llvm-cov") and br < min_branch:
                flagged = True
            if tool == "llvm-cov" and mcdc > 0 and mcdc < min_mcdc:
                flagged = True
            if tool == "pitest" and mt > 0 and mut_pct < min_mutation:
                flagged = True
            if flagged:
                low_coverage.append(
                    {
                        "file": fkey,
                        "tool": tool,
                        "line_pct": round(float(a.get("line_pct", 0.0)), 2),
                        "branch_pct": round(br, 2),
                        "mcdc_pct": round(mcdc, 2),
                        "mutation_pct": round(mut_pct, 2) if mt > 0 else None,
                        "mutants": f"{mk}/{mt}" if mt > 0 else None,
                    }
                )
        return {
            "uncovered_opcodes": uncovered_opcodes,
            "no_fuzzer_opcodes": no_fuzzer_opcodes[:50],
            "no_fuzzer_total": len(no_fuzzer_opcodes),
            "low_coverage_files": sorted(low_coverage, key=lambda x: x["branch_pct"])[:50],
        }

    def query_opcode(self, query: str) -> dict[str, Any]:
        q_up = query.strip().upper()
        matched_oid: int | None = None
        for oid in range(self._dom_count("Opcode")):
            key = self._key("Opcode", oid)
            a = self.attrs.get("Opcode", {}).get(key, {})
            name = str(a.get("name", "")).upper()
            hx = str(a.get("hex_code", "")).upper()
            if q_up in (name, hx, key.upper()) or (not q_up.startswith("OP_") and f"OP_{q_up}" == name):
                matched_oid = oid
                break
        if matched_oid is None:
            return {"found": False, "query": query}

        a = self._node_attrs("Opcode", matched_oid)
        emitters = [self._symbol_info(sid) for sid in self._neighbors_in("EMITS_OPCODE", matched_oid)]
        handlers = [self._symbol_info(sid) for sid in self._neighbors_in("IMPLEMENTS_OPCODE", matched_oid)]
        vectors = []
        for vid in self._neighbors_in("VEC_EXERCISES_OPCODE", matched_oid):
            va = self._node_attrs("SpecVector", vid)
            vectors.append({
                "path": self._key("SpecVector", vid),
                "category": str(va.get("category", "")),
                "expect_status": str(va.get("expect_status", "")),
            })
        suites = [self._key("Suite", sid) for sid in self._neighbors_in("EXERCISES_OPCODE", matched_oid)]
        return {
            "found": True,
            "opcode": str(a.get("name", "")),
            "hex": str(a.get("hex_code", "")),
            "category": str(a.get("category", "")),
            "emitters": emitters,
            "handlers": handlers,
            "spec_vectors": vectors,
            "suites": suites,
        }

    def _find_symbols(self, query: str) -> list[int]:
        q_lo = query.strip().lower()
        exact: list[int] = []
        partial: list[int] = []
        for sid in range(self._dom_count("Symbol")):
            key = self._key("Symbol", sid)
            name = str(self.attrs.get("Symbol", {}).get(key, {}).get("name", ""))
            if key.lower() == q_lo or name.lower() == q_lo:
                exact.append(sid)
            elif q_lo in name.lower() or q_lo in key.lower():
                partial.append(sid)
        return exact if exact else partial[:20]

    def query_call_chain(self, query: str, *, direction: str = "callers", depth: int = 3) -> dict[str, Any]:
        seeds = self._find_symbols(query)
        if not seeds:
            return {"found": False, "query": query, "direction": direction, "nodes": []}
        visited: dict[int, int] = {s: 0 for s in seeds}
        q: deque[tuple[int, int]] = deque((s, 0) for s in seeds)
        edges_out: list[dict[str, str]] = []
        while q:
            cur, d = q.popleft()
            if d >= depth:
                continue
            nbrs = self._neighbors_in("CALLS", cur) if direction == "callers" else self._neighbors_out("CALLS", cur)
            for nxt in nbrs:
                edges_out.append({"from": self._key("Symbol", nxt if direction == "callers" else cur), "to": self._key("Symbol", cur if direction == "callers" else nxt)})
                if nxt not in visited:
                    visited[nxt] = d + 1
                    q.append((nxt, d + 1))
        nodes = [{"depth": d, **self._symbol_info(sid)} for sid, d in sorted(visited.items(), key=lambda kv: (kv[1], kv[0]))]
        return {"found": True, "query": query, "direction": direction, "depth": depth, "nodes": nodes, "edges": edges_out[:200]}

    def query_where(self, query: str) -> dict[str, Any]:
        q_lo = query.strip().lower()
        cabi_matches: list[dict[str, Any]] = []
        for cid in range(self._dom_count("CAbiSymbol")):
            ckey = self._key("CAbiSymbol", cid)
            cname = ckey.removeprefix("cabi:")
            if q_lo == cname.lower() or q_lo in cname.lower():
                ca = self.attrs.get("CAbiSymbol", {}).get(ckey, {})
                impls = [self._symbol_info(sid) for sid in self._neighbors_in("IMPLEMENTS_CABI", cid)]
                callers = [self._symbol_info(sid) for sid in self._neighbors_in("CALLS_CABI", cid)]
                cabi_matches.append(
                    {
                        "symbol": cname,
                        "header": str(ca.get("header_file", "")),
                        "line": int(ca.get("line_start", 0)),
                        "signature": str(ca.get("signature", "")),
                        "implementations": impls,
                        "call_sites": callers,
                    }
                )
        sym_matches = [self._symbol_info(sid) for sid in self._find_symbols(query)]
        return {"query": query, "cabi": cabi_matches, "symbols": sym_matches}

    def query_impacted(self, changed_files: list[str], cat: Catalog) -> dict[str, Any]:
        norm_targets = [f.replace("\\", "/").lstrip("./") for f in changed_files]
        matched_fids: list[int] = []
        matched_repos: set[str] = set()
        for fid in range(self._dom_count("SourceFile")):
            fkey = self._key("SourceFile", fid)
            if any(fkey == t or fkey.endswith("/" + t) or t.endswith("/" + fkey) for t in norm_targets):
                matched_fids.append(fid)
                for rid in self._neighbors_out("FILE_IN_REPO", fid):
                    matched_repos.add(self._key("Repo", rid))

        seed_syms: set[int] = set()
        for fid in matched_fids:
            seed_syms.update(self._neighbors_out("DEFINES_SYMBOL", fid))

        reached_syms: set[int] = set(seed_syms)
        q: deque[tuple[int, int]] = deque((s, 0) for s in seed_syms)
        while q:
            cur, d = q.popleft()
            if d >= 3:
                continue
            for caller in self._neighbors_in("CALLS", cur):
                if caller not in reached_syms:
                    reached_syms.add(caller)
                    q.append((caller, d + 1))

        impacted_opcodes: set[int] = set()
        impacted_suites: set[str] = set()
        for sid in reached_syms:
            impacted_opcodes.update(self._neighbors_out("IMPLEMENTS_OPCODE", sid))
            impacted_opcodes.update(self._neighbors_out("EMITS_OPCODE", sid))
            for suite_nid in self._neighbors_in("COVERS_SYMBOL", sid):
                impacted_suites.add(self._key("Suite", suite_nid))

        for oid in impacted_opcodes:
            for suite_nid in self._neighbors_in("EXERCISES_OPCODE", oid):
                impacted_suites.add(self._key("Suite", suite_nid))

        for rname in matched_repos:
            for s in cat.suites.values():
                if s.repo == rname and s.pillar in ("lint", "unit", "parity"):
                    impacted_suites.add(s.id)

        cost_rank = {"fast": 0, "medium": 1, "slow": 2, "soak": 3}
        ordered_suites = sorted(
            impacted_suites,
            key=lambda sid: (cost_rank.get(cat.suites[sid].cost, 9), sid) if sid in cat.suites else (9, sid),
        )
        return {
            "changed_files": changed_files,
            "matched_source_files": [self._key("SourceFile", fid) for fid in matched_fids],
            "impacted_symbol_count": len(reached_syms),
            "impacted_opcodes": sorted(str(self._node_attrs("Opcode", oid).get("name", "")) for oid in impacted_opcodes),
            "recommended_suites": ordered_suites,
        }

    def query_audit_invariants(self) -> dict[str, Any]:
        """Transitive call-graph invariant audit: hot-path -> alloc/stdio & 3-engine opcode symmetry."""
        hot_roots = [
            sid
            for sid in range(self._dom_count("Symbol"))
            if any(
                tok in str(self._node_attrs("Symbol", sid).get("name", "")).lower()
                for tok in ("impulse_stmt_execute", "impulse_exec", "impulsevm.execute")
            )
        ]
        alloc_violations: list[dict[str, Any]] = []
        for root in hot_roots:
            visited: set[int] = {root}
            q: deque[tuple[int, list[int]]] = deque([(root, [root])])
            while q:
                cur, path = q.popleft()
                if len(path) > 6:
                    continue
                for nxt in self._neighbors_out("CALLS", cur):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    info = self._symbol_info(nxt)
                    if info["is_alloc_or_stdio"] and not info["is_test"]:
                        alloc_violations.append(
                            {
                                "root": self._key("Symbol", root),
                                "target": info["key"],
                                "location": f"{info['file_path']}:{info['line_start']}",
                                "hops": len(path),
                            }
                        )
                    else:
                        q.append((nxt, path + [nxt]))

        asymmetric_opcodes: list[dict[str, Any]] = []
        for oid in range(self._dom_count("Opcode")):
            a = self._node_attrs("Opcode", oid)
            if not int(a.get("active", 1)):
                continue
            handler_repos = {str(self._node_attrs("Symbol", sid).get("repo", "")) for sid in self._neighbors_in("IMPLEMENTS_OPCODE", oid)}
            missing_engines = sorted({"core", "java", "ocaml"} - handler_repos)
            if missing_engines:
                asymmetric_opcodes.append(
                    {
                        "opcode": str(a.get("name", "")),
                        "hex": str(a.get("hex_code", "")),
                        "present_in": sorted(handler_repos),
                        "missing_in": missing_engines,
                    }
                )
        return {
            "hot_roots_checked": len(hot_roots),
            "transitive_alloc_violations": alloc_violations[:50],
            "asymmetric_opcode_count": len(asymmetric_opcodes),
            "asymmetric_opcodes_sample": asymmetric_opcodes[:30],
        }
