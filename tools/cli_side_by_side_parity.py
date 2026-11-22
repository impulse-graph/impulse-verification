#!/usr/bin/env python3
"""
Side-by-Side CLI Differential Parity Suite: Rust `impulse-graph` vs. Java `impulse-j`.

Verifies cross-compatibility, schema/metric parity, bitwise `.impb` assembly identity,
cross-disassembly parity, VM execution parity, and negative error handling across
both CLI implementations.

Exit code:
  0  - 100% of executed parity checks passed (excluding tracked XFAILs unless --strict-xfail)
  1  - One or more parity checks failed
  2  - Environment / CLI binary discovery error
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Binary Discovery & Process Runner
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_JAVA_25_HOMES = [
    "/opt/homebrew/Cellar/openjdk@25/25.0.5.0.3/libexec/openjdk.jdk/Contents/Home",
    "/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home",
    "/Library/Java/JavaVirtualMachines/openjdk-25.jdk/Contents/Home",
]


def _pick_newest(paths: Sequence[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.is_file() and os.access(p, os.X_OK)]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def discover_rust_cli(override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Rust CLI not found at --rust-cli: {p}")
        return p
    env_val = os.environ.get("IMPULSE_RUST_CLI")
    if env_val:
        p = Path(env_val).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Rust CLI not found at IMPULSE_RUST_CLI: {p}")
        return p

    search_roots = [
        REPO_ROOT.parent / "impulse-graph-core" / "impulse-cli",
        Path("/Users/jesse/impulse/impulse-graph-core/impulse-cli"),
    ]
    candidates: List[Path] = []
    for root in search_roots:
        candidates.append(root / "target" / "release" / "impulse-graph")
        candidates.append(root / "target" / "debug" / "impulse-graph")
    picked = _pick_newest(candidates)
    if picked:
        return picked
    which = shutil.which("impulse-graph")
    if which:
        return Path(which)
    raise FileNotFoundError(
        "Could not locate Rust `impulse-graph` CLI. Pass --rust-cli or set IMPULSE_RUST_CLI."
    )


def discover_java_cli(override: Optional[str]) -> List[str]:
    """Return command prefix to invoke Java `impulse-j` (script or `java -jar ...`)."""
    raw = override or os.environ.get("IMPULSE_JAVA_CLI")
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Java CLI not found: {p}")
        if p.suffix == ".jar":
            return _java_jar_cmd(p)
        return [str(p)]

    search_roots = [
        REPO_ROOT.parent / "impulse-graph-java" / "impulse-cli",
        Path("/Users/jesse/impulse/impulse-graph-java/impulse-cli"),
    ]
    script_candidates = [r / "src" / "main" / "scripts" / "impulse-j" for r in search_roots]
    picked = _pick_newest(script_candidates)
    if picked:
        return [str(picked)]

    for r in search_roots:
        target = r / "target"
        if target.is_dir():
            jars = sorted(
                target.glob("impulse-cli-*-all.jar"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if jars:
                return _java_jar_cmd(jars[0])

    which = shutil.which("impulse-j")
    if which:
        return [which]
    raise FileNotFoundError(
        "Could not locate Java `impulse-j` CLI. Pass --java-cli or set IMPULSE_JAVA_CLI."
    )


def _resolve_java_home() -> Optional[str]:
    env_jh = os.environ.get("JAVA_HOME")
    if env_jh and Path(env_jh).is_dir():
        return env_jh
    for candidate in DEFAULT_JAVA_25_HOMES:
        if Path(candidate).is_dir():
            return candidate
    return None


def _java_jar_cmd(jar_path: Path) -> List[str]:
    jh = _resolve_java_home()
    java_bin = str(Path(jh) / "bin" / "java") if jh else "java"
    return [
        java_bin,
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules",
        "jdk.incubator.vector",
        "-jar",
        str(jar_path),
    ]


@dataclasses.dataclass
class CmdResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


class CliRunner:
    def __init__(self, rust_cli: Path, java_cmd: List[str], verbose: bool = False):
        self.rust_cmd = [str(rust_cli)]
        self.java_cmd = list(java_cmd)
        self.verbose = verbose
        self.env = os.environ.copy()
        jh = _resolve_java_home()
        if jh:
            self.env["JAVA_HOME"] = jh

    def run_rust(self, args: Sequence[str], timeout: int = 60) -> CmdResult:
        cmd = self.rust_cmd + list(args)
        p = subprocess.run(cmd, capture_output=True, text=True, env=self.env, timeout=timeout)
        return CmdResult(cmd=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

    def run_java(self, args: Sequence[str], timeout: int = 60) -> CmdResult:
        cmd = self.java_cmd + list(args)
        p = subprocess.run(cmd, capture_output=True, text=True, env=self.env, timeout=timeout)
        return CmdResult(cmd=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


# ---------------------------------------------------------------------------
# Test Accounting (PASS / FAIL / XFAIL / XPASS)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CheckResult:
    category: str
    name: str
    passed: bool
    status: str  # PASS | FAIL | XFAIL | XPASS
    detail: str = ""
    diff: str = ""
    known_core_issue: Optional[str] = None


class SuiteRecorder:
    def __init__(self, verbose: bool = False, strict_xfail: bool = False):
        self.verbose = verbose
        self.strict_xfail = strict_xfail
        self.results: List[CheckResult] = []

    def record(
        self,
        category: str,
        name: str,
        passed: bool,
        detail: str = "",
        diff: str = "",
        known_core_issue: Optional[str] = None,
    ) -> bool:
        if known_core_issue and not self.strict_xfail:
            status = "XPASS" if passed else "XFAIL"
            tag = f"XPASS {known_core_issue}" if passed else f"XFAIL {known_core_issue}"
        else:
            status = "PASS" if passed else "FAIL"
            tag = status

        res = CheckResult(
            category=category,
            name=name,
            passed=passed,
            status=status,
            detail=detail,
            diff=diff,
            known_core_issue=known_core_issue,
        )
        self.results.append(res)
        line = f"  [{tag}] {name}"
        if detail and (self.verbose or status in ("FAIL", "XFAIL", "XPASS")):
            line += f" — {detail}"
        print(line)
        if diff and (self.verbose or status == "FAIL"):
            for dl in diff.splitlines():
                print(f"         {dl}")
        return passed


# ---------------------------------------------------------------------------
# Shared Fixtures & Normalization Helpers
# ---------------------------------------------------------------------------

def create_multi_domain_fixture(base_dir: Path) -> Path:
    """Create a 2-domain, 2-relation (CSC=true & CSC=false) CSV + manifest.json fixture.
    User node 5 has 0 outgoing LIKES edges (used for empty-result traversal parity).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "users.csv").write_text(
        "id,age,score\n"
        "0,25,1.5\n"
        "1,30,2.5\n"
        "2,35,3.5\n"
        "3,40,4.5\n"
        "4,45,5.5\n"
        "5,50,6.5\n",
        encoding="utf-8",
    )
    (base_dir / "items.csv").write_text(
        "id,price\n"
        "0,10.0\n"
        "1,20.0\n"
        "2,30.0\n"
        "3,40.0\n"
        "4,50.0\n",
        encoding="utf-8",
    )
    (base_dir / "follows.csv").write_text(
        "src,tgt,weight\n"
        "0,1,1.0\n"
        "0,2,2.0\n"
        "1,2,3.0\n"
        "1,3,3.5\n"
        "2,3,4.0\n"
        "3,4,5.0\n"
        "4,5,6.0\n"
        "5,0,7.0\n",
        encoding="utf-8",
    )
    (base_dir / "likes.csv").write_text(
        "src,tgt,rating\n"
        "0,0,5\n"
        "0,1,4\n"
        "1,2,3\n"
        "2,3,5\n"
        "3,0,2\n"
        "4,4,5\n",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.10.0",
        "domains": [
            {
                "id": 0,
                "name": "User",
                "key_type": "uint32",
                "file": "users.csv",
                "format": "csv",
                "id_column": "id",
                "attributes": [
                    {"name": "age", "column": "age", "type": "int32"},
                    {"name": "score", "column": "score", "type": "float32"},
                ],
            },
            {
                "id": 1,
                "name": "Item",
                "key_type": "uint32",
                "file": "items.csv",
                "format": "csv",
                "id_column": "id",
                "attributes": [
                    {"name": "price", "column": "price", "type": "float32"},
                ],
            },
        ],
        "relations": [
            {
                "name": "FOLLOWS",
                "src_domain": 0,
                "tgt_domain": 0,
                "encoding": "raw_uint32",
                "include_csc": True,
                "file": "follows.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "weight", "column": "weight", "type": "float32"},
                ],
            },
            {
                "name": "LIKES",
                "src_domain": 0,
                "tgt_domain": 1,
                "encoding": "raw_uint32",
                "include_csc": False,
                "file": "likes.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "rating", "column": "rating", "type": "int32"},
                ],
            },
        ],
    }
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def create_three_domain_nullable_fixture(base_dir: Path) -> Path:
    """Create a 3-domain / 3-relation graph with nullable attributes (empty cells)
    and an isolated zero-degree node (User 3 & Org 2 have 0 edges).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "orgs.csv").write_text(
        "id,employees\n"
        "0,100\n"
        "1,\n"
        "2,50\n",
        encoding="utf-8",
    )
    (base_dir / "users.csv").write_text(
        "id,age,score\n"
        "0,25,1.5\n"
        "1,,2.5\n"
        "2,35,\n"
        "3,40,4.5\n",
        encoding="utf-8",
    )
    (base_dir / "repos.csv").write_text(
        "id,stars\n"
        "0,10\n"
        "1,20\n"
        "2,\n",
        encoding="utf-8",
    )
    (base_dir / "employs.csv").write_text(
        "src,tgt\n"
        "0,0\n"
        "0,1\n"
        "1,2\n",
        encoding="utf-8",
    )
    (base_dir / "follows.csv").write_text(
        "src,tgt\n"
        "0,1\n"
        "1,2\n"
        "2,0\n",
        encoding="utf-8",
    )
    (base_dir / "contribs.csv").write_text(
        "src,tgt,commits\n"
        "0,0,5\n"
        "1,1,\n"
        "2,2,12\n",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.10.0",
        "domains": [
            {
                "id": 0,
                "name": "Org",
                "key_type": "uint32",
                "file": "orgs.csv",
                "format": "csv",
                "id_column": "id",
                "attributes": [
                    {"name": "employees", "column": "employees", "type": "int32"},
                ],
            },
            {
                "id": 1,
                "name": "User",
                "key_type": "uint32",
                "file": "users.csv",
                "format": "csv",
                "id_column": "id",
                "attributes": [
                    {"name": "age", "column": "age", "type": "int32"},
                    {"name": "score", "column": "score", "type": "float32"},
                ],
            },
            {
                "id": 2,
                "name": "Repo",
                "key_type": "uint32",
                "file": "repos.csv",
                "format": "csv",
                "id_column": "id",
                "attributes": [
                    {"name": "stars", "column": "stars", "type": "int32"},
                ],
            },
        ],
        "relations": [
            {
                "name": "EMPLOYS",
                "src_domain": 0,
                "tgt_domain": 1,
                "encoding": "raw_uint32",
                "include_csc": True,
                "file": "employs.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
            {
                "name": "FOLLOWS",
                "src_domain": 1,
                "tgt_domain": 1,
                "encoding": "raw_uint32",
                "include_csc": True,
                "file": "follows.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
            {
                "name": "CONTRIBUTES",
                "src_domain": 1,
                "tgt_domain": 2,
                "encoding": "raw_uint32",
                "include_csc": False,
                "file": "contribs.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "commits", "column": "commits", "type": "int32"},
                ],
            },
        ],
    }
    manifest_path = base_dir / "manifest3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


SYNTHETIC_TOPOLOGIES: List[Tuple[str, List[str]]] = [
    ("social_100", ["--profile", "social", "-n", "100", "--include-csc", "--seed", "42"]),
    ("graph500_s7", ["--profile", "graph500", "--scale", "7", "--edge-factor", "4", "--include-csc", "--seed", "42"]),
    ("grid_6x6", ["--profile", "grid", "--dim-x", "6", "--dim-y", "6", "--include-csc", "--seed", "42"]),
    ("bipartite_30x20", ["--profile", "bipartite", "--src-nodes", "30", "--tgt-nodes", "20", "--edges", "90", "--include-csc", "--seed", "42"]),
]


def _parse_json_output(res: CmdResult) -> Tuple[Optional[Dict[str, Any]], str]:
    if res.returncode != 0:
        return None, f"exit={res.returncode} stderr={res.stderr.strip()}"
    txt = res.stdout.strip()
    if not txt:
        return None, f"empty stdout (stderr={res.stderr.strip()})"
    try:
        return json.loads(txt), ""
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e} (stdout={txt[:200]!r})"


def normalize_inspect_json(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Rust and Java `inspect --format json` payloads to a common schema."""
    hdr = doc.get("header", {})
    version_raw = str(hdr.get("version", ""))
    version_norm = version_raw.split()[0] if version_raw else ""

    domains = []
    for d in doc.get("domains", []):
        domains.append(
            {
                "name": d.get("name"),
                "node_count": d.get("node_count", d.get("nodeCount")),
            }
        )
    domains.sort(key=lambda x: str(x["name"]))

    relations = []
    for r in doc.get("relations", []):
        relations.append(
            {
                "name": r.get("name"),
                "node_count": r.get("node_count", r.get("nodeCount")),
                "edge_count": r.get("edge_count", r.get("edgeCount")),
                "has_csc": bool(r.get("has_csc", r.get("hasCsc"))),
            }
        )
    relations.sort(key=lambda x: str(x["name"]))

    return {
        "magic": hdr.get("magic"),
        "version": version_norm,
        "domain_count": hdr.get("domain_count", hdr.get("domainCount")),
        "relation_count": hdr.get("relation_count", hdr.get("relationCount")),
        "domains": domains,
        "relations": relations,
    }


def normalize_stats_json(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize Rust and Java `stats --format json` payloads by relation name."""
    cbo = doc.get("cbo_sketches") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for idx, r in enumerate(doc.get("relations", [])):
        name = r.get("name")
        rel_id = r.get("relation_id", idx)
        min_deg = r.get("minOutDegree", r.get("min_out_degree"))
        if min_deg is None:
            sketch_key = f"impulse.stats.relation.{rel_id}.out_degree"
            if sketch_key in cbo:
                try:
                    min_deg = json.loads(cbo[sketch_key]).get("min")
                except Exception:
                    min_deg = None
        out[str(name)] = {
            "node_count": r.get("node_count", r.get("nodeCount")),
            "edge_count": r.get("edge_count", r.get("edgeCount")),
            "min_out_degree": min_deg,
            "max_out_degree": r.get("max_out_degree", r.get("maxOutDegree")),
            "avg_out_degree": r.get("avg_out_degree", r.get("avgOutDegree")),
            "supernode_count": r.get("supernode_count", r.get("supernodeCount")),
        }
    return out


def compare_stats_dicts(rn: Dict[str, Dict[str, Any]], jn: Dict[str, Dict[str, Any]]) -> List[str]:
    diffs: List[str] = []
    if set(rn.keys()) != set(jn.keys()):
        diffs.append(f"relation keys differ: rust={sorted(rn.keys())} java={sorted(jn.keys())}")
        return diffs
    for rel_name in sorted(rn.keys()):
        rv, jv = rn[rel_name], jn[rel_name]
        for int_field in ("node_count", "edge_count", "min_out_degree", "max_out_degree", "supernode_count"):
            if rv[int_field] != jv[int_field]:
                diffs.append(f"{rel_name}.{int_field}: rust={rv[int_field]} != java={jv[int_field]}")
        ra, ja = rv["avg_out_degree"], jv["avg_out_degree"]
        if ra is None or ja is None or abs(float(ra) - float(ja)) > 1e-4:
            diffs.append(f"{rel_name}.avg_out_degree: rust={ra} != java={ja}")
    return diffs


def extract_disasm_opcodes(text: str) -> List[str]:
    """Extract ordered OP_* mnemonics from Rust or Java disassembly output."""
    ops: List[str] = []
    for line in text.splitlines():
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        m = re.search(r"\b(OP_[A-Z0-9_]+)\b", code)
        if m:
            ops.append(m.group(1))
    return ops


def parse_rust_run_cardinality(stdout: str) -> Tuple[Optional[int], Optional[List[int]]]:
    card = None
    nodes = None
    m = re.search(r"Cardinality:\s*(\d+)\s*/\s*\d+", stdout)
    if m:
        card = int(m.group(1))
    m2 = re.search(r"Active Nodes \(first \d+\):\s*\[([^\]]*)\]", stdout)
    if m2:
        raw = m2.group(1).strip()
        nodes = [int(x.strip()) for x in raw.split(",") if x.strip()] if raw else []
    elif card == 0:
        nodes = []
    return card, nodes


def parse_rust_run_scalar(stdout: str) -> Optional[int]:
    m = re.search(r"Scalar Return Value:\s*(-?\d+)", stdout)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Category 1: `build` / `snapshot build` Bidirectional Cross-Compatibility
# ---------------------------------------------------------------------------

def _run_bidirectional_build_pair(
    runner: CliRunner,
    rec: SuiteRecorder,
    cat: str,
    prefix: str,
    manifest: Path,
    rust_imps: Path,
    java_imps: Path,
) -> None:
    r_build = runner.run_rust(["build", "-m", str(manifest), "-o", str(rust_imps)])
    rec.record(
        cat,
        f"{prefix}_rust_build",
        r_build.returncode == 0 and rust_imps.is_file(),
        f"rc={r_build.returncode} size={rust_imps.stat().st_size if rust_imps.is_file() else 0}",
        r_build.stderr.strip() if r_build.returncode != 0 else "",
    )

    j_build = runner.run_java(["build", "-m", str(manifest), "-o", str(java_imps)])
    rec.record(
        cat,
        f"{prefix}_java_build",
        j_build.returncode == 0 and java_imps.is_file(),
        f"rc={j_build.returncode} size={java_imps.stat().st_size if java_imps.is_file() else 0}",
        j_build.stderr.strip() if j_build.returncode != 0 else "",
    )

    if rust_imps.is_file():
        jv = runner.run_java(["snapshot", "validate", "--strict-alignment", "--deep-audit", str(rust_imps)])
        rec.record(
            cat,
            f"{prefix}_java_validates_rust_imps",
            jv.returncode == 0,
            f"rc={jv.returncode}",
            f"stdout={jv.stdout.strip()}\nstderr={jv.stderr.strip()}" if jv.returncode != 0 else "",
        )

    if java_imps.is_file():
        rv = runner.run_rust(["snapshot", "validate", "--strict-alignment", "--deep-audit", str(java_imps)])
        rec.record(
            cat,
            f"{prefix}_rust_validates_java_imps",
            rv.returncode == 0,
            f"rc={rv.returncode}",
            f"stdout={rv.stdout.strip()}\nstderr={rv.stderr.strip()}" if rv.returncode != 0 else "",
        )

    for label, snap in [("rust.imps", rust_imps), ("java.imps", java_imps)]:
        if not snap.is_file():
            continue
        r_doc, r_err = _parse_json_output(runner.run_rust(["inspect", "--format", "json", str(snap)]))
        j_doc, j_err = _parse_json_output(runner.run_java(["inspect", "--format", "json", str(snap)]))
        if r_doc is None or j_doc is None:
            rec.record(
                cat,
                f"{prefix}_cross_inspect_{label}",
                False,
                f"parse failure: rust=({r_err}) java=({j_err})",
            )
            continue
        rn = normalize_inspect_json(r_doc)
        jn = normalize_inspect_json(j_doc)
        ok = rn == jn
        rec.record(
            cat,
            f"{prefix}_cross_inspect_{label}",
            ok,
            f"domains={rn['domain_count']} relations={rn['relation_count']}",
            "" if ok else f"RUST: {json.dumps(rn)}\nJAVA: {json.dumps(jn)}",
        )


def run_category_1_build(
    runner: CliRunner, rec: SuiteRecorder, workdir: Path
) -> Dict[str, Path]:
    cat = "cat1_build"
    print("\n=== Category 1: build / snapshot build Bidirectional Cross-Compatibility ===")

    # 1A: 2-domain / 2-relation fixture (CSC=true & CSC=false)
    manifest2 = create_multi_domain_fixture(workdir / "fixture2")
    rust_imps = workdir / "rust.imps"
    java_imps = workdir / "java.imps"
    _run_bidirectional_build_pair(runner, rec, cat, "2dom", manifest2, rust_imps, java_imps)

    # 1B: 3-domain / 3-relation fixture with nullable attributes & isolated zero-degree node
    manifest3 = create_three_domain_nullable_fixture(workdir / "fixture3")
    rust3_imps = workdir / "rust3.imps"
    java3_imps = workdir / "java3.imps"
    _run_bidirectional_build_pair(runner, rec, cat, "3dom_nullable", manifest3, rust3_imps, java3_imps)

    return {
        "manifest": manifest2,
        "rust_imps": rust_imps,
        "java_imps": java_imps,
        "rust3_imps": rust3_imps,
        "java3_imps": java3_imps,
    }


def ensure_synthetic_topologies(
    runner: CliRunner, workdir: Path, artifacts: Dict[str, Path], rec: Optional[SuiteRecorder] = None
) -> None:
    for topo_name, gen_args in SYNTHETIC_TOPOLOGIES:
        key = f"topo_{topo_name}"
        if key in artifacts and artifacts[key].is_file():
            continue
        out_path = workdir / f"{topo_name}.imps"
        g = runner.run_rust(["generate"] + gen_args + ["-o", str(out_path)])
        if rec is not None:
            rec.record(
                "cat2_inspect",
                f"rust_generate_{topo_name}",
                g.returncode == 0 and out_path.is_file(),
                f"rc={g.returncode}",
                g.stderr.strip() if g.returncode != 0 else "",
            )
        if out_path.is_file():
            artifacts[key] = out_path


# ---------------------------------------------------------------------------
# Category 2: `inspect` Side-by-Side Parity
# ---------------------------------------------------------------------------

def run_category_2_inspect(
    runner: CliRunner, rec: SuiteRecorder, workdir: Path, artifacts: Dict[str, Path]
) -> Dict[str, Path]:
    cat = "cat2_inspect"
    print("\n=== Category 2: inspect Side-by-Side Parity ===")

    ensure_synthetic_topologies(runner, workdir, artifacts, rec=rec)

    # JSON inspect parity across built snapshots + all 4 synthetic topologies
    snap_keys = ["rust_imps", "java_imps", "rust3_imps", "java3_imps"] + [
        f"topo_{name}" for name, _ in SYNTHETIC_TOPOLOGIES
    ]
    for label in snap_keys:
        snap = artifacts.get(label)
        if not snap or not snap.is_file():
            continue
        r_doc, r_err = _parse_json_output(runner.run_rust(["inspect", "--format", "json", str(snap)]))
        j_doc, j_err = _parse_json_output(runner.run_java(["inspect", "--format", "json", str(snap)]))
        if r_doc is None or j_doc is None:
            rec.record(cat, f"inspect_json_{label}", False, f"rust=({r_err}) java=({j_err})")
            continue
        rn = normalize_inspect_json(r_doc)
        jn = normalize_inspect_json(j_doc)
        ok = rn == jn
        rec.record(
            cat,
            f"inspect_json_{label}",
            ok,
            f"version={rn['version']} domains={rn['domain_count']} rels={rn['relation_count']}",
            "" if ok else f"RUST: {json.dumps(rn)}\nJAVA: {json.dumps(jn)}",
        )

    # Flag parity on rust.imps: --domain, --relation, --anonymize, --page-map, and --format json --anonymize
    snap = artifacts.get("rust_imps")
    if snap and snap.is_file():
        flag_cases = [
            ("inspect_domain_filter", ["--domain", "User", str(snap)], "User"),
            ("inspect_relation_filter", ["--relation", "FOLLOWS", str(snap)], "FOLLOWS"),
            ("inspect_anonymize_text", ["--anonymize", str(snap)], "ANONYMIZED"),
            ("inspect_page_map", ["--page-map", str(snap)], ""),
        ]
        for cname, args, expect_substr in flag_cases:
            rr = runner.run_rust(["inspect"] + args)
            jr = runner.run_java(["inspect"] + args)
            r_ok = rr.returncode == 0 and bool(rr.stdout.strip())
            j_ok = jr.returncode == 0 and bool(jr.stdout.strip())
            sub_ok = True
            if cname == "inspect_anonymize_text":
                sub_ok = ("User" not in rr.stdout) and ("User" not in jr.stdout)
            elif expect_substr:
                sub_ok = (expect_substr in rr.stdout) and (expect_substr in jr.stdout)
            ok = r_ok and j_ok and sub_ok
            diff = ""
            if not ok:
                diff = (
                    f"RUST rc={rr.returncode} len={len(rr.stdout)} err={rr.stderr.strip()[:160]}\n"
                    f"JAVA rc={jr.returncode} len={len(jr.stdout)} err={jr.stderr.strip()[:160]}"
                )
            rec.record(cat, cname, ok, f"rust_len={len(rr.stdout)} java_len={len(jr.stdout)}", diff)

        # `inspect --format json --anonymize`
        r_anon, r_err = _parse_json_output(
            runner.run_rust(["inspect", "--format", "json", "--anonymize", str(snap)])
        )
        j_anon, j_err = _parse_json_output(
            runner.run_java(["inspect", "--format", "json", "--anonymize", str(snap)])
        )
        if r_anon is None or j_anon is None:
            rec.record(cat, "inspect_json_anonymize", False, f"rust=({r_err}) java=({j_err})")
        else:
            r_txt = json.dumps(r_anon)
            j_txt = json.dumps(j_anon)
            leaked = [
                ident
                for ident in ("User", "Item", "FOLLOWS", "LIKES")
                if (ident in r_txt or ident in j_txt)
            ]
            r_dom_cnt = len(r_anon.get("domains", []))
            j_dom_cnt = len(j_anon.get("domains", []))
            r_rel_cnt = len(r_anon.get("relations", []))
            j_rel_cnt = len(j_anon.get("relations", []))
            ok = (not leaked) and (r_dom_cnt == j_dom_cnt == 2) and (r_rel_cnt == j_rel_cnt == 2)
            rec.record(
                cat,
                "inspect_json_anonymize",
                ok,
                f"domains={r_dom_cnt}/{j_dom_cnt} rels={r_rel_cnt}/{j_rel_cnt} leaked={leaked}",
                "" if ok else f"RUST: {r_txt[:220]}\nJAVA: {j_txt[:220]}",
            )

    # Negative: missing file & corrupted magic
    missing = workdir / "does_not_exist.imps"
    r_miss = runner.run_rust(["inspect", str(missing)])
    j_miss = runner.run_java(["inspect", str(missing)])
    rec.record(
        cat,
        "inspect_negative_missing_file",
        r_miss.returncode != 0 and j_miss.returncode != 0,
        f"rust_rc={r_miss.returncode} java_rc={j_miss.returncode}",
    )

    if snap and snap.is_file():
        bad_magic = workdir / "inspect_bad_magic.imps"
        data = bytearray(snap.read_bytes())
        data[0:4] = b"\xEF\xBE\xAD\xDE"
        bad_magic.write_bytes(data)
        r_bad = runner.run_rust(["inspect", str(bad_magic)])
        j_bad = runner.run_java(["inspect", str(bad_magic)])
        rec.record(
            cat,
            "inspect_negative_bad_magic",
            r_bad.returncode != 0 and j_bad.returncode != 0,
            f"rust_rc={r_bad.returncode} java_rc={j_bad.returncode}",
        )

    return artifacts


# ---------------------------------------------------------------------------
# Category 3: `snapshot validate` Side-by-Side Parity
# ---------------------------------------------------------------------------

def run_category_3_validate(
    runner: CliRunner, rec: SuiteRecorder, workdir: Path, artifacts: Dict[str, Path]
) -> None:
    cat = "cat3_validate"
    print("\n=== Category 3: snapshot validate Side-by-Side Parity ===")
    snap = artifacts.get("rust_imps")
    if not snap or not snap.is_file():
        rec.record(cat, "validate_prerequisite", False, "rust.imps missing")
        return

    for flag_name, flags in [
        ("strict_alignment", ["--strict-alignment"]),
        ("quick_check", ["--quick-check"]),
        ("deep_audit", ["--deep-audit"]),
    ]:
        rr = runner.run_rust(["snapshot", "validate"] + flags + [str(snap)])
        jr = runner.run_java(["snapshot", "validate"] + flags + [str(snap)])
        ok = rr.returncode == 0 and jr.returncode == 0
        rec.record(
            cat,
            f"validate_pos_{flag_name}",
            ok,
            f"rust_rc={rr.returncode} java_rc={jr.returncode}",
            "" if ok else f"RUST: {rr.stderr.strip()}\nJAVA: {jr.stderr.strip()}",
        )

    # Validate all 4 generated synthetic topologies side-by-side
    for topo_name, _ in SYNTHETIC_TOPOLOGIES:
        tsnap = artifacts.get(f"topo_{topo_name}")
        if not tsnap or not tsnap.is_file():
            continue
        rr = runner.run_rust(["snapshot", "validate", "--strict-alignment", "--deep-audit", str(tsnap)])
        jr = runner.run_java(["snapshot", "validate", "--strict-alignment", "--deep-audit", str(tsnap)])
        ok = rr.returncode == 0 and jr.returncode == 0
        rec.record(
            cat,
            f"validate_topo_{topo_name}",
            ok,
            f"rust_rc={rr.returncode} java_rc={jr.returncode}",
            "" if ok else f"RUST: {rr.stderr.strip()}\nJAVA: {jr.stderr.strip()}",
        )

    # Negative 1: corrupted magic (0xDEADBEEF)
    raw = snap.read_bytes()
    bad_magic = workdir / "val_bad_magic.imps"
    b1 = bytearray(raw)
    b1[0:4] = (0xDEADBEEF).to_bytes(4, "little")
    bad_magic.write_bytes(b1)
    rr = runner.run_rust(["snapshot", "validate", str(bad_magic)])
    jr = runner.run_java(["snapshot", "validate", str(bad_magic)])
    rec.record(
        cat,
        "validate_neg_corrupted_magic",
        rr.returncode != 0 and jr.returncode != 0,
        f"rust_rc={rr.returncode} java_rc={jr.returncode}",
    )

    # Negative 2: truncated file (128 bytes)
    trunc = workdir / "val_truncated_128b.imps"
    trunc.write_bytes(raw[:128])
    rr = runner.run_rust(["snapshot", "validate", str(trunc)])
    jr = runner.run_java(["snapshot", "validate", str(trunc)])
    rec.record(
        cat,
        "validate_neg_truncated_128b",
        rr.returncode != 0 and jr.returncode != 0,
        f"rust_rc={rr.returncode} java_rc={jr.returncode}",
    )

    # Negative 3: flipped byte in payload/checksum
    flipped = workdir / "val_flipped_byte.imps"
    b3 = bytearray(raw)
    b3[-1] ^= 0xFF
    if len(b3) > 16400:
        b3[16388] ^= 0xFF
    flipped.write_bytes(b3)
    rr = runner.run_rust(["snapshot", "validate", "--deep-audit", str(flipped)])
    jr = runner.run_java(["snapshot", "validate", "--deep-audit", str(flipped)])
    rec.record(
        cat,
        "validate_neg_flipped_byte",
        rr.returncode != 0 and jr.returncode != 0,
        f"rust_rc={rr.returncode} java_rc={jr.returncode}",
        ""
        if (rr.returncode != 0 and jr.returncode != 0)
        else f"RUST rc={rr.returncode}\nJAVA rc={jr.returncode}",
    )


# ---------------------------------------------------------------------------
# Category 4: `stats` Side-by-Side Parity
# ---------------------------------------------------------------------------

def run_category_4_stats(
    runner: CliRunner, rec: SuiteRecorder, artifacts: Dict[str, Path]
) -> None:
    cat = "cat4_stats"
    print("\n=== Category 4: stats Side-by-Side Parity ===")

    snap_keys = ["rust_imps", "java_imps", "rust3_imps", "java3_imps"] + [
        f"topo_{name}" for name, _ in SYNTHETIC_TOPOLOGIES
    ]
    for label in snap_keys:
        snap = artifacts.get(label)
        if not snap or not snap.is_file():
            continue
        r_doc, r_err = _parse_json_output(runner.run_rust(["stats", "--format", "json", str(snap)]))
        j_doc, j_err = _parse_json_output(runner.run_java(["stats", "--format", "json", str(snap)]))
        if r_doc is None or j_doc is None:
            rec.record(cat, f"stats_json_{label}", False, f"rust=({r_err}) java=({j_err})")
            continue
        rn = normalize_stats_json(r_doc)
        jn = normalize_stats_json(j_doc)
        diffs = compare_stats_dicts(rn, jn)
        rec.record(cat, f"stats_json_{label}", len(diffs) == 0, f"relations={len(rn)}", "\n".join(diffs))

    # Threshold sensitivity: --supernode-threshold 1.5 and 10.0 on rust_imps & topo_social_100
    for label in ("rust_imps", "topo_social_100"):
        snap = artifacts.get(label)
        if not snap or not snap.is_file():
            continue
        for thresh in ("1.5", "10.0"):
            r_doc, r_err = _parse_json_output(
                runner.run_rust(["stats", "--format", "json", "--supernode-threshold", thresh, str(snap)])
            )
            j_doc, j_err = _parse_json_output(
                runner.run_java(["stats", "--format", "json", "--supernode-threshold", thresh, str(snap)])
            )
            if r_doc is None or j_doc is None:
                rec.record(
                    cat,
                    f"stats_thresh_{thresh}_{label}",
                    False,
                    f"rust=({r_err}) java=({j_err})",
                )
                continue
            rn = normalize_stats_json(r_doc)
            jn = normalize_stats_json(j_doc)
            diffs = compare_stats_dicts(rn, jn)
            sc = {k: v["supernode_count"] for k, v in rn.items()}
            rec.record(
                cat,
                f"stats_thresh_{thresh}_{label}",
                len(diffs) == 0,
                f"supernodes={sc}",
                "\n".join(diffs),
            )


# ---------------------------------------------------------------------------
# Category 5: `assemble` & `disassemble` Byte-for-Byte Parity
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ImpasCase:
    name: str
    source: str
    known_core_issue: Optional[str] = None


IMPAS_BATTERY: List[ImpasCase] = [
    ImpasCase(
        "01_csr_walk",
        """\
.domain User = 0
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_COLLECT_BITSET R63, R1
OP_HALT
""",
    ),
    ImpasCase(
        "02_csc_walk_2part_3part",
        """\
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, 0
OP_CSC_WALK R2, R1, 0
OP_CSR_WALK_FILTERED R3, R1, R2, 0
OP_COLLECT_BITSET R63, R3
OP_HALT
""",
    ),
    ImpasCase(
        "03_set_ops",
        """\
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, 0
OP_CSR_WALK R2, R1, 0
OP_SET_UNION R3, R1, R2
OP_SET_INTERSECT R4, R1, R2
OP_SET_DIFFERENCE R5, R3, R4
OP_SET_CARDINALITY R63, R5
OP_HALT
""",
    ),
    ImpasCase(
        "04_labels_and_branches",
        """\
OP_LOAD_CONST_INT R0, 5
loop_top:
OP_LOOP_DECR R0, loop_done
OP_JZ R0, loop_done
OP_JNZ R0, loop_top
OP_JMP loop_top
loop_done:
OP_STABLE_CHECK R1, R0
OP_HALT
""",
    ),
    ImpasCase(
        "05_directives_domain_rel_attr",
        """\
.domain User = 0
.domain Item = 1
.relation LIKES = 1
.attribute rating = 0
OP_INIT_INPUT_NODE R0, DOMAIN_USER
OP_CSR_WALK R1, R0, REL_LIKES
OP_GATHER_EDGE_ATTR R2, R1, ATTR_RATING
OP_COLLECT_BITSET R63, R1
OP_HALT
""",
    ),
    ImpasCase(
        "06_const_and_equ_symbols",
        """\
.const SEED = 3
.equ REL_ID = 0
OP_LOAD_CONST_INT R0, SEED
OP_CSR_WALK R1, R0, REL_ID
OP_HALT
""",
        known_core_issue="impulse-graph-core#148",
    ),
    ImpasCase(
        "07_float_vector_math",
        """\
OP_LOAD_CONST_FLOAT R0, 2.5
OP_FLOAT_VECTOR_SCALE R1, R0, 1
OP_VEC_MATH_UNARY R2, R1, 0
OP_VEC_MATH_BINARY R3, R1, R2, 0
OP_L1_NORM_DIFF R4, R2, R3
OP_VECTOR_REDUCE_SUM R5, R3
OP_HALT
""",
        known_core_issue="impulse-graph-core#148",
    ),
    ImpasCase(
        "08_stream_shaders_0xA0_0xBF",
        """\
OP_STREAM_FUNC_BEGIN R0, 0
OP_STREAM_LOAD_SRC R1, 0
OP_STREAM_LOAD_EDGE R2, 0
OP_STREAM_LOAD_TGT R3, 0
OP_STREAM_MATH_ADD R4, R1, R2
OP_STREAM_MATH_MUL R5, R4, R3
OP_STREAM_CMP_GT R6, R5, R1
OP_STREAM_FILTER R6, 0
OP_STREAM_REDUCE R7, R5, 0
OP_STREAM_YIELD R7, 0
OP_STREAM_FUNC_END R0, 0
OP_CSR_WALK_STREAM R8, R0, 0
OP_HALT
""",
    ),
    ImpasCase(
        "09_multi_lane_0xC0_0xC9",
        """\
OP_LANE_INIT_SEEDS R0, 0
OP_LANE_INIT_GROUPS R1, 0
OP_LANE_SET_EDGE_MASK R2, R0, 0
OP_LANE_CSR_WALK R3, R0, 0
OP_LANE_CSC_WALK R4, R3, 0
OP_LANE_PRUNE_MASK R5, R3, R4
OP_LANE_FILTER_QUORUM R6, R5, 2
OP_LANE_REDUCE_AND R7, R5, R6
OP_LANE_REDUCE_ATTR R8, R7, 0
OP_LANE_SET_COVER R9, R8, 0
OP_HALT
""",
    ),
    ImpasCase(
        "10_named_fields_and_imm64",
        """\
OP_MOV R1, R0
OP_SWAP_REG R1, R2
OP_CLEAR_REG R2
OP_ALLOC_SCRATCH R3, 64
OP_ASSERT_SCRATCH_BYTES R3, 64
OP_SET_MAX_DOP R0, 4
OP_ADAPTIVE_WALK flags=0x03, dst=R4, payload=0x00000000
OP_COLLECT_BITSET flags=0x00, dst=R63, payload=0x00000004
OP_HALT
""",
        known_core_issue="impulse-graph-core#148",
    ),
    ImpasCase(
        "11_csr_degree",
        """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_DEGREE R2, R1, REL_FOLLOWS
OP_MOV R63, R2
OP_HALT
""",
    ),
    ImpasCase(
        "12_set_diff_and_cardinality",
        """\
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, 0
OP_CSR_WALK R2, R1, 0
OP_SET_DIFFERENCE R3, R2, R1
OP_SET_CARDINALITY R63, R3
OP_HALT
""",
    ),
    ImpasCase(
        "13_mxv_semirings",
        """\
OP_INIT_INPUT_NODE R0, 0
OP_MXV R1, R0, 0, SEMIRING_PLUS_TIMES
OP_MXV R2, R1, 0, SEMIRING_MIN_PLUS
OP_VXM R3, R2, 0, SEMIRING_MIN_PLUS
OP_REDUCE R63, R3, BINARY_OP_ADD
OP_HALT
""",
    ),
    ImpasCase(
        "14_cc_afforest",
        """\
OP_CC_AFFOREST R1, 0
OP_COLLECT_ARRAY R63, R1
OP_HALT
""",
    ),
    ImpasCase(
        "15_fixpoint_kleene_star",
        """\
OP_INIT_INPUT_NODE R0, 0
OP_FIXPOINT_KLEENE_STAR R1, R0, 0
OP_COLLECT_BITSET R63, R1
OP_HALT
""",
    ),
    ImpasCase(
        "16_vector_ann_and_rerank",
        """\
OP_VECTOR_ANN_SEARCH R1, 0, 1, 16, 32
OP_VECTOR_RERANK_DOT R2, R1, 0, 10
OP_COLLECT_ARRAY R63, R2
OP_HALT
""",
    ),
    ImpasCase(
        "17_stable_check_and_swap",
        """\
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, 0
OP_CSR_WALK R2, R1, 0
OP_SWAP_REG R1, R2
OP_STABLE_CHECK R3, R1, R2
OP_JZ R3, done
OP_COLLECT_BITSET R63, R1
done:
OP_HALT
""",
    ),
]


def run_category_5_assemble(
    runner: CliRunner, rec: SuiteRecorder, workdir: Path
) -> None:
    cat = "cat5_assemble"
    print("\n=== Category 5: assemble & disassemble Byte-for-Byte Parity ===")
    asm_dir = workdir / "asm"
    asm_dir.mkdir(parents=True, exist_ok=True)

    programs: List[Tuple[str, Path, Optional[str]]] = []
    for case in IMPAS_BATTERY:
        p = asm_dir / f"{case.name}.impas"
        p.write_text(case.source, encoding="utf-8")
        programs.append((case.name, p, case.known_core_issue))

    spec_vectors: List[Tuple[Path, Optional[str]]] = [
        (REPO_ROOT / "spec/test-vectors/vm-impas/01_scalars_and_data/tc00_vm_init_and_halt_pos.impas", None),
        (
            REPO_ROOT / "spec/test-vectors/vm-impas/01_scalars_and_data/tc01_scalar_ops_pos.impas",
            "impulse-graph-core#148",
        ),
    ]
    for sv, issue in spec_vectors:
        if sv.is_file():
            programs.append((f"spec_{sv.stem}", sv, issue))

    for name, impas_path, known_issue in programs:
        r_out = asm_dir / f"{name}.rust.impb"
        j_out = asm_dir / f"{name}.java.impb"
        ra = runner.run_rust(["assemble", "-i", str(impas_path), "-o", str(r_out)])
        ja = runner.run_java(["assemble", "-i", str(impas_path), "-o", str(j_out)])

        if ra.returncode != 0 or ja.returncode != 0:
            rec.record(
                cat,
                f"assemble_bitwise_{name}",
                False,
                f"rust_rc={ra.returncode} java_rc={ja.returncode}",
                f"RUST err: {ra.stderr.strip()}\nJAVA err: {ja.stderr.strip()}",
                known_core_issue=known_issue,
            )
            continue

        rb = r_out.read_bytes()
        jb = j_out.read_bytes()
        eq = rb == jb
        diff = ""
        if not eq:
            n_words = max(len(rb), len(jb)) // 8
            lines = [f"len: rust={len(rb)}B java={len(jb)}B"]
            for i in range(n_words):
                rw = rb[i * 8 : (i + 1) * 8].hex()
                jw = jb[i * 8 : (i + 1) * 8].hex()
                if rw != jw:
                    lines.append(f"  instr[{i}]: rust={rw} vs java={jw}")
            diff = "\n".join(lines[:10])
        rec.record(
            cat,
            f"assemble_bitwise_{name}",
            eq,
            f"{len(rb)} bytes ({len(rb) // 8} instrs)",
            diff,
            known_core_issue=known_issue,
        )

        # Cross-disassemble: Rust disassembles Java .impb, Java disassembles Rust .impb
        r_dis = runner.run_rust(["disassemble", "-i", str(j_out)])
        j_dis = runner.run_java(["disassemble", "-i", str(r_out)])
        r_ops = extract_disasm_opcodes(r_dis.stdout)
        j_ops = extract_disasm_opcodes(j_dis.stdout)
        dis_ok = (
            r_dis.returncode == 0
            and j_dis.returncode == 0
            and len(r_ops) > 0
            and r_ops == j_ops
        )
        dis_diff = ""
        if not dis_ok:
            dis_diff = (
                f"RUST disasm rc={r_dis.returncode} ops={r_ops}\n"
                f"JAVA disasm rc={j_dis.returncode} ops={j_ops} stdout_len={len(j_dis.stdout)}"
            )
        rec.record(
            cat,
            f"cross_disassemble_{name}",
            dis_ok,
            f"rust_ops={len(r_ops)} java_ops={len(j_ops)}",
            dis_diff,
        )

    # Negative assemble tests: invalid opcode & out-of-range register
    neg_cases = [
        ("neg_invalid_opcode", "OP_DOES_NOT_EXIST R0, 1\nOP_HALT\n"),
        ("neg_register_out_of_range", "OP_LOAD_CONST_INT R99, 1\nOP_HALT\n"),
    ]
    for nname, nsrc in neg_cases:
        np = asm_dir / f"{nname}.impas"
        np.write_text(nsrc, encoding="utf-8")
        ra = runner.run_rust(["assemble", "-i", str(np), "-o", str(asm_dir / f"{nname}.r.impb")])
        ja = runner.run_java(["assemble", "-i", str(np), "-o", str(asm_dir / f"{nname}.j.impb")])
        ok = ra.returncode != 0 and ja.returncode != 0
        rec.record(
            cat,
            f"assemble_{nname}",
            ok,
            f"rust_rc={ra.returncode} java_rc={ja.returncode}",
            ""
            if ok
            else f"RUST: {ra.stdout.strip()} | {ra.stderr.strip()}\nJAVA: {ja.stdout.strip()} | {ja.stderr.strip()}",
        )


# ---------------------------------------------------------------------------
# Category 6: `run` & `compile` Side-by-Side Parity
# ---------------------------------------------------------------------------

def run_category_6_run_compile(
    runner: CliRunner, rec: SuiteRecorder, workdir: Path, artifacts: Dict[str, Path]
) -> None:
    cat = "cat6_run_compile"
    print("\n=== Category 6: run & compile Side-by-Side Parity ===")
    run_dir = workdir / "run_prog"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Assemble reusable .impb programs
    # (name, kind, source, seeds)
    run_programs: List[Tuple[str, str, str, Sequence[int]]] = [
        (
            "csr_1hop",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_COLLECT_BITSET R63, R1
OP_HALT
""",
            (0, 1, 2),
        ),
        (
            "csr_2hop",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_WALK R2, R1, REL_FOLLOWS
OP_COLLECT_BITSET R63, R2
OP_HALT
""",
            (0, 1, 2),
        ),
        (
            "csr_3hop",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_WALK R2, R1, REL_FOLLOWS
OP_CSR_WALK R3, R2, REL_FOLLOWS
OP_COLLECT_BITSET R63, R3
OP_HALT
""",
            (0, 1),
        ),
        (
            "csc_reverse_1hop",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
; Single-word packed payload (srcReg=R1=1, relId=0 -> 0x00000001); avoids assemble.rs:516 (63<<16) bug (core#148)
OP_CSC_WALK R2, 0x00000001
OP_COLLECT_BITSET R63, R2
OP_HALT
""",
            (0, 1),
        ),
        (
            "csr_set_union",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_WALK R2, R1, REL_FOLLOWS
OP_SET_UNION R3, R1, R2
OP_COLLECT_BITSET R63, R3
OP_HALT
""",
            (0, 1),
        ),
        (
            "csr_set_intersect",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_WALK R2, R1, REL_FOLLOWS
OP_SET_INTERSECT R3, R1, R2
OP_COLLECT_BITSET R63, R3
OP_HALT
""",
            (0, 1),
        ),
        (
            "csr_set_difference",
            "bitset",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_WALK R2, R1, REL_FOLLOWS
OP_SET_DIFFERENCE R3, R2, R1
OP_COLLECT_BITSET R63, R3
OP_HALT
""",
            (0, 1),
        ),
        (
            "csr_degree",
            "scalar",
            """\
.relation FOLLOWS = 0
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_FOLLOWS
OP_CSR_DEGREE R63, R1, REL_FOLLOWS
OP_HALT
""",
            (0, 2),
        ),
        (
            "empty_isolated_likes_seed5",
            "bitset",
            """\
.relation LIKES = 1
OP_INIT_INPUT_NODE R0, 0
OP_CSR_WALK R1, R0, REL_LIKES
OP_COLLECT_BITSET R63, R1
OP_HALT
""",
            (5,),
        ),
    ]

    assembled: Dict[str, Path] = {}
    for pname, _kind, psrc, _seeds in run_programs:
        impas = run_dir / f"{pname}.impas"
        impb = run_dir / f"{pname}.impb"
        impas.write_text(psrc, encoding="utf-8")
        a = runner.run_rust(["assemble", "-i", str(impas), "-o", str(impb)])
        if a.returncode != 0:
            rec.record(cat, f"run_assemble_{pname}", False, a.stderr.strip())
        else:
            assembled[pname] = impb

    # Run all programs on BOTH rust.imps and java.imps
    for snap_label in ("rust_imps", "java_imps"):
        snap = artifacts.get(snap_label)
        if not snap or not snap.is_file():
            continue
        for pname, kind, _psrc, seeds in run_programs:
            impb = assembled.get(pname)
            if not impb:
                continue
            for seed in seeds:
                rr = runner.run_rust(
                    ["run", "-s", str(snap), "-b", str(impb), "--verify", "-i", str(seed)]
                )
                jr = runner.run_java(
                    [
                        "run",
                        "-s",
                        str(snap),
                        "-b",
                        str(impb),
                        "--verify",
                        "-i",
                        str(seed),
                        "--format",
                        "json",
                    ]
                )
                cname = f"run_{snap_label}_{pname}_seed{seed}"
                if rr.returncode != 0 or jr.returncode != 0:
                    rec.record(
                        cat,
                        cname,
                        False,
                        f"rust_rc={rr.returncode} java_rc={jr.returncode}",
                        f"RUST err: {rr.stderr.strip()}\nJAVA err: {jr.stderr.strip()}",
                    )
                    continue

                j_doc, j_err = _parse_json_output(jr)
                if j_doc is None:
                    rec.record(cat, cname, False, f"java json: {j_err}")
                    continue

                if kind == "bitset":
                    r_card, r_nodes = parse_rust_run_cardinality(rr.stdout)
                    j_card = j_doc.get("cardinality")
                    j_nodes = j_doc.get("sampleNodeIds")
                    nodes_ok = (
                        (r_nodes == j_nodes)
                        if (r_nodes is not None and j_nodes is not None)
                        else True
                    )
                    ok = (r_card is not None) and (r_card == j_card) and nodes_ok
                    rec.record(
                        cat,
                        cname,
                        ok,
                        f"cardinality={r_card} nodes={r_nodes}",
                        ""
                        if ok
                        else f"RUST: card={r_card} nodes={r_nodes}\nJAVA: card={j_card} nodes={j_nodes}",
                    )
                else:
                    # scalar result (e.g. OP_CSR_DEGREE)
                    r_val = parse_rust_run_scalar(rr.stdout)
                    j_val = j_doc.get("scalarValue", j_doc.get("value"))
                    if j_val is None and str(j_doc.get("resultType", "")).lower().find("scalar") >= 0:
                        m = re.search(r"(-?\d+)", str(j_doc.get("summary", "")))
                        if m:
                            j_val = int(m.group(1))
                    ok = (r_val is not None) and (j_val is not None) and (int(r_val) == int(j_val))
                    rec.record(
                        cat,
                        cname,
                        ok,
                        f"rust_scalar={r_val} java_scalar={j_val} (java_type={j_doc.get('resultType')})",
                        "" if ok else f"RUST: {r_val}\nJAVA JSON: {json.dumps(j_doc)}",
                    )

    # Run 1-hop CSR walk across all 4 generated synthetic topologies
    impb_1hop = assembled.get("csr_1hop")
    if impb_1hop:
        for topo_name, _ in SYNTHETIC_TOPOLOGIES:
            tsnap = artifacts.get(f"topo_{topo_name}")
            if not tsnap or not tsnap.is_file():
                continue
            rr = runner.run_rust(["run", "-s", str(tsnap), "-b", str(impb_1hop), "--verify", "-i", "0"])
            jr = runner.run_java(
                ["run", "-s", str(tsnap), "-b", str(impb_1hop), "--verify", "-i", "0", "--format", "json"]
            )
            cname = f"run_topo_{topo_name}_csr_1hop_seed0"
            if rr.returncode != 0 or jr.returncode != 0:
                rec.record(
                    cat,
                    cname,
                    False,
                    f"rust_rc={rr.returncode} java_rc={jr.returncode}",
                    f"RUST err: {rr.stderr.strip()}\nJAVA err: {jr.stderr.strip()}",
                )
                continue
            r_card, _ = parse_rust_run_cardinality(rr.stdout)
            j_doc, j_err = _parse_json_output(jr)
            j_card = j_doc.get("cardinality") if j_doc else None
            ok = (r_card is not None) and (r_card == j_card)
            rec.record(
                cat,
                cname,
                ok,
                f"cardinality: rust={r_card} java={j_card}",
                "" if ok else f"java_err={j_err}",
            )

    # `compile --emit-ir` on .impscm, .impk, .implog
    dsl_samples = [
        ("impscm", "sample.impscm", "(let ((a 1) (b c)) (set! a 3) (return a))\n"),
        ("impk", "sample.impk", "(impk-pipeline (define-query my_q (g:walk-csr edge)))\n"),
        ("implog", "sample.implog", "path(X, Y) :- edge(X, Z), edge(Z, Y).\n"),
    ]
    for lang, fname, src in dsl_samples:
        fp = run_dir / fname
        fp.write_text(src, encoding="utf-8")
        rc_r = runner.run_rust(["compile", "--emit-ir", str(fp)])
        rc_j = runner.run_java(["compile", "--emit-ir", str(fp)])
        ok = (
            rc_r.returncode == 0
            and rc_j.returncode == 0
            and bool(rc_r.stdout.strip())
            and bool(rc_j.stdout.strip())
        )
        rec.record(
            cat,
            f"compile_emit_ir_{lang}",
            ok,
            f"rust_rc={rc_r.returncode} ({len(rc_r.stdout)}B) java_rc={rc_j.returncode} ({len(rc_j.stdout)}B)",
            "" if ok else f"RUST err: {rc_r.stderr.strip()}\nJAVA err: {rc_j.stderr.strip()}",
        )


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

CATEGORY_ALIASES = {
    "1": 1,
    "build": 1,
    "2": 2,
    "inspect": 2,
    "3": 3,
    "validate": 3,
    "4": 4,
    "stats": 4,
    "5": 5,
    "assemble": 5,
    "disassemble": 5,
    "6": 6,
    "run": 6,
    "compile": 6,
}


def parse_categories(raw: Optional[List[str]]) -> set[int]:
    if not raw:
        return {1, 2, 3, 4, 5, 6}
    chosen: set[int] = set()
    for item in raw:
        for tok in item.split(","):
            key = tok.strip().lower()
            if not key:
                continue
            if key not in CATEGORY_ALIASES:
                raise ValueError(
                    f"Unknown category '{tok}'. Valid: 1..6, build, inspect, validate, stats, assemble, run, compile"
                )
            chosen.add(CATEGORY_ALIASES[key])
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-Side CLI Differential Parity Suite (Rust impulse-graph vs. Java impulse-j)"
    )
    parser.add_argument("--rust-cli", help="Path to Rust `impulse-graph` binary (or IMPULSE_RUST_CLI)")
    parser.add_argument("--java-cli", help="Path to Java `impulse-j` script or shaded JAR (or IMPULSE_JAVA_CLI)")
    parser.add_argument(
        "--category",
        action="append",
        help="Filter categories to run (1..6 or build,inspect,validate,stats,assemble,run,compile; comma-separated or repeatable)",
    )
    parser.add_argument(
        "--strict-xfail",
        action="store_true",
        help="Treat tracked known_core_issue XFAIL checks as hard failures",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose check details")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary fixture directory after exit")
    parser.add_argument("--json-report", help="Optional path to write machine-readable JSON results")
    args = parser.parse_args(argv)

    try:
        rust_cli = discover_rust_cli(args.rust_cli)
        java_cmd = discover_java_cli(args.java_cli)
        categories = parse_categories(args.category)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("============================================================")
    print("  Impulse Graph Side-by-Side CLI Parity Suite (Rust vs Java)")
    print("============================================================")
    print(f"  Rust CLI: {rust_cli}")
    print(f"  Java CLI: {' '.join(java_cmd)}")
    print(f"  Categories: {sorted(categories)}")
    if args.strict_xfail:
        print("  Mode: --strict-xfail")

    runner = CliRunner(rust_cli, java_cmd, verbose=args.verbose)
    rec = SuiteRecorder(verbose=args.verbose, strict_xfail=args.strict_xfail)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="impulse_cli_parity_")
    workdir = Path(tmp_ctx.name)
    if args.keep_tmp:
        tmp_ctx._finalizer.detach()  # type: ignore[attr-defined]
        print(f"  Workdir (kept): {workdir}")

    try:
        artifacts: Dict[str, Path] = {}
        if 1 in categories:
            artifacts = run_category_1_build(runner, rec, workdir)
        elif categories & {2, 3, 4, 6}:
            m2 = create_multi_domain_fixture(workdir / "fixture2")
            m3 = create_three_domain_nullable_fixture(workdir / "fixture3")
            rust_imps, java_imps = workdir / "rust.imps", workdir / "java.imps"
            rust3_imps, java3_imps = workdir / "rust3.imps", workdir / "java3.imps"
            runner.run_rust(["build", "-m", str(m2), "-o", str(rust_imps)])
            runner.run_java(["build", "-m", str(m2), "-o", str(java_imps)])
            runner.run_rust(["build", "-m", str(m3), "-o", str(rust3_imps)])
            runner.run_java(["build", "-m", str(m3), "-o", str(java3_imps)])
            artifacts = {
                "manifest": m2,
                "rust_imps": rust_imps,
                "java_imps": java_imps,
                "rust3_imps": rust3_imps,
                "java3_imps": java3_imps,
            }

        if 2 in categories:
            artifacts = run_category_2_inspect(runner, rec, workdir, artifacts)
        elif categories & {3, 4, 6}:
            ensure_synthetic_topologies(runner, workdir, artifacts)

        if 3 in categories:
            run_category_3_validate(runner, rec, workdir, artifacts)
        if 4 in categories:
            run_category_4_stats(runner, rec, artifacts)
        if 5 in categories:
            run_category_5_assemble(runner, rec, workdir)
        if 6 in categories:
            run_category_6_run_compile(runner, rec, workdir, artifacts)
    finally:
        if not args.keep_tmp:
            tmp_ctx.cleanup()

    n_pass = sum(1 for r in rec.results if r.status == "PASS")
    n_fail = sum(1 for r in rec.results if r.status == "FAIL")
    n_xfail = sum(1 for r in rec.results if r.status == "XFAIL")
    n_xpass = sum(1 for r in rec.results if r.status == "XPASS")
    total = len(rec.results)

    print("\n============================================================")
    print(f"  SUMMARY: {n_pass}/{total} PASSED, {n_fail} FAILED, {n_xfail} XFAIL, {n_xpass} XPASS")
    print("============================================================")
    if n_fail:
        print("\nFailed Checks:")
        for r in rec.results:
            if r.status == "FAIL":
                print(f"  - [{r.category}] {r.name}: {r.detail}")
    if n_xfail:
        print("\nTracked XFAIL Checks:")
        for r in rec.results:
            if r.status == "XFAIL":
                print(f"  - [{r.category}] {r.name} ({r.known_core_issue}): {r.detail}")

    if args.json_report:
        report = {
            "rust_cli": str(rust_cli),
            "java_cli": java_cmd,
            "passed": n_pass,
            "failed": n_fail,
            "xfail": n_xfail,
            "xpass": n_xpass,
            "total": total,
            "results": [dataclasses.asdict(r) for r in rec.results],
        }
        Path(args.json_report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
