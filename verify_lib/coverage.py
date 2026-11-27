"""Coverage & opcode catalog parsers (ISA YAML, opcode reports, llvm-cov, JaCoCo, Pitest)."""

from __future__ import annotations

import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpcodeEntry:
    key: str
    name: str
    hex_code: str
    subop: int
    category: str
    active: int


def load_spec_opcodes(spec_dir: Path) -> list[OpcodeEntry]:
    """Load canonical v0.11 opcodes from impulse-graph-spec/v0.11-refactor/isa_v0_11.yaml."""
    isa_yaml = spec_dir / "v0.11-refactor" / "isa_v0_11.yaml"
    isa_loader_path = spec_dir / "tools" / "isa_loader.py"
    if not isa_yaml.is_file() or not isa_loader_path.is_file():
        return []

    spec_mod = importlib.util.spec_from_file_location("_impulse_isa_loader", isa_loader_path)
    if spec_mod is None or spec_mod.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)  # type: ignore[union-attr]

    # isa_loader exposes load_isa(path) or load_yaml(path)
    data: dict[str, Any] = {}
    if hasattr(mod, "load_isa"):
        data = mod.load_isa(isa_yaml)
    elif hasattr(mod, "load_yaml"):
        data = mod.load_yaml(isa_yaml)
    elif hasattr(mod, "parse_yaml_file"):
        data = mod.parse_yaml_file(isa_yaml)

    entries: list[OpcodeEntry] = []
    opcodes_raw = data.get("opcodes", [])
    if isinstance(opcodes_raw, dict):
        items = list(opcodes_raw.values())
    elif isinstance(opcodes_raw, list):
        items = opcodes_raw
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", item.get("mnemonic", ""))).strip()
        if not name:
            continue
        raw_code = item.get("opcode", item.get("code", 0))
        if isinstance(raw_code, str):
            try:
                code_int = int(raw_code, 0)
            except ValueError:
                code_int = 0
        else:
            code_int = int(raw_code or 0)
        hex_code = f"0x{code_int:02X}"
        category = str(item.get("category", item.get("group", "general")))
        status = str(item.get("status", "active")).lower()
        active = 0 if status in ("reserved", "removed", "deprecated") else 1
        key = f"{hex_code}:{name}"
        entries.append(
            OpcodeEntry(
                key=key,
                name=name,
                hex_code=hex_code,
                subop=-1,
                category=category,
                active=active,
            )
        )
    return entries


_OP_TOKEN_RE = re.compile(r"\b(OP_[A-Z0-9_]+)\b")
_HEX_OP_RE = re.compile(r"\b(0x[0-9A-Fa-f]{2})\b")


def parse_opcode_coverage_output(text: str) -> list[str]:
    """Extract covered opcode names / hex codes from test or fuzzer stdout."""
    found: set[str] = set()
    for m in _OP_TOKEN_RE.finditer(text):
        found.add(m.group(1))
    for m in _HEX_OP_RE.finditer(text):
        found.add(m.group(1).upper())
    return sorted(found)


def parse_llvm_cov_json(data: dict[str, Any], repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Parse `llvm-cov export` JSON payload into normalized per-file coverage dicts."""
    out: list[dict[str, Any]] = []
    for entry in data.get("data", []):
        for fobj in entry.get("files", []):
            fname = str(fobj.get("filename", ""))
            if not fname:
                continue
            if repo_root is not None:
                try:
                    fname = str(Path(fname).resolve().relative_to(repo_root.resolve()))
                except ValueError:
                    pass
            summary = fobj.get("summary", {})
            lines = summary.get("lines", {})
            branches = summary.get("branches", {})
            mcdc = summary.get("mcdc", {})
            out.append(
                {
                    "tool": "llvm-cov",
                    "repo": "core",
                    "file_path": fname,
                    "line_pct": float(lines.get("percent", 0.0)),
                    "branch_pct": float(branches.get("percent", 0.0)),
                    "mcdc_pct": float(mcdc.get("percent", 0.0)),
                    "mutants_killed": 0,
                    "mutants_total": 0,
                }
            )
    return out


def collect_llvm_cov_reports(core_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for p in core_dir.rglob("llvm-cov*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.extend(parse_llvm_cov_json(data, repo_root=core_dir))
        except (OSError, json.JSONDecodeError):
            continue
    return results


def parse_jacoco_xml(xml_text: str, module_prefix: str = "") -> list[dict[str, Any]]:
    """Parse JaCoCo XML report text into normalized per-file coverage dicts."""
    root = ET.fromstring(xml_text)
    out: list[dict[str, Any]] = []
    for pkg in root.findall(".//package"):
        pkg_name = pkg.attrib.get("name", "")
        for sf in pkg.findall("sourcefile"):
            sf_name = sf.attrib.get("name", "")
            if not sf_name:
                continue
            rel = f"{pkg_name}/{sf_name}" if pkg_name else sf_name
            if module_prefix:
                rel = f"{module_prefix}/src/main/java/{rel}"
            line_pct = 0.0
            branch_pct = 0.0
            for counter in sf.findall("counter"):
                ctype = counter.attrib.get("type", "")
                missed = int(counter.attrib.get("missed", "0"))
                covered = int(counter.attrib.get("covered", "0"))
                total = missed + covered
                pct = (100.0 * covered / total) if total > 0 else 100.0
                if ctype == "LINE":
                    line_pct = round(pct, 2)
                elif ctype == "BRANCH":
                    branch_pct = round(pct, 2)
            out.append(
                {
                    "tool": "jacoco",
                    "repo": "java",
                    "file_path": rel,
                    "line_pct": line_pct,
                    "branch_pct": branch_pct,
                    "mcdc_pct": 0.0,
                    "mutants_killed": 0,
                    "mutants_total": 0,
                }
            )
    return out


def collect_jacoco_reports(java_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not java_dir.is_dir():
        return results
    for xml_path in java_dir.glob("**/target/site/jacoco/jacoco.xml"):
        try:
            mod_name = xml_path.parents[3].name
            prefix = "" if mod_name == java_dir.name else mod_name
            results.extend(parse_jacoco_xml(xml_path.read_text(encoding="utf-8"), module_prefix=prefix))
        except (OSError, ET.ParseError):
            continue
    return results


def parse_pitest_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse Pitest mutations.xml into normalized per-file mutation coverage dicts."""
    root = ET.fromstring(xml_text)
    stats: dict[str, list[int]] = {}  # file -> [killed, total]
    for mut in root.findall("mutation"):
        src_file = (mut.findtext("sourceFile") or "").strip()
        cls_name = (mut.findtext("mutatedClass") or "").strip()
        if not src_file:
            continue
        pkg_path = "/".join(cls_name.split(".")[:-1])
        rel = f"{pkg_path}/{src_file}" if pkg_path else src_file
        bucket = stats.setdefault(rel, [0, 0])
        bucket[1] += 1
        status = mut.attrib.get("status", "").upper()
        if status in ("KILLED", "TIMED_OUT", "MEMORY_ERROR", "RUN_ERROR"):
            bucket[0] += 1
    out: list[dict[str, Any]] = []
    for rel, (killed, total) in sorted(stats.items()):
        out.append(
            {
                "tool": "pitest",
                "repo": "java",
                "file_path": rel,
                "line_pct": 0.0,
                "branch_pct": 0.0,
                "mcdc_pct": 0.0,
                "mutants_killed": killed,
                "mutants_total": total,
            }
        )
    return out


def collect_pitest_reports(java_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not java_dir.is_dir():
        return results
    for xml_path in java_dir.glob("**/target/pit-reports/**/mutations.xml"):
        try:
            results.extend(parse_pitest_xml(xml_path.read_text(encoding="utf-8")))
        except (OSError, ET.ParseError):
            continue
    return results
