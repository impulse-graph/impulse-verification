"""SCIP-first polyglot static call-chain indexer + C-ABI / Opcode semantic bridge linker."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .catalog import WorkspacePaths
from .coverage import load_spec_opcodes
from .runner import graph_data_dir
from .scip_ingest import (
    decode_scip_file,
    run_available_scip_indexers,
    run_docker_scip_indexers,
    scip_index_to_ir,
)


SKIP_DIRS = frozenset({
    ".git",
    "target",
    "build",
    "build_on",
    "build_off",
    "build-cpp",
    "build-cpp-on",
    "build-cpp-off",
    "build-cpp-fallback",
    "build-old",
    "build-san",
    "build-tsan",
    "build-mcdc",
    "_build",
    "datasets",
    "node_modules",
    "__pycache__",
    ".venv",
    ".idea",
    "spec",
    "fuzz_findings",
})

ALLOC_STDIO_RE = re.compile(
    r"\b(malloc|calloc|realloc|std::vector|std::string|fprintf|printf|stderr|std::cerr|System\.err|System\.out)\b"
)
OP_REF_RE = re.compile(r"\b(OP_[A-Z0-9_]{2,})\b")
CABI_DECL_RE = re.compile(
    r"IMPULSE_API\s+[^;()]+?\b(impulse_[a-z0-9_]+)\s*\([^;]*?\)\s*;",
    re.DOTALL,
)
CABI_CALL_RE = re.compile(r"\b(impulse_[a-z0-9_]+)\s*\(")

# Lightweight fallback function/method header patterns
CPP_FUNC_RE = re.compile(
    r"^(?:template\s*<[^>]*>\s*)?(?:static\s+|inline\s+|extern\s+\"C\"\s+|IMPULSE_API\s+|virtual\s+|constexpr\s+)*"
    r"[A-Za-z_][\w:<>*&\s]+\s+([A-Za-z_~][\w:]*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
)
JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected|static|final|synchronized|abstract|native|\s)+"
    r"(?:<[^>]+>\s+)?[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:throws\s+[\w.,\s]+)?\{"
)
KOTLIN_FN_RE = re.compile(r"^\s*(?:public|private|internal|protected|override|inline|suspend|\s)*fun\s+(?:<[^>]+>\s+)?(?:[\w.<>]+\.)?([A-Za-z_]\w*)\s*\(")
SCALA_DEF_RE = re.compile(r"^\s*(?:override\s+|private\s+|protected\s+|final\s+)*def\s+([A-Za-z_]\w*)")
CLOJURE_DEF_RE = re.compile(r"^\s*\(defn-?\s+([A-Za-z0-9_\-\?\!\*]+)")
RUST_FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?fn\s+([A-Za-z_]\w*)")
GO_FUNC_RE = re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_]\w*)\s*\(")
CSHARP_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected|internal|static|extern|unsafe|virtual|override|async|\s)+"
    r"[\w<>\[\], ?*]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)"
)
TS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s+([A-Za-z_]\w*)\s*\(|([A-Za-z_]\w*)\s*\([^)]*\)\s*(?::\s*[\w<>\[\]| ]+)?\s*\{)")
OCAML_LET_RE = re.compile(r"^\s*(?:let(?:\s+rec)?|and)\s+([a-z_][a-zA-Z0-9_']*)\b")
CALL_TOKEN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "sizeof", "alignof",
    "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast", "new",
    "delete", "throw", "else", "match", "when", "let", "fun", "function",
    "assert", "super", "this", "typeof", "decltype",
})


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _find_brace_end(lines: list[str], start_idx: int) -> int:
    depth = 0
    started = False
    for idx in range(start_idx, min(len(lines), start_idx + 800)):
        line = lines[idx]
        for ch in line:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth <= 0:
                    return idx + 1
    return min(len(lines), start_idx + 1)


def _walk_source_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def _extract_python_file(path: Path, rel: str, repo: str, text: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    syms: list[dict[str, Any]] = []
    raw_calls: list[tuple[str, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return syms, raw_calls
    fkey = f"{repo}/{rel}"
    is_test_file = "test" in rel.lower()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            skey = f"{repo}:python:{rel}:{node.name}"
            line_start = getattr(node, "lineno", 1)
            line_end = getattr(node, "end_lineno", line_start)
            syms.append({
                "key": skey,
                "name": node.name,
                "kind": "function",
                "file_path": fkey,
                "file_key": fkey,
                "line_start": line_start,
                "line_end": line_end,
                "repo": repo,
                "language": "python",
                "is_test": is_test_file or node.name.startswith("test_"),
                "is_alloc_or_stdio": False,
            })
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Name):
                        raw_calls.append((skey, sub.func.id))
                    elif isinstance(sub.func, ast.Attribute):
                        raw_calls.append((skey, sub.func.attr))
    return syms, raw_calls


def _extract_brace_file(
    path: Path,
    rel: str,
    repo: str,
    lang: str,
    pattern: re.Pattern[str],
    text: str,
    *,
    multiline_sig: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    syms: list[dict[str, Any]] = []
    raw_calls: list[tuple[str, str]] = []
    lines = text.splitlines()
    fkey = f"{repo}/{rel}"
    is_test_file = "test" in rel.lower()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = pattern.match(line)
        sig_end = i
        # Handle multi-line C++/C#/Java function signatures (up to 6 continuation lines)
        if not m and multiline_sig and line.strip() and not line.lstrip().startswith(("//", "/*", "*", "#")):
            stripped = line.strip()
            if "(" in stripped and ";" not in stripped and "{" not in stripped:
                joined = stripped
                for j in range(i + 1, min(len(lines), i + 7)):
                    joined += " " + lines[j].strip()
                    sig_end = j
                    if "{" in lines[j] or ";" in lines[j]:
                        break
                if "{" in joined and ";" not in joined.split("{", 1)[0]:
                    m = pattern.match(joined)
        if not m:
            i += 1
            continue
        name = next((g for g in m.groups() if g), "")
        if not name or name in KEYWORDS or len(name) < 2:
            i += 1
            continue
        line_start = i + 1
        line_end = _find_brace_end(lines, sig_end) if "{" in "\n".join(lines[i : sig_end + 1]) else sig_end + 1
        body = "\n".join(lines[i:line_end])
        skey = f"{repo}:{lang}:{rel}:{name}"
        syms.append({
            "key": skey,
            "name": name,
            "kind": "method" if lang in ("java", "kotlin", "scala", "csharp") else "function",
            "file_path": fkey,
            "file_key": fkey,
            "line_start": line_start,
            "line_end": line_end,
            "repo": repo,
            "language": lang,
            "is_test": is_test_file or name.lower().startswith("test"),
            "is_alloc_or_stdio": bool(ALLOC_STDIO_RE.search(body)),
        })
        for cm in CALL_TOKEN_RE.finditer(body):
            callee = cm.group(1)
            if callee != name and callee not in KEYWORDS:
                raw_calls.append((skey, callee))
        i = sig_end + 1
    return syms, raw_calls


def _extract_ocaml_file(rel: str, repo: str, text: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    syms: list[dict[str, Any]] = []
    raw_calls: list[tuple[str, str]] = []
    lines = text.splitlines()
    fkey = f"{repo}/{rel}"
    is_test_file = "test" in rel.lower()
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = OCAML_LET_RE.match(line)
        if m:
            starts.append((i, m.group(1)))
    for idx, (sline, name) in enumerate(starts):
        eline = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        skey = f"{repo}:ocaml:{rel}:{name}"
        syms.append({
            "key": skey,
            "name": name,
            "kind": "function",
            "file_path": fkey,
            "file_key": fkey,
            "line_start": sline + 1,
            "line_end": max(sline + 1, eline),
            "repo": repo,
            "language": "ocaml",
            "is_test": is_test_file,
            "is_alloc_or_stdio": False,
        })
    return syms, raw_calls


def _extract_cabi_declarations(core_dir: Path) -> list[dict[str, Any]]:
    header = core_dir / "impulse-cpp" / "include" / "impulse_graph.h"
    if not header.is_file():
        return []
    text = header.read_text(encoding="utf-8", errors="replace")
    cabi: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in CABI_DECL_RE.finditer(text):
        fname = m.group(1)
        if fname in seen:
            continue
        seen.add(fname)
        line_no = text.count("\n", 0, m.start()) + 1
        sig = " ".join(m.group(0).split())[:180]
        cabi.append({
            "key": f"cabi:{fname}",
            "name": fname,
            "header_file": "core/impulse-cpp/include/impulse_graph.h",
            "line_start": line_no,
            "signature": sig,
        })
    return cabi


def _extract_spec_vectors(spec_dir: Path, valid_opcodes: set[str]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    vec_root = spec_dir / "test-vectors" / "vm-impas"
    if not vec_root.is_dir():
        return [], []
    vectors: list[dict[str, Any]] = []
    vec_edges: list[tuple[str, str]] = []
    for p in sorted(vec_root.rglob("*.impas")):
        rel = str(p.relative_to(spec_dir))
        text = p.read_text(encoding="utf-8", errors="replace")
        expect_m = re.search(r";\s*expect_status\s*:\s*([A-Z0-9_]+)", text, re.IGNORECASE)
        cat = p.parent.name
        vkey = f"vec:{rel}"
        vectors.append({
            "key": vkey,
            "category": cat,
            "expect_status": expect_m.group(1) if expect_m else "IMPULSE_VM_OK",
        })
        for op_tok in set(OP_REF_RE.findall(text)):
            if op_tok in valid_opcodes:
                vec_edges.append((vkey, op_tok))
    return vectors, vec_edges


def build_code_index(
    ws: WorkspacePaths,
    *,
    run_scip: bool = False,
    use_docker: bool = False,
    scip_dir: Path | None = None,
) -> dict[str, Any]:
    """Build unified polyglot CodeIndex (SCIP-first + graceful fallback + C-ABI & Opcode bridge)."""
    s_dir = scip_dir or (graph_data_dir(ws) / "scip")
    scip_run_status: dict[str, str] = {}
    if run_scip:
        if use_docker:
            scip_run_status = run_docker_scip_indexers(ws.root, ws.repos["verification"], s_dir)
        else:
            scip_run_status = run_available_scip_indexers(ws.repos, s_dir)

    source_files: dict[str, dict[str, Any]] = {}
    symbols: dict[str, dict[str, Any]] = {}
    calls: set[tuple[str, str]] = set()
    implements_or_extends: set[tuple[str, str]] = set()
    raw_calls: list[tuple[str, str]] = []
    mode_by_repo: dict[str, str] = {}

    # 1. Ingest any available *.scip files first
    scip_covered_repos: set[str] = set()
    if s_dir.is_dir():
        for sp in sorted(s_dir.glob("*.scip")):
            repo_guess = sp.stem.split("_")[0]
            if repo_guess not in ws.repos:
                continue
            try:
                ir = scip_index_to_ir(decode_scip_file(sp), repo_name=repo_guess)
            except Exception:
                continue
            scip_covered_repos.add(repo_guess)
            mode_by_repo[repo_guess] = f"scip ({sp.name})"
            for sf in ir["source_files"]:
                source_files[sf["key"]] = sf
            for sym in ir["symbols"]:
                symbols[sym["key"]] = sym
            for edge in ir["calls"]:
                calls.add((edge[0], edge[1]))
            for edge in ir["implements_or_extends"]:
                implements_or_extends.add((edge[0], edge[1]))

    # 2. Fallback extraction for repos without SCIP indexes (and always enrich file content_sha256)
    scan_specs = [
        ("core", ws.repos["core"] / "impulse-cpp", {".cpp", ".cc", ".h", ".hpp"}, "cpp"),
        ("core", ws.repos["core"] / "impulse-rust", {".rs"}, "rust"),
        ("core", ws.repos["core"] / "impulse-cli", {".rs"}, "rust"),
        ("core", ws.repos["core"] / "impulse-python", {".py"}, "python"),
        ("core", ws.repos["core"] / "impulse-python" / "src", {".cpp", ".cc", ".h", ".hpp"}, "cpp"),
        ("core", ws.repos["core"] / "impulse-go", {".go"}, "go"),
        ("core", ws.repos["core"] / "impulse-node", {".cpp", ".cc", ".h", ".hpp"}, "cpp"),
        ("core", ws.repos["core"] / "impulse-node", {".ts", ".js"}, "ts"),
        ("core", ws.repos["core"] / "impulse-dotnet", {".cs"}, "csharp"),
        ("java", ws.repos["java"], {".java"}, "java"),
        ("java", ws.repos["java"], {".kt"}, "kotlin"),
        ("java", ws.repos["java"], {".scala"}, "scala"),
        ("java", ws.repos["java"], {".clj"}, "clojure"),
        ("ocaml", ws.repos["ocaml"], {".ml", ".mli"}, "ocaml"),
        ("verification", ws.repos["verification"] / "tools", {".py"}, "python"),
        ("spec", ws.repos["spec"] / "tools", {".py"}, "python"),
    ]

    file_texts: dict[str, str] = {}
    for repo_k, scan_root, exts, lang in scan_specs:
        repo_root = ws.repos[repo_k]
        if not scan_root.is_dir():
            continue
        if repo_k not in scip_covered_repos:
            mode_by_repo.setdefault(repo_k, "fallback")
        for p in _walk_source_files(scan_root, exts):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(p.relative_to(repo_root))
            fkey = f"{repo_k}/{rel}"
            file_texts[fkey] = text
            source_files[fkey] = {
                "key": fkey,
                "repo": repo_k,
                "language": lang,
                "line_count": text.count("\n") + 1,
                "content_sha256": _sha256_text(text),
            }
            if repo_k in scip_covered_repos:
                continue
            if lang == "python":
                fsyms, fcalls = _extract_python_file(p, rel, repo_k, text)
            elif lang == "cpp":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "cpp", CPP_FUNC_RE, text, multiline_sig=True)
            elif lang == "java":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "java", JAVA_METHOD_RE, text, multiline_sig=True)
            elif lang == "kotlin":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "kotlin", KOTLIN_FN_RE, text)
            elif lang == "scala":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "scala", SCALA_DEF_RE, text)
            elif lang == "clojure":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "clojure", CLOJURE_DEF_RE, text)
            elif lang == "rust":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "rust", RUST_FN_RE, text)
            elif lang == "go":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "go", GO_FUNC_RE, text)
            elif lang == "csharp":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "csharp", CSHARP_METHOD_RE, text, multiline_sig=True)
            elif lang == "ts":
                fsyms, fcalls = _extract_brace_file(p, rel, repo_k, "ts", TS_FUNC_RE, text)
            elif lang == "ocaml":
                fsyms, fcalls = _extract_ocaml_file(rel, repo_k, text)
            else:
                fsyms, fcalls = [], []
            for s in fsyms:
                symbols[s["key"]] = s
            raw_calls.extend(fcalls)

    # Resolve fallback raw_calls within same repo (exact name match)
    syms_by_repo_name: dict[tuple[str, str], list[str]] = {}
    for skey, s in symbols.items():
        syms_by_repo_name.setdefault((s["repo"], s["name"]), []).append(skey)
    for caller_key, callee_name in raw_calls:
        repo_k = symbols[caller_key]["repo"]
        targets = syms_by_repo_name.get((repo_k, callee_name), [])
        for tkey in targets[:3]:
            if tkey != caller_key:
                calls.add((caller_key, tkey))

    # 3. Semantic Bridge Linker: C-ABI symbols + Opcode emitters/handlers + SpecVectors
    cabi_symbols = _extract_cabi_declarations(ws.repos["core"])
    cabi_names = {c["name"]: c["key"] for c in cabi_symbols}

    opcodes = load_spec_opcodes(ws.repos["spec"])
    valid_op_names = {op.name for op in opcodes}
    spec_vectors, vec_exercises_opcode = _extract_spec_vectors(ws.repos["spec"], valid_op_names)

    implements_cabi: set[tuple[str, str]] = set()
    calls_cabi: set[tuple[str, str]] = set()
    emits_opcode: set[tuple[str, str]] = set()
    implements_opcode: set[tuple[str, str]] = set()

    # Group symbols by file for fast body-span lookup
    syms_by_file: dict[str, list[dict[str, Any]]] = {}
    for s in symbols.values():
        syms_by_file.setdefault(s["file_key"], []).append(s)

    for fkey, text in file_texts.items():
        flines = text.splitlines()
        fl_lower = fkey.lower()
        is_compiler_file = any(
            tok in fl_lower
            for tok in ("compiler", "impscm", "cypher", "cel", "datalog", "impk", "implog", "emit", "assembler")
        )
        is_vm_handler_file = any(
            tok in fl_lower
            for tok in ("impulse_vm", "impulsevm", "impulsepillar", "impulsejit", "impulse-vm", "impulse-ocaml/lib/", "vm_")
        )
        for s in syms_by_file.get(fkey, []):
            skey = s["key"]
            if s["name"] in cabi_names and s["repo"] == "core" and s["language"] == "cpp":
                implements_cabi.add((skey, cabi_names[s["name"]]))
            body = "\n".join(flines[max(0, s["line_start"] - 1) : min(len(flines), s["line_end"])])
            for cm in CABI_CALL_RE.finditer(body):
                cname = cm.group(1)
                if cname in cabi_names and cname != s["name"]:
                    calls_cabi.add((skey, cabi_names[cname]))
            if is_compiler_file or is_vm_handler_file:
                for op_tok in set(OP_REF_RE.findall(body)):
                    if op_tok in valid_op_names:
                        if is_compiler_file:
                            emits_opcode.add((skey, op_tok))
                        if is_vm_handler_file:
                            implements_opcode.add((skey, op_tok))

    result = {
        "scip_run_status": scip_run_status,
        "mode_by_repo": mode_by_repo,
        "source_files": list(source_files.values()),
        "symbols": list(symbols.values()),
        "cabi_symbols": cabi_symbols,
        "spec_vectors": spec_vectors,
        "calls": sorted(calls),
        "implements_or_extends": sorted(implements_or_extends),
        "implements_cabi": sorted(implements_cabi),
        "calls_cabi": sorted(calls_cabi),
        "emits_opcode": sorted(emits_opcode),
        "implements_opcode": sorted(implements_opcode),
        "vec_exercises_opcode": sorted(set(vec_exercises_opcode)),
        "covers_symbol": [],
    }
    cache_path = graph_data_dir(ws) / "code_index.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    return result
