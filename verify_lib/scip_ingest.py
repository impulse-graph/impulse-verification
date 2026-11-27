"""Zero-dependency SCIP (scip.proto proto3) wire decoder & indexer orchestrator."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# SCIP SymbolRole bitmask (from scip.proto)
ROLE_DEFINITION = 0x1
ROLE_IMPORT = 0x2
ROLE_WRITE_ACCESS = 0x4
ROLE_READ_ACCESS = 0x8

# Subset of SCIP SymbolInformation.Kind enum
SCIP_KIND_NAMES = {
    0: "unspecified",
    7: "class",
    17: "function",
    21: "interface",
    26: "method",
    29: "module",
    49: "struct",
    53: "trait",
    54: "type",
}


@dataclass
class ScipRelationship:
    symbol: str
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    is_definition: bool = False


@dataclass
class ScipSymbolInfo:
    symbol: str
    kind: int = 0
    display_name: str = ""
    enclosing_symbol: str = ""
    relationships: list[ScipRelationship] = field(default_factory=list)


@dataclass
class ScipOccurrence:
    range: list[int]
    symbol: str
    symbol_roles: int = 0
    enclosing_range: list[int] = field(default_factory=list)


@dataclass
class ScipDocument:
    relative_path: str
    language: str = ""
    occurrences: list[ScipOccurrence] = field(default_factory=list)
    symbols: list[ScipSymbolInfo] = field(default_factory=list)


@dataclass
class ScipIndex:
    project_root: str = ""
    documents: list[ScipDocument] = field(default_factory=list)
    external_symbols: list[ScipSymbolInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimal proto3 wire reader (varint, zigzag/int32, length-delimited, packed)
# ---------------------------------------------------------------------------
def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            val, pos = _read_varint(buf, pos)
            yield field_num, wire_type, val
        elif wire_type == 1:
            val = buf[pos : pos + 8]
            pos += 8
            yield field_num, wire_type, val
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            val = buf[pos : pos + length]
            pos += length
            yield field_num, wire_type, val
        elif wire_type == 5:
            val = buf[pos : pos + 4]
            pos += 4
            yield field_num, wire_type, val
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type} at offset {pos}")


def _decode_packed_int32(buf: bytes) -> list[int]:
    vals: list[int] = []
    pos = 0
    end = len(buf)
    while pos < end:
        v, pos = _read_varint(buf, pos)
        if v >= (1 << 31):
            v -= 1 << 32
        vals.append(v)
    return vals


def _decode_relationship(buf: bytes) -> ScipRelationship:
    rel = ScipRelationship(symbol="")
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 1 and wtype == 2:
            rel.symbol = val.decode("utf-8", errors="replace")
        elif fnum == 2 and wtype == 0:
            rel.is_reference = bool(val)
        elif fnum == 3 and wtype == 0:
            rel.is_implementation = bool(val)
        elif fnum == 4 and wtype == 0:
            rel.is_type_definition = bool(val)
        elif fnum == 5 and wtype == 0:
            rel.is_definition = bool(val)
    return rel


def _decode_symbol_info(buf: bytes) -> ScipSymbolInfo:
    info = ScipSymbolInfo(symbol="")
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 1 and wtype == 2:
            info.symbol = val.decode("utf-8", errors="replace")
        elif fnum == 4 and wtype == 2:
            info.relationships.append(_decode_relationship(val))
        elif fnum == 5 and wtype == 0:
            info.kind = int(val)
        elif fnum == 6 and wtype == 2:
            info.display_name = val.decode("utf-8", errors="replace")
        elif fnum == 8 and wtype == 2:
            info.enclosing_symbol = val.decode("utf-8", errors="replace")
    return info


def _decode_occurrence(buf: bytes) -> ScipOccurrence:
    occ = ScipOccurrence(range=[], symbol="")
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 1:
            if wtype == 2:
                occ.range.extend(_decode_packed_int32(val))
            elif wtype == 0:
                occ.range.append(int(val))
        elif fnum == 2 and wtype == 2:
            occ.symbol = val.decode("utf-8", errors="replace")
        elif fnum == 3 and wtype == 0:
            occ.symbol_roles = int(val)
        elif fnum == 7:
            if wtype == 2:
                occ.enclosing_range.extend(_decode_packed_int32(val))
            elif wtype == 0:
                occ.enclosing_range.append(int(val))
    return occ


def _decode_document(buf: bytes) -> ScipDocument:
    doc = ScipDocument(relative_path="")
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 1 and wtype == 2:
            doc.relative_path = val.decode("utf-8", errors="replace")
        elif fnum == 2 and wtype == 2:
            doc.occurrences.append(_decode_occurrence(val))
        elif fnum == 3 and wtype == 2:
            doc.symbols.append(_decode_symbol_info(val))
        elif fnum == 4 and wtype == 2:
            doc.language = val.decode("utf-8", errors="replace")
    return doc


def _decode_metadata(buf: bytes) -> str:
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 3 and wtype == 2:
            return val.decode("utf-8", errors="replace")
    return ""


def decode_scip_bytes(data: bytes) -> ScipIndex:
    """Decode a binary SCIP protobuf Index payload."""
    idx = ScipIndex()
    for fnum, wtype, val in _iter_fields(data):
        if fnum == 1 and wtype == 2:
            idx.project_root = _decode_metadata(val)
        elif fnum == 2 and wtype == 2:
            idx.documents.append(_decode_document(val))
        elif fnum == 3 and wtype == 2:
            idx.external_symbols.append(_decode_symbol_info(val))
    return idx


def decode_scip_file(path: Path) -> ScipIndex:
    return decode_scip_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Minimal proto3 wire encoder (used for tests & fixture generation)
# ---------------------------------------------------------------------------
def _enc_varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 32
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _enc_tag(field_num: int, wire_type: int) -> bytes:
    return _enc_varint((field_num << 3) | wire_type)


def _enc_bytes(field_num: int, payload: bytes) -> bytes:
    return _enc_tag(field_num, 2) + _enc_varint(len(payload)) + payload


def _enc_str(field_num: int, text: str) -> bytes:
    return _enc_bytes(field_num, text.encode("utf-8"))


def _enc_packed_int32(field_num: int, nums: list[int]) -> bytes:
    if not nums:
        return b""
    body = b"".join(_enc_varint(n) for n in nums)
    return _enc_bytes(field_num, body)


def encode_scip_index(idx: ScipIndex) -> bytes:
    """Encode a ScipIndex into standard scip.proto binary wire format."""
    parts: list[bytes] = []
    if idx.project_root:
        meta = _enc_str(3, idx.project_root)
        parts.append(_enc_bytes(1, meta))
    for doc in idx.documents:
        dparts: list[bytes] = []
        if doc.relative_path:
            dparts.append(_enc_str(1, doc.relative_path))
        for occ in doc.occurrences:
            oparts: list[bytes] = []
            if occ.range:
                oparts.append(_enc_packed_int32(1, occ.range))
            if occ.symbol:
                oparts.append(_enc_str(2, occ.symbol))
            if occ.symbol_roles:
                oparts.append(_enc_tag(3, 0) + _enc_varint(occ.symbol_roles))
            if occ.enclosing_range:
                oparts.append(_enc_packed_int32(7, occ.enclosing_range))
            dparts.append(_enc_bytes(2, b"".join(oparts)))
        for sym in doc.symbols:
            sparts: list[bytes] = []
            if sym.symbol:
                sparts.append(_enc_str(1, sym.symbol))
            for rel in sym.relationships:
                rparts: list[bytes] = [_enc_str(1, rel.symbol)]
                if rel.is_reference:
                    rparts.append(_enc_tag(2, 0) + _enc_varint(1))
                if rel.is_implementation:
                    rparts.append(_enc_tag(3, 0) + _enc_varint(1))
                if rel.is_type_definition:
                    rparts.append(_enc_tag(4, 0) + _enc_varint(1))
                if rel.is_definition:
                    rparts.append(_enc_tag(5, 0) + _enc_varint(1))
                sparts.append(_enc_bytes(4, b"".join(rparts)))
            if sym.kind:
                sparts.append(_enc_tag(5, 0) + _enc_varint(sym.kind))
            if sym.display_name:
                sparts.append(_enc_str(6, sym.display_name))
            if sym.enclosing_symbol:
                sparts.append(_enc_str(8, sym.enclosing_symbol))
            dparts.append(_enc_bytes(3, b"".join(sparts)))
        if doc.language:
            dparts.append(_enc_str(4, doc.language))
        parts.append(_enc_bytes(2, b"".join(dparts)))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Convert ScipIndex -> normalized CodeIndex fragment
# ---------------------------------------------------------------------------
def _norm_range(r: list[int]) -> tuple[int, int, int, int]:
    """Normalize 3-int or 4-int SCIP range (0-indexed) into (start_line, start_col, end_line, end_col)."""
    if len(r) == 3:
        return (r[0], r[1], r[0], r[2])
    if len(r) >= 4:
        return (r[0], r[1], r[2], r[3])
    return (0, 0, 0, 0)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    o_sl, o_sc, o_el, o_ec = outer
    i_sl, i_sc, i_el, i_ec = inner
    return (o_sl, o_sc) <= (i_sl, i_sc) and (i_el, i_ec) <= (o_el, o_ec)


def _short_display_name(scip_symbol: str, display_name: str) -> str:
    if display_name:
        return display_name
    cleaned = scip_symbol.rstrip(".#()!")
    for sep in ("#", "/", ".", "::", " "):
        if sep in cleaned:
            cleaned = cleaned.rsplit(sep, 1)[-1]
    return cleaned or scip_symbol


def scip_index_to_ir(idx: ScipIndex, *, repo_name: str) -> dict[str, Any]:
    """Convert a decoded ScipIndex into normalized source_files, symbols, calls, and implements_or_extends."""
    source_files: list[dict[str, Any]] = []
    symbols: dict[str, dict[str, Any]] = {}
    calls: set[tuple[str, str]] = set()
    implements: set[tuple[str, str]] = set()

    for doc in idx.documents:
        rel = doc.relative_path.lstrip("./")
        fkey = f"{repo_name}/{rel}"
        lang = doc.language or "unknown"
        source_files.append(
            {
                "key": fkey,
                "repo": repo_name,
                "language": lang,
                "line_count": 0,
                "content_sha256": "",
            }
        )
        sym_meta: dict[str, ScipSymbolInfo] = {s.symbol: s for s in doc.symbols}

        # Pass 1: collect definition occurrences + enclosing ranges
        defs_in_doc: list[tuple[tuple[int, int, int, int], str]] = []
        for occ in doc.occurrences:
            if not occ.symbol or occ.symbol.startswith("local "):
                continue
            if occ.symbol_roles & ROLE_DEFINITION:
                span = _norm_range(occ.enclosing_range if occ.enclosing_range else occ.range)
                meta = sym_meta.get(occ.symbol)
                kind_name = SCIP_KIND_NAMES.get(meta.kind, "symbol") if meta else "symbol"
                dname = _short_display_name(occ.symbol, meta.display_name if meta else "")
                skey = f"{repo_name}:{lang}:{occ.symbol}"
                symbols[skey] = {
                    "key": skey,
                    "scip_id": occ.symbol,
                    "name": dname,
                    "kind": kind_name,
                    "file_path": f"{repo_name}/{rel}",
                    "file_key": fkey,
                    "line_start": span[0] + 1,
                    "line_end": span[2] + 1,
                    "repo": repo_name,
                    "language": lang,
                    "is_test": ("test" in rel.lower() or "test" in dname.lower()),
                    "is_alloc_or_stdio": False,
                }
                defs_in_doc.append((span, skey))

                if meta:
                    for r in meta.relationships:
                        if r.is_implementation and r.symbol:
                            implements.add((skey, f"{repo_name}:{lang}:{r.symbol}"))

        # Sort definitions smallest-span-first so tightest enclosing scope wins
        defs_in_doc.sort(key=lambda item: (item[0][2] - item[0][0], item[0][3] - item[0][1]))

        # Pass 2: map reference occurrences inside enclosing definition spans -> CALLS
        for occ in doc.occurrences:
            if not occ.symbol or occ.symbol.startswith("local "):
                continue
            if occ.symbol_roles & ROLE_DEFINITION:
                continue
            ref_span = _norm_range(occ.range)
            enclosing_key: str | None = None
            for def_span, def_skey in defs_in_doc:
                if _contains(def_span, ref_span):
                    enclosing_key = def_skey
                    break
            if enclosing_key:
                callee_key = f"{repo_name}:{lang}:{occ.symbol}"
                if callee_key != enclosing_key:
                    calls.add((enclosing_key, callee_key))

    return {
        "source_files": source_files,
        "symbols": list(symbols.values()),
        "calls": sorted(calls),
        "implements_or_extends": sorted(implements),
    }


def run_available_scip_indexers(ws_repos: dict[str, Path], scip_out_dir: Path) -> dict[str, str]:
    """Run any installed official SCIP indexers (scip-clang, scip-java, rust-analyzer, scip-python)."""
    scip_out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    # 1. C++ via scip-clang
    compdb = ws_repos["core"] / "impulse-cpp" / "build_on" / "compile_commands.json"
    if shutil.which("scip-clang") and compdb.is_file():
        out = scip_out_dir / "core_cpp.scip"
        res = subprocess.run(
            ["scip-clang", f"--compdb-path={compdb}", f"--index-output-path={out}"],
            cwd=ws_repos["core"] / "impulse-cpp",
            capture_output=True,
            text=True,
        )
        status["core_cpp"] = "scip-clang OK" if res.returncode == 0 else f"scip-clang failed: {res.stderr[:120]}"
    else:
        status["core_cpp"] = "skipped (scip-clang or compile_commands.json unavailable)"

    # 2. Java/Kotlin/Scala via scip-java
    if shutil.which("scip-java") and ws_repos["java"].is_dir():
        out = scip_out_dir / "java.scip"
        res = subprocess.run(
            ["scip-java", "index", "--output", str(out)],
            cwd=ws_repos["java"],
            capture_output=True,
            text=True,
        )
        status["java"] = "scip-java OK" if res.returncode == 0 else f"scip-java failed: {res.stderr[:120]}"
    else:
        status["java"] = "skipped (scip-java not on PATH)"

    # 3. Rust via rust-analyzer scip
    if shutil.which("rust-analyzer") and (ws_repos["core"] / "impulse-rust").is_dir():
        out = scip_out_dir / "core_rust.scip"
        res = subprocess.run(
            ["rust-analyzer", "scip", ".", "--output", str(out)],
            cwd=ws_repos["core"] / "impulse-rust",
            capture_output=True,
            text=True,
        )
        status["core_rust"] = "rust-analyzer OK" if res.returncode == 0 else f"rust-analyzer failed: {res.stderr[:120]}"
    else:
        status["core_rust"] = "skipped (rust-analyzer not on PATH)"

    # 4. Python via scip-python
    if shutil.which("scip-python") and ws_repos["verification"].is_dir():
        out = scip_out_dir / "verification_py.scip"
        res = subprocess.run(
            ["scip-python", "index", ".", "--project-name", "impulse-verification", "--output", str(out)],
            cwd=ws_repos["verification"],
            capture_output=True,
            text=True,
        )
        status["verification_py"] = "scip-python OK" if res.returncode == 0 else f"scip-python failed: {res.stderr[:120]}"
    else:
        status["verification_py"] = "skipped (scip-python not on PATH)"

    return status


def run_docker_scip_indexers(
    ws_root: Path,
    verification_dir: Path,
    scip_out_dir: Path,
    *,
    image: str = "impulse-scip-indexer:latest",
) -> dict[str, str]:
    """Build (if needed) and run docker/Dockerfile.scip to generate *.scip indexes into scip_out_dir."""
    if not shutil.which("docker"):
        return {"docker": "skipped (docker not on PATH)"}

    scip_out_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = verification_dir / "docker" / "Dockerfile.scip"
    if not dockerfile.is_file():
        return {"docker": f"missing {dockerfile}"}

    # Build image if not already present locally
    inspect = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    if inspect.returncode != 0:
        build_res = subprocess.run(
            [
                "docker", "build", "--platform", "linux/amd64",
                "-f", str(dockerfile), "-t", image, str(verification_dir),
            ],
            capture_output=True,
            text=True,
        )
        if build_res.returncode != 0:
            return {"docker": f"docker build failed: {(build_res.stderr or build_res.stdout)[-240:]}"}

    run_res = subprocess.run(
        [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{ws_root}:/workspace:ro",
            "-v", f"{scip_out_dir}:/out",
            image,
        ],
        capture_output=True,
        text=True,
    )
    generated = sorted(p.name for p in scip_out_dir.glob("*.scip"))
    if run_res.returncode != 0 and not generated:
        return {"docker": f"docker run failed: {(run_res.stderr or run_res.stdout)[-240:]}"}
    return {"docker": f"OK (generated: {', '.join(generated) or 'none'})"}
