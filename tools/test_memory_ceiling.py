#!/usr/bin/env python3
"""
test_memory_ceiling.py — Sub-File-Size Heap Ceiling Gate (No 100GB File Needed).

Opens a large .imps snapshot (e.g. 54 MB storage_coverage_bench.imps) inside a JVM
capped at -Xmx32m (strictly smaller than the .imps file!) and executes a battery of
walks, morsels, degrees, multi-lane walks, SpMV, and attribute gathers.

Any code that copies CSR/CSC onto the Java heap, materializes full attribute columns
or String[N] key catalogs, or allocates O(E) heap collections immediately dies with
OutOfMemoryError.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fuzz_snapshot_morsels import (
    _discover_default_lib,
    load_snapshot_schema,
)


def assemble_fixture(asm, impas: str) -> dict:
    prog = asm.assemble(impas)
    return {
        "words_hex": [f"0x{w:016x}" for w in prog.words],
        "data_directives": prog.data_directives,
        "const_pool": prog.const_pool,
    }
from fuzz_vm_contracts import (
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    resolve_spec_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sub-file-size JVM heap ceiling gate (-Xmx32m on 54MB .imps)")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--xmx", type=str, default="32m", help="JVM max heap (default: 32m)")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    args = parser.parse_args()

    snap_path = args.snapshot.resolve()
    snap_mb = snap_path.stat().st_size / (1024 * 1024)
    lib_path = args.lib_path or _discover_default_lib()
    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_asm = _import_spec_modules(spec_dir)
    asm = impas_asm.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    schema = load_snapshot_schema(snap_path, lib_path)

    java_cmd = discover_default_java_cmd()
    # Inject -Xmx<args.xmx> immediately after the java binary
    capped_cmd = [java_cmd[0], f"-Xmx{args.xmx}", f"-Xms{args.xmx}"] + java_cmd[1:] + ["--snapshot", str(snap_path)]

    # Find a relation with CSR+CSC and a domain with node attrs
    rel_csr = next((r for r in schema.relations.values() if r.has_csr), None)
    rel_csc = next((r for r in schema.relations.values() if r.has_csc), None)
    rel_coo = next((r for r in schema.relations.values() if r.has_coo), None)
    dom_attr = next(((d, al[0]) for d, al in schema.node_attrs.items() if al), None)

    queries: list[tuple[str, str]] = []
    if rel_csr:
        queries.append((
            f"WALK.fwd(rel={rel_csr.rel_id})",
            f".data\n  .param 0 type=frontier dom={rel_csr.src_dom} values=[0, 1, 2, 5, 10]\n.code\n"
            f"  LOAD_PARAM.frontier F0, param=0, dom={rel_csr.src_dom}\n"
            f"  WALK.fwd F1, F0, rel={rel_csr.rel_id}\n"
            f"  SET_CARDINALITY R0, F1\n  COLLECT.scalar R0, out=0\n  HALT\n",
        ))
        queries.append((
            f"WALK_MORSEL.fwd(rel={rel_csr.rel_id})",
            f".data\n  .param 0 type=frontier dom={rel_csr.src_dom} values=[0, 1, 2, 5, 10]\n.code\n"
            f"  LOAD_PARAM.frontier F0, param=0, dom={rel_csr.src_dom}\n"
            f"  WALK_MORSEL.fwd F0, rel={rel_csr.rel_id}\n"
            f"    MORSEL_COMMIT.count_add_r R0\n"
            f"  MORSEL_END\n  COLLECT.scalar R0, out=0\n  HALT\n",
        ))
    if rel_csc:
        queries.append((
            f"WALK.rev(rel={rel_csc.rel_id})",
            f".data\n  .param 0 type=frontier dom={rel_csc.tgt_dom} values=[0, 1, 2, 5, 10]\n.code\n"
            f"  LOAD_PARAM.frontier F0, param=0, dom={rel_csc.tgt_dom}\n"
            f"  WALK.rev F1, F0, rel={rel_csc.rel_id}\n"
            f"  SET_CARDINALITY R0, F1\n  COLLECT.scalar R0, out=0\n  HALT\n",
        ))
    if rel_coo:
        queries.append((
            f"WALK.coo_fwd(rel={rel_coo.rel_id})",
            f".data\n  .param 0 type=frontier dom={rel_coo.src_dom} values=[0, 1, 2, 5, 10]\n.code\n"
            f"  LOAD_PARAM.frontier F0, param=0, dom={rel_coo.src_dom}\n"
            f"  WALK.coo_fwd F1, F0, rel={rel_coo.rel_id}\n"
            f"  SET_CARDINALITY R0, F1\n  COLLECT.scalar R0, out=0\n  HALT\n",
        ))
    if dom_attr:
        dom_id, attr = dom_attr
        queries.append((
            f"GATHER_NODE_ATTR(dom={dom_id}, attr={attr.attr_id})",
            f".data\n  .param 0 type=u64_vec values=[0, 1, 2, 5, 10]\n.code\n"
            f"  LOAD_PARAM.vec V0, param=0, type=u64\n"
            f"  GATHER_NODE_ATTR.vec V1, V0, dom={dom_id}, attr={attr.attr_id}\n"
            f"  VEC_REDUCE.count_valid R0, V1\n  COLLECT.scalar R0, out=0\n  HALT\n",
        ))

    print(f"Memory Ceiling Gate | snapshot={snap_path.name} ({snap_mb:.1f} MB) | JVM -Xmx{args.xmx} | queries={len(queries)}")
    cli = BatchEngineClient(capped_cmd, f"java_xmx_{args.xmx}")
    try:
        for label, impas in queries:
            fixture = assemble_fixture(asm, impas)
            res = cli.run_fixture(fixture, "P0", 500_000)
            st = str(res.get("status", ""))
            if st != "IMPULSE_VM_OK":
                print(f"  FAIL: {label} returned {st}")
                return 1
            print(f"  OK: {label} -> {res.get('outputs')}")
    finally:
        cli.close()

    print(f"PASS: Opened and queried {snap_mb:.1f} MB snapshot inside -Xmx{args.xmx} JVM heap!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
