#!/usr/bin/env python3
"""
Generate a self-describing Metamorphic Storage Coverage .imps (v0.11, 0x000B)
exclusively via the official `impulse-graph build` CLI (AGENTS.md §3.17).

Same deterministic logical graph G = (V, E) is mirrored across:
  - Domains  : EQ_DOM_* (u16/u32/u64 dense IDs, i32/i64/u16/string keys)
  - Relations: EQ_REL_* (CSR-only, CSR+CSC u16/u32/u64, mixed widths, CSC-only,
               COO SRC_MAJOR/TGT_MAJOR/UNSORTED, ALL3, plus EQ_FLAT_M1_U32)
  - Attrs    : EQ_NUM_* (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64),
               EQ_NUM_NULL_* (nullable with identical validity bitmap),
               BOUNDARY_FP_* (+0.0, -0.0, +inf, -inf, NaN, NULL, extremes),
               BOUNDARY_INT_* (min/max/sign boundaries, NULL)
  - Metadata : coverage.* keys describing groups, canonical targets, seeds, gadgets.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/Users/jesse/impulse")
DEFAULT_CLI = ROOT / ".worktrees/v11-m3/impulse-graph-core/impulse-cli/target/release/impulse-graph"
FALLBACK_CLI = ROOT / ".worktrees/v11-m3/impulse-graph-core/impulse-cli/target/debug/impulse-graph"


def generate_graph(n: int, avg_deg: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    # Structural gadgets at fixed IDs:
    # 0 = isolated, 1 = self-loop, 2->3->4->2 = 3-cycle, 5 = supernode, 6 = sink
    edges.add((1, 1))
    edges.add((2, 3))
    edges.add((3, 4))
    edges.add((4, 2))
    # Supernode 5 -> 128 distinct targets (crosses P3 and P0 morsel boundaries)
    for t in range(10, min(n, 138)):
        edges.add((5, t))
    # Sink node 6 receives edges from 10..25
    for s in range(10, min(n, 26)):
        edges.add((s, 6))
    # Bitset word-boundary gadgets (63, 64, 65)
    if n > 70:
        edges.add((63, 64))
        edges.add((64, 65))
        edges.add((65, 63))

    # Deterministic RMAT-like power-law body over nodes 7..n-1 (excluding 0 and out-edges from 6)
    target_edges = n * avg_deg
    # Power-law hub bias using squared uniform
    while len(edges) < target_edges:
        u = 7 + int((rng.random() ** 1.6) * (n - 7))
        v = 7 + int((rng.random() ** 1.2) * (n - 7))
        if u != v and u != 0 and u != 6 and v != 0:
            edges.add((u, v))
    return sorted(edges)


FP_BOUNDARY_CYCLE = [
    "0.0",
    "-0.0",
    "1.0",
    "-1.0",
    "inf",
    "-inf",
    "nan",
    "",          # NULL via empty cell
    "1e-30",
    "3.4028235e38",
    "-3.4028235e38",
    "40.0",
]

I64_BOUNDARY_CYCLE = [
    "0",
    "1",
    "-1",
    "127",
    "-128",
    "32767",
    "-32768",
    "",          # NULL
    "2147483647",
    "-2147483648",
    "9223372036854775807",
    "-9223372036854775808",
]

U64_BOUNDARY_CYCLE = [
    "0",
    "1",
    "40",
    "255",
    "65535",
    "4294967295",
    "9223372036854775807",
    "",          # NULL
    "9223372036854775808",
    "18446744073709551615",
    "42",
    "1024",
]


def is_null_idx(i: int) -> bool:
    return i == 0 or i == 63 or i == 64 or (i % 7 == 0)


def write_node_tsv(path: Path, n: int, key_mode: str) -> None:
    headers = [
        "id",
        "EQ_NUM_I8", "EQ_NUM_I16", "EQ_NUM_I32", "EQ_NUM_I64",
        "EQ_NUM_U8", "EQ_NUM_U16", "EQ_NUM_U32", "EQ_NUM_U64",
        "EQ_NUM_F32", "EQ_NUM_F64",
        "EQ_NUM_NULL_I32", "EQ_NUM_NULL_I64", "EQ_NUM_NULL_F32", "EQ_NUM_NULL_F64",
        "BOUNDARY_FP_F32", "BOUNDARY_FP_F64",
        "BOUNDARY_INT_I64", "BOUNDARY_INT_U64",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for i in range(n):
            key = f"node_{i:05d}" if key_mode == "str" else str(i)
            val = i % 100  # fits cleanly in i8 (-128..127), u8 (0..255), f32, f64; node 40 == 40
            v_str = str(val)
            vf_str = f"{val}.0"
            n_int = "" if is_null_idx(i) else v_str
            n_flt = "" if is_null_idx(i) else vf_str
            fp_b = FP_BOUNDARY_CYCLE[i % len(FP_BOUNDARY_CYCLE)]
            i64_b = I64_BOUNDARY_CYCLE[i % len(I64_BOUNDARY_CYCLE)]
            u64_b = U64_BOUNDARY_CYCLE[i % len(U64_BOUNDARY_CYCLE)]
            row = [
                key,
                v_str, v_str, v_str, v_str,
                v_str, v_str, v_str, v_str,
                vf_str, vf_str,
                n_int, n_int, n_flt, n_flt,
                fp_b, fp_b,
                i64_b, u64_b,
            ]
            f.write("\t".join(row) + "\n")


def write_edge_tsv(path: Path, edges: list[tuple[int, int]], with_attrs: bool) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if with_attrs:
            headers = [
                "src", "tgt",
                "EQ_NUM_I8", "EQ_NUM_I16", "EQ_NUM_I32", "EQ_NUM_I64",
                "EQ_NUM_U8", "EQ_NUM_U16", "EQ_NUM_U32", "EQ_NUM_U64",
                "EQ_NUM_F32", "EQ_NUM_F64",
                "EQ_NUM_NULL_I32", "EQ_NUM_NULL_F64",
                "BOUNDARY_FP_F32", "BOUNDARY_FP_F64",
                "BOUNDARY_INT_I64", "BOUNDARY_INT_U64",
            ]
            f.write("\t".join(headers) + "\n")
            for idx, (u, v) in enumerate(edges):
                val = (u + v) % 100
                v_str = str(val)
                vf_str = f"{val}.0"
                n_int = "" if is_null_idx(idx) else v_str
                n_flt = "" if is_null_idx(idx) else vf_str
                fp_b = FP_BOUNDARY_CYCLE[idx % len(FP_BOUNDARY_CYCLE)]
                i64_b = I64_BOUNDARY_CYCLE[idx % len(I64_BOUNDARY_CYCLE)]
                u64_b = U64_BOUNDARY_CYCLE[idx % len(U64_BOUNDARY_CYCLE)]
                f.write("\t".join([
                    str(u), str(v),
                    v_str, v_str, v_str, v_str,
                    v_str, v_str, v_str, v_str,
                    vf_str, vf_str,
                    n_int, n_flt,
                    fp_b, fp_b,
                    i64_b, u64_b,
                ]) + "\n")
        else:
            f.write("src\ttgt\n")
            for u, v in edges:
                f.write(f"{u}\t{v}\n")


NODE_ATTR_DEFS = [
    {"name": "EQ_NUM_I8", "column": "EQ_NUM_I8", "type": "int8"},
    {"name": "EQ_NUM_I16", "column": "EQ_NUM_I16", "type": "int16"},
    {"name": "EQ_NUM_I32", "column": "EQ_NUM_I32", "type": "int32"},
    {"name": "EQ_NUM_I64", "column": "EQ_NUM_I64", "type": "int64"},
    {"name": "EQ_NUM_U8", "column": "EQ_NUM_U8", "type": "uint8"},
    {"name": "EQ_NUM_U16", "column": "EQ_NUM_U16", "type": "uint16"},
    {"name": "EQ_NUM_U32", "column": "EQ_NUM_U32", "type": "uint32"},
    {"name": "EQ_NUM_U64", "column": "EQ_NUM_U64", "type": "uint64"},
    {"name": "EQ_NUM_F32", "column": "EQ_NUM_F32", "type": "float32"},
    {"name": "EQ_NUM_F64", "column": "EQ_NUM_F64", "type": "float64"},
    {"name": "EQ_NUM_NULL_I32", "column": "EQ_NUM_NULL_I32", "type": "int32"},
    {"name": "EQ_NUM_NULL_I64", "column": "EQ_NUM_NULL_I64", "type": "int64"},
    {"name": "EQ_NUM_NULL_F32", "column": "EQ_NUM_NULL_F32", "type": "float32"},
    {"name": "EQ_NUM_NULL_F64", "column": "EQ_NUM_NULL_F64", "type": "float64"},
    {"name": "BOUNDARY_FP_F32", "column": "BOUNDARY_FP_F32", "type": "float32"},
    {"name": "BOUNDARY_FP_F64", "column": "BOUNDARY_FP_F64", "type": "float64"},
    {"name": "BOUNDARY_INT_I64", "column": "BOUNDARY_INT_I64", "type": "int64"},
    {"name": "BOUNDARY_INT_U64", "column": "BOUNDARY_INT_U64", "type": "uint64"},
]

EDGE_ATTR_DEFS = [
    {"name": "EQ_NUM_I8", "column": "EQ_NUM_I8", "type": "int8"},
    {"name": "EQ_NUM_I16", "column": "EQ_NUM_I16", "type": "int16"},
    {"name": "EQ_NUM_I32", "column": "EQ_NUM_I32", "type": "int32"},
    {"name": "EQ_NUM_I64", "column": "EQ_NUM_I64", "type": "int64"},
    {"name": "EQ_NUM_U8", "column": "EQ_NUM_U8", "type": "uint8"},
    {"name": "EQ_NUM_U16", "column": "EQ_NUM_U16", "type": "uint16"},
    {"name": "EQ_NUM_U32", "column": "EQ_NUM_U32", "type": "uint32"},
    {"name": "EQ_NUM_U64", "column": "EQ_NUM_U64", "type": "uint64"},
    {"name": "EQ_NUM_F32", "column": "EQ_NUM_F32", "type": "float32"},
    {"name": "EQ_NUM_F64", "column": "EQ_NUM_F64", "type": "float64"},
    {"name": "EQ_NUM_NULL_I32", "column": "EQ_NUM_NULL_I32", "type": "int32"},
    {"name": "EQ_NUM_NULL_F64", "column": "EQ_NUM_NULL_F64", "type": "float64"},
    {"name": "BOUNDARY_FP_F32", "column": "BOUNDARY_FP_F32", "type": "float32"},
    {"name": "BOUNDARY_FP_F64", "column": "BOUNDARY_FP_F64", "type": "float64"},
    {"name": "BOUNDARY_INT_I64", "column": "BOUNDARY_INT_I64", "type": "int64"},
    {"name": "BOUNDARY_INT_U64", "column": "BOUNDARY_INT_U64", "type": "uint64"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate self-describing Metamorphic Storage Coverage .imps")
    ap.add_argument("--size", choices=["small", "bench"], default="small")
    ap.add_argument("--output", required=True, help="Output .imps path")
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--cli", default=None, help="Path to impulse-graph CLI binary")
    args = ap.parse_args()

    cli_bin = Path(args.cli) if args.cli else (DEFAULT_CLI if DEFAULT_CLI.exists() else FALLBACK_CLI)
    if not cli_bin.exists():
        print(f"ERROR: impulse-graph CLI not found at {cli_bin}", file=sys.stderr)
        return 1

    n = 1024 if args.size == "small" else 16384
    avg_deg = 8 if args.size == "small" else 16
    edges = generate_graph(n, avg_deg, args.seed)

    # M:1 subset (at most 1 outgoing edge per node)
    m1_edges: list[tuple[int, int]] = []
    seen_src: set[int] = set()
    for u, v in edges:
        if u not in seen_src:
            seen_src.add(u)
            m1_edges.append((u, v))

    # 64 deterministic seeds (gadgets + spread across domain)
    seeds = [1, 2, 3, 4, 5, 6, 7, 10, 40, 63, 64, 65]
    step = max(1, (n - 70) // 52)
    cur = 70
    while len(seeds) < 64 and cur < n:
        seeds.append(cur)
        cur += step

    tmpdir = Path(tempfile.mkdtemp(prefix="impulse_storage_cov_"))
    try:
        nodes_int_tsv = tmpdir / "nodes_int.tsv"
        nodes_str_tsv = tmpdir / "nodes_str.tsv"
        edges_plain_tsv = tmpdir / "edges_plain.tsv"
        edges_attrs_tsv = tmpdir / "edges_attrs.tsv"
        edges_m1_tsv = tmpdir / "edges_m1.tsv"

        write_node_tsv(nodes_int_tsv, n, "int")
        write_node_tsv(nodes_str_tsv, n, "str")
        write_edge_tsv(edges_plain_tsv, edges, with_attrs=False)
        write_edge_tsv(edges_attrs_tsv, edges, with_attrs=True)
        write_edge_tsv(edges_m1_tsv, m1_edges, with_attrs=False)

        manifest = {
            "version": "0.11.0",
            "domains": [
                {
                    "id": 0, "name": "EQ_DOM_U16_I32", "key_type": "int32", "dense_id_width": 2,
                    "file": str(nodes_int_tsv), "id_column": "id", "attributes": NODE_ATTR_DEFS,
                },
                {
                    "id": 1, "name": "EQ_DOM_U32_I32", "key_type": "int32", "dense_id_width": 4,
                    "file": str(nodes_int_tsv), "id_column": "id", "attributes": NODE_ATTR_DEFS,
                },
                {
                    "id": 2, "name": "EQ_DOM_U64_I64", "key_type": "int64", "dense_id_width": 8,
                    "file": str(nodes_int_tsv), "id_column": "id", "attributes": NODE_ATTR_DEFS,
                },
                {
                    "id": 3, "name": "EQ_DOM_U32_STR", "key_type": "string", "dense_id_width": 4,
                    "file": str(nodes_str_tsv), "id_column": "id", "attributes": NODE_ATTR_DEFS,
                },
                {
                    "id": 4, "name": "EQ_DOM_U16_U16", "key_type": "uint16", "dense_id_width": 2,
                    "file": str(nodes_int_tsv), "id_column": "id", "attributes": NODE_ATTR_DEFS,
                },
            ],
            "relations": [
                {
                    "name": "EQ_REL_CSR_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": True, "include_csc": False, "include_coo": False,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_CSR_CSC_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": True, "include_csc": True, "include_coo": False,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_CSR_CSC_U16", "src_domain": 0, "tgt_domain": 0,
                    "node_id_width": 2, "edge_index_width": 4,
                    "include_csr": True, "include_csc": True, "include_coo": False,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_CSR_CSC_U64", "src_domain": 2, "tgt_domain": 2,
                    "node_id_width": 8, "edge_index_width": 8,
                    "include_csr": True, "include_csc": True, "include_coo": False,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_MIXED_U16_U32", "src_domain": 0, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": True, "include_csc": True, "include_coo": True, "coo_order": 1,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_MIXED_U32_U16", "src_domain": 1, "tgt_domain": 0,
                    "node_id_width": 2, "edge_index_width": 4,
                    "include_csr": True, "include_csc": True, "include_coo": True, "coo_order": 3,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_CSC_ONLY_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": False, "include_csc": True, "include_coo": False,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_COO_SRC_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": False, "include_csc": False, "include_coo": True, "coo_order": 1,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_COO_TGT_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": False, "include_csc": False, "include_coo": True, "coo_order": 3,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_COO_UNSORTED_U16", "src_domain": 0, "tgt_domain": 0,
                    "node_id_width": 2, "edge_index_width": 4,
                    "include_csr": False, "include_csc": False, "include_coo": True, "coo_order": 0,
                    "file": str(edges_plain_tsv), "src_column": "src", "tgt_column": "tgt",
                },
                {
                    "name": "EQ_REL_ALL3_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4,
                    "include_csr": True, "include_csc": True, "include_coo": True, "coo_order": 2,
                    "file": str(edges_attrs_tsv), "src_column": "src", "tgt_column": "tgt",
                    "attributes": EDGE_ATTR_DEFS,
                },
                {
                    "name": "EQ_FLAT_M1_U32", "src_domain": 1, "tgt_domain": 1,
                    "node_id_width": 4, "edge_index_width": 4, "cardinality": "m:1",
                    "include_csr": True, "include_csc": False, "include_coo": False,
                    "file": str(edges_m1_tsv), "src_column": "src", "tgt_column": "tgt",
                },
            ],
            "metadata": {
                "coverage.schema_version": "1",
                "coverage.size_preset": args.size,
                "coverage.node_count": str(n),
                "coverage.edge_count": str(len(edges)),
                "coverage.dom_groups": "EQ_DOM_",
                "coverage.rel_groups": "EQ_REL_,EQ_FLAT_",
                "coverage.attr_groups": "EQ_NUM_,EQ_NUM_NULL_",
                "coverage.boundary_attrs": "BOUNDARY_FP_,BOUNDARY_INT_",
                "coverage.canonical_dom": "EQ_DOM_U32_I32",
                "coverage.canonical_rel": "EQ_REL_ALL3_U32",
                "coverage.canonical_attr": "EQ_NUM_F64",
                "coverage.seed_nodes": ",".join(str(s) for s in seeds),
                "coverage.gadget.isolated": "0",
                "coverage.gadget.self_loop": "1",
                "coverage.gadget.supernode": "5",
                "coverage.gadget.sink": "6",
            },
        }

        manifest_path = tmpdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(cli_bin), "build", "--manifest", str(manifest_path), "--output", str(out_path)]
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            return res.returncode
        print(res.stdout.strip())

        val_cmd = [str(cli_bin), "snapshot", "validate", str(out_path)]
        vres = subprocess.run(val_cmd, capture_output=True, text=True)
        if vres.returncode != 0:
            print(vres.stdout)
            print(vres.stderr, file=sys.stderr)
            return vres.returncode
        print(vres.stdout.strip())
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
