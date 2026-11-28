#!/usr/bin/env python3
"""
Generate a self-describing Metamorphic Storage Coverage .imps (v0.11, 0x000B)
exclusively via the official `impulse-graph build` CLI (AGENTS.md §3.17).

Same deterministic logical graph G = (V, E) is mirrored across:
  - Domains  : EQ_DOM_* (u16/u32/u64 dense IDs, i32/i64/u16/string keys)
  - Relations: EQ_REL_* (CSR-only, CSR+CSC u16/u32/u64, mixed widths, CSC-only,
               COO SRC_MAJOR/TGT_MAJOR/UNSORTED, ALL3, plus EQ_FLAT_M1_U32)
  - Attrs    : EQ_NUM_* (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64,
                         decimal64 across P4_S0/P10_S2/P12_S4/P18_S0/P18_S6/P18_S12),
               EQ_NUM_NULL_* (nullable with identical validity bitmap, incl. decimal64),
               BOUNDARY_FP_* (+0.0, -0.0, +inf, -inf, NaN, NULL, extremes),
               BOUNDARY_INT_* (min/max/sign boundaries, NULL),
               BOUNDARY_DEC_* (P18_S0, P18_S6, P6_S6 pure-frac, P1_S0 min-prec, NULL)
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
CLI_CANDIDATES = [
    ROOT / "impulse-graph-core/impulse-cli/target/release/impulse-graph",
    ROOT / "impulse-graph-core/impulse-cli/target/debug/impulse-graph",
]


def resolve_default_cli() -> Path:
    existing = [p for p in CLI_CANDIDATES if p.exists()]
    if existing:
        # Prefer the most recently built CLI binary so newly landed format/type features are present
        return max(existing, key=lambda p: p.stat().st_mtime)
    return CLI_CANDIDATES[0]


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

DEC_P18_S0_BOUNDARY_CYCLE = [
    "0",
    "1",
    "-1",
    "999999999999999999",
    "-999999999999999999",
    "1000000000",
    "-1000000000",
    "",          # NULL
    "123456789012345678",
    "-123456789012345678",
    "40",
    "-40",
]

DEC_P18_S6_BOUNDARY_CYCLE = [
    "0.000000",
    "0.000001",
    "-0.000001",
    "999999999999.999999",
    "-999999999999.999999",
    "0.100000",
    "0.200000",
    "",          # NULL
    "0.300000",
    "-12345.678901",
    "40.000000",
    "-40.000000",
]

DEC_P6_S6_BOUNDARY_CYCLE = [
    "0.000000",
    "0.000001",
    "-0.000001",
    "0.999999",
    "-0.999999",
    "0.500000",
    "-0.500000",
    "",          # NULL
    "0.123456",
    "-0.123456",
    "0.010000",
    "-0.010000",
]

DEC_P1_S0_BOUNDARY_CYCLE = [
    "0",
    "1",
    "-1",
    "9",
    "-9",
    "5",
    "-5",
    "",          # NULL
    "2",
    "-2",
    "8",
    "-8",
]


def is_null_idx(i: int) -> bool:
    return i == 0 or i == 63 or i == 64 or (i % 7 == 0)


def write_node_tsv(path: Path, n: int, key_mode: str) -> None:
    headers = [
        "id",
        "EQ_NUM_I8", "EQ_NUM_I16", "EQ_NUM_I32", "EQ_NUM_I64",
        "EQ_NUM_U8", "EQ_NUM_U16", "EQ_NUM_U32", "EQ_NUM_U64",
        "EQ_NUM_F32", "EQ_NUM_F64",
        "EQ_NUM_DEC_P4_S0", "EQ_NUM_DEC_P10_S2", "EQ_NUM_DEC_P12_S4",
        "EQ_NUM_DEC_P18_S0", "EQ_NUM_DEC_P18_S6", "EQ_NUM_DEC_P18_S12",
        "EQ_NUM_NULL_I32", "EQ_NUM_NULL_I64", "EQ_NUM_NULL_F32", "EQ_NUM_NULL_F64",
        "EQ_NUM_NULL_DEC_P10_S2", "EQ_NUM_NULL_DEC_P18_S6", "EQ_NUM_NULL_DEC_P18_S12",
        "BOUNDARY_FP_F32", "BOUNDARY_FP_F64",
        "BOUNDARY_INT_I64", "BOUNDARY_INT_U64",
        "BOUNDARY_DEC_P18_S0", "BOUNDARY_DEC_P18_S6",
        "BOUNDARY_DEC_P6_S6", "BOUNDARY_DEC_P1_S0",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for i in range(n):
            key = f"node_{i:05d}" if key_mode == "str" else str(i)
            val = i % 100  # fits cleanly in i8 (-128..127), u8 (0..255), f32, f64, decimal64(P-S>=2); node 40 == 40
            v_str = str(val)
            vf_str = f"{val}.0"
            vd2 = f"{val}.00"
            vd4 = f"{val}.0000"
            vd6 = f"{val}.000000"
            vd12 = f"{val}.000000000000"
            is_n = is_null_idx(i)
            n_int = "" if is_n else v_str
            n_flt = "" if is_n else vf_str
            n_d2 = "" if is_n else vd2
            n_d6 = "" if is_n else vd6
            n_d12 = "" if is_n else vd12
            fp_b = FP_BOUNDARY_CYCLE[i % len(FP_BOUNDARY_CYCLE)]
            i64_b = I64_BOUNDARY_CYCLE[i % len(I64_BOUNDARY_CYCLE)]
            u64_b = U64_BOUNDARY_CYCLE[i % len(U64_BOUNDARY_CYCLE)]
            d18_0_b = DEC_P18_S0_BOUNDARY_CYCLE[i % len(DEC_P18_S0_BOUNDARY_CYCLE)]
            d18_6_b = DEC_P18_S6_BOUNDARY_CYCLE[i % len(DEC_P18_S6_BOUNDARY_CYCLE)]
            d6_6_b = DEC_P6_S6_BOUNDARY_CYCLE[i % len(DEC_P6_S6_BOUNDARY_CYCLE)]
            d1_0_b = DEC_P1_S0_BOUNDARY_CYCLE[i % len(DEC_P1_S0_BOUNDARY_CYCLE)]
            row = [
                key,
                v_str, v_str, v_str, v_str,
                v_str, v_str, v_str, v_str,
                vf_str, vf_str,
                v_str, vd2, vd4,
                v_str, vd6, vd12,
                n_int, n_int, n_flt, n_flt,
                n_d2, n_d6, n_d12,
                fp_b, fp_b,
                i64_b, u64_b,
                d18_0_b, d18_6_b,
                d6_6_b, d1_0_b,
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
                "EQ_NUM_DEC_P4_S0", "EQ_NUM_DEC_P10_S2", "EQ_NUM_DEC_P12_S4",
                "EQ_NUM_DEC_P18_S0", "EQ_NUM_DEC_P18_S6", "EQ_NUM_DEC_P18_S12",
                "EQ_NUM_NULL_I32", "EQ_NUM_NULL_F64",
                "EQ_NUM_NULL_DEC_P10_S2", "EQ_NUM_NULL_DEC_P18_S12",
                "BOUNDARY_FP_F32", "BOUNDARY_FP_F64",
                "BOUNDARY_INT_I64", "BOUNDARY_INT_U64",
                "BOUNDARY_DEC_P18_S0", "BOUNDARY_DEC_P18_S6",
                "BOUNDARY_DEC_P6_S6", "BOUNDARY_DEC_P1_S0",
            ]
            f.write("\t".join(headers) + "\n")
            for idx, (u, v) in enumerate(edges):
                val = (u + v) % 100
                v_str = str(val)
                vf_str = f"{val}.0"
                vd2 = f"{val}.00"
                vd4 = f"{val}.0000"
                vd6 = f"{val}.000000"
                vd12 = f"{val}.000000000000"
                is_n = is_null_idx(idx)
                n_int = "" if is_n else v_str
                n_flt = "" if is_n else vf_str
                n_d2 = "" if is_n else vd2
                n_d12 = "" if is_n else vd12
                fp_b = FP_BOUNDARY_CYCLE[idx % len(FP_BOUNDARY_CYCLE)]
                i64_b = I64_BOUNDARY_CYCLE[idx % len(I64_BOUNDARY_CYCLE)]
                u64_b = U64_BOUNDARY_CYCLE[idx % len(U64_BOUNDARY_CYCLE)]
                d18_0_b = DEC_P18_S0_BOUNDARY_CYCLE[idx % len(DEC_P18_S0_BOUNDARY_CYCLE)]
                d18_6_b = DEC_P18_S6_BOUNDARY_CYCLE[idx % len(DEC_P18_S6_BOUNDARY_CYCLE)]
                d6_6_b = DEC_P6_S6_BOUNDARY_CYCLE[idx % len(DEC_P6_S6_BOUNDARY_CYCLE)]
                d1_0_b = DEC_P1_S0_BOUNDARY_CYCLE[idx % len(DEC_P1_S0_BOUNDARY_CYCLE)]
                f.write("\t".join([
                    str(u), str(v),
                    v_str, v_str, v_str, v_str,
                    v_str, v_str, v_str, v_str,
                    vf_str, vf_str,
                    v_str, vd2, vd4,
                    v_str, vd6, vd12,
                    n_int, n_flt,
                    n_d2, n_d12,
                    fp_b, fp_b,
                    i64_b, u64_b,
                    d18_0_b, d18_6_b,
                    d6_6_b, d1_0_b,
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
    {"name": "EQ_NUM_DEC_P4_S0", "column": "EQ_NUM_DEC_P4_S0", "type": "decimal64(4, 0)"},
    {"name": "EQ_NUM_DEC_P10_S2", "column": "EQ_NUM_DEC_P10_S2", "type": "decimal64(10, 2)"},
    {"name": "EQ_NUM_DEC_P12_S4", "column": "EQ_NUM_DEC_P12_S4", "type": "decimal64(12, 4)"},
    {"name": "EQ_NUM_DEC_P18_S0", "column": "EQ_NUM_DEC_P18_S0", "type": "decimal64(18, 0)"},
    {"name": "EQ_NUM_DEC_P18_S6", "column": "EQ_NUM_DEC_P18_S6", "type": "decimal64(18, 6)"},
    {"name": "EQ_NUM_DEC_P18_S12", "column": "EQ_NUM_DEC_P18_S12", "type": "decimal64(18, 12)"},
    {"name": "EQ_NUM_NULL_I32", "column": "EQ_NUM_NULL_I32", "type": "int32"},
    {"name": "EQ_NUM_NULL_I64", "column": "EQ_NUM_NULL_I64", "type": "int64"},
    {"name": "EQ_NUM_NULL_F32", "column": "EQ_NUM_NULL_F32", "type": "float32"},
    {"name": "EQ_NUM_NULL_F64", "column": "EQ_NUM_NULL_F64", "type": "float64"},
    {"name": "EQ_NUM_NULL_DEC_P10_S2", "column": "EQ_NUM_NULL_DEC_P10_S2", "type": "decimal64(10, 2)?"},
    {"name": "EQ_NUM_NULL_DEC_P18_S6", "column": "EQ_NUM_NULL_DEC_P18_S6", "type": "decimal64(18, 6)?"},
    {"name": "EQ_NUM_NULL_DEC_P18_S12", "column": "EQ_NUM_NULL_DEC_P18_S12", "type": "decimal64(18, 12)?"},
    {"name": "BOUNDARY_FP_F32", "column": "BOUNDARY_FP_F32", "type": "float32"},
    {"name": "BOUNDARY_FP_F64", "column": "BOUNDARY_FP_F64", "type": "float64"},
    {"name": "BOUNDARY_INT_I64", "column": "BOUNDARY_INT_I64", "type": "int64"},
    {"name": "BOUNDARY_INT_U64", "column": "BOUNDARY_INT_U64", "type": "uint64"},
    {"name": "BOUNDARY_DEC_P18_S0", "column": "BOUNDARY_DEC_P18_S0", "type": "decimal64(18, 0)?"},
    {"name": "BOUNDARY_DEC_P18_S6", "column": "BOUNDARY_DEC_P18_S6", "type": "decimal64(18, 6)?"},
    {"name": "BOUNDARY_DEC_P6_S6", "column": "BOUNDARY_DEC_P6_S6", "type": "decimal64(6, 6)?"},
    {"name": "BOUNDARY_DEC_P1_S0", "column": "BOUNDARY_DEC_P1_S0", "type": "decimal64(1, 0)?"},
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
    {"name": "EQ_NUM_DEC_P4_S0", "column": "EQ_NUM_DEC_P4_S0", "type": "decimal64(4, 0)"},
    {"name": "EQ_NUM_DEC_P10_S2", "column": "EQ_NUM_DEC_P10_S2", "type": "decimal64(10, 2)"},
    {"name": "EQ_NUM_DEC_P12_S4", "column": "EQ_NUM_DEC_P12_S4", "type": "decimal64(12, 4)"},
    {"name": "EQ_NUM_DEC_P18_S0", "column": "EQ_NUM_DEC_P18_S0", "type": "decimal64(18, 0)"},
    {"name": "EQ_NUM_DEC_P18_S6", "column": "EQ_NUM_DEC_P18_S6", "type": "decimal64(18, 6)"},
    {"name": "EQ_NUM_DEC_P18_S12", "column": "EQ_NUM_DEC_P18_S12", "type": "decimal64(18, 12)"},
    {"name": "EQ_NUM_NULL_I32", "column": "EQ_NUM_NULL_I32", "type": "int32"},
    {"name": "EQ_NUM_NULL_F64", "column": "EQ_NUM_NULL_F64", "type": "float64"},
    {"name": "EQ_NUM_NULL_DEC_P10_S2", "column": "EQ_NUM_NULL_DEC_P10_S2", "type": "decimal64(10, 2)?"},
    {"name": "EQ_NUM_NULL_DEC_P18_S12", "column": "EQ_NUM_NULL_DEC_P18_S12", "type": "decimal64(18, 12)?"},
    {"name": "BOUNDARY_FP_F32", "column": "BOUNDARY_FP_F32", "type": "float32"},
    {"name": "BOUNDARY_FP_F64", "column": "BOUNDARY_FP_F64", "type": "float64"},
    {"name": "BOUNDARY_INT_I64", "column": "BOUNDARY_INT_I64", "type": "int64"},
    {"name": "BOUNDARY_INT_U64", "column": "BOUNDARY_INT_U64", "type": "uint64"},
    {"name": "BOUNDARY_DEC_P18_S0", "column": "BOUNDARY_DEC_P18_S0", "type": "decimal64(18, 0)?"},
    {"name": "BOUNDARY_DEC_P18_S6", "column": "BOUNDARY_DEC_P18_S6", "type": "decimal64(18, 6)?"},
    {"name": "BOUNDARY_DEC_P6_S6", "column": "BOUNDARY_DEC_P6_S6", "type": "decimal64(6, 6)?"},
    {"name": "BOUNDARY_DEC_P1_S0", "column": "BOUNDARY_DEC_P1_S0", "type": "decimal64(1, 0)?"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate self-describing Metamorphic Storage Coverage .imps")
    ap.add_argument("--size", choices=["small", "bench"], default="small")
    ap.add_argument("--output", required=True, help="Output .imps path")
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--cli", default=None, help="Path to impulse-graph CLI binary")
    args = ap.parse_args()

    cli_bin = Path(args.cli) if args.cli else resolve_default_cli()
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
                "coverage.boundary_attrs": "BOUNDARY_FP_,BOUNDARY_INT_,BOUNDARY_DEC_",
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
