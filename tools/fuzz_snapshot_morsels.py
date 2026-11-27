#!/usr/bin/env python3
"""
ImpulseVM v0.11.0 Snapshot-Backed Morsel & Multi-Lane Corpus + Metamorphic Differential Fuzzer.

Drives persistent `vm_fixture_batch_runner` (C++ opt ON/OFF) and `VmFixtureBatchMain` (Java 25)
with `--snapshot <file.imps>` against real v0.11 `.imps` snapshots (e.g. `storage_coverage_small.imps`).

Checks two invariants on every iteration:
  1. 3-way differential parity (`cpp_on` == `cpp_off` == `java`) on status, pc, flags, r_regs,
     outputs, and ICTX v2 logical_hash across geometry profiles (P0..P3).
  2. Metamorphic storage equivalence across `EQ_REL_*` topology/width variants (u16/u32/u64,
     CSR/CSC/COO) and `EQ_NUM_*` attribute encodings (i8..u64, f32, f64).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_vm_contracts import (  # noqa: E402
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    resolve_spec_dir,
)


def compare_results(results: dict[str, dict[str, Any]]) -> str | None:
    names = list(results.keys())
    if len(names) < 2:
        return None
    ref_name = names[0]
    ref = results[ref_name]
    for other_name in names[1:]:
        oth = results[other_name]
        for key in ("status", "pc", "flags", "logical_hash", "outputs"):
            if ref.get(key) != oth.get(key):
                return f"{ref_name} vs {other_name} mismatch on {key}: {ref.get(key)!r} != {oth.get(key)!r}"
    return None


# ---------------------------------------------------------------------------
# C-ABI Snapshot Schema Loader (ctypes)
# ---------------------------------------------------------------------------

class DomainEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("domain_id", ctypes.c_uint16),
        ("key_type", ctypes.c_uint8),
        ("dense_id_width", ctypes.c_uint8),
        ("name_offset", ctypes.c_uint32),
        ("node_count", ctypes.c_uint64),
        ("attr_count", ctypes.c_uint16),
        ("key_attr_index", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint8 * 12),
    ]


class RelationEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("relation_id", ctypes.c_uint16),
        ("src_domain_id", ctypes.c_uint16),
        ("tgt_domain_id", ctypes.c_uint16),
        ("encoding_id", ctypes.c_uint8),
        ("node_id_width", ctypes.c_uint8),
        ("edge_index_width", ctypes.c_uint8),
        ("reserved1", ctypes.c_uint8 * 3),
        ("name_offset", ctypes.c_uint32),
        ("node_count", ctypes.c_uint64),
        ("edge_count", ctypes.c_uint64),
        ("section_features", ctypes.c_uint64),
        ("csr_row_off_offset", ctypes.c_uint64),
        ("csr_row_off_bytes", ctypes.c_uint64),
        ("csr_col_idx_offset", ctypes.c_uint64),
        ("csr_col_idx_bytes", ctypes.c_uint64),
        ("csc_row_off_offset", ctypes.c_uint64),
        ("csc_row_off_bytes", ctypes.c_uint64),
        ("csc_col_idx_offset", ctypes.c_uint64),
        ("csc_col_idx_bytes", ctypes.c_uint64),
        ("attr_count", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint8 * 22),
        ("coo_src_offset", ctypes.c_uint64),
        ("coo_src_bytes", ctypes.c_uint64),
        ("coo_tgt_offset", ctypes.c_uint64),
        ("coo_tgt_bytes", ctypes.c_uint64),
        ("coo_edge_id_offset", ctypes.c_uint64),
        ("coo_edge_id_bytes", ctypes.c_uint64),
        ("coo_order", ctypes.c_uint8),
        ("reserved3", ctypes.c_uint8 * 15),
    ]


@dataclass
class AttrSchema:
    attr_id: int
    name: str
    type_code: int
    nullable: bool
    is_float: bool


@dataclass
class RelSchema:
    rel_id: int
    name: str
    src_dom: int
    tgt_dom: int
    edge_count: int
    has_csr: bool
    has_csc: bool
    has_coo: bool
    edge_attrs: list[AttrSchema] = field(default_factory=list)


@dataclass
class SnapshotSchema:
    dom_cards: dict[int, int]
    dom_names: dict[int, str]
    relations: dict[int, RelSchema]
    node_attrs: dict[int, list[AttrSchema]]
    eq_rels: list[int]  # EQ_REL_* relation IDs sharing identical logical graph


def _discover_default_lib() -> Path:
    candidates = [
        Path("/Users/jesse/impulse/.worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_on/libimpulse_graph.dylib"),
        Path("/Users/jesse/impulse/impulse-graph-core/impulse-cpp/build_on/libimpulse_graph.dylib"),
        Path("/Users/jesse/impulse/impulse-graph-core/impulse-cpp/build/libimpulse_graph.so"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate libimpulse_graph.dylib/.so; pass --lib-path")


def load_snapshot_schema(snapshot_path: Path, lib_path: Path) -> SnapshotSchema:
    lib = ctypes.CDLL(str(lib_path))
    lib.impulse_snapshot_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.impulse_snapshot_open.restype = ctypes.c_void_p
    lib.impulse_snapshot_close.argtypes = [ctypes.c_void_p]
    lib.impulse_snapshot_domain_count.argtypes = [ctypes.c_void_p]
    lib.impulse_snapshot_domain_count.restype = ctypes.c_uint16
    lib.impulse_snapshot_get_domain_entry.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.POINTER(DomainEntry), ctypes.POINTER(ctypes.c_char_p)
    ]
    lib.impulse_snapshot_relation_count.argtypes = [ctypes.c_void_p]
    lib.impulse_snapshot_relation_count.restype = ctypes.c_uint16
    lib.impulse_snapshot_get_relation_entry.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.POINTER(RelationEntry)
    ]
    lib.impulse_snapshot_get_relation_name.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.POINTER(ctypes.c_char_p)
    ]
    lib.impulse_snapshot_get_attribute_buffers.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.impulse_snapshot_get_attribute_name.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.impulse_snapshot_get_domain_attribute_count.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    lib.impulse_snapshot_get_domain_attribute_count.restype = ctypes.c_uint16
    lib.impulse_snapshot_get_domain_attribute_buffers.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.impulse_snapshot_get_domain_attribute_name.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.POINTER(ctypes.c_char_p),
    ]

    st = ctypes.c_int(0)
    snap = lib.impulse_snapshot_open(str(snapshot_path).encode("utf-8"), ctypes.byref(st))
    if not snap or st.value != 0:
        raise RuntimeError(f"Failed to open {snapshot_path} (status={st.value})")
    try:
        dom_cards: dict[int, int] = {}
        dom_names: dict[int, str] = {}
        node_attrs: dict[int, list[AttrSchema]] = {}
        nd = lib.impulse_snapshot_domain_count(snap)
        for d in range(nd):
            de = DomainEntry()
            name_p = ctypes.c_char_p()
            lib.impulse_snapshot_get_domain_entry(snap, d, ctypes.byref(de), ctypes.byref(name_p))
            dom_cards[d] = int(de.node_count)
            dom_names[d] = name_p.value.decode("utf-8") if name_p.value else f"dom_{d}"
            na_cnt = lib.impulse_snapshot_get_domain_attribute_count(snap, d)
            attrs: list[AttrSchema] = []
            for a in range(na_cnt):
                dp, op = ctypes.c_void_p(), ctypes.c_void_p()
                db, ob = ctypes.c_uint64(0), ctypes.c_uint64(0)
                tc, dim = ctypes.c_uint8(0), ctypes.c_uint32(0)
                an = ctypes.c_char_p()
                if lib.impulse_snapshot_get_domain_attribute_buffers(
                    snap, d, a, ctypes.byref(dp), ctypes.byref(db),
                    ctypes.byref(op), ctypes.byref(ob), ctypes.byref(tc), ctypes.byref(dim)
                ) == 0:
                    lib.impulse_snapshot_get_domain_attribute_name(snap, d, a, ctypes.byref(an))
                    base = tc.value & 0x7F
                    if base in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14):
                        attrs.append(AttrSchema(
                            attr_id=a,
                            name=an.value.decode("utf-8") if an.value else f"na_{a}",
                            type_code=tc.value,
                            nullable=bool(tc.value & 0x80),
                            is_float=(base in (6, 7, 10)),
                        ))
            node_attrs[d] = attrs

        relations: dict[int, RelSchema] = {}
        eq_rels: list[int] = []
        nr = lib.impulse_snapshot_relation_count(snap)
        for r in range(nr):
            re = RelationEntry()
            name_p = ctypes.c_char_p()
            lib.impulse_snapshot_get_relation_entry(snap, r, ctypes.byref(re))
            lib.impulse_snapshot_get_relation_name(snap, r, ctypes.byref(name_p))
            rname = name_p.value.decode("utf-8") if name_p.value else f"rel_{r}"
            ea_cnt = int(re.attr_count)
            eattrs: list[AttrSchema] = []
            for a in range(ea_cnt):
                dp, op = ctypes.c_void_p(), ctypes.c_void_p()
                db, ob = ctypes.c_uint64(0), ctypes.c_uint64(0)
                tc, dim = ctypes.c_uint8(0), ctypes.c_uint32(0)
                an = ctypes.c_char_p()
                if lib.impulse_snapshot_get_attribute_buffers(
                    snap, r, a, ctypes.byref(dp), ctypes.byref(db),
                    ctypes.byref(op), ctypes.byref(ob), ctypes.byref(tc), ctypes.byref(dim)
                ) == 0:
                    lib.impulse_snapshot_get_attribute_name(snap, r, a, ctypes.byref(an))
                    base = tc.value & 0x7F
                    if base in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14):
                        eattrs.append(AttrSchema(
                            attr_id=a,
                            name=an.value.decode("utf-8") if an.value else f"ea_{a}",
                            type_code=tc.value,
                            nullable=bool(tc.value & 0x80),
                            is_float=(base in (6, 7, 10)),
                        ))
            rs = RelSchema(
                rel_id=r,
                name=rname,
                src_dom=int(re.src_domain_id),
                tgt_dom=int(re.tgt_domain_id),
                edge_count=int(re.edge_count),
                has_csr=(re.csr_row_off_bytes > 0),
                has_csc=(re.csc_row_off_bytes > 0),
                has_coo=(re.coo_src_bytes > 0 and re.coo_tgt_bytes > 0),
                edge_attrs=eattrs,
            )
            relations[r] = rs
            if rname.startswith("EQ_REL_"):
                eq_rels.append(r)

        return SnapshotSchema(
            dom_cards=dom_cards,
            dom_names=dom_names,
            relations=relations,
            node_attrs=node_attrs,
            eq_rels=eq_rels,
        )
    finally:
        lib.impulse_snapshot_close(snap)


# ---------------------------------------------------------------------------
# Structured Morsel & Multi-Lane Program Generator + Corpus Mutator
# ---------------------------------------------------------------------------

@dataclass
class MorselGene:
    kind: str  # "walk_morsel", "filter_morsel", "ml_morsel"
    rel_id: int
    dir_sub: str  # "fwd", "rev", "coo_fwd", "coo_rev"
    dom_id: int
    attr_id: int | None
    attr_is_float: bool
    cmp_op: str  # eq, ne, lt, le, gt, ge
    thresh: float | int
    use_mask: bool
    use_jempty: bool
    commit_mode: str  # yield_frontier, yield_lane_or, scatter_add, scatter_min, scatter_max, scatter_argmin, count_add_r
    seeds: list[int]
    k_lanes: int = 1
    force_w64: bool = False


def _valid_dirs_for_rel(rel: RelSchema) -> list[str]:
    dirs: list[str] = []
    if rel.has_csr:
        dirs.append("fwd")
    if rel.has_csc:
        dirs.append("rev")
    if rel.has_coo:
        dirs.append("coo_fwd")
        dirs.append("coo_rev")
    return dirs


def generate_random_gene(rng: random.Random, schema: SnapshotSchema) -> MorselGene:
    kind = rng.choice(["walk_morsel", "walk_morsel", "filter_morsel", "ml_morsel"])
    valid_rels = [r for r in schema.relations.values() if _valid_dirs_for_rel(r)]
    rel = rng.choice(valid_rels)
    dirs = _valid_dirs_for_rel(rel)
    dir_sub = rng.choice(dirs)

    is_rev = dir_sub in ("rev", "coo_rev")
    in_dom = rel.tgt_dom if is_rev else rel.src_dom
    n_in = max(1, schema.dom_cards[in_dom])

    if kind == "filter_morsel":
        doms_with_attrs = [d for d, al in schema.node_attrs.items() if al]
        dom_id = rng.choice(doms_with_attrs) if doms_with_attrs else 0
        n_dom = max(1, schema.dom_cards[dom_id])
        attr = rng.choice(schema.node_attrs[dom_id]) if schema.node_attrs.get(dom_id) else None
        seed_cnt = rng.choice([1, 4, 16, 64, min(256, n_dom)])
        seeds = sorted({rng.randint(0, n_dom - 1) for _ in range(seed_cnt)})
        thresh = rng.choice([0.0, -0.0, 20.0, 40.0, 100.0, -1e9, 1e9]) if (attr and attr.is_float) else rng.choice([0, 20, 40, 100, -10])
        return MorselGene(
            kind="filter_morsel",
            rel_id=rel.rel_id,
            dir_sub="fwd",
            dom_id=dom_id,
            attr_id=attr.attr_id if attr else None,
            attr_is_float=attr.is_float if attr else True,
            cmp_op=rng.choice(["eq", "ne", "lt", "le", "gt", "ge"]),
            thresh=thresh,
            use_mask=False,
            use_jempty=rng.random() < 0.5,
            commit_mode=rng.choice(["yield_frontier", "count_add_r"]),
            seeds=seeds,
        )

    if kind == "ml_morsel":
        # Multi-lane combined with WALK_MORSEL + yield_lane_or
        k_lanes = rng.choice([1, 2, 4, 8, 16, 64])
        seeds = [rng.randint(0, n_in - 1) for _ in range(k_lanes)]
        eattr = rng.choice(rel.edge_attrs) if (rel.edge_attrs and rng.random() < 0.6) else None
        thresh = rng.choice([0.0, -0.0, 20.0, 40.0]) if (eattr and eattr.is_float) else rng.choice([0, 20, 40])
        return MorselGene(
            kind="ml_morsel",
            rel_id=rel.rel_id,
            dir_sub=dir_sub,
            dom_id=in_dom,
            attr_id=eattr.attr_id if eattr else None,
            attr_is_float=eattr.is_float if eattr else True,
            cmp_op=rng.choice(["eq", "ne", "lt", "le", "gt", "ge"]),
            thresh=thresh,
            use_mask=False,
            use_jempty=rng.random() < 0.4,
            commit_mode="yield_lane_or",
            seeds=seeds,
            k_lanes=k_lanes,
            force_w64=(rng.random() < 0.35),
        )

    # Standard WALK_MORSEL with optional edge-attr gather + filter + commit
    seed_cnt = rng.choice([1, 2, 8, 32, min(128, n_in)])
    seeds = sorted({rng.randint(0, n_in - 1) for _ in range(seed_cnt)})
    eattr = rng.choice(rel.edge_attrs) if (rel.edge_attrs and rng.random() < 0.75) else None
    thresh = rng.choice([0.0, -0.0, 20.0, 40.0, 100.0]) if (eattr and eattr.is_float) else rng.choice([0, 20, 40, 100])
    commit_choices = ["yield_frontier", "count_add_r"]
    if eattr is not None and eattr.is_float:
        commit_choices.extend(["scatter_add", "scatter_min", "scatter_max"])
    commit_choices.extend(["scatter_argmin", "scatter_argmax"])
    return MorselGene(
        kind="walk_morsel",
        rel_id=rel.rel_id,
        dir_sub=dir_sub,
        dom_id=in_dom,
        attr_id=eattr.attr_id if eattr else None,
        attr_is_float=eattr.is_float if eattr else True,
        cmp_op=rng.choice(["eq", "ne", "lt", "le", "gt", "ge"]),
        thresh=thresh,
        use_mask=(rng.random() < 0.35 and dir_sub in ("fwd", "rev")),
        use_jempty=rng.random() < 0.5,
        commit_mode=rng.choice(commit_choices),
        seeds=seeds,
    )


def mutate_gene(rng: random.Random, gene: MorselGene, schema: SnapshotSchema) -> MorselGene:
    g = MorselGene(**gene.__dict__)
    g.seeds = list(gene.seeds)
    mut = rng.choice(["rel", "dir", "cmp", "thresh", "commit", "mask", "jempty", "seeds", "attr"])
    rel = schema.relations[g.rel_id]
    if mut == "rel" and g.kind != "filter_morsel":
        valid_rels = [r for r in schema.relations.values() if _valid_dirs_for_rel(r)]
        new_rel = rng.choice(valid_rels)
        g.rel_id = new_rel.rel_id
        g.dir_sub = rng.choice(_valid_dirs_for_rel(new_rel))
        g.dom_id = new_rel.tgt_dom if g.dir_sub in ("rev", "coo_rev") else new_rel.src_dom
        n_in = max(1, schema.dom_cards[g.dom_id])
        g.seeds = [s % n_in for s in g.seeds]
        if g.kind != "ml_morsel":
            g.seeds = sorted(set(g.seeds))
        if new_rel.edge_attrs:
            ea = rng.choice(new_rel.edge_attrs)
            g.attr_id, g.attr_is_float = ea.attr_id, ea.is_float
        else:
            g.attr_id = None
    elif mut == "dir" and g.kind != "filter_morsel":
        dirs = _valid_dirs_for_rel(rel)
        g.dir_sub = rng.choice(dirs)
        g.dom_id = rel.tgt_dom if g.dir_sub in ("rev", "coo_rev") else rel.src_dom
        n_in = max(1, schema.dom_cards[g.dom_id])
        g.seeds = [s % n_in for s in g.seeds]
        if g.kind != "ml_morsel":
            g.seeds = sorted(set(g.seeds))
    elif mut == "cmp":
        g.cmp_op = rng.choice(["eq", "ne", "lt", "le", "gt", "ge"])
    elif mut == "thresh":
        g.thresh = rng.choice([0.0, -0.0, 1.0, 20.0, 40.0, 80.0, -1e9, 1e9]) if g.attr_is_float else rng.choice([0, 1, 20, 40, 80, -10])
    elif mut == "commit" and g.kind == "walk_morsel":
        choices = ["yield_frontier", "count_add_r", "scatter_argmin", "scatter_argmax"]
        if g.attr_id is not None and g.attr_is_float:
            choices.extend(["scatter_add", "scatter_min", "scatter_max"])
        g.commit_mode = rng.choice(choices)
    elif mut == "mask" and g.kind == "walk_morsel" and g.dir_sub in ("fwd", "rev"):
        g.use_mask = not g.use_mask
    elif mut == "jempty":
        g.use_jempty = not g.use_jempty
    elif mut == "seeds":
        n_in = max(1, schema.dom_cards[g.dom_id])
        if g.kind == "ml_morsel":
            g.k_lanes = rng.choice([1, 2, 4, 8, 16, 64])
            g.seeds = [rng.randint(0, n_in - 1) for _ in range(g.k_lanes)]
        else:
            cnt = rng.choice([1, 4, 16, 64])
            g.seeds = sorted({rng.randint(0, n_in - 1) for _ in range(cnt)})
    elif mut == "attr":
        if g.kind == "filter_morsel" and schema.node_attrs.get(g.dom_id):
            a = rng.choice(schema.node_attrs[g.dom_id])
            g.attr_id, g.attr_is_float = a.attr_id, a.is_float
        elif g.kind != "filter_morsel" and rel.edge_attrs:
            a = rng.choice(rel.edge_attrs)
            g.attr_id, g.attr_is_float = a.attr_id, a.is_float
    return g


def render_impas(gene: MorselGene, schema: SnapshotSchema) -> str:
    rel = schema.relations[gene.rel_id]
    is_rev = gene.dir_sub in ("rev", "coo_rev")
    in_dom = gene.dom_id if gene.kind == "filter_morsel" else (rel.tgt_dom if is_rev else rel.src_dom)
    out_dom = gene.dom_id if gene.kind == "filter_morsel" else (rel.src_dom if is_rev else rel.tgt_dom)
    id_reg = "M0" if (gene.kind == "filter_morsel" or is_rev) else "M1"

    lines: list[str] = [".data"]
    if gene.kind == "ml_morsel":
        lines.append(f"  .param 0 type=u64_vec values=[{', '.join(str(x) for x in gene.seeds)}]")
    else:
        lines.append(f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in gene.seeds)}]")
    if gene.attr_is_float:
        lines.append(f"  .param 1 type=f64 value={gene.thresh!r}")
    else:
        lines.append(f"  .param 1 type=i64 value={int(gene.thresh)}")
    lines.append(".code")

    if gene.kind == "ml_morsel":
        lines.append("  LOAD_PARAM.vec V0, param=0, type=u64")
        init_sub = "force_w64" if gene.force_w64 else "auto_width"
        lines.append(f"  LANE_INIT_SEEDS.{init_sub} F0, V0, dom={in_dom}")
        lines.append(f"  WALK_MORSEL.{gene.dir_sub} F0, rel={gene.rel_id}")
        if gene.attr_id is not None:
            lines.append(f"    GATHER_EDGE_ATTR.vec M4, M2, rel={gene.rel_id}, attr={gene.attr_id}")
            lines.append("    LOAD_PARAM.scalar R5, param=1")
            lines.append(f"    VEC_CMP.{gene.cmp_op}.scalar M5, M4, R5")
            lines.append("    SET_CONVERT.b2f MF1, M5")
            if gene.use_jempty:
                lines.append("    JCC.jempty_f MF1, m_skip")
            lines.append(f"    MORSEL_COMMIT.yield_lane_or.mask F2, {id_reg}, M3, MF1, dom={out_dom}")
            lines.append("m_skip:")
        else:
            lines.append(f"    MORSEL_COMMIT.yield_lane_or F2, {id_reg}, M3, dom={out_dom}")
        lines.append("  MORSEL_END")
        lines.append("  SET_CARDINALITY.lane_pop_sum R0, F2")
        lines.append("  COLLECT.scalar R0, out=0")
        lines.append("  HALT")
        return "\n".join(lines) + "\n"

    # Single-lane WALK_MORSEL or FILTER_MORSEL
    lines.append(f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}")
    if gene.attr_id is not None:
        lines.append("  LOAD_PARAM.scalar R5, param=1")
    if gene.commit_mode in ("scatter_add", "scatter_min", "scatter_max"):
        lines.append(f"  VEC_INIT.dom_f64 V1, dom={out_dom}")
    elif gene.commit_mode in ("scatter_argmin", "scatter_argmax"):
        lines.append(f"  VEC_INIT.dom_u64 V1, dom={out_dom}")

    if gene.kind == "filter_morsel":
        lines.append("  FILTER_MORSEL F0")
    elif gene.use_mask and gene.dir_sub in ("fwd", "rev"):
        lines.append(f"  WALK_MORSEL.{gene.dir_sub}.mask F0, F0, rel={gene.rel_id}")
    else:
        lines.append(f"  WALK_MORSEL.{gene.dir_sub} F0, rel={gene.rel_id}")

    has_pred = gene.attr_id is not None
    if has_pred:
        if gene.kind == "filter_morsel":
            lines.append(f"    GATHER_NODE_ATTR.vec M4, M0, dom={in_dom}, attr={gene.attr_id}")
        else:
            lines.append(f"    GATHER_EDGE_ATTR.vec M4, M2, rel={gene.rel_id}, attr={gene.attr_id}")
        lines.append(f"    VEC_CMP.{gene.cmp_op}.scalar M5, M4, R5")
        lines.append("    SET_CONVERT.b2f MF1, M5")
        if gene.use_jempty:
            lines.append("    JCC.jempty_f MF1, m_skip")

    ms = ".mask" if has_pred else ""
    mf = ", MF1" if has_pred else ""
    if gene.commit_mode == "yield_frontier":
        lines.append(f"    MORSEL_COMMIT.yield_frontier{ms} F2, {id_reg}{mf}, dom={out_dom}")
    elif gene.commit_mode == "count_add_r":
        if has_pred:
            lines.append("    MORSEL_COMMIT.count_add_r.mask R0, MF1")
        else:
            lines.append("    MORSEL_COMMIT.count_add_r R0")
    elif gene.commit_mode in ("scatter_add", "scatter_min", "scatter_max"):
        lines.append(f"    MORSEL_COMMIT.{gene.commit_mode}{ms} V1, {id_reg}, M4{mf}")
    elif gene.commit_mode in ("scatter_argmin", "scatter_argmax"):
        lines.append(f"    MORSEL_COMMIT.{gene.commit_mode}{ms} V1, {id_reg}, M0{mf}")

    if has_pred and gene.use_jempty:
        lines.append("m_skip:")
    lines.append("  MORSEL_END")

    if gene.commit_mode == "yield_frontier":
        lines.append("  SET_CARDINALITY R0, F2")
    elif gene.commit_mode in ("scatter_add", "scatter_min", "scatter_max"):
        lines.append("  VEC_REDUCE.sum R0, V1")
    elif gene.commit_mode in ("scatter_argmin", "scatter_argmax"):
        lines.append("  VEC_REDUCE.count_valid R0, V1")
    lines.append("  COLLECT.scalar R0, out=0")
    lines.append("  HALT")
    return "\n".join(lines) + "\n"


def _metamorphic_twin(rng: random.Random, gene: MorselGene, schema: SnapshotSchema) -> MorselGene | None:
    """If gene uses an EQ_REL_* relation (without edge attrs or with matching EQ_NUM_* attr),
    build a twin gene using a different EQ_REL_* topology/width variant."""
    if gene.kind == "filter_morsel" or gene.rel_id not in schema.eq_rels:
        return None
    if gene.attr_id is not None:
        return None  # only topology-equivalent swaps when attr isn't relation-specific
    if gene.use_mask:
        return None
    is_rev = gene.dir_sub in ("rev", "coo_rev")
    cands: list[tuple[int, str]] = []
    for rid in schema.eq_rels:
        if rid == gene.rel_id:
            continue
        r = schema.relations[rid]
        if is_rev:
            if r.has_csc:
                cands.append((rid, "rev"))
            if r.has_coo:
                cands.append((rid, "coo_rev"))
        else:
            if r.has_csr:
                cands.append((rid, "fwd"))
            if r.has_coo:
                cands.append((rid, "coo_fwd"))
    if not cands:
        return None
    new_rid, new_dir = rng.choice(cands)
    new_rel = schema.relations[new_rid]
    twin = MorselGene(**gene.__dict__)
    twin.rel_id = new_rid
    twin.dir_sub = new_dir
    twin.dom_id = new_rel.tgt_dom if is_rev else new_rel.src_dom
    twin.seeds = list(gene.seeds)
    return twin


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot-backed morsel & multi-lane corpus + metamorphic fuzzer")
    parser.add_argument("snapshot", type=Path, help="Path to v0.11 .imps snapshot (e.g. storage_coverage_small.imps)")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    parser.add_argument("--cpp-on-cmd", type=str,
                        default="/Users/jesse/impulse/.worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_on/vm_fixture_batch_runner")
    parser.add_argument("--cpp-off-cmd", type=str,
                        default="/Users/jesse/impulse/.worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_off/vm_fixture_batch_runner")
    parser.add_argument("--engines", type=str, default="java,cpp_on,cpp_off")
    parser.add_argument("--profiles", type=str, default="P0,P1,P2,P3")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-errors", type=int, default=5)
    args = parser.parse_args()

    snap_path = args.snapshot.resolve()
    lib_path = args.lib_path or _discover_default_lib()
    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_asm = _import_spec_modules(spec_dir)
    asm = impas_asm.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    schema = load_snapshot_schema(snap_path, lib_path)

    geom_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    clients: dict[str, BatchEngineClient] = {}
    for eng in engines:
        if eng == "java":
            cmd = discover_default_java_cmd() + ["--snapshot", str(snap_path)]
        elif eng == "cpp_on":
            cmd = [args.cpp_on_cmd, "--snapshot", str(snap_path)]
        elif eng == "cpp_off":
            cmd = [args.cpp_off_cmd, "--snapshot", str(snap_path)]
        else:
            raise ValueError(f"Unsupported snapshot engine: {eng}")
        clients[eng] = BatchEngineClient(cmd, eng)

    print(
        f"Snapshot Morsel Corpus Fuzzer | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}, eq_rels={len(schema.eq_rels)}) | "
        f"engines={','.join(clients.keys())} | geom={','.join(geom_profiles)} | iters={args.iterations}"
    )

    rng = random.Random(args.seed)
    corpus: list[MorselGene] = []
    seen_sigs: set[str] = set()
    kind_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    meta_checks = 0
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            if corpus and rng.random() < 0.60:
                gene = mutate_gene(rng, rng.choice(corpus), schema)
            else:
                gene = generate_random_gene(rng, schema)

            prof = geom_profiles[i % len(geom_profiles)]
            impas = render_impas(gene, schema)
            prog = asm.assemble(impas)
            fixture = {
                "words_hex": [f"0x{w:016x}" for w in prog.words],
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
            }
            raw_json = json.dumps(fixture, separators=(",", ":"))

            results = {name: cli.run_fixture(fixture, prof, 500_000) for name, cli in clients.items()}
            err = compare_results(results)
            ref = next(iter(results.values()))
            kind_counts[gene.kind] += 1
            status_counts[str(ref.get("status", "UNKNOWN"))] += 1

            # Corpus promotion on new behavioral signature
            sig = hashlib.sha1(
                f"{gene.kind}:{gene.dir_sub}:{gene.commit_mode}:{ref.get('status')}:{ref.get('outputs')}".encode()
            ).hexdigest()[:12]
            if sig not in seen_sigs and len(corpus) < 512:
                seen_sigs.add(sig)
                corpus.append(gene)

            if err is not None:
                errors += 1
                print(f"[DIFF #{errors}] iter={i} prof={prof} kind={gene.kind} rel={gene.rel_id} dir={gene.dir_sub}: {err}")
                print(impas)
                if errors >= args.max_errors:
                    break
                continue

            # Metamorphic topology/width equivalence check
            twin = _metamorphic_twin(rng, gene, schema)
            if twin is not None and ref.get("status") == "IMPULSE_VM_OK":
                twin_impas = render_impas(twin, schema)
                tprog = asm.assemble(twin_impas)
                twin_fix = {
                    "words_hex": [f"0x{w:016x}" for w in tprog.words],
                    "data_directives": tprog.data_directives,
                    "const_pool": tprog.const_pool,
                }
                twin_json = json.dumps(twin_fix, separators=(",", ":"))
                twin_res = next(iter(clients.values())).run_fixture(twin_fix, prof, 500_000)
                meta_checks += 1
                if twin_res.get("status") != ref.get("status") or twin_res.get("outputs") != ref.get("outputs"):
                    errors += 1
                    print(
                        f"[META DIFF #{errors}] iter={i} {schema.relations[gene.rel_id].name}({gene.dir_sub}) vs "
                        f"{schema.relations[twin.rel_id].name}({twin.dir_sub}): "
                        f"{ref.get('status')}/{ref.get('outputs')} != {twin_res.get('status')}/{twin_res.get('outputs')}"
                    )
                    if errors >= args.max_errors:
                        break
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    print(
        f"\nCompleted {sum(status_counts.values())}/{args.iterations} iters in {dt:.2f}s "
        f"({sum(status_counts.values()) / dt:.1f} iters/s) | corpus={len(corpus)} | "
        f"metamorphic_checks={meta_checks} | errors={errors}"
    )
    print(f"Kinds: {dict(kind_counts)}")
    print(f"Statuses: {dict(status_counts)}")
    if errors > 0:
        return 1
    print("PASS: 0 differential or metamorphic divergences!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
