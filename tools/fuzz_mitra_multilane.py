#!/usr/bin/env python3
"""
Suite #2: Multi-Lane Semiring & Witness (`MITra` / `OP_LANE_STATE_*` & `OP_LANE_WALK`)
Differential + 1-Lane vs K-Lane Metamorphic Fuzzer.

Tests across K in {1, 2, 4, 8, 16, 64}:
  - `LANE_INIT_SEEDS.{auto_width,force_w64}`
  - `LANE_WALK.{fwd,rev}` (1-hop & 2-hop) + `SET_CARDINALITY.lane_pop_sum` + `LANE_BITWISE.extract_lane`
  - `WALK_MORSEL.{fwd,rev,coo_fwd,coo_rev}` + `MORSEL_COMMIT.yield_lane_or` (1-hop & 2-hop)
  - `LANE_STATE_INIT.{from_frontier_seeds,with_witness}` + `LANE_STATE_WALK.*` +
    `LANE_STATE_OP.*` + `LANE_STATE_REDUCE.*`
  - 1-Lane vs K-Lane Metamorphic Oracle: for K in {2, 4, 8}, compares K-lane `lane_pop_sum`
    against the sum of K independent 1-lane `WALK` / `WALK_MORSEL` runs on identical seeds,
    and checks `LANE_BITWISE.extract_lane` cardinality against the corresponding 1-lane run.
  - `EQ_REL_*` metamorphic relation-swap equivalence checks.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_snapshot_morsels import (  # noqa: E402
    RelSchema,
    SnapshotSchema,
    _discover_default_lib,
    _valid_dirs_for_rel,
    compare_results,
    load_snapshot_schema,
)
from fuzz_vm_contracts import (  # noqa: E402
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    resolve_spec_dir,
)

K_CHOICES = [1, 2, 4, 8, 16, 64]
STATE_FWD_SUBOPS = ["fwd_min_plus", "fwd_max_plus", "fwd_max_mul", "fwd_sum_mul", "fwd_min_max"]
STATE_REV_SUBOPS = ["rev_min_plus", "rev_max_plus", "rev_max_mul", "rev_sum_mul", "rev_min_max"]
STATE_REDUCE_SUBOPS = ["min", "max", "sum", "mean", "argmin", "argmax"]


def run_on_engines(
    clients: dict[str, BatchEngineClient],
    fixture: dict[str, Any],
    profile: str,
    gas: int = 500_000,
) -> dict[str, dict[str, Any]]:
    return {name: cli.run_fixture(fixture, profile, gas) for name, cli in clients.items()}


# ---------------------------------------------------------------------------
# Gene Representation
# ---------------------------------------------------------------------------

@dataclass
class LaneHop:
    rel_id: int
    dir_sub: str  # fwd, rev, coo_fwd, coo_rev (coo_* only for lane_morsel)
    in_dom: int
    out_dom: int
    semiring: str = "fwd_min_plus"  # used by mitra_state
    attr_id: int | None = None  # edge attr for mitra_state


@dataclass
class MitraGene:
    kind: str  # "lane_walk", "lane_morsel", "mitra_state"
    k_lanes: int
    force_w64: bool
    dom0: int
    seeds: list[int]
    hops: list[LaneHop] = field(default_factory=list)
    extract_lane_idx: int | None = None  # for lane_walk / lane_morsel
    with_witness: bool = False  # for mitra_state
    use_accum: bool = False  # for mitra_state hop2 if endo
    use_mask: bool = False  # for mitra_state
    state_op: str = "none"  # none, project_affine, prune_cutoff, to_lane_frontier, to_quorum_frontier
    state_reduce: str = "sum"  # min, max, sum, mean, argmin, argmax, trace_path


def _rels_for_kind(schema: SnapshotSchema, dom_id: int, allow_coo: bool) -> list[tuple[RelSchema, str]]:
    out: list[tuple[RelSchema, str]] = []
    for rel in schema.relations.values():
        dirs = _valid_dirs_for_rel(rel)
        for d in dirs:
            if not allow_coo and d in ("coo_fwd", "coo_rev"):
                continue
            in_d = rel.tgt_dom if d in ("rev", "coo_rev") else rel.src_dom
            if in_d == dom_id:
                out.append((rel, d))
    return out


def _make_lane_hop(
    rng: random.Random,
    schema: SnapshotSchema,
    cur_dom: int,
    allow_coo: bool,
    for_state: bool,
) -> LaneHop | None:
    cands = _rels_for_kind(schema, cur_dom, allow_coo=allow_coo)
    if not cands:
        return None
    rel, d = rng.choice(cands)
    is_rev = d in ("rev", "coo_rev")
    out_dom = rel.src_dom if is_rev else rel.tgt_dom
    semiring = rng.choice(STATE_REV_SUBOPS if is_rev else STATE_FWD_SUBOPS)
    attr_id: int | None = None
    if for_state and rel.edge_attrs and rng.random() < 0.55:
        attr_id = rng.choice(rel.edge_attrs).attr_id
    return LaneHop(
        rel_id=rel.rel_id,
        dir_sub=d,
        in_dom=cur_dom,
        out_dom=out_dom,
        semiring=semiring,
        attr_id=attr_id,
    )


def generate_random_mitra(rng: random.Random, schema: SnapshotSchema) -> MitraGene:
    kind = rng.choice(["lane_walk", "lane_morsel", "mitra_state"])
    allow_coo = (kind == "lane_morsel")
    for_state = (kind == "mitra_state")

    cand_doms = [d for d in schema.dom_cards if _rels_for_kind(schema, d, allow_coo=allow_coo)]
    dom0 = rng.choice(cand_doms) if cand_doms else 0
    n0 = max(1, schema.dom_cards[dom0])

    # Weight K in {2, 4, 8} slightly higher so 1-lane vs K-lane metamorphic checks fire often
    k_lanes = rng.choice([1, 2, 2, 4, 4, 8, 8, 16, 64])
    seeds = [rng.randint(0, n0 - 1) for _ in range(k_lanes)]
    force_w64 = (rng.random() < 0.35)

    num_hops = rng.choice([1, 2])
    hops: list[LaneHop] = []
    cur_dom = dom0
    for _ in range(num_hops):
        hop = _make_lane_hop(rng, schema, cur_dom, allow_coo=allow_coo, for_state=for_state)
        if hop is None:
            break
        hops.append(hop)
        cur_dom = hop.out_dom

    extract_lane_idx: int | None = None
    if kind in ("lane_walk", "lane_morsel") and rng.random() < 0.65:
        extract_lane_idx = rng.randrange(k_lanes)

    with_witness = for_state and (rng.random() < 0.45)
    state_op = rng.choice(["none", "project_affine", "prune_cutoff", "to_lane_frontier", "to_quorum_frontier"]) if for_state else "none"
    reduce_choices = list(STATE_REDUCE_SUBOPS)
    if with_witness:
        reduce_choices.append("trace_path")
    state_reduce = rng.choice(reduce_choices) if for_state else "sum"

    return MitraGene(
        kind=kind,
        k_lanes=k_lanes,
        force_w64=force_w64,
        dom0=dom0,
        seeds=seeds,
        hops=hops,
        extract_lane_idx=extract_lane_idx,
        with_witness=with_witness,
        use_accum=(for_state and len(hops) == 2 and hops[0].out_dom == hops[1].out_dom and rng.random() < 0.4),
        use_mask=(for_state and rng.random() < 0.35),
        state_op=state_op,
        state_reduce=state_reduce,
    )


def mutate_mitra(rng: random.Random, gene: MitraGene, schema: SnapshotSchema) -> MitraGene:
    g = MitraGene(
        kind=gene.kind,
        k_lanes=gene.k_lanes,
        force_w64=gene.force_w64,
        dom0=gene.dom0,
        seeds=list(gene.seeds),
        hops=[LaneHop(**h.__dict__) for h in gene.hops],
        extract_lane_idx=gene.extract_lane_idx,
        with_witness=gene.with_witness,
        use_accum=gene.use_accum,
        use_mask=gene.use_mask,
        state_op=gene.state_op,
        state_reduce=gene.state_reduce,
    )
    allow_coo = (g.kind == "lane_morsel")
    for_state = (g.kind == "mitra_state")
    mut = rng.choice(["k_lanes", "force_w64", "hop_rel", "semiring", "state_op", "state_reduce", "extract", "witness"])

    if mut == "k_lanes":
        g.k_lanes = rng.choice(K_CHOICES)
        n0 = max(1, schema.dom_cards[g.dom0])
        g.seeds = [rng.randint(0, n0 - 1) for _ in range(g.k_lanes)]
        if g.extract_lane_idx is not None:
            g.extract_lane_idx = rng.randrange(g.k_lanes)
    elif mut == "force_w64":
        g.force_w64 = not g.force_w64
    elif mut == "hop_rel" and g.hops:
        idx = rng.randrange(len(g.hops))
        h = g.hops[idx]
        same_io = [
            (r, d)
            for r, d in _rels_for_kind(schema, h.in_dom, allow_coo=allow_coo)
            if (r.src_dom if d in ("rev", "coo_rev") else r.tgt_dom) == h.out_dom
        ]
        if same_io:
            nr, nd = rng.choice(same_io)
            h.rel_id = nr.rel_id
            h.dir_sub = nd
            is_rev = nd in ("rev", "coo_rev")
            h.semiring = rng.choice(STATE_REV_SUBOPS if is_rev else STATE_FWD_SUBOPS)
            if for_state and nr.edge_attrs and rng.random() < 0.5:
                h.attr_id = rng.choice(nr.edge_attrs).attr_id
            else:
                h.attr_id = None
    elif mut == "semiring" and for_state and g.hops:
        h = rng.choice(g.hops)
        is_rev = h.dir_sub in ("rev", "coo_rev")
        h.semiring = rng.choice(STATE_REV_SUBOPS if is_rev else STATE_FWD_SUBOPS)
    elif mut == "state_op" and for_state:
        g.state_op = rng.choice(["none", "project_affine", "prune_cutoff", "to_lane_frontier", "to_quorum_frontier"])
    elif mut == "state_reduce" and for_state:
        choices = list(STATE_REDUCE_SUBOPS) + (["trace_path"] if g.with_witness else [])
        g.state_reduce = rng.choice(choices)
    elif mut == "extract" and g.kind in ("lane_walk", "lane_morsel"):
        g.extract_lane_idx = None if g.extract_lane_idx is not None else rng.randrange(g.k_lanes)
    elif mut == "witness" and for_state:
        g.with_witness = not g.with_witness
        if not g.with_witness and g.state_reduce == "trace_path":
            g.state_reduce = rng.choice(STATE_REDUCE_SUBOPS)
    return g


# ---------------------------------------------------------------------------
# .impas Renderers
# ---------------------------------------------------------------------------

def render_mitra_impas(gene: MitraGene) -> str:
    lines: list[str] = [
        ".data",
        f"  .param 0 type=u64_vec values=[{', '.join(str(x) for x in gene.seeds)}]",
        ".code",
        "  LOAD_PARAM.vec V0, param=0, type=u64",
    ]
    init_sub = "force_w64" if gene.force_w64 else "auto_width"
    lines.append(f"  LANE_INIT_SEEDS.{init_sub} F0, V0, dom={gene.dom0}")

    if gene.kind == "lane_walk":
        cur_f = "F0"
        for idx, h in enumerate(gene.hops):
            nxt_f = f"F{idx + 1}"
            lines.append(f"  LANE_WALK.{h.dir_sub} {nxt_f}, {cur_f}, rel={h.rel_id}")
            cur_f = nxt_f
        lines.append(f"  SET_CARDINALITY.lane_pop_sum R0, {cur_f}")
        lines.append("  COLLECT.scalar R0, out=0")
        if gene.extract_lane_idx is not None:
            lines.append(f"  LANE_BITWISE.extract_lane F6, {cur_f}, {gene.extract_lane_idx}")
            lines.append("  SET_CARDINALITY.card_1bit R1, F6")
            lines.append("  COLLECT.scalar R1, out=1")
        lines.append("  HALT")
        return "\n".join(lines) + "\n"

    if gene.kind == "lane_morsel":
        cur_f = "F0"
        for idx, h in enumerate(gene.hops):
            nxt_f = f"F{idx + 1}"
            is_rev = h.dir_sub in ("rev", "coo_rev")
            id_reg = "M0" if is_rev else "M1"
            lines.append(f"  WALK_MORSEL.{h.dir_sub} {cur_f}, rel={h.rel_id}")
            lines.append(f"    MORSEL_COMMIT.yield_lane_or {nxt_f}, {id_reg}, M3, dom={h.out_dom}")
            lines.append("  MORSEL_END")
            cur_f = nxt_f
        lines.append(f"  SET_CARDINALITY.lane_pop_sum R0, {cur_f}")
        lines.append("  COLLECT.scalar R0, out=0")
        if gene.extract_lane_idx is not None:
            lines.append(f"  LANE_BITWISE.extract_lane F6, {cur_f}, {gene.extract_lane_idx}")
            lines.append("  SET_CARDINALITY.card_1bit R1, F6")
            lines.append("  COLLECT.scalar R1, out=1")
        lines.append("  HALT")
        return "\n".join(lines) + "\n"

    # mitra_state
    if gene.with_witness:
        lines.append("  LOAD_CONST.f32 R10, 1.0")
        lines.append(f"  LANE_STATE_INIT.with_witness.scalar V1, F0, R10, dom={gene.dom0}, k={gene.k_lanes}")
    else:
        lines.append(f"  LANE_STATE_INIT.from_frontier_seeds V1, F0, dom={gene.dom0}, k={gene.k_lanes}")

    cur_v = "V1"
    for idx, h in enumerate(gene.hops):
        nxt_v = f"V{idx + 2}"
        attr_tok = "none" if h.attr_id is None else str(h.attr_id)
        if gene.use_accum and idx == 1 and h.in_dom == h.out_dom:
            lines.append(f"  LANE_STATE_WALK.{h.semiring}.accum {cur_v}, {cur_v}, rel={h.rel_id}, attr={attr_tok}")
        elif gene.use_mask and idx == 0:
            lines.append(f"  INIT_DOMAIN_SPAN.all_span F3, dom={h.out_dom}")
            lines.append(f"  LANE_STATE_WALK.{h.semiring}.mask {nxt_v}, {cur_v}, F3, rel={h.rel_id}, attr={attr_tok}")
            cur_v = nxt_v
        else:
            lines.append(f"  LANE_STATE_WALK.{h.semiring} {nxt_v}, {cur_v}, rel={h.rel_id}, attr={attr_tok}")
            cur_v = nxt_v

    # Optional LANE_STATE_OP
    if gene.state_op == "project_affine":
        lines.append("  LOAD_CONST.f32 R11, 2.0")
        lines.append("  LOAD_CONST.f32 R12, 0.5")
        lines.append(f"  LANE_STATE_OP.project_affine V5, {cur_v}, R11, R12")
        cur_v = "V5"
    elif gene.state_op == "prune_cutoff":
        lines.append("  LOAD_CONST.f32 R11, 50.0")
        lines.append(f"  LANE_STATE_OP.prune_cutoff V5, {cur_v}, R11")
        cur_v = "V5"
    elif gene.state_op == "to_lane_frontier":
        lines.append(f"  LANE_STATE_OP.to_lane_frontier F4, {cur_v}")
        lines.append("  SET_CARDINALITY.lane_pop_sum R1, F4")
        lines.append("  COLLECT.scalar R1, out=1")
    elif gene.state_op == "to_quorum_frontier":
        lines.append(f"  LANE_STATE_OP.to_quorum_frontier F4, {cur_v}, 1")
        lines.append("  SET_CARDINALITY.card_1bit R1, F4")
        lines.append("  COLLECT.scalar R1, out=1")

    # LANE_STATE_REDUCE
    if gene.state_reduce == "trace_path":
        lines.append("  LOAD_CONST.u32 R13, 0")
        lines.append(f"  LANE_STATE_REDUCE.trace_path.scalar V6, {cur_v}, R13")
        lines.append("  VEC_REDUCE.count_valid R0, V6")
    elif gene.state_reduce in ("argmin", "argmax"):
        lines.append(f"  LANE_STATE_REDUCE.{gene.state_reduce} V6, {cur_v}")
        lines.append("  VEC_REDUCE.count_valid R0, V6")
    else:
        lines.append(f"  LANE_STATE_REDUCE.{gene.state_reduce} V6, {cur_v}")
        lines.append("  VEC_REDUCE.sum R0, V6")
    lines.append("  COLLECT.scalar R0, out=0")
    lines.append("  HALT")
    return "\n".join(lines) + "\n"


def render_single_lane_oracle_impas(gene: MitraGene, seed_node: int) -> str:
    """Render an equivalent 1-lane (1-bit frontier) program for a single seed node."""
    lines: list[str] = [
        ".data",
        f"  .param 0 type=frontier dom={gene.dom0} values=[{seed_node}]",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={gene.dom0}",
    ]
    cur_f = "F0"
    for idx, h in enumerate(gene.hops):
        nxt_f = f"F{idx + 1}"
        if gene.kind == "lane_walk":
            lines.append(f"  WALK.{h.dir_sub} {nxt_f}, {cur_f}, rel={h.rel_id}")
        else:
            is_rev = h.dir_sub in ("rev", "coo_rev")
            id_reg = "M0" if is_rev else "M1"
            lines.append(f"  WALK_MORSEL.{h.dir_sub} {cur_f}, rel={h.rel_id}")
            lines.append(f"    MORSEL_COMMIT.yield_frontier {nxt_f}, {id_reg}, dom={h.out_dom}")
            lines.append("  MORSEL_END")
        cur_f = nxt_f
    lines.append(f"  SET_CARDINALITY R0, {cur_f}")
    lines.append("  COLLECT.scalar R0, out=0")
    lines.append("  HALT")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metamorphic EQ_REL_* Swap Oracle
# ---------------------------------------------------------------------------

def _metamorphic_eq_rel_twin(
    rng: random.Random, gene: MitraGene, schema: SnapshotSchema
) -> MitraGene | None:
    allow_coo = (gene.kind == "lane_morsel")
    swappable = [
        i for i, h in enumerate(gene.hops)
        if h.rel_id in schema.eq_rels and h.attr_id is None
    ]
    if not swappable:
        return None
    twin = MitraGene(
        kind=gene.kind,
        k_lanes=gene.k_lanes,
        force_w64=gene.force_w64,
        dom0=gene.dom0,
        seeds=list(gene.seeds),
        hops=[LaneHop(**h.__dict__) for h in gene.hops],
        extract_lane_idx=gene.extract_lane_idx,
        with_witness=gene.with_witness,
        use_accum=gene.use_accum,
        use_mask=gene.use_mask,
        state_op=gene.state_op,
        state_reduce=gene.state_reduce,
    )
    changed = False
    for idx in swappable:
        h = twin.hops[idx]
        is_rev = h.dir_sub in ("rev", "coo_rev")
        cands: list[tuple[int, str]] = []
        for rid in schema.eq_rels:
            if rid == h.rel_id:
                continue
            r = schema.relations[rid]
            if is_rev:
                if r.src_dom != h.out_dom or r.tgt_dom != h.in_dom:
                    continue
                if r.has_csc:
                    cands.append((rid, "rev"))
                if allow_coo and r.has_coo:
                    cands.append((rid, "coo_rev"))
            else:
                if r.src_dom != h.in_dom or r.tgt_dom != h.out_dom:
                    continue
                if r.has_csr:
                    cands.append((rid, "fwd"))
                if allow_coo and r.has_coo:
                    cands.append((rid, "coo_fwd"))
        if cands:
            h.rel_id, h.dir_sub = rng.choice(cands)
            changed = True
    return twin if changed else None


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------

def _extract_scalar_out(res: dict[str, Any], out_idx: int) -> int | None:
    outs = res.get("outputs")
    if isinstance(outs, dict):
        entry = outs.get(str(out_idx))
        if entry is None:
            entry = outs.get(out_idx)
        if isinstance(entry, list) and len(entry) >= 1 and entry[0] is not None:
            return int(entry[0])
        if isinstance(entry, dict) and "value" in entry:
            return int(entry["value"])
        if isinstance(entry, int):
            return entry
    elif isinstance(outs, list):
        for item in outs:
            if isinstance(item, dict) and item.get("out") == out_idx:
                val = item.get("value", item.get("scalar"))
                if val is not None:
                    return int(val)
        if out_idx < len(outs) and isinstance(outs[out_idx], dict):
            val = outs[out_idx].get("value", outs[out_idx].get("scalar"))
            if val is not None:
                return int(val)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Suite #2: Multi-lane MITra & 1-vs-K metamorphic fuzzer")
    parser.add_argument("snapshot", type=Path, help="Path to v0.11 .imps snapshot")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    parser.add_argument(
        "--cpp-on-cmd",
        type=str,
        default="/Users/jesse/impulse/.worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_on/vm_fixture_batch_runner",
    )
    parser.add_argument(
        "--cpp-off-cmd",
        type=str,
        default="/Users/jesse/impulse/.worktrees/v11-m3/impulse-graph-core/impulse-cpp/build_off/vm_fixture_batch_runner",
    )
    parser.add_argument("--engines", type=str, default="java,cpp_on,cpp_off")
    parser.add_argument("--profiles", type=str, default="P0,P1,P2,P3")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4026)
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
            raise ValueError(f"Unsupported engine: {eng}")
        clients[eng] = BatchEngineClient(cmd, eng)

    print(
        f"Suite #2 Multi-Lane & MITra Fuzzer | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}, eq_rels={len(schema.eq_rels)}) | "
        f"engines={','.join(clients.keys())} | geom={','.join(geom_profiles)} | iters={args.iterations}"
    )

    rng = random.Random(args.seed)
    corpus: list[MitraGene] = []
    seen_sigs: set[str] = set()
    kind_counts: Counter[str] = Counter()
    k_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    lane_meta_checks = 0
    eq_rel_checks = 0
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            if corpus and rng.random() < 0.60:
                gene = mutate_mitra(rng, rng.choice(corpus), schema)
            else:
                gene = generate_random_mitra(rng, schema)

            prof = geom_profiles[i % len(geom_profiles)]
            impas = render_mitra_impas(gene)
            prog = asm.assemble(impas)
            fixture = {
                "words_hex": [f"0x{w:016x}" for w in prog.words],
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
            }

            results = run_on_engines(clients, fixture, prof, 500_000)
            err = compare_results(results)
            ref = next(iter(results.values()))
            kind_counts[gene.kind] += 1
            k_counts[gene.k_lanes] += 1
            status_counts[str(ref.get("status", "UNKNOWN"))] += 1

            sig = hashlib.sha1(
                f"{gene.kind}:{gene.k_lanes}:{gene.force_w64}:{len(gene.hops)}:{ref.get('status')}:{ref.get('outputs')}".encode()
            ).hexdigest()[:12]
            if sig not in seen_sigs and len(corpus) < 512:
                seen_sigs.add(sig)
                corpus.append(gene)

            if err is not None:
                errors += 1
                print(
                    f"[DIFF #{errors}] iter={i} prof={prof} kind={gene.kind} K={gene.k_lanes} "
                    f"hops={len(gene.hops)}: {err}"
                )
                print(impas)
                if errors >= args.max_errors:
                    break
                continue

            ref_cli = next(iter(clients.values()))

            # 1-Lane vs K-Lane Metamorphic Oracle for K in {2, 4, 8}
            if (
                gene.kind in ("lane_walk", "lane_morsel")
                and gene.k_lanes in (2, 4, 8)
                and ref.get("status") == "IMPULSE_VM_OK"
            ):
                k_pop_sum = _extract_scalar_out(ref, 0)
                single_cards: list[int] = []
                oracle_ok = True
                for s_node in gene.seeds:
                    s_impas = render_single_lane_oracle_impas(gene, s_node)
                    s_prog = asm.assemble(s_impas)
                    s_fix = {
                        "words_hex": [f"0x{w:016x}" for w in s_prog.words],
                        "data_directives": s_prog.data_directives,
                        "const_pool": s_prog.const_pool,
                    }
                    s_res = ref_cli.run_fixture(s_fix, prof, 500_000)
                    if s_res.get("status") != "IMPULSE_VM_OK":
                        oracle_ok = False
                        errors += 1
                        print(
                            f"[1vK ORACLE STATUS #{errors}] iter={i} seed={s_node}: "
                            f"K-lane OK but 1-lane returned {s_res.get('status')}"
                        )
                        break
                    val = _extract_scalar_out(s_res, 0)
                    if val is None:
                        oracle_ok = False
                        break
                    single_cards.append(val)

                if oracle_ok and k_pop_sum is not None:
                    lane_meta_checks += 1
                    sum_1lane = sum(single_cards)
                    if sum_1lane != k_pop_sum:
                        errors += 1
                        print(
                            f"[1vK META DIFF #{errors}] iter={i} prof={prof} kind={gene.kind} K={gene.k_lanes}: "
                            f"lane_pop_sum={k_pop_sum} != sum(1-lane)={sum_1lane} (per_lane={single_cards})"
                        )
                        print(impas)
                        if errors >= args.max_errors:
                            break
                    elif gene.extract_lane_idx is not None:
                        ext_val = _extract_scalar_out(ref, 1)
                        if ext_val is not None and ext_val != single_cards[gene.extract_lane_idx]:
                            errors += 1
                            print(
                                f"[1vK EXTRACT DIFF #{errors}] iter={i} lane={gene.extract_lane_idx}: "
                                f"extract_lane={ext_val} != 1-lane={single_cards[gene.extract_lane_idx]}"
                            )
                            print(impas)
                            if errors >= args.max_errors:
                                break

            # EQ_REL_* Metamorphic Swap Oracle
            twin = _metamorphic_eq_rel_twin(rng, gene, schema)
            if twin is not None and ref.get("status") == "IMPULSE_VM_OK":
                twin_impas = render_mitra_impas(twin)
                tprog = asm.assemble(twin_impas)
                twin_fix = {
                    "words_hex": [f"0x{w:016x}" for w in tprog.words],
                    "data_directives": tprog.data_directives,
                    "const_pool": tprog.const_pool,
                }
                twin_res = ref_cli.run_fixture(twin_fix, prof, 500_000)
                eq_rel_checks += 1
                if twin_res.get("status") != ref.get("status") or twin_res.get("outputs") != ref.get("outputs"):
                    errors += 1
                    print(
                        f"[EQ_REL META DIFF #{errors}] iter={i} prof={prof} kind={gene.kind} K={gene.k_lanes}: "
                        f"{ref.get('status')}/{ref.get('outputs')} != "
                        f"{twin_res.get('status')}/{twin_res.get('outputs')}"
                    )
                    print("--- ORIG ---\n" + impas + "--- TWIN ---\n" + twin_impas)
                    if errors >= args.max_errors:
                        break
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    total = sum(status_counts.values())
    print(
        f"\nCompleted {total}/{args.iterations} iters in {dt:.2f}s ({total / dt:.1f} iters/s) | "
        f"corpus={len(corpus)} | 1vK_meta={lane_meta_checks} | eq_rel_meta={eq_rel_checks} | errors={errors}"
    )
    print(f"Kinds: {dict(kind_counts)}")
    print(f"K_lanes: {dict(sorted(k_counts.items()))}")
    print(f"Statuses: {dict(status_counts)}")
    if errors > 0:
        return 1
    print("PASS: 0 differential or metamorphic divergences!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
