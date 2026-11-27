#!/usr/bin/env python3
"""
Suite #1: Multi-Hop Chained Morsel & Pipelined DAG Differential + Metamorphic Fuzzer.

Generates and mutates 2-to-4-stage chained morsel programs in `.impas`:
  - `FILTER_MORSEL` -> `WALK_MORSEL.{fwd,rev,coo_fwd,coo_rev}[.mask]`
  - Inter-stage `SET_OP.{diff,intersect,union}` and `JCC.jempty_f` short-circuits
  - Final `MORSEL_COMMIT.{yield_frontier,count_add_r,scatter_add,scatter_min,scatter_max,
    scatter_argmin,scatter_argmax}` feeding downstream `VEC_REDUCE` + `COLLECT.scalar`
  - 3-way differential parity (`java` == `cpp_on` == `cpp_off`) across geometry profiles P0..P3
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


def run_on_engines(
    clients: dict[str, BatchEngineClient],
    fixture: dict[str, Any],
    profile: str,
    gas: int = 500_000,
) -> dict[str, dict[str, Any]]:
    return {name: cli.run_fixture(fixture, profile, gas) for name, cli in clients.items()}


# ---------------------------------------------------------------------------
# Multi-Hop DAG Gene Representation
# ---------------------------------------------------------------------------

@dataclass
class HopStage:
    kind: str  # "filter_morsel" or "walk_morsel"
    in_dom: int
    out_dom: int
    rel_id: int | None = None
    dir_sub: str = "fwd"  # fwd, rev, coo_fwd, coo_rev
    attr_id: int | None = None
    attr_is_float: bool = True
    cmp_op: str = "ge"
    thresh: float | int = 0.0
    use_mask: bool = False
    use_jempty_inner: bool = False
    post_set_op: str = "none"  # none, diff, intersect, union
    use_jempty_outer: bool = False


@dataclass
class MultiHopGene:
    dom0: int
    seeds: list[int]
    aux_seeds: dict[int, list[int]] = field(default_factory=dict)  # dom_id -> auxiliary frontier seeds
    stages: list[HopStage] = field(default_factory=list)
    final_commit: str = "yield_frontier"
    final_reduce: str = "sum"  # sum, min, max, count_valid


def _rels_from_dom(schema: SnapshotSchema, dom_id: int) -> list[tuple[RelSchema, str]]:
    out: list[tuple[RelSchema, str]] = []
    for rel in schema.relations.values():
        for d in _valid_dirs_for_rel(rel):
            in_d = rel.tgt_dom if d in ("rev", "coo_rev") else rel.src_dom
            if in_d == dom_id:
                out.append((rel, d))
    return out


def _pick_thresh(rng: random.Random, is_float: bool) -> float | int:
    if is_float:
        return rng.choice([0.0, -0.0, 1.0, 20.0, 40.0, 80.0, -1e9, 1e9])
    return rng.choice([0, 1, 20, 40, 80, -10])


def _make_stage(
    rng: random.Random,
    schema: SnapshotSchema,
    cur_dom: int,
    allow_filter: bool,
    prefer_walk: bool,
) -> HopStage:
    walk_cands = _rels_from_dom(schema, cur_dom)
    can_filter = allow_filter and bool(schema.node_attrs.get(cur_dom))
    if not walk_cands or (can_filter and not prefer_walk and rng.random() < 0.30):
        attrs = schema.node_attrs.get(cur_dom, [])
        attr = rng.choice(attrs) if attrs else None
        is_flt = attr.is_float if attr else True
        return HopStage(
            kind="filter_morsel",
            in_dom=cur_dom,
            out_dom=cur_dom,
            rel_id=None,
            dir_sub="fwd",
            attr_id=attr.attr_id if attr else None,
            attr_is_float=is_flt,
            cmp_op=rng.choice(["eq", "ne", "lt", "le", "gt", "ge"]),
            thresh=_pick_thresh(rng, is_flt),
            use_mask=False,
            use_jempty_inner=(rng.random() < 0.5),
            post_set_op=rng.choice(["none", "none", "diff", "intersect"]),
            use_jempty_outer=(rng.random() < 0.4),
        )

    rel, dir_sub = rng.choice(walk_cands)
    is_rev = dir_sub in ("rev", "coo_rev")
    out_dom = rel.src_dom if is_rev else rel.tgt_dom
    eattr = rng.choice(rel.edge_attrs) if (rel.edge_attrs and rng.random() < 0.60) else None
    is_flt = eattr.is_float if eattr else True
    return HopStage(
        kind="walk_morsel",
        in_dom=cur_dom,
        out_dom=out_dom,
        rel_id=rel.rel_id,
        dir_sub=dir_sub,
        attr_id=eattr.attr_id if eattr else None,
        attr_is_float=is_flt,
        cmp_op=rng.choice(["eq", "ne", "lt", "le", "gt", "ge"]),
        thresh=_pick_thresh(rng, is_flt),
        use_mask=(rng.random() < 0.35 and dir_sub in ("fwd", "rev")),
        use_jempty_inner=(rng.random() < 0.45),
        post_set_op=rng.choice(["none", "none", "diff", "intersect", "union"]),
        use_jempty_outer=(rng.random() < 0.40),
    )


def _valid_final_commits(last: HopStage) -> list[str]:
    choices = ["yield_frontier", "count_add_r"]
    if last.kind == "walk_morsel":
        choices.extend(["scatter_argmin", "scatter_argmax"])
        if last.attr_id is not None and last.attr_is_float:
            choices.extend(["scatter_add", "scatter_min", "scatter_max"])
    elif last.kind == "filter_morsel":
        if last.attr_id is not None and last.attr_is_float:
            choices.extend(["scatter_add", "scatter_min", "scatter_max"])
    return choices


def _ensure_aux_seeds(rng: random.Random, gene: MultiHopGene, schema: SnapshotSchema) -> None:
    for st in gene.stages:
        if st.post_set_op != "none" or st.use_mask:
            d = st.out_dom
            if d not in gene.aux_seeds:
                n = max(1, schema.dom_cards[d])
                cnt = rng.choice([2, 8, 32, min(128, n)])
                gene.aux_seeds[d] = sorted({rng.randint(0, n - 1) for _ in range(cnt)})


def generate_random_multihop(rng: random.Random, schema: SnapshotSchema) -> MultiHopGene:
    # Pick a starting domain with outgoing or incoming relations
    cand_doms = [d for d in schema.dom_cards if _rels_from_dom(schema, d)]
    dom0 = rng.choice(cand_doms) if cand_doms else 0
    n0 = max(1, schema.dom_cards[dom0])
    seed_cnt = rng.choice([1, 4, 16, 48, min(128, n0)])
    seeds = sorted({rng.randint(0, n0 - 1) for _ in range(seed_cnt)})

    num_stages = rng.choice([2, 2, 3, 3, 4])
    stages: list[HopStage] = []
    cur_dom = dom0
    for idx in range(num_stages):
        st = _make_stage(
            rng,
            schema,
            cur_dom,
            allow_filter=(idx == 0 or rng.random() < 0.4),
            prefer_walk=(idx == num_stages - 1),
        )
        stages.append(st)
        cur_dom = st.out_dom

    # Disable post_set_op/outer jempty on final stage when final_commit is not yield_frontier
    last = stages[-1]
    final_commit = rng.choice(_valid_final_commits(last))
    if final_commit != "yield_frontier":
        last.post_set_op = "none"
        last.use_jempty_outer = False

    final_reduce = rng.choice(["sum", "min", "max", "count_valid"]) if final_commit in (
        "scatter_add", "scatter_min", "scatter_max"
    ) else "count_valid"

    gene = MultiHopGene(
        dom0=dom0,
        seeds=seeds,
        stages=stages,
        final_commit=final_commit,
        final_reduce=final_reduce,
    )
    _ensure_aux_seeds(rng, gene, schema)
    return gene


def mutate_multihop(rng: random.Random, gene: MultiHopGene, schema: SnapshotSchema) -> MultiHopGene:
    g = MultiHopGene(
        dom0=gene.dom0,
        seeds=list(gene.seeds),
        aux_seeds={k: list(v) for k, v in gene.aux_seeds.items()},
        stages=[HopStage(**s.__dict__) for s in gene.stages],
        final_commit=gene.final_commit,
        final_reduce=gene.final_reduce,
    )
    mut = rng.choice([
        "stage_dir", "stage_attr", "stage_cmp", "stage_mask",
        "stage_setop", "stage_jempty", "final_commit", "seeds", "add_or_drop_stage",
    ])
    idx = rng.randrange(len(g.stages))
    st = g.stages[idx]

    if mut == "stage_dir" and st.kind == "walk_morsel" and st.rel_id is not None:
        # Pick another relation/dir with same (in_dom, out_dom) to keep downstream stages valid
        same_io = [
            (r, d)
            for r, d in _rels_from_dom(schema, st.in_dom)
            if (r.src_dom if d in ("rev", "coo_rev") else r.tgt_dom) == st.out_dom
        ]
        if same_io:
            nr, nd = rng.choice(same_io)
            st.rel_id = nr.rel_id
            st.dir_sub = nd
            if nd not in ("fwd", "rev"):
                st.use_mask = False
            if nr.edge_attrs and rng.random() < 0.6:
                ea = rng.choice(nr.edge_attrs)
                st.attr_id, st.attr_is_float = ea.attr_id, ea.is_float
            else:
                st.attr_id = None
    elif mut == "stage_attr":
        if st.kind == "filter_morsel" and schema.node_attrs.get(st.in_dom):
            a = rng.choice(schema.node_attrs[st.in_dom])
            st.attr_id, st.attr_is_float = a.attr_id, a.is_float
            st.thresh = _pick_thresh(rng, a.is_float)
        elif st.kind == "walk_morsel" and st.rel_id is not None:
            rel = schema.relations[st.rel_id]
            if rel.edge_attrs and rng.random() < 0.7:
                a = rng.choice(rel.edge_attrs)
                st.attr_id, st.attr_is_float = a.attr_id, a.is_float
                st.thresh = _pick_thresh(rng, a.is_float)
            else:
                st.attr_id = None
    elif mut == "stage_cmp":
        st.cmp_op = rng.choice(["eq", "ne", "lt", "le", "gt", "ge"])
        st.thresh = _pick_thresh(rng, st.attr_is_float)
    elif mut == "stage_mask" and st.kind == "walk_morsel" and st.dir_sub in ("fwd", "rev"):
        st.use_mask = not st.use_mask
    elif mut == "stage_setop":
        st.post_set_op = rng.choice(["none", "diff", "intersect", "union"])
    elif mut == "stage_jempty":
        if rng.random() < 0.5:
            st.use_jempty_inner = not st.use_jempty_inner
        else:
            st.use_jempty_outer = not st.use_jempty_outer
    elif mut == "final_commit":
        g.final_commit = rng.choice(_valid_final_commits(g.stages[-1]))
        if g.final_commit in ("scatter_add", "scatter_min", "scatter_max"):
            g.final_reduce = rng.choice(["sum", "min", "max", "count_valid"])
    elif mut == "seeds":
        n0 = max(1, schema.dom_cards[g.dom0])
        cnt = rng.choice([1, 4, 16, 64])
        g.seeds = sorted({rng.randint(0, n0 - 1) for _ in range(cnt)})
    elif mut == "add_or_drop_stage":
        if len(g.stages) < 4 and rng.random() < 0.5:
            tail_dom = g.stages[-1].out_dom
            g.stages.append(_make_stage(rng, schema, tail_dom, allow_filter=True, prefer_walk=True))
        elif len(g.stages) > 2:
            g.stages.pop()

    # Re-validate final_commit against last stage
    valid_fc = _valid_final_commits(g.stages[-1])
    if g.final_commit not in valid_fc:
        g.final_commit = rng.choice(valid_fc)
    if g.final_commit != "yield_frontier":
        g.stages[-1].post_set_op = "none"
        g.stages[-1].use_jempty_outer = False
    _ensure_aux_seeds(rng, g, schema)
    return g


# ---------------------------------------------------------------------------
# .impas Renderer
# ---------------------------------------------------------------------------

def render_multihop_impas(gene: MultiHopGene, schema: SnapshotSchema) -> str:
    data_lines: list[str] = [".data"]
    code_lines: list[str] = [".code"]

    # Param 0: initial frontier
    data_lines.append(
        f"  .param 0 type=frontier dom={gene.dom0} values=[{', '.join(str(x) for x in gene.seeds)}]"
    )
    next_param = 1

    # Allocate auxiliary frontiers in F4..F7 (s3 MASK_F requires F0..F7!)
    needed_aux_doms: list[int] = []
    for st in gene.stages:
        if (st.post_set_op != "none" or st.use_mask) and st.out_dom not in needed_aux_doms:
            needed_aux_doms.append(st.out_dom)

    aux_freg: dict[int, str] = {}
    aux_param: dict[int, int] = {}
    for i, d in enumerate(needed_aux_doms[:4]):
        pidx = next_param
        next_param += 1
        aux_param[d] = pidx
        aux_freg[d] = f"F{4 + i}"
        vals = gene.aux_seeds.get(d, [0])
        data_lines.append(
            f"  .param {pidx} type=frontier dom={d} values=[{', '.join(str(x) for x in vals)}]"
        )

    # Allocate f64 scalar threshold params for float stages (int stages use LOAD_CONST.i32)
    thresh_param: dict[int, int] = {}
    for idx, st in enumerate(gene.stages):
        if st.attr_id is not None and st.attr_is_float:
            pidx = next_param
            next_param += 1
            thresh_param[idx] = pidx
            data_lines.append(f"  .param {pidx} type=f64 value={st.thresh!r}")

    # Load F0 and auxiliary frontiers
    code_lines.append(f"  LOAD_PARAM.frontier F0, param=0, dom={gene.dom0}")
    for d, freg in aux_freg.items():
        code_lines.append(f"  LOAD_PARAM.frontier {freg}, param={aux_param[d]}, dom={d}")

    # Pre-initialize R0 = 0 and final sink (so early JCC.jempty_f short-circuit always has defined output)
    code_lines.append("  LOAD_CONST.u32 R0, 0")
    last_out_dom = gene.stages[-1].out_dom
    if gene.final_commit in ("scatter_add", "scatter_min", "scatter_max"):
        code_lines.append(f"  VEC_INIT.dom_f64 V1, dom={last_out_dom}")
    elif gene.final_commit in ("scatter_argmin", "scatter_argmax"):
        code_lines.append(f"  VEC_INIT.dom_u64 V1, dom={last_out_dom}")

    # Stage frontier register pool strictly in F0..F3 (F8..F15 are morsel-local MF0..MF7; F4..F7 are aux)
    stage_f_pool = ["F1", "F2", "F3"]
    cur_f = "F0"
    pool_pos = 0
    dom_history: dict[int, str] = {gene.dom0: "F0"}
    has_outer_shortcircuit = False

    def _next_stage_f(avoid: str) -> str:
        nonlocal pool_pos
        for _ in range(len(stage_f_pool)):
            cand = stage_f_pool[pool_pos % len(stage_f_pool)]
            pool_pos += 1
            if cand != avoid:
                return cand
        return "F1"

    for idx, st in enumerate(gene.stages):
        is_last = idx == len(gene.stages) - 1
        is_rev = st.dir_sub in ("rev", "coo_rev")
        id_reg = "M0" if (st.kind == "filter_morsel" or is_rev) else "M1"

        if st.attr_id is not None:
            if st.attr_is_float:
                code_lines.append(f"  LOAD_PARAM.scalar R5, param={thresh_param[idx]}")
            else:
                code_lines.append(f"  LOAD_CONST.i32 R5, {int(st.thresh)}")

        # Choose output frontier register if committing a frontier (never aliasing cur_f)
        out_f = _next_stage_f(cur_f)

        # Emit morsel header
        if st.kind == "filter_morsel":
            code_lines.append(f"  FILTER_MORSEL {cur_f}")
        elif st.use_mask and st.dir_sub in ("fwd", "rev") and st.out_dom in aux_freg:
            mask_f = aux_freg[st.out_dom]
            code_lines.append(f"  WALK_MORSEL.{st.dir_sub}.mask {cur_f}, {mask_f}, rel={st.rel_id}")
        else:
            code_lines.append(f"  WALK_MORSEL.{st.dir_sub} {cur_f}, rel={st.rel_id}")

        # Emit morsel body
        has_pred = st.attr_id is not None
        if has_pred:
            if st.kind == "filter_morsel":
                code_lines.append(f"    GATHER_NODE_ATTR.vec M4, M0, dom={st.in_dom}, attr={st.attr_id}")
            else:
                code_lines.append(f"    GATHER_EDGE_ATTR.vec M4, M2, rel={st.rel_id}, attr={st.attr_id}")
            code_lines.append(f"    VEC_CMP.{st.cmp_op}.scalar M5, M4, R5")
            code_lines.append("    SET_CONVERT.b2f MF1, M5")
            if st.use_jempty_inner:
                code_lines.append(f"    JCC.jempty_f MF1, m_skip_{idx}")

        ms = ".mask" if has_pred else ""
        mf = ", MF1" if has_pred else ""
        commit_mode = gene.final_commit if is_last else "yield_frontier"

        if commit_mode == "yield_frontier":
            code_lines.append(f"    MORSEL_COMMIT.yield_frontier{ms} {out_f}, {id_reg}{mf}, dom={st.out_dom}")
        elif commit_mode == "count_add_r":
            if has_pred:
                code_lines.append("    MORSEL_COMMIT.count_add_r.mask R0, MF1")
            else:
                code_lines.append("    MORSEL_COMMIT.count_add_r R0")
        elif commit_mode in ("scatter_add", "scatter_min", "scatter_max"):
            code_lines.append(f"    MORSEL_COMMIT.{commit_mode}{ms} V1, {id_reg}, M4{mf}")
        elif commit_mode in ("scatter_argmin", "scatter_argmax"):
            code_lines.append(f"    MORSEL_COMMIT.{commit_mode}{ms} V1, {id_reg}, M0{mf}")

        if has_pred and st.use_jempty_inner:
            code_lines.append(f"m_skip_{idx}:")
        code_lines.append("  MORSEL_END")

        if commit_mode == "yield_frontier":
            cur_f = out_f
            # Optional inter-stage SET_OP.{diff, intersect, union}
            if st.post_set_op != "none":
                rhs_f = aux_freg.get(st.out_dom) or dom_history.get(st.out_dom)
                if rhs_f is not None:
                    set_out = _next_stage_f(cur_f)
                    code_lines.append(f"  SET_OP.{st.post_set_op} {set_out}, {cur_f}, {rhs_f}")
                    cur_f = set_out
            dom_history[st.out_dom] = cur_f

            # Optional outer JCC.jempty_f short-circuit between stages
            if not is_last and st.use_jempty_outer:
                has_outer_shortcircuit = True
                code_lines.append(f"  JCC.jempty_f {cur_f}, early_done")

    # Downstream reduction / materialization
    if gene.final_commit == "yield_frontier":
        code_lines.append(f"  SET_CARDINALITY R0, {cur_f}")
    elif gene.final_commit in ("scatter_add", "scatter_min", "scatter_max"):
        code_lines.append(f"  VEC_REDUCE.{gene.final_reduce} R0, V1")
    elif gene.final_commit in ("scatter_argmin", "scatter_argmax"):
        code_lines.append("  VEC_REDUCE.count_valid R0, V1")

    if has_outer_shortcircuit:
        code_lines.append("early_done:")
    code_lines.append("  COLLECT.scalar R0, out=0")
    code_lines.append("  HALT")
    return "\n".join(data_lines + code_lines) + "\n"


# ---------------------------------------------------------------------------
# Metamorphic EQ_REL_* Swap Oracle
# ---------------------------------------------------------------------------

def _metamorphic_twin(
    rng: random.Random, gene: MultiHopGene, schema: SnapshotSchema
) -> MultiHopGene | None:
    swappable: list[int] = []
    for i, st in enumerate(gene.stages):
        if (
            st.kind == "walk_morsel"
            and st.rel_id is not None
            and st.rel_id in schema.eq_rels
            and st.attr_id is None
            and not st.use_mask
        ):
            swappable.append(i)
    if not swappable:
        return None

    twin = MultiHopGene(
        dom0=gene.dom0,
        seeds=list(gene.seeds),
        aux_seeds={k: list(v) for k, v in gene.aux_seeds.items()},
        stages=[HopStage(**s.__dict__) for s in gene.stages],
        final_commit=gene.final_commit,
        final_reduce=gene.final_reduce,
    )
    changed = False
    for idx in swappable:
        st = twin.stages[idx]
        assert st.rel_id is not None
        is_rev = st.dir_sub in ("rev", "coo_rev")
        cands: list[tuple[int, str]] = []
        for rid in schema.eq_rels:
            if rid == st.rel_id:
                continue
            r = schema.relations[rid]
            if is_rev:
                if r.src_dom != st.out_dom or r.tgt_dom != st.in_dom:
                    continue
                if r.has_csc:
                    cands.append((rid, "rev"))
                if r.has_coo:
                    cands.append((rid, "coo_rev"))
            else:
                if r.src_dom != st.in_dom or r.tgt_dom != st.out_dom:
                    continue
                if r.has_csr:
                    cands.append((rid, "fwd"))
                if r.has_coo:
                    cands.append((rid, "coo_fwd"))
        if cands:
            new_rid, new_dir = rng.choice(cands)
            st.rel_id = new_rid
            st.dir_sub = new_dir
            changed = True
    return twin if changed else None


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Suite #1: Multi-hop chained morsel & DAG fuzzer")
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
    parser.add_argument("--seed", type=int, default=3026)
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
        f"Suite #1 Multi-Hop Morsel DAG Fuzzer | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}, eq_rels={len(schema.eq_rels)}) | "
        f"engines={','.join(clients.keys())} | geom={','.join(geom_profiles)} | iters={args.iterations}"
    )

    rng = random.Random(args.seed)
    corpus: list[MultiHopGene] = []
    seen_sigs: set[str] = set()
    hop_counts: Counter[int] = Counter()
    commit_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    meta_checks = 0
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            if corpus and rng.random() < 0.60:
                gene = mutate_multihop(rng, rng.choice(corpus), schema)
            else:
                gene = generate_random_multihop(rng, schema)

            prof = geom_profiles[i % len(geom_profiles)]
            impas = render_multihop_impas(gene, schema)
            prog = asm.assemble(impas)
            fixture = {
                "words_hex": [f"0x{w:016x}" for w in prog.words],
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
            }

            results = run_on_engines(clients, fixture, prof, 500_000)
            err = compare_results(results)
            ref = next(iter(results.values()))
            hop_counts[len(gene.stages)] += 1
            commit_counts[gene.final_commit] += 1
            status_counts[str(ref.get("status", "UNKNOWN"))] += 1

            sig = hashlib.sha1(
                f"{len(gene.stages)}:{gene.final_commit}:{ref.get('status')}:{ref.get('outputs')}".encode()
            ).hexdigest()[:12]
            if sig not in seen_sigs and len(corpus) < 512:
                seen_sigs.add(sig)
                corpus.append(gene)

            if err is not None:
                errors += 1
                print(
                    f"[DIFF #{errors}] iter={i} prof={prof} hops={len(gene.stages)} "
                    f"commit={gene.final_commit}: {err}"
                )
                print(impas)
                if errors >= args.max_errors:
                    break
                continue

            # EQ_REL_* metamorphic swap check
            twin = _metamorphic_twin(rng, gene, schema)
            if twin is not None and ref.get("status") == "IMPULSE_VM_OK":
                twin_impas = render_multihop_impas(twin, schema)
                tprog = asm.assemble(twin_impas)
                twin_fix = {
                    "words_hex": [f"0x{w:016x}" for w in tprog.words],
                    "data_directives": tprog.data_directives,
                    "const_pool": tprog.const_pool,
                }
                twin_res = next(iter(clients.values())).run_fixture(twin_fix, prof, 500_000)
                meta_checks += 1
                if twin_res.get("status") != ref.get("status") or twin_res.get("outputs") != ref.get("outputs"):
                    errors += 1
                    print(
                        f"[META DIFF #{errors}] iter={i} prof={prof}: "
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
        f"corpus={len(corpus)} | eq_rel_meta={meta_checks} | errors={errors}"
    )
    print(f"Hops: {dict(sorted(hop_counts.items()))}")
    print(f"FinalCommits: {dict(commit_counts)}")
    print(f"Statuses: {dict(status_counts)}")
    if errors > 0:
        return 1
    print("PASS: 0 differential or metamorphic divergences!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
