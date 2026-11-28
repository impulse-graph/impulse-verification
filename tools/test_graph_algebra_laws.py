#!/usr/bin/env python3
"""
Suite #4: Graph-Algebraic & Semiring Law Property Tester + Exhaustive Deterministic
EQ_REL_* & EQ_DOM_* Storage Equivalence Matrix.

Part A (Exhaustive Deterministic Matrix on storage_coverage_small.imps):
  - All EQ_REL_* relations x all valid directions (fwd, rev, coo_fwd, coo_rev):
      * WALK single-hop (plus fixpoint fwd_plus/fwd_star/rev_plus/rev_star/fwd_bounded for endo rels)
      * DEGREE (out / in / total)
      * WALK_MORSEL (yield_frontier / count_add_r)
    Verifying BOTH 3-way (java, cpp_on, cpp_off) parity AND cross-EQ_REL_* equivalence.
  - All EQ_DOM_* domains x all numeric node attributes:
      * GATHER_NODE_ATTR.vec + VEC_REDUCE.count_valid / sum
    Verifying BOTH 3-way parity AND cross-EQ_DOM_* equivalence.

Part B (Algebraic & Semiring Law Checks across P0..P3):
  - Reversibility: v in WALK.fwd(u, R) <=> u in WALK.rev(v, R) across CSR/CSC/COO.
  - Distributivity: WALK(A U B, R) == WALK(A, R) U WALK(B, R).
  - Kleene Identities: fwd_star(A) == A U fwd_plus(A) and fwd_star(fwd_star(A)) == fwd_star(A)
    (and rev_star/rev_plus when CSC is present).
  - De Morgan / Set Algebra: A \\ (B U C) == (A \\ B) & (A \\ C) and
    complement(A U B) == complement(A) & complement(B) across P0..P3 tier boundaries
    (0, 1, 63, 64, 65, 255, 256).
  - Degree-Walk Identity: sum(out_degree(u) for u in S) == WALK_MORSEL.count_add_r(S)
    (and in-degree for rev).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_snapshot_morsels import (  # noqa: E402
    BatchEngineClient,
    SnapshotSchema,
    _discover_default_lib,
    _import_spec_modules,
    _valid_dirs_for_rel,
    compare_results,
    discover_default_java_cmd,
    load_snapshot_schema,
    resolve_spec_dir,
)


def assemble_fixture(asm: Any, impas: str) -> dict[str, Any]:
    prog = asm.assemble(impas)
    return {
        "words_hex": [f"0x{w:016x}" for w in prog.words],
        "data_directives": prog.data_directives,
        "const_pool": prog.const_pool,
    }


def run_on_engines(
    clients: dict[str, BatchEngineClient],
    fixture: dict[str, Any],
    profile: str,
    gas: int = 500_000,
) -> dict[str, dict[str, Any]]:
    return {name: cli.run_fixture(fixture, profile, gas) for name, cli in clients.items()}


def _seed_list_for_dom(n: int) -> list[int]:
    raw = [0, 1, 2, 7, 15, 63, 64, 65, 127, 255, 256, 511, n - 1]
    return sorted({x for x in raw if 0 <= x < n})


def _small_seed_list(n: int) -> list[int]:
    raw = [0, 7, 64, 255]
    return sorted({x for x in raw if 0 <= x < n})


def _make_boundary_frontier(rng: random.Random, n: int, target_size: int) -> list[int]:
    size = min(target_size, n)
    if size <= 0:
        return []
    if size >= n:
        return list(range(n))
    return sorted(rng.sample(range(n), size))


# ---------------------------------------------------------------------------
# Part A: Exhaustive Deterministic EQ_REL_* & EQ_DOM_* Matrix
# ---------------------------------------------------------------------------

def run_part_a(
    asm: Any,
    clients: dict[str, BatchEngineClient],
    schema: SnapshotSchema,
    profiles: list[str],
    max_errors: int,
) -> tuple[int, int, Counter[str]]:
    checks = 0
    errors = 0
    counts: Counter[str] = Counter()

    def exec_and_check(label: str, impas: str, prof: str) -> dict[str, Any] | None:
        nonlocal checks, errors
        checks += 1
        fixture = assemble_fixture(asm, impas)
        results = run_on_engines(clients, fixture, prof)
        diff = compare_results(results)
        ref = next(iter(results.values()))
        if diff is not None:
            errors += 1
            print(f"[PART-A 3-WAY DIFF #{errors}] {label} prof={prof}: {diff}")
            print(impas)
            return None
        if ref.get("status") != "IMPULSE_VM_OK":
            errors += 1
            print(f"[PART-A STATUS ERR #{errors}] {label} prof={prof}: status={ref.get('status')}")
            print(impas)
            return None
        return ref

    # 1. All EQ_REL_* relations x valid directions
    # Equivalence buckets: key -> (first_rel_label, outputs)
    eq_walk_fwd: dict[str, tuple[str, Any]] = {}
    eq_walk_rev: dict[str, tuple[str, Any]] = {}
    eq_fixpoint: dict[str, tuple[str, Any]] = {}
    eq_degree: dict[str, tuple[str, Any]] = {}
    eq_morsel_fwd: dict[str, tuple[str, Any]] = {}
    eq_morsel_rev: dict[str, tuple[str, Any]] = {}

    prof_idx = 0

    for rid in schema.eq_rels:
        if errors >= max_errors:
            break
        rel = schema.relations[rid]
        dirs = _valid_dirs_for_rel(rel)
        is_endo = rel.src_dom == rel.tgt_dom

        for dsub in dirs:
            if errors >= max_errors:
                break
            prof = profiles[prof_idx % len(profiles)]
            prof_idx += 1
            is_rev = dsub in ("rev", "coo_rev")
            in_dom = rel.tgt_dom if is_rev else rel.src_dom
            out_dom = rel.src_dom if is_rev else rel.tgt_dom
            seeds = _seed_list_for_dom(schema.dom_cards[in_dom])

            # 1a. WALK.<dir> single-hop
            impas = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
                f"  WALK.{dsub} F1, F0, rel={rid}",
                "  SET_CARDINALITY R0, F1",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.frontier_ids F1, out=1",
                "  HALT",
            ]) + "\n"
            lbl = f"WALK.{dsub}({rel.name})"
            ref = exec_and_check(lbl, impas, prof)
            counts["eq_rel_walk_1hop"] += 1
            if ref is not None:
                bucket = eq_walk_rev if is_rev else eq_walk_fwd
                if "1hop" not in bucket:
                    bucket["1hop"] = (lbl, ref.get("outputs"))
                elif bucket["1hop"][1] != ref.get("outputs"):
                    errors += 1
                    print(
                        f"[PART-A EQ_REL DIFF #{errors}] {lbl} != {bucket['1hop'][0]}: "
                        f"{ref.get('outputs')!r} != {bucket['1hop'][1]!r}"
                    )

            # 1b. WALK_MORSEL.<dir> (yield_frontier + count_add_r)
            id_reg = "M0" if is_rev else "M1"
            impas_m = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
                f"  WALK_MORSEL.{dsub} F0, rel={rid}",
                f"    MORSEL_COMMIT.yield_frontier F2, {id_reg}, dom={out_dom}",
                "    MORSEL_COMMIT.count_add_r R1",
                "  MORSEL_END",
                "  SET_CARDINALITY R0, F2",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.frontier_ids F2, out=1",
                "  COLLECT.scalar R1, out=2",
                "  HALT",
            ]) + "\n"
            lbl_m = f"WALK_MORSEL.{dsub}({rel.name})"
            ref_m = exec_and_check(lbl_m, impas_m, prof)
            counts["eq_rel_walk_morsel"] += 1
            if ref_m is not None:
                # Cross-check WALK_MORSEL yield_frontier matches WALK.<dir>
                if ref is not None:
                    walk_outs = ref.get("outputs", {})
                    morsel_outs = ref_m.get("outputs", {})
                    if walk_outs.get("0") != morsel_outs.get("0") or walk_outs.get("1") != morsel_outs.get("1"):
                        errors += 1
                        print(
                            f"[PART-A WALK vs MORSEL DIFF #{errors}] {lbl_m} != {lbl}: "
                            f"{morsel_outs!r} vs {walk_outs!r}"
                        )
                mbucket = eq_morsel_rev if is_rev else eq_morsel_fwd
                if "morsel" not in mbucket:
                    mbucket["morsel"] = (lbl_m, ref_m.get("outputs"))
                elif mbucket["morsel"][1] != ref_m.get("outputs"):
                    errors += 1
                    print(
                        f"[PART-A EQ_REL MORSEL DIFF #{errors}] {lbl_m} != {mbucket['morsel'][0]}: "
                        f"{ref_m.get('outputs')!r} != {mbucket['morsel'][1]!r}"
                    )

        # 1c. Fixpoint & bounded WALK for endo EQ_REL_*
        if is_endo and errors < max_errors:
            seeds_fp = _small_seed_list(schema.dom_cards[rel.src_dom])
            fp_ops: list[tuple[str, str]] = []
            if rel.has_csr:
                fp_ops.extend([
                    ("fwd_fixpoint_plus", f"WALK.fwd_fixpoint_plus F1, F0, rel={rid}"),
                    ("fwd_fixpoint_star", f"WALK.fwd_fixpoint_star F1, F0, rel={rid}"),
                    ("fwd_bounded_2", f"WALK.fwd_bounded F1, F0, rel={rid}, k=2"),
                ])
            if rel.has_csc:
                fp_ops.extend([
                    ("rev_fixpoint_plus", f"WALK.rev_fixpoint_plus F1, F0, rel={rid}"),
                    ("rev_fixpoint_star", f"WALK.rev_fixpoint_star F1, F0, rel={rid}"),
                ])
            for fp_key, fp_inst in fp_ops:
                if errors >= max_errors:
                    break
                prof = profiles[prof_idx % len(profiles)]
                prof_idx += 1
                impas_fp = "\n".join([
                    ".data",
                    f"  .param 0 type=frontier dom={rel.src_dom} values=[{', '.join(str(x) for x in seeds_fp)}]",
                    ".code",
                    f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}",
                    f"  {fp_inst}",
                    "  SET_CARDINALITY R0, F1",
                    "  COLLECT.scalar R0, out=0",
                    "  COLLECT.frontier_ids F1, out=1",
                    "  HALT",
                ]) + "\n"
                lbl_fp = f"WALK.{fp_key}({rel.name})"
                ref_fp = exec_and_check(lbl_fp, impas_fp, prof)
                counts["eq_rel_fixpoint"] += 1
                if ref_fp is not None:
                    if fp_key not in eq_fixpoint:
                        eq_fixpoint[fp_key] = (lbl_fp, ref_fp.get("outputs"))
                    elif eq_fixpoint[fp_key][1] != ref_fp.get("outputs"):
                        errors += 1
                        print(
                            f"[PART-A EQ_REL FIXPOINT DIFF #{errors}] {lbl_fp} != {eq_fixpoint[fp_key][0]}"
                        )

        # 1d. DEGREE (out, in, total)
        if errors < max_errors:
            seeds_out = _seed_list_for_dom(schema.dom_cards[rel.src_dom])
            seeds_in = _seed_list_for_dom(schema.dom_cards[rel.tgt_dom])
            if rel.has_csr:
                prof = profiles[prof_idx % len(profiles)]
                prof_idx += 1
                impas_deg = "\n".join([
                    ".data",
                    f"  .param 0 type=frontier dom={rel.src_dom} values=[{', '.join(str(x) for x in seeds_out)}]",
                    ".code",
                    f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}",
                    f"  DEGREE.sum_fwd_r R0, F0, rel={rid}",
                    f"  DEGREE.vec_fwd_v V0, rel={rid}",
                    "  VEC_REDUCE.sum R1, V0",
                    "  COLLECT.scalar R0, out=0",
                    "  COLLECT.scalar R1, out=1",
                    "  HALT",
                ]) + "\n"
                lbl_d = f"DEGREE.out({rel.name})"
                ref_d = exec_and_check(lbl_d, impas_deg, prof)
                counts["eq_rel_degree"] += 1
                if ref_d is not None:
                    if "out" not in eq_degree:
                        eq_degree["out"] = (lbl_d, ref_d.get("outputs"))
                    elif eq_degree["out"][1] != ref_d.get("outputs"):
                        errors += 1
                        print(
                            f"[PART-A EQ_REL DEGREE DIFF #{errors}] {lbl_d} != {eq_degree['out'][0]}: "
                            f"{ref_d.get('outputs')!r} != {eq_degree['out'][1]!r}"
                        )

            if rel.has_csc and errors < max_errors:
                prof = profiles[prof_idx % len(profiles)]
                prof_idx += 1
                impas_deg = "\n".join([
                    ".data",
                    f"  .param 0 type=frontier dom={rel.tgt_dom} values=[{', '.join(str(x) for x in seeds_in)}]",
                    ".code",
                    f"  LOAD_PARAM.frontier F0, param=0, dom={rel.tgt_dom}",
                    f"  DEGREE.sum_rev_r R0, F0, rel={rid}",
                    f"  DEGREE.vec_rev_v V0, rel={rid}",
                    "  VEC_REDUCE.sum R1, V0",
                    "  COLLECT.scalar R0, out=0",
                    "  COLLECT.scalar R1, out=1",
                    "  HALT",
                ]) + "\n"
                lbl_d = f"DEGREE.in({rel.name})"
                ref_d = exec_and_check(lbl_d, impas_deg, prof)
                counts["eq_rel_degree"] += 1
                if ref_d is not None:
                    if "in" not in eq_degree:
                        eq_degree["in"] = (lbl_d, ref_d.get("outputs"))
                    elif eq_degree["in"][1] != ref_d.get("outputs"):
                        errors += 1
                        print(
                            f"[PART-A EQ_REL DEGREE DIFF #{errors}] {lbl_d} != {eq_degree['in'][0]}: "
                            f"{ref_d.get('outputs')!r} != {eq_degree['in'][1]!r}"
                        )

            if rel.has_csr and rel.has_csc and is_endo and errors < max_errors:
                prof = profiles[prof_idx % len(profiles)]
                prof_idx += 1
                impas_tot = "\n".join([
                    ".data",
                    f"  .param 0 type=frontier dom={rel.src_dom} values=[{', '.join(str(x) for x in seeds_out)}]",
                    ".code",
                    f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}",
                    f"  DEGREE.sum_fwd_r R0, F0, rel={rid}",
                    f"  DEGREE.sum_rev_r R1, F0, rel={rid}",
                    "  SCALAR_INT_ALU.add R2, R0, R1",
                    "  COLLECT.scalar R2, out=0",
                    "  HALT",
                ]) + "\n"
                lbl_t = f"DEGREE.total({rel.name})"
                ref_t = exec_and_check(lbl_t, impas_tot, prof)
                counts["eq_rel_degree"] += 1
                if ref_t is not None:
                    if "total" not in eq_degree:
                        eq_degree["total"] = (lbl_t, ref_t.get("outputs"))
                    elif eq_degree["total"][1] != ref_t.get("outputs"):
                        errors += 1
                        print(
                            f"[PART-A EQ_REL DEGREE DIFF #{errors}] {lbl_t} != {eq_degree['total'][0]}: "
                            f"{ref_t.get('outputs')!r} != {eq_degree['total'][1]!r}"
                        )

    # 2. All EQ_DOM_* domains x node attributes
    eq_doms = [d for d, name in schema.dom_names.items() if name.startswith("EQ_DOM_")]
    eq_dom_attr_ref: dict[str, tuple[str, Any]] = {}

    for dom_id in eq_doms:
        if errors >= max_errors:
            break
        dname = schema.dom_names[dom_id]
        ids = _seed_list_for_dom(schema.dom_cards[dom_id])
        for attr in schema.node_attrs.get(dom_id, []):
            if errors >= max_errors:
                break
            prof = profiles[prof_idx % len(profiles)]
            prof_idx += 1
            impas_na = "\n".join([
                ".data",
                f"  .param 0 type=u64_vec values=[{', '.join(str(x) for x in ids)}]",
                ".code",
                "  LOAD_PARAM.vec V0, param=0, type=u64",
                f"  GATHER_NODE_ATTR.vec V1, V0, dom={dom_id}, attr={attr.attr_id}",
                "  VEC_REDUCE.count_valid R0, V1",
                "  VEC_REDUCE.sum R1, V1",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.scalar R1, out=1",
                "  HALT",
            ]) + "\n"
            lbl_na = f"GATHER_NODE_ATTR({dname}.{attr.name})"
            ref_na = exec_and_check(lbl_na, impas_na, prof)
            counts["eq_dom_node_attr"] += 1
            if ref_na is not None:
                if attr.name not in eq_dom_attr_ref:
                    eq_dom_attr_ref[attr.name] = (lbl_na, ref_na.get("outputs"))
                elif eq_dom_attr_ref[attr.name][1] != ref_na.get("outputs"):
                    errors += 1
                    print(
                        f"[PART-A EQ_DOM DIFF #{errors}] {lbl_na} != {eq_dom_attr_ref[attr.name][0]}: "
                        f"{ref_na.get('outputs')!r} != {eq_dom_attr_ref[attr.name][1]!r}"
                    )

    return checks, errors, counts


# ---------------------------------------------------------------------------
# Part B: Algebraic & Semiring Law Checks across P0..P3
# ---------------------------------------------------------------------------

def run_part_b(
    asm: Any,
    clients: dict[str, BatchEngineClient],
    schema: SnapshotSchema,
    profiles: list[str],
    trials: int,
    seed: int,
    max_errors: int,
) -> tuple[int, int, Counter[str]]:
    rng = random.Random(seed)
    checks = 0
    errors = 0
    counts: Counter[str] = Counter()

    valid_rels = [r for r in schema.relations.values() if _valid_dirs_for_rel(r)]
    bidir_rels = [
        r for r in valid_rels
        if (r.has_csr or r.has_coo) and (r.has_csc or r.has_coo)
    ]
    endo_csr_rels = [r for r in valid_rels if r.src_dom == r.tgt_dom and r.has_csr]
    endo_csc_rels = [r for r in valid_rels if r.src_dom == r.tgt_dom and r.has_csc]
    csr_or_csc_rels = [r for r in valid_rels if r.has_csr or r.has_csc]
    all_doms = list(schema.dom_cards.keys())
    tier_boundaries = [0, 1, 63, 64, 65, 255, 256]

    def exec_law(label: str, impas: str, prof: str) -> dict[str, Any] | None:
        nonlocal checks, errors
        checks += 1
        fixture = assemble_fixture(asm, impas)
        results = run_on_engines(clients, fixture, prof)
        diff = compare_results(results)
        ref = next(iter(results.values()))
        if diff is not None:
            errors += 1
            print(f"[PART-B 3-WAY DIFF #{errors}] {label} prof={prof}: {diff}")
            print(impas)
            return None
        if ref.get("status") != "IMPULSE_VM_OK":
            errors += 1
            print(f"[PART-B LAW VIOLATION #{errors}] {label} prof={prof}: status={ref.get('status')}")
            print(impas)
            return None
        return ref

    for t in range(trials):
        if errors >= max_errors:
            break
        prof = profiles[t % len(profiles)]

        # Law 1: Reversibility — v in WALK.fwd(u, R) <=> u in WALK.rev(v, R)
        if bidir_rels and errors < max_errors:
            rel = rng.choice(bidir_rels)
            fwd_dirs = [d for d in ("fwd", "coo_fwd") if (d == "fwd" and rel.has_csr) or (d == "coo_fwd" and rel.has_coo)]
            rev_dirs = [d for d in ("rev", "coo_rev") if (d == "rev" and rel.has_csc) or (d == "coo_rev" and rel.has_coo)]
            fd = rng.choice(fwd_dirs)
            rd = rng.choice(rev_dirs)
            n_src = schema.dom_cards[rel.src_dom]
            n_tgt = schema.dom_cards[rel.tgt_dom]
            u = rng.randint(0, n_src - 1)
            # First get neighbors of u so half the trials test a real edge (True==True)
            # and half test a random node (often False==False).
            if t % 2 == 0:
                probe_impas = "\n".join([
                    ".data",
                    f"  .param 0 type=frontier dom={rel.src_dom} values=[{u}]",
                    ".code",
                    f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}",
                    f"  WALK.{fd} F1, F0, rel={rel.rel_id}",
                    "  COLLECT.frontier_ids F1, out=0",
                    "  HALT",
                ]) + "\n"
                p_res = next(iter(clients.values())).run_fixture(assemble_fixture(asm, probe_impas), prof, 500_000)
                nbrs = p_res.get("outputs", {}).get("0", [])
                v = rng.choice(nbrs) if (isinstance(nbrs, list) and nbrs) else rng.randint(0, n_tgt - 1)
            else:
                v = rng.randint(0, n_tgt - 1)

            impas = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={rel.src_dom} values=[{u}]",
                f"  .param 1 type=frontier dom={rel.tgt_dom} values=[{v}]",
                f"  .param 2 type=u64 value={u}",
                f"  .param 3 type=u64 value={v}",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}",
                f"  LOAD_PARAM.frontier F1, param=1, dom={rel.tgt_dom}",
                "  LOAD_PARAM.scalar R2, param=2, type=u64",
                "  LOAD_PARAM.scalar R3, param=3, type=u64",
                f"  WALK.{fd} F2, F0, rel={rel.rel_id}",
                f"  WALK.{rd} F3, F1, rel={rel.rel_id}",
                "  SET_CONTAINS.contains_1bit R0, F2, R3",
                "  SET_CONTAINS.contains_1bit R1, F3, R2",
                "  SCALAR_3VL.and R4, R0, R1",
                "  SCALAR_3VL.not R5, R0",
                "  SCALAR_3VL.not R6, R1",
                "  SCALAR_3VL.and R7, R5, R6",
                "  SCALAR_3VL.or R8, R4, R7",
                "  ASSERT.true_r R8, 101",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.scalar R1, out=1",
                "  HALT",
            ]) + "\n"
            exec_law(f"reversibility(t={t}, {rel.name}, {fd}<->{rd}, u={u}, v={v})", impas, prof)
            counts["reversibility"] += 1

        # Law 2: Distributivity — WALK(A U B, R) == WALK(A, R) U WALK(B, R)
        if valid_rels and errors < max_errors:
            rel = rng.choice(valid_rels)
            dsub = rng.choice(_valid_dirs_for_rel(rel))
            in_dom = rel.tgt_dom if dsub in ("rev", "coo_rev") else rel.src_dom
            n_in = schema.dom_cards[in_dom]
            a = _make_boundary_frontier(rng, n_in, rng.choice([0, 1, 5, 32, 65]))
            b = _make_boundary_frontier(rng, n_in, rng.choice([0, 1, 5, 32, 65]))
            impas = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in a)}]",
                f"  .param 1 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in b)}]",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
                f"  LOAD_PARAM.frontier F1, param=1, dom={in_dom}",
                "  SET_OP.union F2, F0, F1",
                f"  WALK.{dsub} F3, F2, rel={rel.rel_id}",
                f"  WALK.{dsub} F4, F0, rel={rel.rel_id}",
                f"  WALK.{dsub} F5, F1, rel={rel.rel_id}",
                "  SET_OP.union F6, F4, F5",
                "  STABLE_CHECK.frontier_eq F3, F6",
                "  ASSERT.st_set 201",
                "  SET_CARDINALITY R0, F3",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.frontier_ids F3, out=1",
                "  HALT",
            ]) + "\n"
            exec_law(f"distributivity(t={t}, {rel.name}.{dsub})", impas, prof)
            counts["distributivity"] += 1

        # Law 3: Kleene Identities — star(A) == A U plus(A) and star(star(A)) == star(A)
        if (endo_csr_rels or endo_csc_rels) and errors < max_errors:
            use_rev = bool(endo_csc_rels) and (not endo_csr_rels or (t % 2 == 1))
            rel = rng.choice(endo_csc_rels if use_rev else endo_csr_rels)
            prefix = "rev" if use_rev else "fwd"
            n_dom = schema.dom_cards[rel.src_dom]
            a = _make_boundary_frontier(rng, n_dom, rng.choice([0, 1, 3, 8]))
            impas = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={rel.src_dom} values=[{', '.join(str(x) for x in a)}]",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={rel.src_dom}",
                f"  WALK.{prefix}_fixpoint_plus F1, F0, rel={rel.rel_id}",
                f"  WALK.{prefix}_fixpoint_star F2, F0, rel={rel.rel_id}",
                "  SET_OP.union F3, F0, F1",
                "  STABLE_CHECK.frontier_eq F2, F3",
                "  ASSERT.st_set 301",
                f"  WALK.{prefix}_fixpoint_star F4, F2, rel={rel.rel_id}",
                "  STABLE_CHECK.frontier_eq F2, F4",
                "  ASSERT.st_set 302",
                "  SET_CARDINALITY R0, F2",
                "  COLLECT.scalar R0, out=0",
                "  HALT",
            ]) + "\n"
            exec_law(f"kleene_{prefix}(t={t}, {rel.name})", impas, prof)
            counts["kleene"] += 1

        # Law 4: De Morgan / Set Algebra across P0..P3 tier boundaries (0, 1, 63, 64, 65, 255, 256)
        if all_doms and errors < max_errors:
            dom_id = rng.choice(all_doms)
            n_dom = schema.dom_cards[dom_id]
            sa = tier_boundaries[t % len(tier_boundaries)]
            sb = rng.choice(tier_boundaries)
            sc = rng.choice(tier_boundaries)
            a = _make_boundary_frontier(rng, n_dom, sa)
            b = _make_boundary_frontier(rng, n_dom, sb)
            c = _make_boundary_frontier(rng, n_dom, sc)
            impas = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={dom_id} values=[{', '.join(str(x) for x in a)}]",
                f"  .param 1 type=frontier dom={dom_id} values=[{', '.join(str(x) for x in b)}]",
                f"  .param 2 type=frontier dom={dom_id} values=[{', '.join(str(x) for x in c)}]",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={dom_id}",
                f"  LOAD_PARAM.frontier F1, param=1, dom={dom_id}",
                f"  LOAD_PARAM.frontier F2, param=2, dom={dom_id}",
                # Law 4a: A \ (B U C) == (A \ B) & (A \ C)
                "  SET_OP.union F3, F1, F2",
                "  SET_OP.diff F4, F0, F3",
                "  SET_OP.diff F5, F0, F1",
                "  SET_OP.diff F6, F0, F2",
                "  SET_OP.intersect F7, F5, F6",
                "  STABLE_CHECK.frontier_eq F4, F7",
                "  ASSERT.st_set 401",
                # Law 4b: ~(A U B) == (~A) & (~B)
                "  SET_OP.union F8, F0, F1",
                "  SET_COMPLEMENT F9, F8",
                "  SET_COMPLEMENT F10, F0",
                "  SET_COMPLEMENT F11, F1",
                "  SET_OP.intersect F12, F10, F11",
                "  STABLE_CHECK.frontier_eq F9, F12",
                "  ASSERT.st_set 402",
                "  SET_CARDINALITY R0, F4",
                "  SET_CARDINALITY R1, F9",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.scalar R1, out=1",
                "  HALT",
            ]) + "\n"
            exec_law(f"demorgan(t={t}, dom={dom_id}, |A|={len(a)}, |B|={len(b)}, |C|={len(c)})", impas, prof)
            counts["demorgan"] += 1

        # Law 5: Degree-Walk Identity — sum(deg(u) for u in S) == WALK_MORSEL.count_add_r(S)
        if csr_or_csc_rels and errors < max_errors:
            rel = rng.choice(csr_or_csc_rels)
            cands: list[tuple[str, str, int]] = []
            if rel.has_csr:
                cands.append(("sum_fwd_r", "fwd", rel.src_dom))
                if rel.has_coo:
                    cands.append(("sum_fwd_r", "coo_fwd", rel.src_dom))
            if rel.has_csc:
                cands.append(("sum_rev_r", "rev", rel.tgt_dom))
                if rel.has_coo:
                    cands.append(("sum_rev_r", "coo_rev", rel.tgt_dom))
            deg_sub, morsel_dir, in_dom = rng.choice(cands)
            n_in = schema.dom_cards[in_dom]
            s = _make_boundary_frontier(rng, n_in, rng.choice(tier_boundaries))
            impas = "\n".join([
                ".data",
                f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in s)}]",
                ".code",
                f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
                f"  DEGREE.{deg_sub} R0, F0, rel={rel.rel_id}",
                f"  WALK_MORSEL.{morsel_dir} F0, rel={rel.rel_id}",
                "    MORSEL_COMMIT.count_add_r R1",
                "  MORSEL_END",
                "  SCALAR_3VL.cmp_eq_i64 R2, R0, R1",
                "  ASSERT.true_r R2, 501",
                "  COLLECT.scalar R0, out=0",
                "  COLLECT.scalar R1, out=1",
                "  HALT",
            ]) + "\n"
            exec_law(f"degree_walk(t={t}, {rel.name}, {deg_sub}=={morsel_dir}, |S|={len(s)})", impas, prof)
            counts["degree_walk"] += 1

    return checks, errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite #4: Graph-Algebraic & Semiring Law Property Tester + Exhaustive EQ_REL/EQ_DOM Matrix"
    )
    parser.add_argument("snapshot", type=Path, help="Path to v0.11 .imps snapshot")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--lib-path", type=Path, default=None)
    parser.add_argument(
        "--cpp-on-cmd",
        type=str,
        default="/Users/jesse/impulse/impulse-graph-core/impulse-cpp/build_on/vm_fixture_batch_runner",
    )
    parser.add_argument(
        "--cpp-off-cmd",
        type=str,
        default="/Users/jesse/impulse/impulse-graph-core/impulse-cpp/build_off/vm_fixture_batch_runner",
    )
    parser.add_argument("--engines", type=str, default="java,cpp_on,cpp_off")
    parser.add_argument("--profiles", type=str, default="P0,P1,P2,P3")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()

    snap_path = args.snapshot.resolve()
    lib_path = args.lib_path or _discover_default_lib()
    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_asm = _import_spec_modules(spec_dir)
    asm = impas_asm.ImpAsm(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    schema = load_snapshot_schema(snap_path, lib_path)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
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
        f"Suite #4 Graph-Algebraic Laws | snapshot={snap_path.name} "
        f"(doms={len(schema.dom_cards)}, rels={len(schema.relations)}, eq_rels={len(schema.eq_rels)}) | "
        f"engines={','.join(clients.keys())} | profiles={','.join(profiles)} | trials={args.trials} | seed={args.seed}"
    )

    t0 = time.perf_counter()
    try:
        a_checks, a_errors, a_counts = run_part_a(asm, clients, schema, profiles, args.max_errors)
        rem_errors = max(1, args.max_errors - a_errors) if a_errors < args.max_errors else 0
        if rem_errors > 0:
            b_checks, b_errors, b_counts = run_part_b(
                asm, clients, schema, profiles, args.trials, args.seed, rem_errors
            )
        else:
            b_checks, b_errors, b_counts = 0, 0, Counter()
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    total_checks = a_checks + b_checks
    total_errors = a_errors + b_errors
    print(
        f"\nCompleted {total_checks} checks in {dt:.2f}s ({total_checks / dt:.1f} checks/s) | "
        f"Part A={a_checks} (errs={a_errors}) | Part B={b_checks} (errs={b_errors})"
    )
    print(f"Part A breakdown: {dict(a_counts)}")
    print(f"Part B breakdown: {dict(b_counts)}")
    if total_errors > 0:
        return 1
    print("PASS: 0 parity, equivalence, or algebraic law violations!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
