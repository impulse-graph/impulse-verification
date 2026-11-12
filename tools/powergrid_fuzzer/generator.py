"""
Schema-Driven Deterministic Seeded Query Generator for Phase B PowerGrid Differential Fuzzing.

Generates queries across all 8 canonical families:
  - f1_cel_filters
  - f2_cross_domain_encodings
  - f3_kleisli_multihop_setops
  - f4_edge_predicates
  - f5_reachability_fixpoint
  - f6_scalar_aggregations
  - f7_multilane_quorum_cover
  - f8_vector_rerank_topk
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .schema import (
    ALL_FAMILIES,
    AttrMeta,
    DomainMeta,
    FixtureSchema,
    RelationMeta,
    quantize_f32,
)
from .sqlite_oracle import SQLiteOracle


def fmt_float(v: float) -> str:
    fv = float(v)
    if abs(fv) < 1e-12:
        return "0.0"
    s = f"{fv:.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def fmt_cel_str(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def fmt_scm_str(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def anchor_to_impscm(anchor: Dict[str, Any]) -> str:
    dom = fmt_scm_str(str(anchor["domain"]))
    mode = str(anchor["mode"])
    if mode == "all":
        return f"(domain {dom})"
    if mode == "seeds":
        seeds = " ".join(str(int(u)) for u in anchor.get("seeds", []))
        return f"(anchor {dom} [{seeds}])"
    if mode == "multilane_seeds":
        seeds = " ".join(str(int(u)) for u in anchor.get("seeds", []))
        return f"(from-seeds {dom} [{seeds}])"
    if mode == "multilane_groups":
        pairs = " ".join(f"({int(u)} {int(g)})" for u, g in anchor.get("groups", []))
        return f"(from-groups {dom} [{pairs}])"
    raise ValueError(f"Unsupported anchor mode '{mode}'")


def step_to_impscm(step: Dict[str, Any]) -> str:
    op = str(step["op"])
    if op == "filter":
        return f"(filter {step['sexpr']})"
    if op == "out":
        return f"(out {fmt_scm_str(str(step['relation']))})"
    if op == "in":
        return f"(in {fmt_scm_str(str(step['relation']))})"
    if op == "both":
        return f"(both {fmt_scm_str(str(step['relation']))})"
    if op == "out_union":
        rels = " ".join(fmt_scm_str(str(r)) for r in step["relations"])
        return f"(out-union [{rels}])"
    if op == "in_union":
        rels = " ".join(fmt_scm_str(str(r)) for r in step["relations"])
        return f"(in-union [{rels}])"
    if op == "out_where":
        return f"(out-where {fmt_scm_str(str(step['relation']))} {step['sexpr']})"
    if op == "in_where":
        return f"(in-where {fmt_scm_str(str(step['relation']))} {step['sexpr']})"
    if op == "repeat_out":
        return f"(repeat-hops (out {fmt_scm_str(str(step['relation']))}) {int(step['hops'])})"
    if op == "fixpoint_out":
        return f"(repeat (out {fmt_scm_str(str(step['relation']))}) :until-stable :max-hops {int(step['max_hops'])})"
    if op == "set_op":
        kind = str(step["set_op"])
        br = step["branch"]
        sub_parts = [anchor_to_impscm(br["anchor"])] + [step_to_impscm(s) for s in br.get("steps", [])]
        sub_scm = "(pipeline " + " ".join(sub_parts) + ")"
        return f"(set-{kind} {sub_scm})"
    if op == "where_quorum":
        return (
            f"(where-quorum {int(step['min_k'])} "
            f":req {int(step['req_mask'])} "
            f":forbid {int(step['forbid_mask'])})"
        )
    raise ValueError(f"Unsupported step op '{op}'")


def terminal_to_impscm(term: Dict[str, Any]) -> str:
    op = str(term["op"])
    if op == "nodes":
        return "(collect :nodes)"
    if op == "keys":
        return "(collect :keys)"
    if op == "count":
        return "(collect :count)"
    if op in ("sum", "min", "max"):
        return f"(reduce :{op} (prop {fmt_scm_str(str(term['attr']))}))"
    if op == "lane_masks":
        return "(collect :lanes)"
    if op == "set_cover":
        return f"(set-cover :target {int(term['target_mask'])})"
    if op == "top_k":
        vec_str = " ".join(fmt_float(float(x)) for x in term["query_vec"])
        return (
            f"(top-k {fmt_scm_str(str(term['attr']))} [{vec_str}] "
            f":metric {int(term['metric'])} :k {int(term['top_k'])})"
        )
    raise ValueError(f"Unsupported terminal op '{op}'")


def build_impscm(anchor: Dict[str, Any], steps: Sequence[Dict[str, Any]], terminal: Dict[str, Any]) -> str:
    parts = [anchor_to_impscm(anchor)]
    parts.extend(step_to_impscm(s) for s in steps)
    parts.append(terminal_to_impscm(terminal))
    return "(pipeline " + " ".join(parts) + ")"


class QueryGenerator:
    """Deterministic schema-aware generator across all 8 Phase B query families."""

    def __init__(
        self,
        schema: FixtureSchema,
        seed: int = 42,
        oracle: Optional[SQLiteOracle] = None,
    ) -> None:
        self.schema = schema
        self.rng = random.Random(seed)
        self.oracle = oracle
        self._domain_names = list(schema.domains.keys())
        self._relation_names = list(schema.relations.keys())
        self._edge_pred_relations = [
            r.name for r in schema.relations.values() if len(r.filterable_attrs) > 0
        ]
        self._homo_relations = [r.name for r in schema.homogeneous_relations("Bus")]
        self._vector_targets: List[Tuple[str, AttrMeta]] = [
            (d.name, va) for d in schema.domains.values() for va in d.vector_attrs
        ]
        self._arith_thresh_cache: Dict[Tuple[str, str], List[Union[int, float]]] = {}

    # -------------------------------------------------------------------------
    # Anchor & Predicate Primitives
    # -------------------------------------------------------------------------

    def _make_anchor(
        self,
        domain_name: str,
        *,
        allow_empty: bool = True,
        force_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        dom = self.schema.domains[domain_name]
        n = dom.node_count
        mode = force_mode or ("all" if self.rng.random() < 0.55 else "seeds")
        if mode == "all":
            return {"domain": domain_name, "mode": "all"}

        r = self.rng.random()
        if allow_empty and r < 0.10:
            seeds: List[int] = []
        elif r < 0.30:
            seeds = [self.rng.randrange(n)]
        else:
            k = self.rng.randint(2, min(25, max(2, n)))
            seeds = sorted(self.rng.sample(range(n), k))
        return {"domain": domain_name, "mode": "seeds", "seeds": seeds}

    def _atomic_predicate(
        self,
        attrs: Sequence[AttrMeta],
        table_name: str,
        *,
        prefer_nullable: bool = False,
        allow_arith: bool = True,
    ) -> Tuple[str, str]:
        """Returns (cel_expr, sexpr) for a single atomic predicate."""
        nullable_pool = [a for a in attrs if a.nullable]
        if prefer_nullable and nullable_pool and self.rng.random() < 0.65:
            attr = self.rng.choice(nullable_pool)
        else:
            attr = self.rng.choice(list(attrs))

        # 1. Explicit NULL / NOT NULL checks on nullable columns
        if attr.nullable and self.rng.random() < 0.32:
            if self.rng.random() < 0.5:
                return f"is_null({attr.name})", f"(null? {attr.name})"
            return f"is_not_null({attr.name})", f"(not-null? {attr.name})"

        # 2. String predicates (==, !=, starts_with)
        if attr.is_string:
            roll = self.rng.random()
            if roll < 0.45 and attr.stats.string_prefixes:
                pref = self.rng.choice(attr.stats.string_prefixes)
                return (
                    f"starts_with({attr.name}, {fmt_cel_str(pref)})",
                    f"(starts-with {attr.name} {fmt_scm_str(pref)})",
                )
            op = self.rng.choice(["==", "!="])
            if attr.stats.distinct_strings and self.rng.random() < 0.85:
                lit = self.rng.choice(attr.stats.distinct_strings)
            else:
                lit = "NO_SUCH_GRID_ENTITY"
            return f"{attr.name} {op} {fmt_cel_str(lit)}", f"({op} {attr.name} {fmt_scm_str(lit)})"

        # 3. Arithmetic predicate (+, -, *) with verified safe non-boundary threshold
        if allow_arith and attr.is_numeric and not attr.is_high_ts_ns and self.rng.random() < 0.25:
            return self._arithmetic_predicate(attr, attrs, table_name)

        # 4. Integer relational predicate
        if attr.is_integer:
            if attr.is_high_ts_ns:
                op = self.rng.choice(["<", "<=", ">", ">="])
                thresh = int(self.rng.choice(attr.stats.safe_thresholds))
            else:
                op = self.rng.choice(["==", "!=", "<", "<=", ">", ">="])
                if op in ("==", "!=") and attr.stats.distinct_ints and self.rng.random() < 0.85:
                    thresh = int(self.rng.choice(attr.stats.distinct_ints))
                else:
                    thresh = int(self.rng.choice(attr.stats.safe_thresholds))
            cel_rhs = f"({thresh})" if thresh < 0 else str(thresh)
            return f"{attr.name} {op} {cel_rhs}", f"({op} {attr.name} {thresh})"

        # 5. Float relational predicate (strictly <, <=, >, >= with safe epsilon margin;
        #    also != against safe non-matching literal on nullable floats to test Kleene NULL!)
        op = self.rng.choice(["<", "<=", ">", ">=", "!="] if attr.nullable else ["<", "<=", ">", ">="])
        f_thresh = float(self.rng.choice(attr.stats.safe_thresholds))
        f_lit = fmt_float(f_thresh)
        cel_rhs = f"({f_lit})" if f_thresh < 0 else f_lit
        return f"{attr.name} {op} {cel_rhs}", f"({op} {attr.name} {f_lit})"

    def _arithmetic_predicate(
        self,
        attr: AttrMeta,
        all_attrs: Sequence[AttrMeta],
        table_name: str,
    ) -> Tuple[str, str]:
        arith_op = self.rng.choice(["+", "-", "*"])
        cmp_op = self.rng.choice(["<", "<=", ">", ">="])

        if attr.is_integer:
            # Small integer arithmetic that never overflows int64 or float64
            peer_ints = [
                a for a in all_attrs
                if a.is_exact_eq_int and a.name != attr.name and (not a.stats.distinct_ints or max(abs(x) for x in a.stats.distinct_ints) < 100_000)
            ]
            max_abs = max((abs(x) for x in attr.stats.distinct_ints), default=10)
            if max_abs < 100_000 and peer_ints and arith_op in ("+", "-") and self.rng.random() < 0.45:
                peer = self.rng.choice(peer_ints)
                cel_lhs = f"({attr.name} {arith_op} {peer.name})"
                scm_lhs = f"({arith_op} {attr.name} {peer.name})"
                sql_expr = f"{attr.name} {arith_op} {peer.name}"
            else:
                const_i = self.rng.choice([1, 2, 3, 5, 10]) if arith_op != "*" or max_abs < 1_000_000 else 1
                if arith_op == "*" and max_abs >= 1_000_000:
                    arith_op = "+"
                cel_lhs = f"({attr.name} {arith_op} {const_i})"
                scm_lhs = f"({arith_op} {attr.name} {const_i})"
                sql_expr = f"{attr.name} {arith_op} {const_i}"

            thresh_i = int(self.schema.evaluate_safe_arith_threshold(self.rng, table_name, sql_expr, True))
            cel_rhs = f"({thresh_i})" if thresh_i < 0 else str(thresh_i)
            return f"{cel_lhs} {cmp_op} {cel_rhs}", f"({cmp_op} {scm_lhs} {thresh_i})"

        # Float arithmetic
        peer_floats = [
            a for a in all_attrs
            if a.is_float and a.base_type != "float16" and a.name != attr.name
        ]
        if attr.base_type != "float16" and peer_floats and arith_op in ("+", "-") and self.rng.random() < 0.45:
            peer_f = self.rng.choice(peer_floats)
            cel_lhs = f"({attr.name} {arith_op} {peer_f.name})"
            scm_lhs = f"({arith_op} {attr.name} {peer_f.name})"
            sql_expr = f"{attr.name} {arith_op} {peer_f.name}"
        else:
            const_f = self.rng.choice([0.5, 1.5, 2.0, 10.0])
            cf_str = fmt_float(const_f)
            cel_lhs = f"({attr.name} {arith_op} {cf_str})"
            scm_lhs = f"({arith_op} {attr.name} {cf_str})"
            sql_expr = f"{attr.name} {arith_op} {cf_str}"

        thresh_f = float(self.schema.evaluate_safe_arith_threshold(self.rng, table_name, sql_expr, False))
        tf_str = fmt_float(thresh_f)
        cel_rhs = f"({tf_str})" if thresh_f < 0 else tf_str
        return f"{cel_lhs} {cmp_op} {cel_rhs}", f"({cmp_op} {scm_lhs} {tf_str})"

    def _compound_predicate(
        self,
        attrs: Sequence[AttrMeta],
        table_name: str,
        *,
        max_clauses: int = 3,
        prefer_nullable: bool = False,
    ) -> Tuple[str, str]:
        num_clauses = self.rng.randint(1, max_clauses)
        clauses: List[Tuple[str, str]] = []
        for _ in range(num_clauses):
            c_cel, c_scm = self._atomic_predicate(
                attrs, table_name, prefer_nullable=prefer_nullable
            )
            if self.rng.random() < 0.22:
                c_cel = f"!({c_cel})"
                c_scm = f"(not {c_scm})"
            clauses.append((c_cel, c_scm))

        while len(clauses) > 1:
            c1, s1 = clauses.pop(0)
            c2, s2 = clauses.pop(0)
            if self.rng.random() < 0.55:
                comb = (f"({c1}) && ({c2})", f"(and {s1} {s2})")
            else:
                comb = (f"({c1}) || ({c2})", f"(or {s1} {s2})")
            if self.rng.random() < 0.12:
                comb = (f"!({comb[0]})", f"(not {comb[1]})")
            clauses.insert(0, comb)

        return clauses[0]

    def _basic_terminal(self) -> Dict[str, Any]:
        return {"op": self.rng.choice(["nodes", "keys", "count"])}

    def _random_hop_from(self, curr_dom: str) -> Tuple[Dict[str, Any], str]:
        """Picks a valid forward or reverse traversal step from `curr_dom`."""
        outs = self.schema.relations_from(curr_dom)
        ins = self.schema.relations_into(curr_dom)
        choices: List[Tuple[str, RelationMeta]] = [("out", r) for r in outs] + [("in", r) for r in ins]
        direction, rel = self.rng.choice(choices)
        if direction == "out":
            return {"op": "out", "relation": rel.name}, rel.tgt_domain
        return {"op": "in", "relation": rel.name}, rel.src_domain

    # -------------------------------------------------------------------------
    # Family 1: f1_cel_filters
    # -------------------------------------------------------------------------

    def _gen_f1_cel_filters(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        dom_name = self._domain_names[seq_idx % len(self._domain_names)] if seq_idx < 18 else self.rng.choice(self._domain_names)
        dom = self.schema.domains[dom_name]
        anchor = self._make_anchor(dom_name, force_mode="all" if seq_idx < 12 else None)

        num_filters = 1 if self.rng.random() < 0.75 else 2
        steps: List[Dict[str, Any]] = []
        for _ in range(num_filters):
            cel_s, scm_s = self._compound_predicate(
                dom.filterable_attrs,
                f"dom_{dom_name}",
                max_clauses=3,
                prefer_nullable=(len(dom.nullable_attrs) > 0 and self.rng.random() < 0.5),
            )
            steps.append({"op": "filter", "cel": cel_s, "sexpr": scm_s})

        term = {"op": ["nodes", "keys", "count"][seq_idx % 3]} if seq_idx < 18 else self._basic_terminal()
        return anchor, steps, term

    # -------------------------------------------------------------------------
    # Family 2: f2_cross_domain_encodings
    # -------------------------------------------------------------------------

    def _gen_f2_cross_domain_encodings(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        # Deterministically cycle through all 10 relations + out/in/both/out_union/in_union + sentinel boundaries
        rel_name = self._relation_names[seq_idx % len(self._relation_names)]
        rel = self.schema.relations[rel_name]

        roll = self.rng.random()
        steps: List[Dict[str, Any]] = []

        if rel.is_homogeneous and roll < 0.22:
            # Test `both`, `out_union`, or `in_union` on Bus -> Bus
            sub_choice = self.rng.choice(["both", "out_union", "in_union"])
            anchor = self._make_anchor(rel.src_domain)
            if sub_choice == "both":
                steps.append({"op": "both", "relation": rel.name})
            else:
                k_rels = self.rng.randint(2, 3)
                picked_rels = sorted(self.rng.sample(self._homo_relations, k_rels))
                steps.append({"op": sub_choice, "relations": picked_rels})
            curr_dom = rel.tgt_domain
        elif seq_idx % 2 == 0 or (roll < 0.52):
            # Forward `out` — explicitly target sentinel / parallel-edge / empty cases too!
            if rel.unlinked_src_nodes and self.rng.random() < 0.35:
                # Sample mix of unlinked sentinel nodes and linked nodes (or purely unlinked!)
                if self.rng.random() < 0.45:
                    seeds = self.rng.sample(rel.unlinked_src_nodes, min(4, len(rel.unlinked_src_nodes)))
                else:
                    u_unl = self.rng.sample(rel.unlinked_src_nodes, min(3, len(rel.unlinked_src_nodes)))
                    u_lnk = self.rng.sample(rel.linked_src_nodes, min(3, len(rel.linked_src_nodes))) if rel.linked_src_nodes else []
                    seeds = sorted(set(u_unl + u_lnk))
                anchor = {"domain": rel.src_domain, "mode": "seeds", "seeds": sorted(seeds)}
            elif rel.multi_edge_src_nodes and self.rng.random() < 0.30:
                seeds = sorted(set(self.rng.sample(rel.multi_edge_src_nodes, min(3, len(rel.multi_edge_src_nodes)))))
                anchor = {"domain": rel.src_domain, "mode": "seeds", "seeds": seeds}
            else:
                anchor = self._make_anchor(rel.src_domain)
            steps.append({"op": "out", "relation": rel.name})
            curr_dom = rel.tgt_domain
        else:
            # Reverse `in` — explicitly target reverse 1:1 / 1:M sentinels!
            if rel.unlinked_tgt_nodes and self.rng.random() < 0.35:
                if self.rng.random() < 0.45:
                    seeds = self.rng.sample(rel.unlinked_tgt_nodes, min(4, len(rel.unlinked_tgt_nodes)))
                else:
                    v_unl = self.rng.sample(rel.unlinked_tgt_nodes, min(3, len(rel.unlinked_tgt_nodes)))
                    v_lnk = self.rng.sample(rel.linked_tgt_nodes, min(3, len(rel.linked_tgt_nodes))) if rel.linked_tgt_nodes else []
                    seeds = sorted(set(v_unl + v_lnk))
                anchor = {"domain": rel.tgt_domain, "mode": "seeds", "seeds": sorted(seeds)}
            else:
                anchor = self._make_anchor(rel.tgt_domain)
            steps.append({"op": "in", "relation": rel.name})
            curr_dom = rel.src_domain

        # Optionally add a 2nd hop (50% of the time)
        if self.rng.random() < 0.50:
            hop2, curr_dom = self._random_hop_from(curr_dom)
            steps.append(hop2)

        return anchor, steps, self._basic_terminal()

    # -------------------------------------------------------------------------
    # Family 3: f3_kleisli_multihop_setops
    # -------------------------------------------------------------------------

    def _build_branch_ending_at(self, target_dom: str) -> Dict[str, Any]:
        """Builds a compatible sub-pipeline whose final domain equals `target_dom`."""
        in_rels = self.schema.relations_into(target_dom)
        out_rels = self.schema.relations_from(target_dom)

        r = self.rng.random()
        if in_rels and r < 0.45:
            # 1-hop forward into target_dom: SrcDom -> out(rel) -> target_dom
            rel = self.rng.choice(in_rels)
            br_anchor = self._make_anchor(rel.src_domain, allow_empty=False)
            br_steps: List[Dict[str, Any]] = []
            if self.rng.random() < 0.5:
                s_dom = self.schema.domains[rel.src_domain]
                c_cel, c_scm = self._compound_predicate(s_dom.filterable_attrs, f"dom_{s_dom.name}", max_clauses=2)
                br_steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})
            br_steps.append({"op": "out", "relation": rel.name})
            return {"anchor": br_anchor, "steps": br_steps}

        if out_rels and r < 0.85:
            # 1-hop reverse into target_dom: TgtDom -> in(rel) -> target_dom
            rel = self.rng.choice(out_rels)
            br_anchor = self._make_anchor(rel.tgt_domain, allow_empty=False)
            br_steps = []
            if self.rng.random() < 0.5:
                t_dom = self.schema.domains[rel.tgt_domain]
                c_cel, c_scm = self._compound_predicate(t_dom.filterable_attrs, f"dom_{t_dom.name}", max_clauses=2)
                br_steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})
            br_steps.append({"op": "in", "relation": rel.name})
            return {"anchor": br_anchor, "steps": br_steps}

        # Direct filtered anchor in target_dom
        br_anchor = self._make_anchor(target_dom, allow_empty=False)
        dom = self.schema.domains[target_dom]
        c_cel, c_scm = self._compound_predicate(dom.filterable_attrs, f"dom_{dom.name}", max_clauses=2)
        return {"anchor": br_anchor, "steps": [{"op": "filter", "cel": c_cel, "sexpr": c_scm}]}

    def _gen_f3_kleisli_multihop_setops(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        curr_dom = self.rng.choice(self._domain_names)
        anchor = self._make_anchor(curr_dom, allow_empty=False)
        steps: List[Dict[str, Any]] = []

        # Initial filter
        dom0 = self.schema.domains[curr_dom]
        c_cel, c_scm = self._compound_predicate(dom0.filterable_attrs, f"dom_{curr_dom}", max_clauses=2)
        steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})

        # 1..2 traversal hops
        n_pre_hops = self.rng.randint(1, 2)
        for _ in range(n_pre_hops):
            hop, curr_dom = self._random_hop_from(curr_dom)
            steps.append(hop)

        # Mid-pipeline filter
        dom_mid = self.schema.domains[curr_dom]
        c_cel2, c_scm2 = self._compound_predicate(dom_mid.filterable_attrs, f"dom_{curr_dom}", max_clauses=2)
        steps.append({"op": "filter", "cel": c_cel2, "sexpr": c_scm2})

        # Set operation (union, intersect, difference)
        set_kind = ["union", "intersect", "difference"][seq_idx % 3]
        branch = self._build_branch_ending_at(curr_dom)
        steps.append({"op": "set_op", "set_op": set_kind, "branch": branch})

        # Optional 3rd/4th hop after set_op
        if self.rng.random() < 0.55:
            hop_post, curr_dom = self._random_hop_from(curr_dom)
            steps.append(hop_post)

        return anchor, steps, self._basic_terminal()

    # -------------------------------------------------------------------------
    # Family 4: f4_edge_predicates
    # -------------------------------------------------------------------------

    def _gen_f4_edge_predicates(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        # Pick primary edge-attributed relation (bias 90% to non-empty AC_LINE, TRANSFORMER, SHUNT_SELF_LOOP; 10% HVDC_TIE_LINE)
        non_empty_ep = [rn for rn in self._edge_pred_relations if self.schema.relations[rn].edge_count > 0]
        if seq_idx % 9 == 8:
            rel_name = "HVDC_TIE_LINE"
        else:
            rel_name = non_empty_ep[seq_idx % len(non_empty_ep)]
        rel = self.schema.relations[rel_name]

        steps: List[Dict[str, Any]] = []
        # Optionally start from Generator / PMU / Substation -> Bus before edge-filtered hop!
        if self.rng.random() < 0.40:
            entry_rel = self.rng.choice(["GEN_AT_BUS", "PMU_MONITORS_BUS", "SUBSTATION_HAS_BUS"])
            er = self.schema.relations[entry_rel]
            anchor = self._make_anchor(er.src_domain, allow_empty=False)
            steps.append({"op": "out", "relation": er.name})
        else:
            anchor = self._make_anchor("Bus")

        ep_op = "out_where" if (seq_idx % 2 == 0) else "in_where"
        e_cel, e_scm = self._compound_predicate(
            rel.filterable_attrs,
            f"rel_{rel.name}",
            max_clauses=2,
            prefer_nullable=(len(rel.nullable_attrs) > 0 and self.rng.random() < 0.55),
        )
        steps.append({"op": ep_op, "relation": rel.name, "cel": e_cel, "sexpr": e_scm})
        curr_dom = rel.tgt_domain if ep_op == "out_where" else rel.src_domain

        # Optionally chain into BUS_IN_ZONE, BUS_RISK_TIER, or reverse GEN_AT_BUS
        if self.rng.random() < 0.40:
            exit_choice = self.rng.choice([("out", "BUS_IN_ZONE"), ("out", "BUS_RISK_TIER"), ("in", "GEN_AT_BUS")])
            steps.append({"op": exit_choice[0], "relation": exit_choice[1]})

        return anchor, steps, self._basic_terminal()

    # -------------------------------------------------------------------------
    # Family 5: f5_reachability_fixpoint
    # -------------------------------------------------------------------------

    def _gen_f5_reachability_fixpoint(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        rel_name = self._homo_relations[seq_idx % len(self._homo_relations)]
        steps: List[Dict[str, Any]] = []

        if self.rng.random() < 0.25:
            # Cross-domain seed into Bus first (e.g. Substation -> Bus or Generator -> Bus)
            er = self.schema.relations[self.rng.choice(["GEN_AT_BUS", "PMU_MONITORS_BUS", "SUBSTATION_HAS_BUS"])]
            anchor = self._make_anchor(er.src_domain, force_mode="seeds")
            steps.append({"op": "out", "relation": er.name})
        else:
            anchor = self._make_anchor("Bus")

        if seq_idx % 2 == 0:
            hops = self.rng.randint(1, 4)
            steps.append({"op": "repeat_out", "relation": rel_name, "hops": hops})
        else:
            max_hops = self.rng.randint(5, 50)
            steps.append({"op": "fixpoint_out", "relation": rel_name, "max_hops": max_hops})

        if self.rng.random() < 0.35:
            post = self.rng.choice([
                {"op": "out", "relation": "BUS_IN_ZONE"},
                {"op": "out", "relation": "BUS_RISK_TIER"},
                {"op": "in", "relation": "PMU_MONITORS_BUS"},
            ])
            steps.append(post)

        return anchor, steps, self._basic_terminal()

    # -------------------------------------------------------------------------
    # Family 6: f6_scalar_aggregations
    # -------------------------------------------------------------------------

    def _gen_f6_scalar_aggregations(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        agg_op = ["sum", "min", "max"][seq_idx % 3]
        dom_name = self._domain_names[(seq_idx // 3) % len(self._domain_names)] if seq_idx < 18 else self.rng.choice(self._domain_names)
        dom = self.schema.domains[dom_name]
        agg_attrs = dom.aggregatable_attrs
        nullable_agg_attrs = [a for a in agg_attrs if a.nullable and a.stats.null_count > 0]

        steps: List[Dict[str, Any]] = []
        mode_roll = seq_idx % 7

        if mode_roll == 0:
            # Explicit empty frontier boundary test
            anchor: Dict[str, Any] = {"domain": dom_name, "mode": "seeds", "seeds": []}
            target_attr = self.rng.choice(agg_attrs)
        elif mode_roll == 1 and nullable_agg_attrs:
            # Explicit all-NULL frontier boundary test!
            target_attr = self.rng.choice(nullable_agg_attrs)
            anchor = {"domain": dom_name, "mode": "all"}
            steps.append({
                "op": "filter",
                "cel": f"is_null({target_attr.name})",
                "sexpr": f"(null? {target_attr.name})",
            })
        elif mode_roll in (2, 3) and nullable_agg_attrs:
            # Mixed NULL + non-NULL frontier
            target_attr = self.rng.choice(nullable_agg_attrs)
            anchor = self._make_anchor(dom_name, allow_empty=False)
            if self.rng.random() < 0.6:
                c_cel, c_scm = self._atomic_predicate(dom.filterable_attrs, f"dom_{dom_name}")
                steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})
        else:
            # Multi-hop Kleisli pipeline ending in an aggregatable domain
            anchor = self._make_anchor(dom_name, allow_empty=False)
            curr_dom = dom_name
            for _ in range(self.rng.randint(0, 2)):
                if self.rng.random() < 0.5:
                    d_cur = self.schema.domains[curr_dom]
                    c_cel, c_scm = self._compound_predicate(d_cur.filterable_attrs, f"dom_{curr_dom}", max_clauses=2)
                    steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})
                else:
                    hop, curr_dom = self._random_hop_from(curr_dom)
                    steps.append(hop)
            target_attr = self.rng.choice(self.schema.domains[curr_dom].aggregatable_attrs)

        term = {"op": agg_op, "attr": target_attr.name}
        return anchor, steps, term

    # -------------------------------------------------------------------------
    # Family 7: f7_multilane_quorum_cover
    # -------------------------------------------------------------------------

    def _gen_f7_multilane_quorum_cover(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        # Start predominantly from Bus, Substation, Generator, or PMU for rich multi-lane fanout
        start_pool = ["Bus", "Bus", "Substation", "Generator", "PMU", "LoadZone"]
        curr_dom = start_pool[seq_idx % len(start_pool)]
        n = self.schema.domains[curr_dom].node_count

        if seq_idx % 2 == 0:
            num_lanes = self.rng.randint(1, min(16, n))
            seeds = sorted(self.rng.sample(range(n), num_lanes))
            anchor: Dict[str, Any] = {
                "domain": curr_dom,
                "mode": "multilane_seeds",
                "seeds": seeds,
            }
        else:
            num_lanes = self.rng.randint(2, min(8, max(2, n)))
            groups: List[List[int]] = []
            # Ensure every group 0 .. num_lanes-1 has 1..4 assigned seed nodes
            for g in range(num_lanes):
                cnt = self.rng.randint(1, min(4, n))
                for u in sorted(self.rng.sample(range(n), cnt)):
                    groups.append([u, g])
            anchor = {
                "domain": curr_dom,
                "mode": "multilane_groups",
                "groups": groups,
            }

        active_mask = (1 << num_lanes) - 1
        steps: List[Dict[str, Any]] = []
        n_steps = 0 if (seq_idx % 8 == 0) else self.rng.randint(1, 3)
        used_quorum = False
        has_hop = False

        for step_i in range(n_steps):
            roll = self.rng.random()
            if (not used_quorum) and (step_i == n_steps - 1 or roll < 0.30) and num_lanes >= 2:
                # Pick non-zero req_mask and disjoint small forbid_mask
                req_bits = self.rng.sample(range(num_lanes), self.rng.randint(1, num_lanes))
                req_mask = sum(1 << b for b in req_bits)
                rem_bits = [b for b in range(num_lanes) if b not in req_bits]
                if rem_bits and self.rng.random() < 0.45:
                    f_bits = self.rng.sample(rem_bits, self.rng.randint(1, min(2, len(rem_bits))))
                    forbid_mask = sum(1 << b for b in f_bits)
                else:
                    forbid_mask = 0
                max_useful_k = min(3, len(req_bits)) if (has_hop or anchor["mode"] == "multilane_groups") else 1
                min_k = self.rng.randint(1, max(1, max_useful_k))
                steps.append({
                    "op": "where_quorum",
                    "min_k": min_k,
                    "req_mask": req_mask,
                    "forbid_mask": forbid_mask,
                })
                used_quorum = True
            elif roll < 0.72:
                hop, curr_dom = self._random_hop_from(curr_dom)
                steps.append(hop)
                has_hop = True
            else:
                d_meta = self.schema.domains[curr_dom]
                c_cel, c_scm = self._atomic_predicate(d_meta.filterable_attrs, f"dom_{curr_dom}")
                steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})

        term_choice = ["lane_masks", "set_cover", "nodes", "count"][seq_idx % 4]
        if term_choice == "set_cover":
            term: Dict[str, Any] = {"op": "set_cover", "target_mask": active_mask}
        else:
            term = {"op": term_choice}
        return anchor, steps, term

    # -------------------------------------------------------------------------
    # Family 8: f8_vector_rerank_topk
    # -------------------------------------------------------------------------

    def _pick_well_separated_query_vec(
        self,
        dom_name: str,
        attr_name: str,
        dim: int,
        metric: int,
        top_k: int,
        cand_nodes: Optional[Sequence[int]],
    ) -> List[float]:
        vecs = self.schema.domains[dom_name].vectors[attr_name]
        nodes = list(cand_nodes) if cand_nodes is not None else list(range(len(vecs)))

        for _ in range(25):
            raw_q = [round(self.rng.uniform(-1.0, 1.0), 3) for _ in range(dim)]
            # Ensure non-trivial vector norm
            if sum(x * x for x in raw_q) < 0.20:
                raw_q[0] = 0.75
            if not nodes:
                return raw_q

            qf = [quantize_f32(x) for x in raw_q]
            nq_f = sum(x * x for x in qf)
            nq_d = sum(float(x) * float(x) for x in raw_q)

            scored_f32: List[Tuple[float, int]] = []
            scored_f64: List[Tuple[float, int]] = []
            for u in nodes:
                v = vecs[u]
                if metric == 0:
                    sf = quantize_f32(sum(quantize_f32(qf[i] * v[i]) for i in range(dim)))
                    sd = sum(float(raw_q[i]) * float(v[i]) for i in range(dim))
                elif metric == 1:
                    sf = -quantize_f32(sum(quantize_f32((qf[i] - v[i]) ** 2) for i in range(dim)))
                    sd = -sum((float(raw_q[i]) - float(v[i])) ** 2 for i in range(dim))
                else:
                    dot_f = sum(qf[i] * v[i] for i in range(dim))
                    nv_f = sum(v[i] * v[i] for i in range(dim))
                    den_f = math.sqrt(nq_f) * math.sqrt(nv_f)
                    sf = 0.0 if den_f <= 1e-12 else (dot_f / den_f)

                    dot_d = sum(float(raw_q[i]) * float(v[i]) for i in range(dim))
                    nv_d = sum(float(v[i]) * float(v[i]) for i in range(dim))
                    den_d = math.sqrt(nq_d) * math.sqrt(nv_d)
                    sd = 0.0 if den_d <= 1e-12 else (dot_d / den_d)

                scored_f32.append((sf, u))
                scored_f64.append((sd, u))

            scored_f32.sort(key=lambda it: (-it[0], it[1]))
            scored_f64.sort(key=lambda it: (-it[0], it[1]))

            top_f32 = [u for _, u in scored_f32[:top_k]]
            top_f64 = [u for _, u in scored_f64[:top_k]]
            if top_f32 != top_f64:
                continue

            # Ensure non-identical adjacent vectors near top_k boundary have safe score separation
            check_lim = min(len(scored_f64), top_k + 3)
            safe = True
            for i in range(check_lim - 1):
                s_a, u_a = scored_f64[i]
                s_b, u_b = scored_f64[i + 1]
                if vecs[u_a] != vecs[u_b] and abs(s_a - s_b) < 5e-4:
                    safe = False
                    break
            if safe:
                return raw_q

        return [round(0.25 * (i + 1) * (-1 if i % 2 else 1), 3) for i in range(dim)]

    def _gen_f8_vector_rerank_topk(self, seq_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        target_dom, vec_attr = self._vector_targets[seq_idx % len(self._vector_targets)]
        metric = seq_idx % 3
        top_k = self.rng.randint(1, 15)

        steps: List[Dict[str, Any]] = []
        r = self.rng.random()
        if seq_idx == 7:
            # Explicit empty candidate frontier boundary test
            anchor: Dict[str, Any] = {"domain": target_dom, "mode": "seeds", "seeds": []}
        elif r < 0.45:
            # Hop into target_dom from a neighbor domain
            in_rels = [rel for rel in self.schema.relations_into(target_dom) if rel.edge_count > 0]
            out_rels = [rel for rel in self.schema.relations_from(target_dom) if rel.edge_count > 0]
            if in_rels and self.rng.random() < 0.5:
                rel = self.rng.choice(in_rels)
                anchor = self._make_anchor(rel.src_domain, allow_empty=False)
                steps.append({"op": "out", "relation": rel.name})
            elif out_rels:
                rel = self.rng.choice(out_rels)
                anchor = self._make_anchor(rel.tgt_domain, allow_empty=False)
                steps.append({"op": "in", "relation": rel.name})
            else:
                anchor = self._make_anchor(target_dom, allow_empty=False)
        else:
            anchor = self._make_anchor(target_dom, allow_empty=False)

        if self.rng.random() < 0.55 and not (anchor.get("mode") == "seeds" and anchor.get("seeds") == []):
            dom_meta = self.schema.domains[target_dom]
            c_cel, c_scm = self._compound_predicate(dom_meta.filterable_attrs, f"dom_{target_dom}", max_clauses=2)
            steps.append({"op": "filter", "cel": c_cel, "sexpr": c_scm})

        cand_nodes: Optional[List[int]] = None
        if self.oracle is not None:
            probe_q = {"id": "_probe", "anchor": anchor, "steps": steps, "terminal": {"op": "nodes"}}
            cand_nodes = self.oracle.evaluate(probe_q)["nodes"]

        q_vec = self._pick_well_separated_query_vec(
            target_dom, vec_attr.name, vec_attr.dimension, metric, top_k, cand_nodes
        )
        term = {
            "op": "top_k",
            "attr": vec_attr.name,
            "query_vec": q_vec,
            "metric": metric,
            "top_k": top_k,
        }
        return anchor, steps, term

    # -------------------------------------------------------------------------
    # Public Batch Generation Entrypoint
    # -------------------------------------------------------------------------

    def generate_one(self, family: str, query_id: str, family_seq_idx: int) -> Dict[str, Any]:
        dispatch = {
            "f1_cel_filters": self._gen_f1_cel_filters,
            "f2_cross_domain_encodings": self._gen_f2_cross_domain_encodings,
            "f3_kleisli_multihop_setops": self._gen_f3_kleisli_multihop_setops,
            "f4_edge_predicates": self._gen_f4_edge_predicates,
            "f5_reachability_fixpoint": self._gen_f5_reachability_fixpoint,
            "f6_scalar_aggregations": self._gen_f6_scalar_aggregations,
            "f7_multilane_quorum_cover": self._gen_f7_multilane_quorum_cover,
            "f8_vector_rerank_topk": self._gen_f8_vector_rerank_topk,
        }
        if family not in dispatch:
            raise ValueError(f"Unknown query family '{family}'. Valid families: {ALL_FAMILIES}")

        anchor, steps, terminal = dispatch[family](family_seq_idx)
        impscm = build_impscm(anchor, steps, terminal)
        return {
            "id": query_id,
            "family": family,
            "impscm": impscm,
            "anchor": anchor,
            "steps": steps,
            "terminal": terminal,
        }

    def generate_batch(
        self,
        count: int,
        families: Optional[Sequence[str]] = None,
        id_prefix: str = "q",
    ) -> List[Dict[str, Any]]:
        fam_list = list(families) if families else list(ALL_FAMILIES)
        for f in fam_list:
            if f not in ALL_FAMILIES:
                raise ValueError(f"Unknown family '{f}'. Supported: {ALL_FAMILIES}")

        per_fam_counter: Dict[str, int] = {f: 0 for f in fam_list}
        queries: List[Dict[str, Any]] = []
        for idx in range(count):
            fam = fam_list[idx % len(fam_list)]
            seq_i = per_fam_counter[fam]
            per_fam_counter[fam] += 1
            qid = f"{id_prefix}_{idx + 1:04d}"
            queries.append(self.generate_one(fam, qid, seq_i))
        return queries
