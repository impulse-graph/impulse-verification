"""
Exact SQLite Differential Oracle for Phase B PowerGrid Queries.

Translates every Kleisli, CEL/ImpScheme, Multi-Lane, Reachability/Fixpoint, Aggregation,
and Vector Reranking query into SQLite CTE SQL against `powergrid_ground_truth.sqlite`
(in-memory read-only clone with CSR/CSC helper indexes for sub-millisecond evaluation).
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .schema import FixtureSchema, quantize_f16, quantize_f32


# -----------------------------------------------------------------------------
# Minimal Deterministic S-Expression Parser (for canonical `sexpr` -> SQLite SQL)
# -----------------------------------------------------------------------------

SExpr = Union[int, float, str, Tuple[str, str], List["SExpr"]]


def _tokenize_sexpr(src: str) -> List[Tuple[str, Any]]:
    tokens: List[Tuple[str, Any]] = []
    n = len(src)
    i = 0
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif c in ("(", "["):
            tokens.append(("LP", c))
            i += 1
        elif c in (")", "]"):
            tokens.append(("RP", c))
            i += 1
        elif c == '"':
            i += 1
            buf: List[str] = []
            while i < n and src[i] != '"':
                if src[i] == "\\" and i + 1 < n:
                    i += 1
                    esc = src[i]
                    buf.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(esc, esc))
                else:
                    buf.append(src[i])
                i += 1
            if i >= n:
                raise ValueError(f"Unclosed string literal in sexpr: {src}")
            i += 1
            tokens.append(("STR", "".join(buf)))
        else:
            start = i
            while i < n and (not src[i].isspace()) and src[i] not in ("(", ")", "[", "]", '"'):
                i += 1
            sym = src[start:i]
            if sym in ("true", "#t"):
                tokens.append(("BOOL", True))
            elif sym in ("false", "#f"):
                tokens.append(("BOOL", False))
            elif sym in ("null", "nil"):
                tokens.append(("NULL", None))
            else:
                try:
                    tokens.append(("INT", int(sym)))
                    continue
                except ValueError:
                    pass
                if "." in sym or "e" in sym or "E" in sym:
                    try:
                        tokens.append(("FLOAT", float(sym)))
                        continue
                    except ValueError:
                        pass
                tokens.append(("SYM", sym))
    return tokens


def parse_sexpr(src: str) -> Any:
    tokens = _tokenize_sexpr(src)
    pos = 0

    def read_one() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"Unexpected EOF in sexpr: {src}")
        kind, val = tokens[pos]
        pos += 1
        if kind == "LP":
            items: List[Any] = []
            while pos < len(tokens) and tokens[pos][0] != "RP":
                items.append(read_one())
            if pos >= len(tokens):
                raise ValueError(f"Unclosed list in sexpr: {src}")
            pos += 1
            return items
        if kind == "RP":
            raise ValueError(f"Unexpected ')' in sexpr: {src}")
        return (kind, val)

    ast = read_one()
    if pos != len(tokens):
        raise ValueError(f"Trailing tokens in sexpr: {src}")
    return ast


def _sql_quote_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def sexpr_to_sqlite_sql(ast: Any, row_alias: str, valid_cols: Optional[set] = None) -> str:
    """Translates a parsed ImpScheme predicate/expression AST to SQLite 3-valued SQL."""
    if isinstance(ast, tuple):
        kind, val = ast
        if kind == "INT":
            return str(int(val))
        if kind == "FLOAT":
            return repr(float(val))
        if kind == "STR":
            return _sql_quote_str(str(val))
        if kind == "BOOL":
            return "1" if val else "0"
        if kind == "NULL":
            return "NULL"
        if kind == "SYM":
            col = str(val)
            if valid_cols is not None and col not in valid_cols:
                raise ValueError(f"Unknown column '{col}' for alias '{row_alias}'")
            return f"{row_alias}.{col}"
        raise ValueError(f"Unhandled token {ast}")

    if isinstance(ast, list):
        if not ast:
            return "NULL"
        head = ast[0]
        if not (isinstance(head, tuple) and head[0] == "SYM"):
            raise ValueError(f"Expected operator symbol at head of {ast}")
        op = str(head[1])
        args = ast[1:]

        if op == "prop" and len(args) == 1:
            ak, av = args[0]
            if ak in ("STR", "SYM"):
                if valid_cols is not None and str(av) not in valid_cols:
                    raise ValueError(f"Unknown prop '{av}'")
                return f"{row_alias}.{av}"
            raise ValueError(f"Invalid prop argument: {args[0]}")

        if op in ("null?", "is_null") and len(args) == 1:
            inner = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            return f"({inner} IS NULL)"

        if op in ("not-null?", "is_not_null") and len(args) == 1:
            inner = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            return f"({inner} IS NOT NULL)"

        if op in ("starts-with", "starts_with") and len(args) == 2:
            col_sql = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            pk, pv = args[1]
            if pk != "STR":
                raise ValueError(f"starts-with requires string literal prefix, got {args[1]}")
            prefix = str(pv)
            plen = len(prefix)
            if plen == 0:
                # Empty prefix is true iff string is non-null (Kleene NULL -> NULL)
                return f"(CASE WHEN {col_sql} IS NULL THEN NULL ELSE 1 END)"
            return f"(substr({col_sql}, 1, {plen}) = {_sql_quote_str(prefix)})"

        if op == "not" and len(args) == 1:
            inner = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            return f"(NOT ({inner}))"

        if op == "and":
            if not args:
                return "1"
            parts = [f"({sexpr_to_sqlite_sql(a, row_alias, valid_cols)})" for a in args]
            return "(" + " AND ".join(parts) + ")"

        if op == "or":
            if not args:
                return "0"
            parts = [f"({sexpr_to_sqlite_sql(a, row_alias, valid_cols)})" for a in args]
            return "(" + " OR ".join(parts) + ")"

        if op in ("==", "=") and len(args) == 2:
            lhs = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            rhs = sexpr_to_sqlite_sql(args[1], row_alias, valid_cols)
            return f"({lhs} = {rhs})"

        if op in ("!=", "<>") and len(args) == 2:
            lhs = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            rhs = sexpr_to_sqlite_sql(args[1], row_alias, valid_cols)
            return f"({lhs} != {rhs})"

        if op in ("<", "<=", ">", ">=") and len(args) == 2:
            lhs = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            rhs = sexpr_to_sqlite_sql(args[1], row_alias, valid_cols)
            return f"({lhs} {op} {rhs})"

        if op in ("+", "-", "*") and len(args) == 2:
            lhs = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            rhs = sexpr_to_sqlite_sql(args[1], row_alias, valid_cols)
            return f"({lhs} {op} {rhs})"

        if op == "-" and len(args) == 1:
            inner = sexpr_to_sqlite_sql(args[0], row_alias, valid_cols)
            return f"(-({inner}))"

        raise ValueError(f"Unsupported S-expression operator '{op}' in {ast}")

    raise ValueError(f"Malformed S-expression AST: {ast}")


# -----------------------------------------------------------------------------
# SQLite Oracle Engine
# -----------------------------------------------------------------------------


class SQLiteOracle:
    """In-memory indexed SQLite differential oracle over `powergrid_ground_truth.sqlite`."""

    def __init__(self, schema: FixtureSchema) -> None:
        self.schema = schema
        disk_conn = sqlite3.connect(f"file:{schema.sqlite_path}?mode=ro", uri=True)
        self.conn = sqlite3.connect(":memory:")
        try:
            disk_conn.backup(self.conn)
        finally:
            disk_conn.close()

        # Register IEEE fp16/fp32 helper functions for exact scalar reduction parity
        self.conn.create_function(
            "ieee_f16",
            1,
            lambda x: None if x is None else quantize_f16(float(x)),
            deterministic=True,
        )
        self.conn.create_function(
            "ieee_f32",
            1,
            lambda x: None if x is None else quantize_f32(float(x)),
            deterministic=True,
        )

        # Build CSR/CSC covering indexes in RAM so 500+ CTE queries finish in <0.25s
        cur = self.conn.cursor()
        for rel in schema.relations.values():
            if not rel.is_virtual:
                cur.execute(f"CREATE INDEX idx_{rel.name}_src ON rel_{rel.name}(src_dense, tgt_dense)")
                cur.execute(f"CREATE INDEX idx_{rel.name}_tgt ON rel_{rel.name}(tgt_dense, src_dense)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _compile_pipeline_ctes(
        self,
        anchor: Dict[str, Any],
        steps: Sequence[Dict[str, Any]],
        prefix: str,
        ctes: List[str],
        counter: List[int],
    ) -> Tuple[str, str, bool, int]:
        """
        Appends CTE definitions into `ctes` and returns:
          (final_cte_name, final_domain_name, is_multilane, active_mask)
        """
        dom_name = str(anchor["domain"])
        if dom_name not in self.schema.domains:
            raise ValueError(f"Unknown anchor domain '{dom_name}'")
        mode = str(anchor["mode"])

        def next_name(tag: str) -> str:
            idx = counter[0]
            counter[0] += 1
            return f"{prefix}_{tag}_{idx}"

        is_multilane = mode in ("multilane_seeds", "multilane_groups")
        active_mask = 0

        curr_cte = next_name("anchor")
        if mode == "all":
            ctes.append(f"{curr_cte}(node_id) AS (\n  SELECT dense_id AS node_id FROM dom_{dom_name}\n)")
        elif mode == "seeds":
            seeds = [int(u) for u in anchor.get("seeds", [])]
            if not seeds:
                ctes.append(f"{curr_cte}(node_id) AS (\n  SELECT dense_id AS node_id FROM dom_{dom_name} WHERE 0\n)")
            else:
                in_list = ", ".join(str(u) for u in sorted(set(seeds)))
                ctes.append(
                    f"{curr_cte}(node_id) AS (\n"
                    f"  SELECT dense_id AS node_id FROM dom_{dom_name} WHERE dense_id IN ({in_list})\n)"
                )
        elif mode == "multilane_seeds":
            seeds = [int(u) for u in anchor.get("seeds", [])][:64]
            k = len(seeds)
            active_mask = (1 << k) - 1 if k > 0 else 0
            if k == 0:
                ctes.append(
                    f"{curr_cte}(node_id, lane_idx) AS (\n"
                    f"  SELECT dense_id AS node_id, 0 AS lane_idx FROM dom_{dom_name} WHERE 0\n)"
                )
            else:
                raw_cte = next_name("seed_vals")
                vals = ", ".join(f"({u}, {b})" for b, u in enumerate(seeds))
                ctes.append(f"{raw_cte}(node_id, lane_idx) AS (\n  VALUES {vals}\n)")
                ctes.append(
                    f"{curr_cte}(node_id, lane_idx) AS (\n"
                    f"  SELECT v.node_id, v.lane_idx\n"
                    f"  FROM {raw_cte} v\n"
                    f"  JOIN dom_{dom_name} d ON d.dense_id = v.node_id\n"
                    f"  GROUP BY v.node_id, v.lane_idx\n)"
                )
        elif mode == "multilane_groups":
            groups = [(int(pair[0]), int(pair[1])) for pair in anchor.get("groups", [])]
            if not groups:
                active_mask = 0
                ctes.append(
                    f"{curr_cte}(node_id, lane_idx) AS (\n"
                    f"  SELECT dense_id AS node_id, 0 AS lane_idx FROM dom_{dom_name} WHERE 0\n)"
                )
            else:
                max_g = max(g for _, g in groups)
                k = min(64, max(0, max_g + 1))
                active_mask = (1 << k) - 1 if k > 0 else 0
                vals = ", ".join(f"({u}, {g})" for u, g in groups if 0 <= g < k)
                if not vals:
                    ctes.append(
                        f"{curr_cte}(node_id, lane_idx) AS (\n"
                        f"  SELECT dense_id AS node_id, 0 AS lane_idx FROM dom_{dom_name} WHERE 0\n)"
                    )
                else:
                    raw_cte = next_name("group_vals")
                    ctes.append(f"{raw_cte}(node_id, lane_idx) AS (\n  VALUES {vals}\n)")
                    ctes.append(
                        f"{curr_cte}(node_id, lane_idx) AS (\n"
                        f"  SELECT v.node_id, v.lane_idx\n"
                        f"  FROM {raw_cte} v\n"
                        f"  JOIN dom_{dom_name} d ON d.dense_id = v.node_id\n"
                        f"  GROUP BY v.node_id, v.lane_idx\n)"
                    )
        else:
            raise ValueError(f"Unsupported anchor.mode '{mode}'")

        curr_dom = dom_name

        for step in steps:
            op = str(step["op"])

            if op == "filter":
                dom_meta = self.schema.domains[curr_dom]
                pred_sql = sexpr_to_sqlite_sql(
                    parse_sexpr(str(step["sexpr"])),
                    "d",
                    set(dom_meta.attrs.keys()),
                )
                nxt = next_name("filter")
                if is_multilane:
                    ctes.append(
                        f"{nxt}(node_id, lane_idx) AS (\n"
                        f"  SELECT s.node_id, s.lane_idx\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN dom_{curr_dom} d ON d.dense_id = s.node_id\n"
                        f"  WHERE {pred_sql}\n)"
                    )
                else:
                    ctes.append(
                        f"{nxt}(node_id) AS (\n"
                        f"  SELECT s.node_id\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN dom_{curr_dom} d ON d.dense_id = s.node_id\n"
                        f"  WHERE {pred_sql}\n)"
                    )
                curr_cte = nxt

            elif op == "out":
                rel = self.schema.relations[str(step["relation"])]
                if rel.src_domain != curr_dom:
                    raise ValueError(f"out({rel.name}) domain mismatch: expected {rel.src_domain}, got {curr_dom}")
                nxt = next_name("out")
                if is_multilane:
                    ctes.append(
                        f"{nxt}(node_id, lane_idx) AS (\n"
                        f"  SELECT DISTINCT r.tgt_dense AS node_id, s.lane_idx\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN rel_{rel.name} r ON r.src_dense = s.node_id\n)"
                    )
                else:
                    ctes.append(
                        f"{nxt}(node_id) AS (\n"
                        f"  SELECT DISTINCT r.tgt_dense AS node_id\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN rel_{rel.name} r ON r.src_dense = s.node_id\n)"
                    )
                curr_cte = nxt
                curr_dom = rel.tgt_domain

            elif op == "in":
                rel = self.schema.relations[str(step["relation"])]
                if rel.tgt_domain != curr_dom:
                    raise ValueError(f"in({rel.name}) domain mismatch: expected {rel.tgt_domain}, got {curr_dom}")
                nxt = next_name("in")
                if is_multilane:
                    ctes.append(
                        f"{nxt}(node_id, lane_idx) AS (\n"
                        f"  SELECT DISTINCT r.src_dense AS node_id, s.lane_idx\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN rel_{rel.name} r ON r.tgt_dense = s.node_id\n)"
                    )
                else:
                    ctes.append(
                        f"{nxt}(node_id) AS (\n"
                        f"  SELECT DISTINCT r.src_dense AS node_id\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN rel_{rel.name} r ON r.tgt_dense = s.node_id\n)"
                    )
                curr_cte = nxt
                curr_dom = rel.src_domain

            elif op == "both":
                if is_multilane:
                    raise ValueError("both not used in multilane mode")
                rel = self.schema.relations[str(step["relation"])]
                if rel.src_domain != curr_dom or rel.tgt_domain != curr_dom:
                    raise ValueError(f"both({rel.name}) requires homogeneous relation on {curr_dom}")
                nxt = next_name("both")
                ctes.append(
                    f"{nxt}(node_id) AS (\n"
                    f"  SELECT r.tgt_dense AS node_id FROM {curr_cte} s JOIN rel_{rel.name} r ON r.src_dense = s.node_id\n"
                    f"  UNION\n"
                    f"  SELECT r.src_dense AS node_id FROM {curr_cte} s JOIN rel_{rel.name} r ON r.tgt_dense = s.node_id\n)"
                )
                curr_cte = nxt

            elif op == "out_union":
                if is_multilane:
                    raise ValueError("out_union not used in multilane mode")
                rels = [self.schema.relations[str(rn)] for rn in step["relations"]]
                if not rels:
                    raise ValueError("out_union requires >= 1 relation")
                tgt_dom = rels[0].tgt_domain
                for r in rels:
                    if r.src_domain != curr_dom or r.tgt_domain != tgt_dom:
                        raise ValueError(f"out_union domain mismatch on {r.name}")
                nxt = next_name("out_union")
                arms = [
                    f"  SELECT DISTINCT r.tgt_dense AS node_id FROM {curr_cte} s JOIN rel_{r.name} r ON r.src_dense = s.node_id"
                    for r in rels
                ]
                ctes.append(f"{nxt}(node_id) AS (\n" + "\n  UNION\n".join(arms) + "\n)")
                curr_cte = nxt
                curr_dom = tgt_dom

            elif op == "in_union":
                if is_multilane:
                    raise ValueError("in_union not used in multilane mode")
                rels = [self.schema.relations[str(rn)] for rn in step["relations"]]
                if not rels:
                    raise ValueError("in_union requires >= 1 relation")
                src_dom = rels[0].src_domain
                for r in rels:
                    if r.tgt_domain != curr_dom or r.src_domain != src_dom:
                        raise ValueError(f"in_union domain mismatch on {r.name}")
                nxt = next_name("in_union")
                arms = [
                    f"  SELECT DISTINCT r.src_dense AS node_id FROM {curr_cte} s JOIN rel_{r.name} r ON r.tgt_dense = s.node_id"
                    for r in rels
                ]
                ctes.append(f"{nxt}(node_id) AS (\n" + "\n  UNION\n".join(arms) + "\n)")
                curr_cte = nxt
                curr_dom = src_dom

            elif op == "out_where":
                if is_multilane:
                    raise ValueError("out_where not used in multilane mode")
                rel = self.schema.relations[str(step["relation"])]
                if rel.src_domain != curr_dom:
                    raise ValueError(f"out_where({rel.name}) domain mismatch")
                pred_sql = sexpr_to_sqlite_sql(
                    parse_sexpr(str(step["sexpr"])),
                    "r",
                    set(rel.attrs.keys()),
                )
                nxt = next_name("out_where")
                ctes.append(
                    f"{nxt}(node_id) AS (\n"
                    f"  SELECT DISTINCT r.tgt_dense AS node_id\n"
                    f"  FROM {curr_cte} s\n"
                    f"  JOIN rel_{rel.name} r ON r.src_dense = s.node_id\n"
                    f"  WHERE {pred_sql}\n)"
                )
                curr_cte = nxt
                curr_dom = rel.tgt_domain

            elif op == "in_where":
                if is_multilane:
                    raise ValueError("in_where not used in multilane mode")
                rel = self.schema.relations[str(step["relation"])]
                if rel.tgt_domain != curr_dom:
                    raise ValueError(f"in_where({rel.name}) domain mismatch")
                pred_sql = sexpr_to_sqlite_sql(
                    parse_sexpr(str(step["sexpr"])),
                    "r",
                    set(rel.attrs.keys()),
                )
                nxt = next_name("in_where")
                ctes.append(
                    f"{nxt}(node_id) AS (\n"
                    f"  SELECT DISTINCT r.src_dense AS node_id\n"
                    f"  FROM {curr_cte} s\n"
                    f"  JOIN rel_{rel.name} r ON r.tgt_dense = s.node_id\n"
                    f"  WHERE {pred_sql}\n)"
                )
                curr_cte = nxt
                curr_dom = rel.src_domain

            elif op == "repeat_out":
                if is_multilane:
                    raise ValueError("repeat_out not used in multilane mode")
                rel = self.schema.relations[str(step["relation"])]
                hops = int(step["hops"])
                if rel.src_domain != curr_dom or rel.tgt_domain != curr_dom:
                    raise ValueError(f"repeat_out({rel.name}) requires homogeneous relation")
                for _ in range(hops):
                    nxt = next_name("rep")
                    ctes.append(
                        f"{nxt}(node_id) AS (\n"
                        f"  SELECT DISTINCT r.tgt_dense AS node_id\n"
                        f"  FROM {curr_cte} s\n"
                        f"  JOIN rel_{rel.name} r ON r.src_dense = s.node_id\n)"
                    )
                    curr_cte = nxt

            elif op == "fixpoint_out":
                if is_multilane:
                    raise ValueError("fixpoint_out not used in multilane mode")
                rel = self.schema.relations[str(step["relation"])]
                max_hops = int(step["max_hops"])
                if rel.src_domain != curr_dom or rel.tgt_domain != curr_dom:
                    raise ValueError(f"fixpoint_out({rel.name}) requires homogeneous relation")
                rec_cte = next_name("fp_rec")
                nxt = next_name("fp")
                ctes.append(
                    f"{rec_cte}(node_id, depth) AS (\n"
                    f"  SELECT node_id, 0 FROM {curr_cte}\n"
                    f"  UNION\n"
                    f"  SELECT r.tgt_dense, {rec_cte}.depth + 1\n"
                    f"  FROM {rec_cte}\n"
                    f"  JOIN rel_{rel.name} r ON r.src_dense = {rec_cte}.node_id\n"
                    f"  WHERE {rec_cte}.depth < {max_hops}\n)"
                )
                ctes.append(
                    f"{nxt}(node_id) AS (\n"
                    f"  SELECT DISTINCT node_id FROM {rec_cte}\n)"
                )
                curr_cte = nxt

            elif op == "set_op":
                if is_multilane:
                    raise ValueError("set_op not used in multilane mode")
                set_kind = str(step["set_op"])
                branch = step["branch"]
                br_cte, br_dom, br_ml, _ = self._compile_pipeline_ctes(
                    branch["anchor"],
                    branch.get("steps", []),
                    f"{prefix}_br",
                    ctes,
                    counter,
                )
                if br_ml or br_dom != curr_dom:
                    raise ValueError(f"set_op branch domain mismatch: {br_dom} vs {curr_dom}")
                nxt = next_name("setop")
                if set_kind == "union":
                    sql_op = "UNION"
                elif set_kind == "intersect":
                    sql_op = "INTERSECT"
                elif set_kind == "difference":
                    sql_op = "EXCEPT"
                else:
                    raise ValueError(f"Unsupported set_op '{set_kind}'")
                ctes.append(
                    f"{nxt}(node_id) AS (\n"
                    f"  SELECT node_id FROM {curr_cte}\n"
                    f"  {sql_op}\n"
                    f"  SELECT node_id FROM {br_cte}\n)"
                )
                curr_cte = nxt

            elif op == "where_quorum":
                if not is_multilane:
                    raise ValueError("where_quorum requires multilane anchor")
                min_k = int(step["min_k"])
                req_mask = int(step.get("req_mask", active_mask)) & active_mask
                forbid_mask = int(step.get("forbid_mask", 0)) & active_mask
                if min_k <= 0:
                    raise ValueError("Generated where_quorum requires min_k >= 1")
                q_keep = next_name("quorum_keep")
                nxt = next_name("quorum")
                ctes.append(
                    f"{q_keep}(node_id) AS (\n"
                    f"  SELECT s.node_id\n"
                    f"  FROM {curr_cte} s\n"
                    f"  GROUP BY s.node_id\n"
                    f"  HAVING SUM(CASE WHEN (({req_mask} >> s.lane_idx) & 1) = 1 THEN 1 ELSE 0 END) >= {min_k}\n"
                    f"     AND SUM(CASE WHEN (({forbid_mask} >> s.lane_idx) & 1) = 1 THEN 1 ELSE 0 END) = 0\n)"
                )
                ctes.append(
                    f"{nxt}(node_id, lane_idx) AS (\n"
                    f"  SELECT s.node_id, s.lane_idx\n"
                    f"  FROM {curr_cte} s\n"
                    f"  JOIN {q_keep} q ON q.node_id = s.node_id\n)"
                )
                curr_cte = nxt

            else:
                raise ValueError(f"Unsupported step.op '{op}'")

        return curr_cte, curr_dom, is_multilane, active_mask

    def translate_to_sql(self, query: Dict[str, Any], *, use_udf: bool = False) -> Tuple[str, str, bool, int]:
        """
        Compiles a Phase B JSON query specification into a single SQLite `WITH RECURSIVE` SQL query.
        Returns (sql_string, final_domain_name, is_multilane, active_mask).
        """
        ctes: List[str] = []
        counter = [0]
        final_cte, final_dom, is_multilane, active_mask = self._compile_pipeline_ctes(
            query["anchor"],
            query.get("steps", []),
            "s",
            ctes,
            counter,
        )
        term = query["terminal"]
        top = str(term["op"])
        dom_meta = self.schema.domains[final_dom]

        if top == "nodes":
            if is_multilane:
                select_sql = f"SELECT DISTINCT node_id FROM {final_cte} ORDER BY node_id ASC"
            else:
                select_sql = f"SELECT node_id FROM {final_cte} ORDER BY node_id ASC"
        elif top == "keys":
            if is_multilane:
                select_sql = (
                    f"SELECT CAST(d.{dom_meta.key_col} AS TEXT)\n"
                    f"FROM (SELECT DISTINCT node_id FROM {final_cte}) f\n"
                    f"JOIN dom_{final_dom} d ON d.dense_id = f.node_id\n"
                    f"ORDER BY f.node_id ASC"
                )
            else:
                select_sql = (
                    f"SELECT CAST(d.{dom_meta.key_col} AS TEXT)\n"
                    f"FROM {final_cte} f\n"
                    f"JOIN dom_{final_dom} d ON d.dense_id = f.node_id\n"
                    f"ORDER BY f.node_id ASC"
                )
        elif top == "count":
            if is_multilane:
                select_sql = f"SELECT COUNT(DISTINCT node_id) FROM {final_cte}"
            else:
                select_sql = f"SELECT COUNT(*) FROM {final_cte}"
        elif top in ("sum", "min", "max"):
            attr_name = str(term["attr"])
            attr_meta = dom_meta.attrs[attr_name]
            col_expr = f"d.{attr_name}"
            if use_udf and attr_meta.base_type == "float16":
                col_expr = f"ieee_f16(d.{attr_name})"
            elif use_udf and attr_meta.base_type == "float32":
                col_expr = f"ieee_f32(d.{attr_name})"
            src_rel = f"(SELECT DISTINCT node_id FROM {final_cte})" if is_multilane else final_cte
            agg_fn = top.upper()
            if top == "sum":
                select_sql = (
                    f"SELECT COALESCE(SUM(CAST({col_expr} AS REAL)), 0.0)\n"
                    f"FROM {src_rel} f\n"
                    f"JOIN dom_{final_dom} d ON d.dense_id = f.node_id"
                )
            else:
                select_sql = (
                    f"SELECT {agg_fn}(CAST({col_expr} AS REAL))\n"
                    f"FROM {src_rel} f\n"
                    f"JOIN dom_{final_dom} d ON d.dense_id = f.node_id"
                )
        elif top in ("lane_masks", "set_cover"):
            if not is_multilane:
                raise ValueError(f"Terminal '{top}' requires multilane pipeline")
            select_sql = (
                f"SELECT node_id, SUM(1 << lane_idx) AS lane_mask\n"
                f"FROM {final_cte}\n"
                f"GROUP BY node_id\n"
                f"ORDER BY node_id ASC"
            )
        elif top == "top_k":
            src_rel = f"SELECT DISTINCT node_id FROM {final_cte}" if is_multilane else f"SELECT node_id FROM {final_cte}"
            select_sql = f"{src_rel} ORDER BY node_id ASC"
        else:
            raise ValueError(f"Unsupported terminal.op '{top}'")

        full_sql = "WITH RECURSIVE\n" + ",\n".join(ctes) + "\n" + select_sql + ";"
        return full_sql, final_dom, is_multilane, active_mask

    def evaluate(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Executes `query` against the ground-truth SQLite database and returns canonical JSON output."""
        qid = str(query["id"])
        term = query["terminal"]
        top = str(term["op"])

        sql, final_dom, _, active_mask = self.translate_to_sql(query, use_udf=True)
        rows = self.conn.execute(sql).fetchall()

        if top == "nodes":
            return {
                "id": qid,
                "status": "ok",
                "result_type": "nodes",
                "nodes": [int(r[0]) for r in rows],
            }

        if top == "keys":
            return {
                "id": qid,
                "status": "ok",
                "result_type": "keys",
                "keys": [str(r[0]) for r in rows],
            }

        if top == "count":
            return {
                "id": qid,
                "status": "ok",
                "result_type": "count",
                "count": int(rows[0][0]) if rows else 0,
            }

        if top in ("sum", "min", "max"):
            raw_val = rows[0][0] if rows else None
            if top == "sum":
                scalar_out: Optional[float] = 0.0 if raw_val is None else float(raw_val)
            else:
                scalar_out = None if raw_val is None else float(raw_val)
            return {
                "id": qid,
                "status": "ok",
                "result_type": "scalar",
                "scalar": scalar_out,
            }

        if top == "lane_masks":
            lanes = [[int(r[0]), int(r[1]) & active_mask] for r in rows if (int(r[1]) & active_mask) != 0]
            return {
                "id": qid,
                "status": "ok",
                "result_type": "lane_masks",
                "lanes": lanes,
            }

        if top == "set_cover":
            target_mask = int(term.get("target_mask", active_mask)) & active_mask
            lane_pairs = [(int(r[0]), int(r[1]) & active_mask) for r in rows if (int(r[1]) & active_mask) != 0]
            uncovered = target_mask
            selected: List[int] = []
            steps = 0
            while uncovered != 0 and steps < 64:
                best_u = -1
                best_gain = 0
                for u, m in lane_pairs:
                    gain = (m & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                if best_u < 0 or best_gain == 0:
                    break
                selected.append(best_u)
                # Clear covered bits
                for u, m in lane_pairs:
                    if u == best_u:
                        uncovered &= ~m
                        break
                steps += 1
            return {
                "id": qid,
                "status": "ok",
                "result_type": "nodes",
                "nodes": selected,
            }

        if top == "top_k":
            cand_nodes = [int(r[0]) for r in rows]
            attr_name = str(term["attr"])
            q_vec = [quantize_f32(float(x)) for x in term["query_vec"]]
            metric = int(term["metric"])
            k = int(term["top_k"])
            dom_vecs = self.schema.domains[final_dom].vectors[attr_name]

            scored: List[Tuple[float, int]] = []
            nq = sum(x * x for x in q_vec)
            for u in cand_nodes:
                v = dom_vecs[u]
                d_len = min(len(q_vec), len(v))
                if metric == 0:  # Dot product
                    score = sum(q_vec[i] * v[i] for i in range(d_len))
                elif metric == 1:  # Negative squared L2: -||q - v||^2
                    score = -sum((q_vec[i] - v[i]) * (q_vec[i] - v[i]) for i in range(d_len))
                elif metric == 2:  # Cosine similarity
                    dot = sum(q_vec[i] * v[i] for i in range(d_len))
                    nv = sum(v[i] * v[i] for i in range(d_len))
                    denom = math.sqrt(nq) * math.sqrt(nv)
                    score = 0.0 if denom <= 1e-12 else (dot / denom)
                else:
                    raise ValueError(f"Unsupported top_k metric {metric}")
                scored.append((score, u))

            scored.sort(key=lambda item: (-item[0], item[1]))
            top_items = scored[:k]
            return {
                "id": qid,
                "status": "ok",
                "result_type": "top_k",
                "nodes": [u for _, u in top_items],
                "scores": [round(float(s), 6) for s, _ in top_items],
            }

        raise ValueError(f"Unhandled terminal.op '{top}'")
