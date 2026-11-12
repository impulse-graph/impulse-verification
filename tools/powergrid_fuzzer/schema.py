"""
Schema introspection & value distribution profiler for PowerGrid Spec v0.10.0 fixtures.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union


ALL_FAMILIES: Tuple[str, ...] = (
    "f1_cel_filters",
    "f2_cross_domain_encodings",
    "f3_kleisli_multihop_setops",
    "f4_edge_predicates",
    "f5_reachability_fixpoint",
    "f6_scalar_aggregations",
    "f7_multilane_quorum_cover",
    "f8_vector_rerank_topk",
)

INT_TYPES: Set[str] = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "date32",
    "timestamp_ms",
    "timestamp_us",
    "timestamp_ns",
}

FLOAT_TYPES: Set[str] = {"float16", "float32", "float64"}
STRING_TYPES: Set[str] = {"var_string", "string"}

DOMAIN_CSV_FILES: Dict[str, str] = {
    "Bus": "buses.csv",
    "Generator": "generators.csv",
    "PMU": "pmus.csv",
    "LoadZone": "load_zones.csv",
    "Substation": "substations.csv",
    "ContingencyTier": "contingency_tiers.csv",
}


def quantize_f16(val: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(val)))[0]


def quantize_f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(val)))[0]


@dataclass
class AttrStats:
    total_rows: int = 0
    null_count: int = 0
    non_null_count: int = 0
    distinct_ints: List[int] = field(default_factory=list)
    distinct_floats: List[float] = field(default_factory=list)
    distinct_strings: List[str] = field(default_factory=list)
    string_prefixes: List[str] = field(default_factory=list)
    safe_thresholds: List[Union[int, float]] = field(default_factory=list)


@dataclass
class AttrMeta:
    name: str
    base_type: str
    nullable: bool
    dimension: int = 1
    stats: AttrStats = field(default_factory=AttrStats)

    @property
    def is_vector(self) -> bool:
        return self.dimension > 1 and self.base_type in FLOAT_TYPES

    @property
    def is_integer(self) -> bool:
        return self.dimension == 1 and self.base_type in INT_TYPES

    @property
    def is_high_ts_ns(self) -> bool:
        return self.dimension == 1 and self.base_type == "timestamp_ns"

    @property
    def is_exact_eq_int(self) -> bool:
        return self.is_integer and not self.is_high_ts_ns

    @property
    def is_float(self) -> bool:
        return self.dimension == 1 and self.base_type in FLOAT_TYPES

    @property
    def is_numeric(self) -> bool:
        return self.is_integer or self.is_float

    @property
    def is_aggregatable(self) -> bool:
        # Exclude > 2^53 nanosecond timestamps from float64 scalar sum/min/max to avoid precision loss
        return (self.is_float or (self.is_integer and not self.is_high_ts_ns))

    @property
    def is_string(self) -> bool:
        return self.dimension == 1 and self.base_type in STRING_TYPES

    @property
    def is_filterable(self) -> bool:
        return self.is_numeric or self.is_string


@dataclass
class DomainMeta:
    id: int
    name: str
    key_type: str
    key_col: str
    node_count: int
    attrs: Dict[str, AttrMeta] = field(default_factory=dict)
    vectors: Dict[str, List[Tuple[float, ...]]] = field(default_factory=dict)

    @property
    def filterable_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_filterable]

    @property
    def nullable_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_filterable and a.nullable]

    @property
    def string_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_string]

    @property
    def numeric_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_numeric]

    @property
    def aggregatable_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_aggregatable]

    @property
    def vector_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_vector]


@dataclass
class RelationMeta:
    id: int
    name: str
    src_domain: str
    tgt_domain: str
    src_domain_id: int
    tgt_domain_id: int
    encoding: str
    cardinality: str
    is_virtual: bool
    virtual_components: List[str] = field(default_factory=list)
    edge_count: int = 0
    attrs: Dict[str, AttrMeta] = field(default_factory=dict)
    linked_src_nodes: List[int] = field(default_factory=list)
    unlinked_src_nodes: List[int] = field(default_factory=list)
    linked_tgt_nodes: List[int] = field(default_factory=list)
    unlinked_tgt_nodes: List[int] = field(default_factory=list)
    multi_edge_src_nodes: List[int] = field(default_factory=list)

    @property
    def is_homogeneous(self) -> bool:
        return self.src_domain == self.tgt_domain

    @property
    def filterable_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_filterable]

    @property
    def nullable_attrs(self) -> List[AttrMeta]:
        return [a for a in self.attrs.values() if a.is_filterable and a.nullable]


def _compute_safe_float_thresholds(raw_vals: Sequence[float], base_type: str) -> List[float]:
    if not raw_vals:
        return [-1.0, 0.0, 1.0, 10.0, 100.0]

    # Build combined exclusion set of raw SQLite float64 AND IEEE fp16/fp32 quantized values
    excl: List[float] = []
    for v in raw_vals:
        fv = float(v)
        excl.append(fv)
        if base_type == "float16":
            excl.append(quantize_f16(fv))
        elif base_type == "float32":
            excl.append(quantize_f32(fv))

    sorted_excl = sorted(set(excl))
    vmin = sorted_excl[0]
    vmax = sorted_excl[-1]
    span = max(abs(vmax - vmin), 1.0)

    def min_margin(val: float) -> float:
        if base_type == "float16":
            return max(5e-3, 1.5e-3 * abs(val))
        return max(2e-3, 2e-4 * abs(val))

    def is_safe(cand: float) -> bool:
        req = min_margin(cand)
        # Binary search in sorted_excl
        lo, hi = 0, len(sorted_excl)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_excl[mid] < cand:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(sorted_excl) and abs(sorted_excl[lo] - cand) < req:
            return False
        if lo > 0 and abs(cand - sorted_excl[lo - 1]) < req:
            return False
        return True

    candidates: List[float] = []
    # Lower and upper boundary thresholds (0% / 100% selectivity)
    low_pad = max(min_margin(vmin) * 4.0, 0.05 * span, 0.25)
    high_pad = max(min_margin(vmax) * 4.0, 0.05 * span, 0.25)
    for b_cand in (round(vmin - low_pad, 4), round(vmax + high_pad, 4)):
        if is_safe(b_cand):
            candidates.append(b_cand)

    # Interior gaps across quantiles
    gaps: List[Tuple[int, float]] = []
    for i in range(len(sorted_excl) - 1):
        a, b = sorted_excl[i], sorted_excl[i + 1]
        mid = 0.5 * (a + b)
        cand = round(mid, 4)
        if not is_safe(cand):
            cand = round(mid, 5)
        if is_safe(cand):
            gaps.append((i, cand))

    if gaps:
        # Subsample up to 24 evenly spaced interior thresholds across the distribution
        if len(gaps) <= 24:
            candidates.extend(c for _, c in gaps)
        else:
            for k in range(24):
                idx = (k * (len(gaps) - 1)) // 23
                candidates.append(gaps[idx][1])
    else:
        # Fallback offsets around vmin / vmax
        for mult in (-2.0, -0.5, 0.5, 2.0):
            cand = round(vmin + mult * max(1.0, abs(vmin) * 0.1), 4)
            if is_safe(cand):
                candidates.append(cand)

    dedup = sorted(set(candidates))
    return dedup if dedup else [round(vmin - 1.0, 4), round(vmax + 1.0, 4)]


def _compute_safe_int_thresholds(distinct_ints: Sequence[int], is_high_ts_ns: bool) -> List[int]:
    if not distinct_ints:
        return [-1, 0, 1, 10, 100]
    s = sorted(set(distinct_ints))
    vmin, vmax = s[0], s[-1]
    if is_high_ts_ns:
        margin = 5_000_000  # 5ms >> float64 ULP (~200ns at 1.77e18)
        out: List[int] = [vmin - margin * 2, vmax + margin * 2]
        step = max(1, len(s) // 16)
        for i in range(0, len(s) - 1, step):
            mid = (s[i] + s[i + 1]) // 2
            if (mid - s[i]) >= margin and (s[i + 1] - mid) >= margin:
                out.append(mid)
        return sorted(set(out))

    out = [vmin - 1, vmin, vmax, vmax + 1]
    if len(s) <= 24:
        out.extend(s)
    else:
        for k in range(24):
            idx = (k * (len(s) - 1)) // 23
            out.append(s[idx])
    return sorted(set(out))


def _compute_string_prefixes(strings: Sequence[str]) -> List[str]:
    prefixes: Set[str] = set()
    sample = list(strings)[:200]
    for s in sample:
        for plen in (3, 4, 6, 8, 10):
            if len(s) >= plen:
                prefixes.add(s[:plen])
    # Always include a non-matching prefix for 0% selectivity testing
    prefixes.add("ZZZ_NOMATCH_")
    return sorted(prefixes)


class FixtureSchema:
    """Introspected PowerGrid fixture schema, value distributions, and exact vector ground truth."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir.resolve()
        self.case_name = self.case_dir.name
        self.manifest_path = self.case_dir / "manifest.json"
        self.sqlite_path = self.case_dir / "powergrid_ground_truth.sqlite"
        imps_candidates = sorted(self.case_dir.glob("*_v010_fixture.imps"))
        if not imps_candidates:
            raise FileNotFoundError(f"No *_v010_fixture.imps found in {self.case_dir}")
        self.imps_path = imps_candidates[0]

        raw_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.domains: Dict[str, DomainMeta] = {}
        self.domains_by_id: Dict[int, DomainMeta] = {}
        self.relations: Dict[str, RelationMeta] = {}
        self._arith_cache: Dict[Tuple[str, str, bool], List[Union[int, float]]] = {}

        conn = sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True)
        try:
            self._load_domains(raw_manifest, conn)
            self._load_relations(raw_manifest, conn)
            self._load_vectors_from_csv()
        finally:
            conn.close()

    def _parse_type(self, raw_type: str) -> Tuple[str, bool]:
        if raw_type.endswith("?"):
            return raw_type[:-1], True
        return raw_type, False

    def _profile_attr(self, conn: sqlite3.Connection, table_name: str, attr: AttrMeta) -> None:
        cur = conn.cursor()
        # Check column exists in SQLite table (intervals / vectors / bytes may be transformed or omitted)
        cols = {row[1] for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if attr.name not in cols:
            return

        total_rows = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        null_count = cur.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {attr.name} IS NULL"
        ).fetchone()[0]
        non_null_vals = [
            r[0]
            for r in cur.execute(
                f"SELECT {attr.name} FROM {table_name} WHERE {attr.name} IS NOT NULL"
            ).fetchall()
        ]
        st = AttrStats(
            total_rows=total_rows,
            null_count=null_count,
            non_null_count=len(non_null_vals),
        )
        if attr.is_integer:
            ints = sorted({int(v) for v in non_null_vals})
            st.distinct_ints = ints
            st.safe_thresholds = _compute_safe_int_thresholds(ints, attr.is_high_ts_ns)
        elif attr.is_float:
            floats = sorted({float(v) for v in non_null_vals})
            st.distinct_floats = floats
            st.safe_thresholds = _compute_safe_float_thresholds(floats, attr.base_type)
        elif attr.is_string:
            strs = sorted({str(v) for v in non_null_vals})
            st.distinct_strings = strs
            st.string_prefixes = _compute_string_prefixes(strs)
        attr.stats = st

    def _load_domains(self, raw_manifest: dict, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        for d in raw_manifest["domains"]:
            dom_id = int(d["id"])
            name = str(d["name"])
            tbl = f"dom_{name}"
            cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            dom = DomainMeta(
                id=dom_id,
                name=name,
                key_type=str(d["key_type"]),
                key_col=str(d["key_attribute"]),
                node_count=cnt,
            )
            sql_cols = {row[1] for row in cur.execute(f"PRAGMA table_info({tbl})").fetchall()}
            for a in d.get("attributes", []):
                btype, nullable = self._parse_type(str(a["type"]))
                dim = int(a.get("dimension", 1))
                am = AttrMeta(name=str(a["name"]), base_type=btype, nullable=nullable, dimension=dim)
                # Only mark scalar/string filterable if backed by a 1:1 SQLite column
                if am.dimension == 1 and am.name in sql_cols:
                    self._profile_attr(conn, tbl, am)
                    dom.attrs[am.name] = am
                elif am.is_vector:
                    dom.attrs[am.name] = am
            self.domains[name] = dom
            self.domains_by_id[dom_id] = dom

    def _load_relations(self, raw_manifest: dict, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        for r_idx, r in enumerate(raw_manifest["relations"]):
            name = str(r["name"])
            src_id = int(r["src_domain"])
            tgt_id = int(r["tgt_domain"])
            src_dom = self.domains_by_id[src_id]
            tgt_dom = self.domains_by_id[tgt_id]
            v_comps = [str(x) for x in r.get("virtual_components", [])]
            is_virt = len(v_comps) > 0
            tbl = f"rel_{name}"
            ecount = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

            rel = RelationMeta(
                id=r_idx,
                name=name,
                src_domain=src_dom.name,
                tgt_domain=tgt_dom.name,
                src_domain_id=src_id,
                tgt_domain_id=tgt_id,
                encoding="virtual" if is_virt else str(r.get("encoding", "csr")),
                cardinality="m:n" if is_virt else str(r.get("cardinality", "m:n")),
                is_virtual=is_virt,
                virtual_components=v_comps,
                edge_count=ecount,
            )

            linked_src = {
                int(row[0]) for row in cur.execute(f"SELECT DISTINCT src_dense FROM {tbl}").fetchall()
            }
            linked_tgt = {
                int(row[0]) for row in cur.execute(f"SELECT DISTINCT tgt_dense FROM {tbl}").fetchall()
            }
            rel.linked_src_nodes = sorted(linked_src)
            rel.unlinked_src_nodes = [u for u in range(src_dom.node_count) if u not in linked_src]
            rel.linked_tgt_nodes = sorted(linked_tgt)
            rel.unlinked_tgt_nodes = [v for v in range(tgt_dom.node_count) if v not in linked_tgt]
            rel.multi_edge_src_nodes = [
                int(row[0])
                for row in cur.execute(
                    f"SELECT src_dense FROM {tbl} GROUP BY src_dense, tgt_dense HAVING COUNT(*) > 1"
                ).fetchall()
            ]

            if not is_virt:
                sql_cols = {row[1] for row in cur.execute(f"PRAGMA table_info({tbl})").fetchall()}
                for a in r.get("attributes", []):
                    btype, nullable = self._parse_type(str(a["type"]))
                    dim = int(a.get("dimension", 1))
                    am = AttrMeta(name=str(a["name"]), base_type=btype, nullable=nullable, dimension=dim)
                    if am.dimension == 1 and am.name in sql_cols:
                        self._profile_attr(conn, tbl, am)
                        rel.attrs[am.name] = am

            self.relations[name] = rel

    def _load_vectors_from_csv(self) -> None:
        for dom_name, dom in self.domains.items():
            vec_attrs = dom.vector_attrs
            if not vec_attrs:
                continue
            csv_file = self.case_dir / "domains" / DOMAIN_CSV_FILES[dom_name]
            for va in vec_attrs:
                dom.vectors[va.name] = []
            with open(csv_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for va in vec_attrs:
                        raw_str = row[va.name].strip()
                        inner = raw_str.strip("[]")
                        parts = [float(x) for x in inner.split(";") if x.strip()]
                        if va.base_type == "float16":
                            q_parts = tuple(quantize_f16(x) for x in parts)
                        else:
                            q_parts = tuple(quantize_f32(x) for x in parts)
                        dom.vectors[va.name].append(q_parts)
                    # Verify 1:1 row count after loop
            for va in vec_attrs:
                if len(dom.vectors[va.name]) != dom.node_count:
                    raise ValueError(
                        f"Vector row count mismatch on {dom_name}.{va.name}: "
                        f"{len(dom.vectors[va.name])} != {dom.node_count}"
                    )

    def relations_from(self, domain_name: str) -> List[RelationMeta]:
        return [r for r in self.relations.values() if r.src_domain == domain_name]

    def relations_into(self, domain_name: str) -> List[RelationMeta]:
        return [r for r in self.relations.values() if r.tgt_domain == domain_name]

    def homogeneous_relations(self, domain_name: str = "Bus") -> List[RelationMeta]:
        return [
            r
            for r in self.relations.values()
            if r.src_domain == domain_name and r.tgt_domain == domain_name
        ]

    def evaluate_safe_arith_threshold(
        self,
        rng: random.Random,
        domain_or_rel_table: str,
        sql_arith_expr: str,
        is_int_expr: bool,
    ) -> Union[int, float]:
        """Samples actual expression values on the SQLite table and returns a non-boundary threshold."""
        cache_key = (domain_or_rel_table, sql_arith_expr, is_int_expr)
        if cache_key not in self._arith_cache:
            conn = sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    f"SELECT ({sql_arith_expr}) AS v FROM {domain_or_rel_table} WHERE ({sql_arith_expr}) IS NOT NULL"
                ).fetchall()
            finally:
                conn.close()
            vals = [r[0] for r in rows if r[0] is not None]
            if is_int_expr:
                self._arith_cache[cache_key] = _compute_safe_int_thresholds([int(v) for v in vals], False)
            else:
                self._arith_cache[cache_key] = _compute_safe_float_thresholds([float(v) for v in vals], "float32")
        cands = self._arith_cache[cache_key]
        return int(rng.choice(cands)) if is_int_expr else float(rng.choice(cands))
