"""
PowerGrid Spec v0.10.0 Schema-Aware Differential Query Fuzzer & Multi-Engine Oracle.

Modules:
  - schema: Introspects manifest.json + powergrid_ground_truth.sqlite + vector CSVs.
  - generator: Seeded deterministic query generator across all 8 Kleisli/CEL/MultiLane/Vector families.
  - sqlite_oracle: Translates every generated query to SQLite CTE SQL + ground-truth evaluation.
  - comparator: Exact + float-tolerant multi-engine result comparator and reproducer writer.
"""

from .schema import ALL_FAMILIES, AttrMeta, DomainMeta, FixtureSchema, RelationMeta
from .generator import QueryGenerator
from .sqlite_oracle import SQLiteOracle
from .comparator import compare_engine_outputs, save_reproducer

__all__ = [
    "ALL_FAMILIES",
    "AttrMeta",
    "DomainMeta",
    "FixtureSchema",
    "RelationMeta",
    "QueryGenerator",
    "SQLiteOracle",
    "compare_engine_outputs",
    "save_reproducer",
]
