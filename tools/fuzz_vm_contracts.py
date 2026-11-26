#!/usr/bin/env python3
"""
fuzz_vm_contracts.py — ImpulseVM v0.11.0 Contract-Guided & Validator Differential Fuzzer.

Drives generation directly from the frozen v0.11 ISA (`v0.11-refactor/isa_v0_11.yaml` via
`v11_isa_loader.load_isa()`) and assembles programs via `impas_v11_assembler.ImpAsmV11`
(zero duplicated opcode tables).

Features:
  - Pillar-scoped fuzzing (`--pillars 1`, `--pillars 1,2`, `--pillars all`) across all 5 pillars
  - Full register-bank type & domain tracking (`R0..R15`, `F0..F15`, `V0..V15`, `MF0..MF7`, `M0..M7`)
  - 9 specialized fuzz profiles (`mixed`, `boundary`, `aliasing`, `controlflow`, `morsel`,
    `precision_3vl`, `multilane`, `negative`, `validator`)
  - Pluggable multi-engine persistent JSONL harness (`ocaml`, `java`, `cpp_on`, `cpp_off`)
  - Cross-engine consensus (`status`, `pc`, `user_throw_code`, `flags`, `ICTX v2 logical_hash`)
  - Metamorphic `P0..P3` `logical_hash` invariance checking per engine
  - 1-minimal delta-debugging shrinker for both `.impas` and raw-word `validator` divergences
  - Multiprocessing (`--workers`, `--chunk-size`) and ISA opcode/subop coverage reporting
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
ROOT_DIR = REPO_DIR.parent

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

ALL_PROFILES = (
    "mixed",
    "boundary",
    "aliasing",
    "controlflow",
    "morsel",
    "precision_3vl",
    "multilane",
    "negative",
    "validator",
    "corpus",
)

TRANSCENDENTAL_OPCODES = frozenset({0x49, 0x4A, 0x4B})


def resolve_spec_dir(cli_spec_dir: Path | None = None) -> Path:
    candidates: list[Path] = []
    if cli_spec_dir is not None:
        candidates.append(cli_spec_dir)
    env_spec = os.environ.get("IMPULSE_SPEC_DIR")
    if env_spec:
        candidates.append(Path(env_spec))
    candidates.append(REPO_DIR / "spec")
    candidates.append(ROOT_DIR / "impulse-graph-spec")

    for c in candidates:
        if (c / "v0.11-refactor" / "isa_v0_11.yaml").exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not locate v0.11 spec directory containing v0.11-refactor/isa_v0_11.yaml. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _import_spec_modules(spec_dir: Path) -> tuple[Any, Any]:
    spec_tools = str(spec_dir / "tools")
    if spec_tools not in sys.path:
        sys.path.insert(0, spec_tools)
    import v11_isa_loader  # type: ignore
    import impas_v11_assembler  # type: ignore

    return v11_isa_loader, impas_v11_assembler


def discover_default_ocaml_bin() -> Path:
    candidates = [
        ROOT_DIR / "impulse-ocaml" / "_build" / "default" / "bin" / "main.exe",
        REPO_DIR / ".." / "impulse-ocaml" / "_build" / "default" / "bin" / "main.exe",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


def discover_default_java_cmd() -> list[str]:
    java_root = ROOT_DIR / "impulse-graph-java"
    modules = [
        java_root / "impulse-vm" / "target" / "classes",
        java_root / "impulse-vm" / "target" / "test-classes",
        java_root / "impulse-core" / "target" / "classes",
        java_root / "impulse-api" / "target" / "classes",
        java_root / "impulse-spec" / "target" / "classes",
        java_root / "impulse-storage" / "target" / "classes",
    ]
    cp = os.pathsep.join(str(p) for p in modules if p.exists())
    if not cp:
        raise FileNotFoundError(
            f"Could not auto-discover Java impulse-vm classpath under {java_root}. "
            "Pass --java-cmd explicitly."
        )
    return [
        "java",
        "--add-modules",
        "jdk.incubator.vector",
        "-cp",
        cp,
        "org.impulsegraph.vm.VmFixtureBatchMain",
    ]


class BatchEngineClient:
    """Persistent stdin/stdout JSONL client for a v0.11 VM engine."""

    def __init__(self, cmd: list[str], name: str):
        self.name = name
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def restart(self) -> None:
        self.close()
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def run_fixture(
        self, payload: dict[str, Any], profile: str = "P0", max_steps: int | None = None
    ) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        ms = str(max_steps) if max_steps is not None else "-"
        line = f"{profile}\t{ms}\t{json.dumps(payload, separators=(',', ':'))}\n"
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline()
        except BrokenPipeError:
            resp = ""
        if not resp:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            self.restart()
            raise RuntimeError(f"Engine '{self.name}' crashed: {err.strip()}")
        return json.loads(resp)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Register Bank State & Fixture Environment
# ---------------------------------------------------------------------------

@dataclass
class RState:
    ty: str = "unset"  # unset, i64, u64, f64, bool_3vl, str, f32_raw, null
    is_null: bool = True
    is_small_nonneg: bool = False  # guaranteed in 0..3 (safe for indexing/node ids/shifts)


@dataclass
class FState:
    dom: int = 0
    is_multilane: bool = False
    k_lanes: int = 1


@dataclass
class VState:
    ty: str = "f64"  # i64, u64, i32, u32, f64, f32, bool_3vl, str, lane_state_f32, lane_state_f64, emb
    length: int = 4
    dom: int | None = None
    is_lane_state: bool = False
    k_lanes: int = 0
    is_node_ids_for_dom: int | None = None


@dataclass
class FixtureEnv:
    dom_cards: dict[int, int] = field(default_factory=dict)
    # rel_id -> (src_dom, tgt_dom, edge_count)
    relations: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # (dom, attr) -> type
    node_attrs: dict[tuple[int, int], str] = field(default_factory=dict)
    # (rel, attr) -> type
    edge_attrs: dict[tuple[int, int], str] = field(default_factory=dict)
    # (dom, attr) with sorted index
    indexes: list[tuple[int, int]] = field(default_factory=list)
    key_catalogs: list[int] = field(default_factory=list)
    # param_id -> (kind, meta_dict)
    params: dict[int, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    # pool_id -> (kind, meta_dict)
    pools: dict[int, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    data_lines: list[str] = field(default_factory=list)
    pool_lines: list[str] = field(default_factory=list)


@dataclass
class V11RegTracker:
    r: dict[int, RState] = field(default_factory=lambda: {i: RState() for i in range(16)})
    f: dict[int, FState | None] = field(default_factory=lambda: {i: None for i in range(16)})
    v: dict[int, VState | None] = field(default_factory=lambda: {i: None for i in range(16)})
    # spill slot -> ("R"|"F"|"V", state_copy)
    spills: dict[int, tuple[str, Any]] = field(default_factory=dict)

    def r_with_types(self, types: set[str], non_null_only: bool = False, small_only: bool = False) -> list[int]:
        out = []
        for idx, st in self.r.items():
            if st.ty not in types:
                continue
            if non_null_only and st.is_null:
                continue
            if small_only and not st.is_small_nonneg:
                continue
            out.append(idx)
        return out

    def f_matching(self, dom: int | None = None, multilane: bool | None = None) -> list[int]:
        out = []
        for idx, st in self.f.items():
            if st is None:
                continue
            if dom is not None and st.dom != dom:
                continue
            if multilane is not None and st.is_multilane != multilane:
                continue
            out.append(idx)
        return out

    def v_matching(
        self,
        types: set[str] | None = None,
        length: int | None = None,
        lane_state: bool | None = None,
        node_ids_dom: int | None = None,
    ) -> list[int]:
        out = []
        for idx, st in self.v.items():
            if st is None:
                continue
            if types is not None and st.ty not in types:
                continue
            if length is not None and st.length != length:
                continue
            if lane_state is not None and st.is_lane_state != lane_state:
                continue
            if node_ids_dom is not None and st.is_node_ids_for_dom != node_ids_dom:
                continue
            out.append(idx)
        return out


# ---------------------------------------------------------------------------
# Constants & Helper Utilities
# ---------------------------------------------------------------------------

BOUNDARY_I32 = [0, 1, -1, 2, 7, 63, 64, 2147483647, -2147483648]
BOUNDARY_U32 = [0, 1, 2, 7, 63, 64, 0x7FFFFFFF, 0xFFFFFFFF]
BOUNDARY_I64 = [
    "0",
    "1",
    "-1",
    "63",
    "64",
    "0x7FFFFFFFFFFFFFFF",
    "0x8000000000000000",
    "1234567890123",
    "-1234567890123",
]
BOUNDARY_U64 = [
    "0",
    "1",
    "63",
    "64",
    "0x7FFFFFFFFFFFFFFF",
    "0x8000000000000000",
    "0xFFFFFFFFFFFFFFFF",
]
BOUNDARY_F64 = [
    "0.0",
    "-0.0",
    "1.0",
    "-1.0",
    "0.5",
    "-2.5",
    "3.5",
    "100.0",
    "0x7FF0000000000000",  # +Inf
    "0xFFF0000000000000",  # -Inf
    "0x7FF8000000000000",  # qNaN
    "0x0000000000000001",  # min subnormal
    "0x7FEFFFFFFFFFFFFF",  # DBL_MAX
]
BOUNDARY_F32 = [
    "0.0",
    "-0.0",
    "1.0",
    "-1.0",
    "0.5",
    "2.5",
    "0x7F800000",  # +Inf
    "0xFF800000",  # -Inf
    "0x7FC00000",  # qNaN
]

NORMAL_I32 = [0, 1, 2, 3, 5, 10, -1, -5, 42]
NORMAL_F64 = ["0.0", "-0.0", "1.0", "-1.0", "2.5", "-3.5", "0.25", "10.0"]
NORMAL_F32 = ["0.0", "1.0", "-1.0", "0.5", "2.0", "3.5"]


def _fmt_list(vals: list[Any]) -> str:
    parts = []
    for v in vals:
        if v is None:
            parts.append("null")
        elif isinstance(v, bool):
            parts.append("true" if v else "false")
        elif isinstance(v, str):
            parts.append(f'"{v}"')
        elif isinstance(v, list):
            parts.append(_fmt_list(v))
        else:
            parts.append(str(v))
    return "[" + ", ".join(parts) + "]"


def _build_csr(n_src: int, n_tgt: int, rng: random.Random, dense_edges: bool = False) -> tuple[list[int], list[int]]:
    offsets = [0]
    targets: list[int] = []
    for u in range(n_src):
        if n_tgt == 0:
            offsets.append(len(targets))
            continue
        if u in (0, 1, 2, 63, 64, 65, n_src - 1) and u < n_src:
            k = rng.randint(1, min(3, n_tgt))
            nbrs = sorted({(u + j) % n_tgt for j in range(1, k + 1)})
        elif dense_edges or rng.random() < 0.45:
            k = rng.randint(1, min(3, n_tgt))
            nbrs = sorted({rng.randint(0, n_tgt - 1) for _ in range(k)})
        else:
            nbrs = []
        targets.extend(nbrs)
        offsets.append(len(targets))
    # Guarantee at least 2 edges if possible
    if len(targets) == 0 and n_src > 0 and n_tgt > 0:
        targets = [0, min(1, n_tgt - 1)]
        offsets = [0] + [len(targets)] * n_src
    return offsets, targets


def _build_csc_from_csr(n_src: int, n_tgt: int, offsets: list[int], targets: list[int]) -> tuple[list[int], list[int]]:
    buckets: list[list[int]] = [[] for _ in range(n_tgt)]
    for u in range(n_src):
        for idx in range(offsets[u], offsets[u + 1]):
            v = targets[idx]
            if 0 <= v < n_tgt:
                buckets[v].append(u)
    csc_off = [0]
    csc_src: list[int] = []
    for v in range(n_tgt):
        csc_src.extend(buckets[v])
        csc_off.append(len(csc_src))
    return csc_off, csc_src


# ---------------------------------------------------------------------------
# Contract-Guided Program Generator (All 5 Pillars)
# ---------------------------------------------------------------------------

class V11ContractFuzzer:
    def __init__(self, isa: dict[str, Any], asm: Any, allowed_pillars: set[int]):
        self.isa = isa
        self.asm = asm
        self.allowed_pillars = set(allowed_pillars)
        self.by_code: dict[int, dict[str, Any]] = {op["code"]: op for op in isa["opcodes"]}
        self.active_ops: list[dict[str, Any]] = [
            op for op in isa["opcodes"] if op["code"] != 0xFF and op["pillar"] in self.allowed_pillars
        ]
        self.active_codes: set[int] = {op["code"] for op in self.active_ops}

    def available_profiles(self) -> list[str]:
        profs = ["mixed", "boundary", "aliasing", "controlflow", "precision_3vl", "negative", "validator"]
        if 3 in self.allowed_pillars:
            profs.append("morsel")
        if 5 in self.allowed_pillars:
            profs.append("multilane")
        return profs

    def pick_profile(self, seed: int, requested: str) -> str:
        if requested != "all":
            return requested
        rng = random.Random(seed ^ 0x5A5A5A5A)
        avail = self.available_profiles()
        weights = {
            "mixed": 0.18,
            "boundary": 0.14,
            "aliasing": 0.12,
            "controlflow": 0.12,
            "morsel": 0.11,
            "precision_3vl": 0.12,
            "multilane": 0.11,
            "negative": 0.05,
            "validator": 0.05,
        }
        w_list = [weights[p] for p in avail]
        return rng.choices(avail, weights=w_list, k=1)[0]

    # ---- Fixture Environment Builder ----

    def _build_fixture_env(self, rng: random.Random, profile: str) -> FixtureEnv:
        env = FixtureEnv()
        is_boundary = profile == "boundary" and rng.random() < 0.45
        if is_boundary and (2 in self.allowed_pillars or 3 in self.allowed_pillars or 4 in self.allowed_pillars):
            n0 = rng.choice([65, 67, 132])
            n1 = rng.choice([5, 65])
        else:
            n0 = rng.choice([4, 5, 6, 8])
            n1 = rng.choice([3, 4, 5])

        env.dom_cards = {0: n0, 1: n1}
        env.data_lines.append(f'  .domain 0 name="User" count={n0} width=u32')
        env.data_lines.append(f'  .domain 1 name="Item" count={n1} width=u32')

        need_graph = bool(self.allowed_pillars & {3, 4, 5})
        if need_graph:
            # Relation 0: endomorphic 0 -> 0 with CSR + CSC
            off0, tgt0 = _build_csr(n0, n0, rng)
            csc_off0, csc_src0 = _build_csc_from_csr(n0, n0, off0, tgt0)
            env.relations[0] = (0, 0, len(tgt0))
            env.data_lines.append('  .relation 0 name="FOLLOWS" src_dom=0 tgt_dom=0')
            env.data_lines.append(f"  .csr_inline rel=0 offsets={_fmt_list(off0)} targets={_fmt_list(tgt0)}")
            env.data_lines.append(f"  .csc_inline rel=0 offsets={_fmt_list(csc_off0)} sources={_fmt_list(csc_src0)}")

            # Relation 1: bipartite 0 -> 1 with CSR
            off1, tgt1 = _build_csr(n0, n1, rng)
            env.relations[1] = (0, 1, len(tgt1))
            env.data_lines.append('  .relation 1 name="LIKES" src_dom=0 tgt_dom=1')
            env.data_lines.append(f"  .csr_inline rel=1 offsets={_fmt_list(off1)} targets={_fmt_list(tgt1)}")

            # Edge attrs for rel 0 & rel 1
            e0_f64 = [None if (i % 5 == 4) else rng.choice([1.0, 2.0, 0.5, 3.5, 10.0]) for i in range(len(tgt0))]
            e0_i64 = [None if (i % 6 == 5) else rng.choice([1, 2, 5, 10, 20]) for i in range(len(tgt0))]
            e1_f64 = [None if (i % 5 == 4) else rng.choice([1.5, 2.5, 4.0, 8.0]) for i in range(len(tgt1))]
            env.edge_attrs[(0, 0)] = "f64"
            env.edge_attrs[(0, 1)] = "i64"
            env.edge_attrs[(1, 0)] = "f64"
            env.data_lines.append(f'  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_list(e0_f64)}')
            env.data_lines.append(f'  .edge_attr rel=0 attr=1 name="ei" type=i64 values={_fmt_list(e0_i64)}')
            env.data_lines.append(f'  .edge_attr rel=1 attr=0 name="amt" type=f64 values={_fmt_list(e1_f64)}')

            # Node attrs for dom 0 & dom 1
            na0_i64 = [None if (i == n0 - 1 and n0 > 2) else (10 * (i + 1)) for i in range(n0)]
            na0_f64 = [
                None if (i % 7 == 6) else rng.choice([0.0, -0.0, 1.5, 2.5, -3.5, 10.0, 42.0])
                for i in range(n0)
            ]
            na0_str = [None if (i == n0 - 1 and n0 > 2) else f"u{i}" for i in range(n0)]
            na1_f64 = [rng.choice([1.0, 2.0, 5.0, -1.0]) for _ in range(n1)]
            env.node_attrs[(0, 0)] = "i64"
            env.node_attrs[(0, 1)] = "f64"
            env.node_attrs[(0, 2)] = "str"
            env.node_attrs[(1, 0)] = "f64"
            env.data_lines.append(f'  .node_attr dom=0 attr=0 name="age" type=i64 values={_fmt_list(na0_i64)}')
            env.data_lines.append(f'  .node_attr dom=0 attr=1 name="score" type=f64 values={_fmt_list(na0_f64)}')
            env.data_lines.append(f'  .node_attr dom=0 attr=2 name="tag" type=str values={_fmt_list(na0_str)}')
            env.data_lines.append(f'  .node_attr dom=1 attr=0 name="price" type=f64 values={_fmt_list(na1_f64)}')

            # Sorted index on (0, 0) and key catalogs on dom 0 and dom 1
            env.indexes.append((0, 0))
            env.data_lines.append("  .index dom=0 attr=0 kind=sorted")
            keys0 = [f"k{i}" for i in range(n0)]
            keys1 = [f"k{(i * 2) % max(1, n0)}" for i in range(n1)]
            env.key_catalogs = [0, 1]
            env.data_lines.append(f'  .key_catalog dom=0 type=str keys={_fmt_list(keys0)}')
            env.data_lines.append(f'  .key_catalog dom=1 type=str keys={_fmt_list(keys1)}')

            # Embedding attribute for Pillar 5 ANN/Rerank/Pool
            if 5 in self.allowed_pillars:
                embs = []
                for i in range(n0):
                    if i == n0 - 1 and n0 > 3:
                        embs.append(None)
                    else:
                        a = round(float((i % 3) + 1) * 0.5, 3)
                        b = round(float(((i + 1) % 3)) * 0.5, 3)
                        c = round(float(((i + 2) % 2)) * 0.5, 3)
                        embs.append([a, b, c])
                env.node_attrs[(0, 3)] = "f32_emb"
                env.data_lines.append(f'  .node_attr dom=0 attr=3 name="emb" type=f32_emb values={_fmt_list(embs)}')

        # Params (always available — used by Pillar 1 LOAD_PARAM and Pillars 2-5)
        s_nodes0 = sorted({rng.randint(0, n0 - 1) for _ in range(min(3, n0))})
        s_nodes1 = sorted({rng.randint(0, n1 - 1) for _ in range(min(2, n1))})
        v_f64 = [None if (i == n0 - 1 and n0 > 2) else rng.choice([0.0, -0.0, 1.0, 2.5, -1.5, 4.0]) for i in range(n0)]
        v_i64 = [None if (i == n0 - 1 and n0 > 2) else rng.choice([0, 1, 2, 10, -5, 42]) for i in range(n0)]
        v_u64_ids = [i % n0 for i in range(n0)]
        v_bool = [rng.choice([True, False, None]) for _ in range(n0)]
        v_str = [f"k{i % n0}" if i % 4 != 3 else None for i in range(n0)]
        v_emb_q = [1.0, 0.5, 0.0]

        env.data_lines.append("  .param 0 type=i64 value=42")
        env.data_lines.append("  .param 1 type=f64 value=3.5")
        env.data_lines.append("  .param 2 type=i64 value=null")
        env.data_lines.append(f"  .param 3 type=frontier dom=0 values={_fmt_list(s_nodes0)}")
        env.data_lines.append(f"  .param 4 type=frontier dom=1 values={_fmt_list(s_nodes1)}")
        env.data_lines.append(f"  .param 5 type=f64_vec values={_fmt_list(v_f64)}")
        env.data_lines.append(f"  .param 6 type=i64_vec values={_fmt_list(v_i64)}")
        env.data_lines.append(f"  .param 7 type=u64_vec values={_fmt_list(v_u64_ids)}")
        env.data_lines.append(f"  .param 8 type=bool_3vl_vec values={_fmt_list(v_bool)}")
        env.data_lines.append(f"  .param 9 type=str_vec values={_fmt_list(v_str)}")
        env.data_lines.append(f"  .param 10 type=f64_vec values={_fmt_list(v_emb_q)}")
        # Small seed vector (len 2..3) for LANE_INIT_SEEDS & LANE_INIT_GROUPS
        seed_ids = [0, min(1, n0 - 1)]
        grp_ids = [0, 1]
        env.data_lines.append(f"  .param 11 type=u64_vec values={_fmt_list(seed_ids)}")
        env.data_lines.append(f"  .param 12 type=u64_vec values={_fmt_list(grp_ids)}")
        # Edge lane mask vector for rel 0
        e0_len = env.relations[0][2] if 0 in env.relations else 2
        edge_masks = [rng.choice([1, 2, 3]) for _ in range(e0_len)]
        env.data_lines.append(f"  .param 13 type=u64_vec values={_fmt_list(edge_masks)}")

        env.params = {
            0: ("scalar", {"type": "i64", "is_null": False}),
            1: ("scalar", {"type": "f64", "is_null": False}),
            2: ("scalar", {"type": "i64", "is_null": True}),
            3: ("frontier", {"dom": 0}),
            4: ("frontier", {"dom": 1}),
            5: ("vec", {"type": "f64", "len": n0}),
            6: ("vec", {"type": "i64", "len": n0}),
            7: ("vec", {"type": "u64", "len": n0, "node_ids_dom": 0}),
            8: ("vec", {"type": "bool_3vl", "len": n0}),
            9: ("vec", {"type": "str", "len": n0}),
            10: ("vec", {"type": "f64", "len": 3}),
            11: ("vec", {"type": "u64", "len": len(seed_ids), "node_ids_dom": 0}),
            12: ("vec", {"type": "u64", "len": len(grp_ids)}),
            13: ("vec", {"type": "u64", "len": e0_len}),
        }

        # Const pool entries
        env.pool_lines.append('  .pool 0 type=str value="k0"')
        env.pool_lines.append('  .pool 1 type=str value="k1"')
        env.pool_lines.append(f"  .pool 2 type=i64_vec values={_fmt_list(v_i64)}")
        env.pool_lines.append(f"  .pool 3 type=frontier values={_fmt_list(s_nodes0)}")
        env.pool_lines.append(f"  .pool 4 type=frontier values={_fmt_list(s_nodes1)}")
        env.pools = {
            0: ("str", {}),
            1: ("str", {}),
            2: ("vec", {"type": "i64", "len": n0}),
            3: ("frontier", {"dom": 0}),
            4: ("frontier", {"dom": 1}),
        }
        return env

    # ---- Preamble Builder ----

    def _emit_preamble(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker
    ) -> list[str]:
        lines: list[str] = []
        is_b = profile == "boundary"
        n0 = env.dom_cards[0]

        # R0..R11 initialized with diverse scalar types (Pillar 1 only)
        i32_v = rng.choice(BOUNDARY_I32 if is_b else NORMAL_I32)
        u32_v = rng.choice(BOUNDARY_U32 if is_b else [0, 1, 2, 3, 7])
        f32_v = rng.choice(BOUNDARY_F32 if is_b else NORMAL_F32)
        i64_v = rng.choice(BOUNDARY_I64 if is_b else ["10", "-10", "42", "0", "1"])
        u64_v = rng.choice(BOUNDARY_U64 if is_b else ["1", "2", "15", "0x123456789ABCDEF0"])
        f64_v = rng.choice(BOUNDARY_F64 if is_b else NORMAL_F64)
        f64_v2 = rng.choice(BOUNDARY_F64 if is_b else NORMAL_F64)
        small_n = rng.randint(0, max(0, min(3, n0 - 1)))

        lines.append(f"  LOAD_CONST.i32 R0, {i32_v}")
        st.r[0] = RState("i64", False, 0 <= i32_v <= 3)

        lines.append(f"  LOAD_CONST.u32 R1, {u32_v}")
        st.r[1] = RState("u64", False, 0 <= u32_v <= 3)

        lines.append(f"  LOAD_CONST.f32 R2, {f32_v}")
        st.r[2] = RState("f64", False, False)

        lines.append(f"  LOAD_CONST.f32_raw R3, {f32_v}")
        st.r[3] = RState("f32_raw", False, False)

        lines.append(f"  LOAD_CONST.i64.ext R4, {i64_v}")
        st.r[4] = RState("i64", False, False)

        lines.append(f"  LOAD_CONST.u64.ext R5, {u64_v}")
        st.r[5] = RState("u64", False, False)

        lines.append(f"  LOAD_CONST.f64.ext R6, {f64_v}")
        st.r[6] = RState("f64", False, False)

        lines.append("  LOAD_CONST.null R7")
        st.r[7] = RState("null", True, False)

        lines.append("  LOAD_CONST_POOL.str R8, pool=0")
        st.r[8] = RState("str", False, False)

        lines.append(f"  LOAD_CONST.u32 R9, {small_n}")
        st.r[9] = RState("u64", False, True)

        lines.append(f"  LOAD_CONST.f64.ext R10, {f64_v2}")
        st.r[10] = RState("f64", False, False)

        lines.append("  SCALAR_3VL.cmp_eq_i64 R11, R0, R4")
        st.r[11] = RState("bool_3vl", False, False)

        # F0..F3 initialized via Pillar 1 LOAD_CONST_POOL / LOAD_PARAM
        lines.append("  LOAD_CONST_POOL.frontier F0, pool=3, dom=0")
        st.f[0] = FState(dom=0, is_multilane=False, k_lanes=1)

        lines.append("  LOAD_PARAM.frontier F1, param=3, dom=0")
        st.f[1] = FState(dom=0, is_multilane=False, k_lanes=1)

        lines.append("  LOAD_PARAM.frontier F2, param=4, dom=1")
        st.f[2] = FState(dom=1, is_multilane=False, k_lanes=1)

        lines.append("  MOV.clear_f F3, dom=0")
        st.f[3] = FState(dom=0, is_multilane=False, k_lanes=1)

        # V0..V6 initialized via Pillar 1 LOAD_PARAM / LOAD_CONST_POOL
        lines.append("  LOAD_PARAM.vec V0, param=5, type=f64")
        st.v[0] = VState("f64", n0, dom=0)

        lines.append("  LOAD_PARAM.vec V1, param=6, type=i64")
        st.v[1] = VState("i64", n0, dom=0)

        lines.append("  LOAD_PARAM.vec V2, param=7, type=u64")
        st.v[2] = VState("u64", n0, dom=0, is_node_ids_for_dom=0)

        lines.append("  LOAD_PARAM.vec V3, param=8, type=bool")
        st.v[3] = VState("bool_3vl", n0, dom=0)

        lines.append("  LOAD_PARAM.vec V4, param=9, type=str")
        st.v[4] = VState("str", n0, dom=0)

        lines.append("  LOAD_PARAM.vec V5, param=10, type=f64")
        st.v[5] = VState("f64", 3)

        lines.append("  LOAD_PARAM.vec V6, param=11, type=u64")
        st.v[6] = VState("u64", 2, is_node_ids_for_dom=0)

        # Pillar 5 multi-lane & lane-state seeds when Pillar 5 is enabled
        if 5 in self.allowed_pillars:
            lines.append("  LANE_INIT_SEEDS.auto_width F4, V6, dom=0")
            st.f[4] = FState(dom=0, is_multilane=True, k_lanes=2)

            lines.append("  LANE_INIT_SEEDS.force_w64 F5, V6, dom=0")
            st.f[5] = FState(dom=0, is_multilane=True, k_lanes=2)

            lines.append("  LANE_STATE_INIT.from_frontier_seeds V7, F4, dom=0, k=2")
            st.v[7] = VState("lane_state_f32", n0, dom=0, is_lane_state=True, k_lanes=2)

        return lines

    # ---- Helper for Aliasing-Aware Destination Selection ----

    def _pick_dst(self, rng: random.Random, profile: str, sources: list[int], max_reg: int = 15) -> int:
        if profile == "aliasing" and sources and rng.random() < 0.65:
            return rng.choice(sources)
        return rng.randint(0, max_reg)

    # ---- Pillar 1 Instruction Emitter ----

    def _emit_pillar1_op(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker
    ) -> list[str]:
        candidates = [0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x14]
        if profile == "precision_3vl":
            code = rng.choices([0x0D, 0x0E, 0x0C, 0x05], weights=[0.35, 0.35, 0.15, 0.15])[0]
        elif profile == "aliasing":
            code = rng.choices([0x08, 0x09, 0x0C, 0x0D, 0x0E, 0x0A, 0x0B], weights=[0.2, 0.15, 0.25, 0.2, 0.1, 0.05, 0.05])[0]
        else:
            code = rng.choice(candidates)

        is_b = profile == "boundary"

        if code == 0x01:  # NOP
            return ["  NOP"]

        if code == 0x04:  # ASSERT (positive only — pick guaranteed-true assertions)
            kind = rng.choice(["not_null_r", "is_null_r", "empty_f", "nonempty_f", "zf_set", "zf_clear"])
            if kind == "not_null_r":
                not_null_r = [i for i, r in st.r.items() if not r.is_null]
                if not_null_r:
                    return [f"  ASSERT.not_null_r R{rng.choice(not_null_r)}, code=1"]
            elif kind == "is_null_r":
                null_r = [i for i, r in st.r.items() if r.is_null]
                if null_r:
                    return [f"  ASSERT.is_null_r R{rng.choice(null_r)}, code=2"]
            elif kind == "empty_f":
                return [
                    "  MOV.clear_f F15, dom=0",
                    "  ASSERT.empty_f F15, code=3",
                ]
            elif kind == "nonempty_f":
                return [
                    "  LOAD_CONST_POOL.frontier F15, pool=3, dom=0",
                    "  ASSERT.nonempty_f F15, code=4",
                ]
            elif kind == "zf_set":
                st.r[15] = RState("i64", False, True)
                return [
                    "  LOAD_CONST.i32 R15, 0",
                    "  ASSERT.zf_set code=5",
                ]
            elif kind == "zf_clear":
                st.r[15] = RState("i64", False, True)
                return [
                    "  LOAD_CONST.i32 R15, 1",
                    "  ASSERT.zf_clear code=6",
                ]
            return ["  NOP"]

        if code == 0x05:  # LOAD_CONST
            subop = rng.randint(0, 7)
            dst = rng.randint(0, 15)
            if subop == 0:
                v = rng.choice(BOUNDARY_I32 if is_b else NORMAL_I32)
                st.r[dst] = RState("i64", False, 0 <= v <= 3)
                return [f"  LOAD_CONST.i32 R{dst}, {v}"]
            if subop == 1:
                v = rng.choice(BOUNDARY_U32 if is_b else [0, 1, 2, 3, 10])
                st.r[dst] = RState("u64", False, 0 <= v <= 3)
                return [f"  LOAD_CONST.u32 R{dst}, {v}"]
            if subop == 2:
                v = rng.choice(BOUNDARY_F32 if is_b else NORMAL_F32)
                st.r[dst] = RState("f64", False, False)
                return [f"  LOAD_CONST.f32 R{dst}, {v}"]
            if subop == 3:
                v = rng.choice(BOUNDARY_F32 if is_b else NORMAL_F32)
                st.r[dst] = RState("f32_raw", False, False)
                return [f"  LOAD_CONST.f32_raw R{dst}, {v}"]
            if subop == 4:
                v = rng.choice(BOUNDARY_I64 if is_b else ["0", "1", "-1", "42", "99999999999"])
                st.r[dst] = RState("i64", False, v in ("0", "1", "2", "3"))
                return [f"  LOAD_CONST.i64.ext R{dst}, {v}"]
            if subop == 5:
                v = rng.choice(BOUNDARY_U64 if is_b else ["0", "1", "2", "100000000000"])
                st.r[dst] = RState("u64", False, v in ("0", "1", "2", "3"))
                return [f"  LOAD_CONST.u64.ext R{dst}, {v}"]
            if subop == 6:
                v = rng.choice(BOUNDARY_F64 if is_b else NORMAL_F64)
                st.r[dst] = RState("f64", False, False)
                return [f"  LOAD_CONST.f64.ext R{dst}, {v}"]
            st.r[dst] = RState("null", True, False)
            return [f"  LOAD_CONST.null R{dst}"]

        if code == 0x06:  # LOAD_CONST_POOL
            subop = rng.choice([0, 1, 2])
            if subop == 0:
                dst = rng.randint(0, 15)
                st.v[dst] = VState("i64", env.dom_cards[0], dom=0)
                return [f"  LOAD_CONST_POOL.vec V{dst}, pool=2"]
            if subop == 1:
                dom = rng.choice([0, 1])
                pid = 3 if dom == 0 else 4
                accum = False
                f_same = st.f_matching(dom=dom, multilane=False)
                if f_same and rng.random() < 0.45:
                    dst = rng.choice(f_same)
                    accum = True
                else:
                    dst = rng.randint(0, 15)
                st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
                mod = ".accum" if accum else ""
                return [f"  LOAD_CONST_POOL.frontier{mod} F{dst}, pool={pid}, dom={dom}"]
            dst = rng.randint(0, 15)
            pid = rng.choice([0, 1])
            st.r[dst] = RState("str", False, False)
            return [f"  LOAD_CONST_POOL.str R{dst}, pool={pid}"]

        if code == 0x07:  # LOAD_PARAM
            subop = rng.choice([0, 1, 2])
            if subop == 0:
                pid = rng.choice([0, 1, 2])
                meta = env.params[pid][1]
                dst = rng.randint(0, 15)
                ty = meta["type"]
                st.r[dst] = RState("null" if meta["is_null"] else ty, meta["is_null"], False)
                return [f"  LOAD_PARAM.scalar R{dst}, param={pid}, type={ty}"]
            if subop == 1:
                pid = rng.choice([3, 4])
                dom = env.params[pid][1]["dom"]
                f_same = st.f_matching(dom=dom, multilane=False)
                accum = bool(f_same) and rng.random() < 0.45
                dst = rng.choice(f_same) if accum else rng.randint(0, 15)
                st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
                mod = ".accum" if accum else ""
                return [f"  LOAD_PARAM.frontier{mod} F{dst}, param={pid}, dom={dom}"]
            pid = rng.choice([5, 6, 7, 8, 9])
            meta = env.params[pid][1]
            dst = rng.randint(0, 15)
            ty = meta["type"]
            asm_ty = "bool" if ty == "bool_3vl" else ty
            st.v[dst] = VState(ty, meta["len"], dom=0, is_node_ids_for_dom=meta.get("node_ids_dom"))
            return [f"  LOAD_PARAM.vec V{dst}, param={pid}, type={asm_ty}"]

        if code == 0x08:  # MOV
            subop = rng.randint(0, 5)
            if subop == 0:
                s1 = rng.randint(0, 15)
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = copy.copy(st.r[s1])
                return [f"  MOV.r R{dst}, R{s1}"]
            if subop == 1:
                fs = st.f_matching()
                if not fs:
                    return ["  NOP"]
                s1 = rng.choice(fs)
                dst = self._pick_dst(rng, profile, [s1])
                st.f[dst] = copy.copy(st.f[s1])
                return [f"  MOV.f F{dst}, F{s1}"]
            if subop == 2:
                vs = st.v_matching()
                if not vs:
                    return ["  NOP"]
                s1 = rng.choice(vs)
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = copy.copy(st.v[s1])
                return [f"  MOV.v V{dst}, V{s1}"]
            if subop == 3:
                dst = rng.randint(0, 15)
                st.r[dst] = RState("null", True, False)
                return [f"  MOV.clear_r R{dst}"]
            if subop == 4:
                dst = rng.randint(0, 15)
                dom = rng.choice([0, 1])
                st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
                return [f"  MOV.clear_f F{dst}, dom={dom}"]
            dst = rng.randint(0, 15)
            st.v[dst] = None
            return [f"  MOV.clear_v V{dst}"]

        if code == 0x09:  # SWAP
            subop = rng.choice([0, 1, 2])
            if subop == 0:
                a = rng.randint(0, 15)
                b = a if (profile == "aliasing" and rng.random() < 0.3) else rng.randint(0, 15)
                st.r[a], st.r[b] = st.r[b], st.r[a]
                return [f"  SWAP.r R{a}, R{b}"]
            if subop == 1:
                fs = st.f_matching()
                if not fs:
                    return ["  NOP"]
                a = rng.choice(fs)
                b = a if (profile == "aliasing" and rng.random() < 0.3) else rng.choice(fs)
                st.f[a], st.f[b] = st.f[b], st.f[a]
                return [f"  SWAP.f F{a}, F{b}"]
            vs = st.v_matching()
            if not vs:
                return ["  NOP"]
            a = rng.choice(vs)
            b = a if (profile == "aliasing" and rng.random() < 0.3) else rng.choice(vs)
            st.v[a], st.v[b] = st.v[b], st.v[a]
            return [f"  SWAP.v V{a}, V{b}"]

        if code == 0x0A:  # SPILL
            slot = rng.randint(0, 15)
            bank_choices = ["R"]
            if st.f_matching():
                bank_choices.append("F")
            if st.v_matching():
                bank_choices.append("V")
            bank = rng.choice(bank_choices)
            if bank == "R":
                s1 = rng.randint(0, 15)
                st.spills[slot] = ("R", copy.copy(st.r[s1]))
                return [f"  SPILL.r R{s1}, slot={slot}"]
            if bank == "F":
                s1 = rng.choice(st.f_matching())
                st.spills[slot] = ("F", copy.copy(st.f[s1]))
                return [f"  SPILL.f F{s1}, slot={slot}"]
            s1 = rng.choice(st.v_matching())
            st.spills[slot] = ("V", copy.copy(st.v[s1]))
            return [f"  SPILL.v V{s1}, slot={slot}"]

        if code == 0x0B:  # RELOAD
            if not st.spills:
                # Emit a spill first so reload is valid
                slot = rng.randint(0, 7)
                st.spills[slot] = ("R", copy.copy(st.r[0]))
                dst = rng.randint(0, 15)
                st.r[dst] = copy.copy(st.r[0])
                return [f"  SPILL.r R0, slot={slot}", f"  RELOAD.r R{dst}, slot={slot}"]
            slot = rng.choice(list(st.spills.keys()))
            bank, saved = st.spills[slot]
            dst = rng.randint(0, 15)
            if bank == "R":
                st.r[dst] = copy.copy(saved)
                return [f"  RELOAD.r R{dst}, slot={slot}"]
            if bank == "F":
                st.f[dst] = copy.copy(saved)
                return [f"  RELOAD.f F{dst}, slot={slot}"]
            st.v[dst] = copy.copy(saved)
            return [f"  RELOAD.v V{dst}, slot={slot}"]

        if code == 0x0C:  # SCALAR_INT_ALU
            subop = rng.randint(0, 15)
            sname = self.by_code[0x0C]["subops"][subop]
            int_regs = st.r_with_types({"i64", "u64", "null"})
            if not int_regs:
                int_regs = [0]
            s1 = rng.choice(int_regs)
            if subop == 10:  # not (unary)
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState(st.r[s1].ty if not st.r[s1].is_null else "null", st.r[s1].is_null, False)
                return [f"  SCALAR_INT_ALU.not R{dst}, R{s1}"]
            use_imm = rng.random() < 0.35
            if use_imm:
                # Avoid div/rem by zero in positive mode
                imm = rng.choice([1, 2, 3, 7, -3, 63] if subop in (3, 4, 5, 6) else (BOUNDARY_I32 if is_b else NORMAL_I32))
                dst = self._pick_dst(rng, profile, [s1])
                res_ty = "u64" if subop in (4, 6, 12) else ("i64" if not st.r[s1].is_null else "null")
                st.r[dst] = RState(res_ty, st.r[s1].is_null, False)
                return [f"  SCALAR_INT_ALU.{sname}.imm R{dst}, R{s1}, {imm}"]
            s2 = s1 if (profile == "aliasing" and rng.random() < 0.3) else rng.choice(int_regs)
            dst = self._pick_dst(rng, profile, [s1, s2])
            res_null = st.r[s1].is_null or st.r[s2].is_null
            res_ty = "null" if res_null else ("u64" if subop in (4, 6, 12) else "i64")
            st.r[dst] = RState(res_ty, res_null, False)
            return [f"  SCALAR_INT_ALU.{sname} R{dst}, R{s1}, R{s2}"]

        if code == 0x0D:  # SCALAR_FP_ALU
            subop = rng.randint(0, 15)
            sname = self.by_code[0x0D]["subops"][subop]
            fp_regs = st.r_with_types({"f64", "f32_raw", "null"})
            int_regs = st.r_with_types({"i64", "u64", "null"})
            if subop == 13:  # i64_to_f64
                s1 = rng.choice(int_regs or [0])
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState("null" if st.r[s1].is_null else "f64", st.r[s1].is_null, False)
                return [f"  SCALAR_FP_ALU.i64_to_f64 R{dst}, R{s1}"]
            if subop == 14:  # f64_to_i64
                s1 = rng.choice(fp_regs or [6])
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState("null" if st.r[s1].is_null else "i64", st.r[s1].is_null, False)
                return [f"  SCALAR_FP_ALU.f64_to_i64 R{dst}, R{s1}"]
            if subop in (4, 5, 6, 10, 11, 12):  # unary fp
                s1 = rng.choice(fp_regs or [6])
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState("null" if st.r[s1].is_null else "f64", st.r[s1].is_null, False)
                return [f"  SCALAR_FP_ALU.{sname} R{dst}, R{s1}"]
            if subop in (9, 15):  # ternary: fma, safe_div
                s1 = rng.choice(fp_regs or [6])
                s2 = rng.choice(fp_regs or [10])
                s3 = rng.choice(fp_regs or [2])
                dst = self._pick_dst(rng, profile, [s1, s2, s3])
                is_n = st.r[s1].is_null or st.r[s2].is_null or (subop == 9 and st.r[s3].is_null)
                st.r[dst] = RState("null" if is_n else "f64", is_n, False)
                return [f"  SCALAR_FP_ALU.{sname} R{dst}, R{s1}, R{s2}, R{s3}"]
            # binary fp
            s1 = rng.choice(fp_regs or [6])
            s2 = s1 if (profile == "aliasing" and rng.random() < 0.3) else rng.choice(fp_regs or [10])
            dst = self._pick_dst(rng, profile, [s1, s2])
            is_n = st.r[s1].is_null or st.r[s2].is_null
            st.r[dst] = RState("null" if is_n else "f64", is_n, False)
            return [f"  SCALAR_FP_ALU.{sname} R{dst}, R{s1}, R{s2}"]

        if code == 0x0E:  # SCALAR_3VL
            subop = rng.randint(0, 15)
            sname = self.by_code[0x0E]["subops"][subop]
            # Occasionally use canonical pseudo-commuted mnemonics
            pseudo_map = {2: "cmp_gt_i64", 3: "cmp_ge_i64", 4: "cmp_gt_u64", 5: "cmp_ge_u64", 8: "cmp_gt_f64", 9: "cmp_ge_f64"}
            if subop in pseudo_map and rng.random() < 0.25:
                sname = pseudo_map[subop]
            if subop in (0, 1, 2, 3, 4, 5):  # i64/u64 cmp
                pool = st.r_with_types({"i64", "u64", "null"}) or [0, 1]
                s1, s2 = rng.choice(pool), rng.choice(pool)
                dst = self._pick_dst(rng, profile, [s1, s2])
                is_n = st.r[s1].is_null or st.r[s2].is_null
                st.r[dst] = RState("null" if is_n else "bool_3vl", is_n, False)
                return [f"  SCALAR_3VL.{sname} R{dst}, R{s1}, R{s2}"]
            if subop in (6, 7, 8, 9):  # f64 cmp
                pool = st.r_with_types({"f64", "f32_raw", "null"}) or [6, 10]
                s1, s2 = rng.choice(pool), rng.choice(pool)
                dst = self._pick_dst(rng, profile, [s1, s2])
                is_n = st.r[s1].is_null or st.r[s2].is_null
                st.r[dst] = RState("null" if is_n else "bool_3vl", is_n, False)
                return [f"  SCALAR_3VL.{sname} R{dst}, R{s1}, R{s2}"]
            if subop in (10, 11):  # and, or
                pool = st.r_with_types({"bool_3vl", "null"}) or [11, 7]
                s1, s2 = rng.choice(pool), rng.choice(pool)
                dst = self._pick_dst(rng, profile, [s1, s2])
                st.r[dst] = RState("bool_3vl", False, False)
                return [f"  SCALAR_3VL.{sname} R{dst}, R{s1}, R{s2}"]
            if subop == 12:  # not
                pool = st.r_with_types({"bool_3vl", "null"}) or [11, 7]
                s1 = rng.choice(pool)
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState("null" if st.r[s1].is_null else "bool_3vl", st.r[s1].is_null, False)
                return [f"  SCALAR_3VL.not R{dst}, R{s1}"]
            if subop in (13, 14):  # is_null, is_not_null
                s1 = rng.randint(0, 15)
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState("bool_3vl", False, False)
                return [f"  SCALAR_3VL.{sname} R{dst}, R{s1}"]
            # 15: coalesce
            s1 = rng.randint(0, 15)
            s2 = rng.randint(0, 15)
            dst = self._pick_dst(rng, profile, [s1, s2])
            chosen = st.r[s2] if st.r[s1].is_null else st.r[s1]
            st.r[dst] = copy.copy(chosen)
            return [f"  SCALAR_3VL.coalesce R{dst}, R{s1}, R{s2}"]

        if code == 0x14:  # SET_MAX_DOP
            if rng.random() < 0.5:
                dop = rng.choice([1, 2, 4, 8])
                return [f"  SET_MAX_DOP.imm {dop}"]
            cand = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if cand:
                return [f"  SET_MAX_DOP.reg R{rng.choice(cand)}"]
            return ["  SET_MAX_DOP.imm 2"]

        return ["  NOP"]

    # ---- Pillar 2 Instruction Emitter ----

    def _emit_pillar2_op(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker
    ) -> list[str]:
        code = rng.choice([0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27])
        dom = rng.choice([0, 1])
        card = env.dom_cards[dom]

        if code == 0x20:  # FSEED
            subop = rng.randint(0, 4)
            sname = self.by_code[0x20]["subops"][subop]
            if subop == 0:  # empty
                dst = rng.randint(0, 15)
                st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
                return [f"  FSEED.empty F{dst}, dom={dom}"]
            if subop == 1:  # single_imm
                node = rng.randint(0, card - 1)
                f_same = st.f_matching(dom=dom, multilane=False)
                accum = bool(f_same) and rng.random() < 0.45
                dst = rng.choice(f_same) if accum else rng.randint(0, 15)
                st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
                mod = ".accum" if accum else ""
                return [f"  FSEED.single_imm{mod} F{dst}, node={node}, dom={dom}"]
            # subops 2, 3, 4 take R register with valid node id
            small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if not small_r:
                return [f"  FSEED.single_imm F{rng.randint(0, 15)}, node=0, dom={dom}"]
            s1 = rng.choice(small_r)
            if subop in (3, 4):
                f_same = st.f_matching(dom=0, multilane=False)
                if not f_same:
                    return [f"  FSEED.single_reg F0, R{s1}, dom=0"]
                dst = rng.choice(f_same)
                return [f"  FSEED.{sname} F{dst}, R{s1}, dom=0"]
            f_same = st.f_matching(dom=0, multilane=False)
            accum = bool(f_same) and rng.random() < 0.4
            dst = rng.choice(f_same) if accum else rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            mod = ".accum" if accum else ""
            return [f"  FSEED.single_reg{mod} F{dst}, R{s1}, dom=0"]

        if code == 0x21:  # INIT_DOMAIN_SPAN
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
            return [f"  INIT_DOMAIN_SPAN.all_span F{dst}, dom={dom}"]

        if code == 0x22:  # SET_OP
            subop = rng.randint(0, 3)
            sname = self.by_code[0x22]["subops"][subop]
            # Pick either 1-bit pair or multi-lane pair with matching dom & k_lanes
            ml_dom0 = st.f_matching(dom=0, multilane=True)
            if ml_dom0 and rng.random() < 0.35:
                s1 = rng.choice(ml_dom0)
                same_k = [i for i in ml_dom0 if st.f[i].k_lanes == st.f[s1].k_lanes]
                s2 = rng.choice(same_k)
                dst = self._pick_dst(rng, profile, [s1, s2])
                st.f[dst] = copy.copy(st.f[s1])
                return [f"  SET_OP.{sname} F{dst}, F{s1}, F{s2}"]
            f1b = st.f_matching(dom=dom, multilane=False)
            if not f1b:
                f1b = st.f_matching(dom=0, multilane=False)
                dom = 0
            if not f1b:
                return ["  NOP"]
            s1 = rng.choice(f1b)
            s2 = s1 if (profile == "aliasing" and rng.random() < 0.35) else rng.choice(f1b)
            dst = self._pick_dst(rng, profile, [s1, s2])
            st.f[dst] = FState(dom=dom, is_multilane=False, k_lanes=1)
            return [f"  SET_OP.{sname} F{dst}, F{s1}, F{s2}"]

        if code == 0x23:  # SET_COMPLEMENT
            f1b = st.f_matching(multilane=False)
            if not f1b:
                return ["  NOP"]
            s1 = rng.choice(f1b)
            dst = self._pick_dst(rng, profile, [s1])
            st.f[dst] = copy.copy(st.f[s1])
            return [f"  SET_COMPLEMENT F{dst}, F{s1}"]

        if code == 0x24:  # SET_CARDINALITY
            ml = st.f_matching(multilane=True)
            f1b = st.f_matching(multilane=False)
            if ml and rng.random() < 0.5:
                s1 = rng.choice(ml)
                subop = rng.choice([1, 2, 3, 4])
                sname = self.by_code[0x24]["subops"][subop]
                dst = rng.randint(0, 15)
                st.r[dst] = RState("u64", False, False)
                if subop == 4:
                    lane = rng.randint(0, max(0, st.f[s1].k_lanes - 1))
                    return [f"  SET_CARDINALITY.single_lane R{dst}, F{s1}, lane={lane}"]
                return [f"  SET_CARDINALITY.{sname} R{dst}, F{s1}"]
            if not f1b:
                return ["  NOP"]
            s1 = rng.choice(f1b)
            dst = rng.randint(0, 15)
            st.r[dst] = RState("u64", False, False)
            return [f"  SET_CARDINALITY.card_1bit R{dst}, F{s1}"]

        if code == 0x25:  # SET_CONTAINS
            small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if not small_r:
                return ["  NOP"]
            s2 = rng.choice(small_r)
            ml = st.f_matching(dom=0, multilane=True)
            if ml and rng.random() < 0.45:
                s1 = rng.choice(ml)
                dst = self._pick_dst(rng, profile, [s2])
                st.r[dst] = RState("u64", False, False)
                return [f"  SET_CONTAINS.get_lane_mask R{dst}, F{s1}, R{s2}"]
            f1b = st.f_matching(dom=0, multilane=False)
            if not f1b:
                return ["  NOP"]
            s1 = rng.choice(f1b)
            dst = self._pick_dst(rng, profile, [s2])
            st.r[dst] = RState("bool_3vl", False, False)
            return [f"  SET_CONTAINS.contains_1bit R{dst}, F{s1}, R{s2}"]

        if code == 0x26:  # SET_CONVERT
            subop = rng.randint(0, 3)
            if subop == 0:  # f2v_ids
                f1b = st.f_matching(multilane=False)
                if not f1b:
                    return ["  NOP"]
                s1 = rng.choice(f1b)
                dst = rng.randint(0, 15)
                st.v[dst] = VState("u64", env.dom_cards[st.f[s1].dom], dom=st.f[s1].dom, is_node_ids_for_dom=st.f[s1].dom)
                return [f"  SET_CONVERT.f2v_ids V{dst}, F{s1}"]
            if subop == 1:  # v2f_ids
                v_ids = st.v_matching(node_ids_dom=0)
                if not v_ids:
                    return ["  NOP"]
                s1 = rng.choice(v_ids)
                f_same = st.f_matching(dom=0, multilane=False)
                accum = bool(f_same) and rng.random() < 0.4
                dst = rng.choice(f_same) if accum else rng.randint(0, 15)
                st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
                mod = ".accum" if accum else ""
                return [f"  SET_CONVERT.v2f_ids{mod} F{dst}, V{s1}, dom=0"]
            if subop == 2:  # b2f
                v_bools = st.v_matching(types={"bool_3vl"}, length=env.dom_cards[0])
                if not v_bools:
                    return ["  NOP"]
                s1 = rng.choice(v_bools)
                f_same = st.f_matching(dom=0, multilane=False)
                accum = bool(f_same) and rng.random() < 0.4
                dst = rng.choice(f_same) if accum else rng.randint(0, 15)
                st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
                mod = ".accum" if accum else ""
                return [f"  SET_CONVERT.b2f{mod} F{dst}, V{s1}, dom=0"]
            # 3: f2b
            f1b = st.f_matching(dom=0, multilane=False)
            if not f1b:
                return ["  NOP"]
            s1 = rng.choice(f1b)
            dst = rng.randint(0, 15)
            st.v[dst] = VState("bool_3vl", env.dom_cards[0], dom=0)
            return [f"  SET_CONVERT.f2b V{dst}, F{s1}"]

        if code == 0x27:  # STABLE_CHECK
            if rng.random() < 0.5:
                f1b = st.f_matching(dom=0, multilane=False)
                if len(f1b) >= 1:
                    s1 = rng.choice(f1b)
                    s2 = rng.choice(f1b)
                    return [f"  STABLE_CHECK.frontier_eq F{s1}, F{s2}"]
            vf = st.v_matching(types={"f64"}, length=env.dom_cards[0])
            if vf:
                s1 = rng.choice(vf)
                s2 = rng.choice(vf)
                dst = rng.randint(0, 15)
                st.r[dst] = RState("f64", False, False)
                fp_r = st.r_with_types({"f64"}, non_null_only=True)
                if fp_r and rng.random() < 0.5:
                    return [f"  STABLE_CHECK.vec_l1 R{dst}, V{s1}, V{s2}, R{rng.choice(fp_r)}"]
                return [f"  STABLE_CHECK.vec_l1 R{dst}, V{s1}, V{s2}"]
            return ["  NOP"]

        return ["  NOP"]

    # ---- Pillar 3 Instruction Emitter ----

    def _emit_morsel_block(
        self, rng: random.Random, env: FixtureEnv, st: V11RegTracker, label_id: int
    ) -> list[str]:
        """Emit a valid lexical WALK_MORSEL or FILTER_MORSEL block."""
        lines: list[str] = []
        f0_list = st.f_matching(dom=0, multilane=False)
        if not f0_list:
            lines.append("  INIT_DOMAIN_SPAN.all_span F0, dom=0")
            st.f[0] = FState(dom=0, is_multilane=False, k_lanes=1)
            f0_list = [0]

        is_walk = rng.random() < 0.65
        skip_lbl = f"m_skip_{label_id}"
        # Prepare a sink vector V0 (f64 len n0) and V1 (u64 len n0)
        n0 = env.dom_cards[0]
        f64_sinks = st.v_matching(types={"f64"}, length=n0)
        u64_sinks = st.v_matching(types={"u64"}, length=n0)
        sink_f64 = f64_sinks[0] if f64_sinks else 0
        if not f64_sinks:
            lines.append("  LOAD_PARAM.vec V0, param=5, type=f64")
            st.v[0] = VState("f64", n0, dom=0)
        sink_u64 = u64_sinks[0] if u64_sinks else 2
        if not u64_sinks:
            lines.append("  LOAD_PARAM.vec V2, param=7, type=u64")
            st.v[2] = VState("u64", n0, dom=0, is_node_ids_for_dom=0)

        # Ensure sink regs are < 8 (outer bank)
        if sink_f64 >= 8:
            lines.append(f"  MOV.v V0, V{sink_f64}")
            st.v[0] = copy.copy(st.v[sink_f64])
            sink_f64 = 0
        if sink_u64 >= 8:
            lines.append(f"  MOV.v V2, V{sink_u64}")
            st.v[2] = copy.copy(st.v[sink_u64])
            sink_u64 = 2

        src_f = rng.choice([f for f in f0_list if f < 8] or [0])
        if st.f[src_f] is None or st.f[src_f].dom != 0:
            lines.append(f"  INIT_DOMAIN_SPAN.all_span F{src_f}, dom=0")
            st.f[src_f] = FState(dom=0, is_multilane=False, k_lanes=1)

        # Choose outer frontier dst != src_f and < 8
        out_f = rng.choice([i for i in range(8) if i != src_f])

        if is_walk:
            dir_sub = rng.choice(["fwd", "rev"])
            use_mask = rng.random() < 0.4
            if use_mask:
                mask_f = rng.choice([i for i in f0_list if i < 8] or [src_f])
                lines.append(f"  WALK_MORSEL.{dir_sub}.mask F{src_f}, F{mask_f}, rel=0")
            else:
                lines.append(f"  WALK_MORSEL.{dir_sub} F{src_f}, rel=0")

            id_reg = "M1" if dir_sub == "fwd" else "M0"
            if 4 in self.allowed_pillars:
                lines.append("    GATHER_EDGE_ATTR.vec M4, M2, rel=0, attr=0")
                fp_r = st.r_with_types({"f64"}, non_null_only=True)
                cmp_r = rng.choice(fp_r) if fp_r else 6
                lines.append(f"    VEC_CMP.ge.scalar M5, M4, R{cmp_r}")
                lines.append("    SET_CONVERT.b2f MF1, M5")
                if rng.random() < 0.5:
                    lines.append(f"    JCC.jempty_f MF1, {skip_lbl}")
                lines.append(f"    MORSEL_COMMIT.yield_frontier.mask F{out_f}, {id_reg}, MF1, dom=0")
                st.f[out_f] = FState(dom=0, is_multilane=False, k_lanes=1)
                commit_sub = rng.choice(["yield_lane_or", "scatter_add", "scatter_min", "scatter_max", "scatter_min_src", "scatter_argmin", "scatter_argmax", "count_add_r"])
                if commit_sub == "yield_lane_or":
                    lane_f = rng.choice([i for i in range(8) if i != src_f and i != out_f])
                    st.f[lane_f] = FState(dom=0, is_multilane=False, k_lanes=1)
                    lines.append(f"    MORSEL_COMMIT.yield_lane_or.mask F{lane_f}, {id_reg}, M3, MF1, dom=0")
                elif commit_sub in ("scatter_add", "scatter_min", "scatter_max"):
                    lines.append(f"    MORSEL_COMMIT.{commit_sub}.mask V{sink_f64}, {id_reg}, M4, MF1")
                elif commit_sub in ("scatter_min_src", "scatter_argmin", "scatter_argmax"):
                    lines.append(f"    MORSEL_COMMIT.{commit_sub}.mask V{sink_u64}, {id_reg}, M0, MF1")
                else:
                    cnt_r = rng.randint(0, 7)
                    st.r[cnt_r] = RState("i64", False, False)
                    lines.append(f"    MORSEL_COMMIT.count_add_r.mask R{cnt_r}, MF1")
                lines.append(f"{skip_lbl}:")
            else:
                lines.append(f"    MORSEL_COMMIT.yield_frontier F{out_f}, {id_reg}, dom=0")
                st.f[out_f] = FState(dom=0, is_multilane=False, k_lanes=1)
                cnt_r = rng.randint(0, 7)
                st.r[cnt_r] = RState("i64", False, False)
                lines.append(f"    MORSEL_COMMIT.count_add_r R{cnt_r}")
            lines.append("  MORSEL_END")
        else:
            lines.append(f"  FILTER_MORSEL F{src_f}")
            if 4 in self.allowed_pillars:
                lines.append("    GATHER_NODE_ATTR.vec M4, M0, dom=0, attr=1")
                fp_r = st.r_with_types({"f64"}, non_null_only=True)
                cmp_r = rng.choice(fp_r) if fp_r else 6
                lines.append(f"    VEC_CMP.gt.scalar M5, M4, R{cmp_r}")
                lines.append("    SET_CONVERT.b2f MF1, M5")
                lines.append(f"    MORSEL_COMMIT.yield_frontier.mask F{out_f}, M0, MF1, dom=0")
                st.f[out_f] = FState(dom=0, is_multilane=False, k_lanes=1)
            else:
                lines.append(f"    MORSEL_COMMIT.yield_frontier F{out_f}, M0, dom=0")
                st.f[out_f] = FState(dom=0, is_multilane=False, k_lanes=1)
            lines.append("  MORSEL_END")
        return lines

    def _emit_pillar3_op(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker, label_id: int
    ) -> list[str]:
        if profile == "morsel" and rng.random() < 0.55:
            return self._emit_morsel_block(rng, env, st, label_id)

        code = rng.choice([0x30, 0x31, 0x33, 0x34, 0x35, 0x39, 0x3A, 0x3B])
        if code == 0x35:
            return self._emit_morsel_block(rng, env, st, label_id)

        f0_list = st.f_matching(dom=0, multilane=False)
        f1_list = st.f_matching(dom=1, multilane=False)

        if code == 0x30:  # WALK
            subop = rng.randint(0, 7)
            sname = self.by_code[0x30]["subops"][subop]
            # Fixpoint/both require endomorphic rel 0
            rel = 0 if subop >= 2 else rng.choice([0, 1])
            src_dom, tgt_dom, _ = env.relations[rel]
            in_dom = tgt_dom if subop in (1, 5, 6) else src_dom
            out_dom = src_dom if subop in (1, 5, 6) else tgt_dom
            src_pool = st.f_matching(dom=in_dom, multilane=False)
            if not src_pool:
                return ["  NOP"]
            s1 = rng.choice(src_pool)
            out_pool = st.f_matching(dom=out_dom, multilane=False)
            use_accum = bool(out_pool) and rng.random() < 0.3
            use_mask = bool(out_pool) and rng.random() < 0.35
            dst = rng.choice(out_pool) if use_accum else self._pick_dst(rng, profile, [s1] if in_dom == out_dom else [])
            st.f[dst] = FState(dom=out_dom, is_multilane=False, k_lanes=1)
            mods = (".accum" if use_accum else "") + (".mask" if use_mask else "")
            mask_arg = f", F{rng.choice(out_pool)}" if use_mask else ""
            k_arg = f", k={rng.randint(1, 3)}" if subop == 7 else ""
            return [f"  WALK.{sname}{mods} F{dst}, F{s1}{mask_arg}, rel={rel}{k_arg}"]

        if code == 0x31:  # WALK_FILTERED
            subop = rng.randint(0, 15)
            sname = self.by_code[0x31]["subops"][subop]
            is_rev = subop >= 8
            is_null_op = (subop % 8) in (6, 7)
            if not f0_list:
                return ["  NOP"]
            s1 = rng.choice(f0_list)
            dst = self._pick_dst(rng, profile, [s1])
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            use_mask = (not is_null_op) and rng.random() < 0.3
            mods = ".mask" if use_mask else ""
            mask_arg = f", F{rng.choice(f0_list)}" if use_mask else ""
            if is_null_op:
                return [f"  WALK_FILTERED.{sname}{mods} F{dst}, F{s1}{mask_arg}, rel=0, attr=0"]
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            s2 = rng.choice(fp_r) if fp_r else 6
            return [f"  WALK_FILTERED.{sname}{mods} F{dst}, F{s1}, R{s2}{mask_arg}, rel=0, attr=0"]

        if code == 0x33:  # DEGREE
            subop = rng.randint(0, 5)
            sname = self.by_code[0x33]["subops"][subop]
            if subop in (0, 1):
                if not f0_list:
                    return ["  NOP"]
                s1 = rng.choice(f0_list)
                dst = rng.randint(0, 15)
                st.r[dst] = RState("u64", False, False)
                return [f"  DEGREE.{sname} R{dst}, F{s1}, rel=0"]
            if subop in (2, 3):
                dst = rng.randint(0, 15)
                st.v[dst] = VState("u64", env.dom_cards[0], dom=0)
                return [f"  DEGREE.{sname} V{dst}, rel=0"]
            small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if not small_r:
                return ["  NOP"]
            s1 = rng.choice(small_r)
            dst = self._pick_dst(rng, profile, [s1])
            st.r[dst] = RState("u64", False, False)
            return [f"  DEGREE.{sname} R{dst}, R{s1}, rel=0"]

        if code == 0x34:  # SPMV
            subop = rng.randint(0, 8)
            sname = self.by_code[0x34]["subops"][subop]
            if not f0_list:
                return ["  NOP"]
            s1 = rng.choice(f0_list)
            n0 = env.dom_cards[0]
            if subop == 7:  # fwd_first_src
                dst = rng.randint(0, 15)
                st.v[dst] = VState("u64", n0, dom=0, is_node_ids_for_dom=0)
                return [f"  SPMV.fwd_first_src V{dst}, F{s1}, rel=0, attr=0xFFFF"]
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            s2 = rng.choice(vf)
            use_accum = (subop < 7) and rng.random() < 0.3
            use_mask = rng.random() < 0.3
            dst = rng.choice(vf) if use_accum else self._pick_dst(rng, profile, [s2])
            res_ty = "u64" if subop == 8 else "f64"
            st.v[dst] = VState(res_ty, n0, dom=0, is_node_ids_for_dom=(0 if subop == 8 else None))
            mods = (".accum" if use_accum else "") + (".mask" if use_mask else "")
            mask_arg = f", F{rng.choice(f0_list)}" if use_mask else ""
            return [f"  SPMV.{sname}{mods} V{dst}, F{s1}, V{s2}{mask_arg}, rel=0, attr=0"]

        if code == 0x39:  # SAMPLE_NEIGHBORS
            subop = rng.randint(0, 3)
            sname = self.by_code[0x39]["subops"][subop]
            seed_r = st.r_with_types({"u64", "i64"}, non_null_only=True)
            if not seed_r:
                return ["  NOP"]
            s2 = rng.choice(seed_r)
            k = rng.randint(1, 3)
            if subop == 0:
                if not f0_list:
                    return ["  NOP"]
                s1 = rng.choice(f0_list)
                dst = self._pick_dst(rng, profile, [s1])
                st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
                return [f"  SAMPLE_NEIGHBORS.frontier_wor F{dst}, F{s1}, R{s2}, rel=0, k={k}"]
            v_ids = st.v_matching(node_ids_dom=0)
            if not v_ids:
                return ["  NOP"]
            s1 = rng.choice(v_ids)
            dst = self._pick_dst(rng, profile, [s1])
            out_len = st.v[s1].length * k if subop == 1 else st.v[s1].length
            st.v[dst] = VState("u64", out_len, dom=0, is_node_ids_for_dom=0)
            return [f"  SAMPLE_NEIGHBORS.{sname} V{dst}, V{s1}, R{s2}, rel=0, k={k}"]

        if code == 0x3A:  # HAS_TOPOLOGY
            subop = rng.randint(0, 9)
            sname = self.by_code[0x3A]["subops"][subop]
            dst = rng.randint(0, 15)
            st.r[dst] = RState("u64" if subop in (8, 9) else "bool_3vl", False, False)
            lo = 0 if subop in (8, 9) else rng.choice([0, 1, 99])
            hi = 0
            return [f"  HAS_TOPOLOGY.{sname} R{dst}, lo={lo}, hi={hi}"]

        if code == 0x3B:  # GRAPH_KERNEL
            subop = rng.randint(0, 4)
            sname = self.by_code[0x3B]["subops"][subop]
            use_mask = bool(f0_list) and rng.random() < 0.35
            mod = ".mask" if use_mask else ""
            mask_arg = f", F{rng.choice(f0_list)}" if use_mask else ""
            if subop in (0, 1, 4):
                dst = rng.randint(0, 15)
                st.v[dst] = VState("u64", env.dom_cards[0], dom=0)
                return [f"  GRAPH_KERNEL.{sname}{mod} V{dst}{mask_arg}, rel=0"]
            if subop == 2:
                dst = rng.randint(0, 15)
                st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
                return [f"  GRAPH_KERNEL.kcore_frontier{mod} F{dst}{mask_arg}, rel=0, k={rng.randint(1, 3)}"]
            dst = rng.randint(0, 15)
            st.r[dst] = RState("u64", False, False)
            return [f"  GRAPH_KERNEL.tri_count_total{mod} R{dst}{mask_arg}, rel=0"]

        return ["  NOP"]

    # ---- Pillar 4 Instruction Emitter ----

    def _emit_pillar4_op(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker
    ) -> list[str]:
        if profile == "precision_3vl":
            code = rng.choice([0x45, 0x46, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x52])
        else:
            code = rng.choice([
                0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
                0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
                0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
            ])

        n0 = env.dom_cards[0]
        f0_list = st.f_matching(dom=0, multilane=False)

        if code == 0x40:  # LOAD_COLUMN
            if rng.random() < 0.6:
                attr = rng.choice([0, 1, 2])
                ty = env.node_attrs[(0, attr)]
                dst = rng.randint(0, 15)
                st.v[dst] = VState(ty, n0, dom=0)
                return [f"  LOAD_COLUMN.node_attr V{dst}, dom=0, attr={attr}"]
            attr = rng.choice([0, 1])
            ty = env.edge_attrs[(0, attr)]
            dst = rng.randint(0, 15)
            st.v[dst] = VState(ty, env.relations[0][2])
            return [f"  LOAD_COLUMN.edge_attr V{dst}, rel=0, attr={attr}"]

        if code == 0x41:  # GATHER_NODE_ATTR
            subop = rng.choice([0, 1, 2])
            attr = rng.choice([0, 1, 2])
            ty = env.node_attrs[(0, attr)]
            if subop == 0:
                v_ids = st.v_matching(node_ids_dom=0)
                if not v_ids:
                    return ["  NOP"]
                s1 = rng.choice(v_ids)
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = VState(ty, st.v[s1].length, dom=0)
                return [f"  GATHER_NODE_ATTR.vec V{dst}, V{s1}, dom=0, attr={attr}"]
            if subop == 1:
                if not f0_list:
                    return ["  NOP"]
                s1 = rng.choice(f0_list)
                dst = rng.randint(0, 15)
                st.v[dst] = VState(ty, n0, dom=0)
                return [f"  GATHER_NODE_ATTR.frontier_compact V{dst}, F{s1}, dom=0, attr={attr}"]
            small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if not small_r:
                return ["  NOP"]
            s1 = rng.choice(small_r)
            dst = self._pick_dst(rng, profile, [s1])
            st.r[dst] = RState(ty, False, False)
            return [f"  GATHER_NODE_ATTR.scalar R{dst}, R{s1}, dom=0, attr={attr}"]

        if code == 0x42:  # GATHER_EDGE_ATTR
            if rng.random() < 0.5:
                small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
                if small_r and env.relations[0][2] >= 4:
                    s1 = rng.choice(small_r)
                    dst = self._pick_dst(rng, profile, [s1])
                    st.r[dst] = RState("f64", False, False)
                    return [f"  GATHER_EDGE_ATTR.scalar R{dst}, R{s1}, rel=0, attr=0"]
            return ["  LOAD_COLUMN.edge_attr V0, rel=0, attr=0"]

        if code == 0x43:  # GATHER_VEC
            v_ids = st.v_matching(node_ids_dom=0)
            v_src = st.v_matching(types={"f64", "i64", "u64", "bool_3vl", "str"}, length=n0)
            if not v_ids or not v_src:
                return ["  NOP"]
            s1 = rng.choice(v_src)
            s2 = rng.choice(v_ids)
            dst = self._pick_dst(rng, profile, [s1, s2])
            st.v[dst] = VState(st.v[s1].ty, st.v[s2].length)
            return [f"  GATHER_VEC V{dst}, V{s1}, V{s2}"]

        if code == 0x44:  # NODE_FILTER
            if not f0_list:
                return ["  NOP"]
            subop = rng.randint(0, 8)
            sname = self.by_code[0x44]["subops"][subop]
            s1 = rng.choice(f0_list)
            dst = self._pick_dst(rng, profile, [s1])
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            if subop in (7, 8):
                return [f"  NODE_FILTER.{sname} F{dst}, F{s1}, dom=0, attr=1"]
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            if not fp_r:
                return ["  NOP"]
            r1 = rng.choice(fp_r)
            if subop == 6:  # between
                r2 = rng.choice(fp_r)
                return [f"  NODE_FILTER.between.scalar F{dst}, F{s1}, R{r1}, R{r2}, dom=0, attr=1"]
            return [f"  NODE_FILTER.{sname}.scalar F{dst}, F{s1}, R{r1}, dom=0, attr=1"]

        if code == 0x45:  # VEC_CMP
            subop = rng.randint(0, 7)
            sname = self.by_code[0x45]["subops"][subop]
            if subop == 7:  # in_set
                v_ids = st.v_matching(node_ids_dom=0)
                if not v_ids or not f0_list:
                    return ["  NOP"]
                s1 = rng.choice(v_ids)
                s2 = rng.choice(f0_list)
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = VState("bool_3vl", st.v[s1].length, dom=0)
                return [f"  VEC_CMP.in_set V{dst}, V{s1}, F{s2}"]
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            s1 = rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("bool_3vl", n0, dom=0)
            if subop == 6:  # between
                fp_r = st.r_with_types({"f64"}, non_null_only=True)
                if fp_r and rng.random() < 0.5:
                    return [f"  VEC_CMP.between.scalar V{dst}, V{s1}, R{rng.choice(fp_r)}, R{rng.choice(fp_r)}"]
                return [f"  VEC_CMP.between V{dst}, V{s1}, V{rng.choice(vf)}, V{rng.choice(vf)}"]
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            if fp_r and rng.random() < 0.5:
                return [f"  VEC_CMP.{sname}.scalar V{dst}, V{s1}, R{rng.choice(fp_r)}"]
            return [f"  VEC_CMP.{sname} V{dst}, V{s1}, V{rng.choice(vf)}"]

        if code == 0x46:  # VEC_3VL_OP
            subop = rng.randint(0, 11)
            sname = self.by_code[0x46]["subops"][subop]
            if subop == 11:  # assert_finite on a freshly initialized finite vector
                st.v[15] = VState("f64", n0, dom=0)
                return [
                    f"  VEC_INIT.f64 V15, len={n0}",
                    "  VEC_3VL_OP.assert_finite V15",
                ]
            if subop in (8, 9, 10):  # is_nan, is_inf, is_finite
                vf = st.v_matching(types={"f64", "f32"}, length=n0)
                if not vf:
                    return ["  NOP"]
                s1 = rng.choice(vf)
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = VState("bool_3vl", n0, dom=0)
                return [f"  VEC_3VL_OP.{sname} V{dst}, V{s1}"]
            vb = st.v_matching(types={"bool_3vl"}, length=n0)
            if not vb:
                return ["  NOP"]
            s1 = rng.choice(vb)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("bool_3vl", n0, dom=0)
            if subop in (0, 1, 3):  # binary 3vl
                return [f"  VEC_3VL_OP.{sname} V{dst}, V{s1}, V{rng.choice(vb)}"]
            return [f"  VEC_3VL_OP.{sname} V{dst}, V{s1}"]

        if code == 0x47:  # VEC_STR_PRED
            vs = st.v_matching(types={"str"}, length=n0)
            if not vs:
                return ["  NOP"]
            subop = rng.randint(0, 9)
            sname = self.by_code[0x47]["subops"][subop]
            s1 = rng.choice(vs)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("bool_3vl", n0, dom=0)
            if subop == 9:  # len_eq takes i64 vec or scalar
                vi = st.v_matching(types={"i64"}, length=n0)
                if vi:
                    return [f"  VEC_STR_PRED.len_eq V{dst}, V{s1}, V{rng.choice(vi)}"]
                return ["  NOP"]
            str_r = st.r_with_types({"str"}, non_null_only=True)
            if str_r and rng.random() < 0.5:
                return [f"  VEC_STR_PRED.{sname}.scalar V{dst}, V{s1}, R{rng.choice(str_r)}"]
            return [f"  VEC_STR_PRED.{sname} V{dst}, V{s1}, V{rng.choice(vs)}"]

        if code == 0x48:  # VEC_MATH_ARITH
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            subop = rng.randint(0, 15)
            sname = self.by_code[0x48]["subops"][subop]
            s1 = rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("f64", n0, dom=0)
            if subop >= 8:
                return [f"  VEC_MATH_ARITH.{sname} V{dst}, V{s1}"]
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            if fp_r and rng.random() < 0.45:
                return [f"  VEC_MATH_ARITH.{sname}.scalar V{dst}, V{s1}, R{rng.choice(fp_r)}"]
            return [f"  VEC_MATH_ARITH.{sname} V{dst}, V{s1}, V{rng.choice(vf)}"]

        if code in (0x49, 0x4A, 0x4B):  # VEC_MATH_EXP_LOG, TRIG_HYP, ACT
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            op = self.by_code[code]
            subop = rng.choice(list(op["subops"].keys()))
            sname = op["subops"][subop]
            s1 = rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("f64", n0, dom=0)
            is_bin = (code == 0x49 and subop in (8, 10)) or (code == 0x4A and subop == 6) or (code == 0x4B and subop == 1)
            if is_bin:
                fp_r = st.r_with_types({"f64"}, non_null_only=True)
                if fp_r and rng.random() < 0.5:
                    return [f"  {op['mnemonic']}.{sname}.scalar V{dst}, V{s1}, R{rng.choice(fp_r)}"]
                return [f"  {op['mnemonic']}.{sname} V{dst}, V{s1}, V{rng.choice(vf)}"]
            return [f"  {op['mnemonic']}.{sname} V{dst}, V{s1}"]

        if code == 0x4C:  # VEC_INT_BITWISE
            vi = st.v_matching(types={"i64", "u64"}, length=n0)
            if not vi:
                return ["  NOP"]
            subop = rng.randint(0, 13)
            sname = self.by_code[0x4C]["subops"][subop]
            s1 = rng.choice(vi)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState(st.v[s1].ty, n0, dom=0)
            if subop in (3, 10, 11, 12, 13):
                return [f"  VEC_INT_BITWISE.{sname} V{dst}, V{s1}"]
            same_ty = [i for i in vi if st.v[i].ty == st.v[s1].ty]
            int_r = st.r_with_types({"i64", "u64"}, non_null_only=True)
            if int_r and rng.random() < 0.45:
                return [f"  VEC_INT_BITWISE.{sname}.scalar V{dst}, V{s1}, R{rng.choice(int_r)}"]
            return [f"  VEC_INT_BITWISE.{sname} V{dst}, V{s1}, V{rng.choice(same_ty)}"]

        if code == 0x4D:  # VEC_TERNARY
            subop = rng.randint(0, 9)
            sname = self.by_code[0x4D]["subops"][subop]
            if subop in (4, 5):  # clamp_v, clamp_r with guaranteed lo <= hi
                vf = st.v_matching(types={"f64"}, length=n0)
                if not vf:
                    return ["  NOP"]
                s1 = rng.choice(vf)
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = VState("f64", n0, dom=0)
                if subop == 4:
                    st.v[14] = VState("f64", n0, dom=0)
                    st.v[15] = VState("f64", n0, dom=0)
                    return [
                        "  LOAD_CONST.f64.ext R14, -10.0",
                        "  LOAD_CONST.f64.ext R15, 10.0",
                        f"  VEC_INIT.f64.scalar V14, R14, len={n0}",
                        f"  VEC_INIT.f64.scalar V15, R15, len={n0}",
                        f"  VEC_TERNARY.clamp_v V{dst}, V{s1}, V14, V15",
                    ]
                st.v[14] = VState("f64", n0, dom=0)
                return [
                    "  LOAD_CONST.f64.ext R14, -10.0",
                    "  LOAD_CONST.f64.ext R15, 10.0",
                    f"  VEC_INIT.f64.scalar V14, R14, len={n0}",
                    f"  VEC_TERNARY.clamp_r V{dst}, V{s1}, V14, R15",
                ]
            if subop in (8, 9):
                vi = st.v_matching(types={"i64"}, length=n0)
                if not vi:
                    return ["  NOP"]
                s1, s2 = rng.choice(vi), rng.choice(vi)
                dst = self._pick_dst(rng, profile, [s1, s2])
                if subop == 8:
                    ir = st.r_with_types({"i64"}, non_null_only=True)
                    if not ir:
                        return ["  NOP"]
                    st.v[dst] = VState("bool_3vl", n0, dom=0)
                    return [f"  VEC_TERNARY.time_valid_at_r V{dst}, V{s1}, V{s2}, R{rng.choice(ir)}"]
                # mul_div_i64_r: ensure non-zero divisor
                st.v[dst] = VState("i64", n0, dom=0)
                return [
                    "  LOAD_CONST.i32 R15, 10",
                    f"  VEC_TERNARY.mul_div_i64_r V{dst}, V{s1}, V{s2}, R15",
                ]
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            s1, s2 = rng.choice(vf), rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s1, s2])
            st.v[dst] = VState("f64", n0, dom=0)
            if subop in (0, 2, 6):
                return [f"  VEC_TERNARY.{sname} V{dst}, V{s1}, V{s2}, V{rng.choice(vf)}"]
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            if not fp_r:
                return ["  NOP"]
            return [f"  VEC_TERNARY.{sname} V{dst}, V{s1}, V{s2}, R{rng.choice(fp_r)}"]

        if code == 0x4E:  # VEC_BLEND
            vb = st.v_matching(types={"bool_3vl"}, length=n0)
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vb or not vf:
                return ["  NOP"]
            subop = rng.randint(0, 3)
            sname = self.by_code[0x4E]["subops"][subop]
            s1 = rng.choice(vb)
            s2 = rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s2])
            st.v[dst] = VState("f64", n0, dom=0)
            if subop in (0, 2):
                return [f"  VEC_BLEND.{sname} V{dst}, V{s1}, V{s2}, V{rng.choice(vf)}"]
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            if not fp_r:
                return ["  NOP"]
            return [f"  VEC_BLEND.{sname} V{dst}, V{s1}, V{s2}, R{rng.choice(fp_r)}"]

        if code == 0x4F:  # COALESCE
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            s1 = rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("f64", n0, dom=0)
            fp_r = st.r_with_types({"f64"}, non_null_only=True)
            if fp_r and rng.random() < 0.5:
                return [f"  COALESCE.scalar V{dst}, V{s1}, R{rng.choice(fp_r)}"]
            return [f"  COALESCE V{dst}, V{s1}, V{rng.choice(vf)}"]

        if code == 0x50:  # VEC_INIT
            op = self.by_code[0x50]
            subop = rng.choice(list(op["subops"].keys()))
            sname = op["subops"][subop]
            dst = rng.randint(0, 15)
            ty_map = {0: "i64", 1: "u64", 2: "i32", 3: "u32", 4: "f64", 5: "f32", 6: "bool_3vl"}
            base_ty = ty_map[subop % 8]
            if subop >= 8:
                st.v[dst] = VState(base_ty, n0, dom=0)
                return [f"  VEC_INIT.{sname} V{dst}, dom=0"]
            st.v[dst] = VState(base_ty, n0)
            return [f"  VEC_INIT.{sname} V{dst}, len={n0}"]

        if code == 0x51:  # VEC_POINT
            subop = rng.randint(0, 3)
            vs = st.v_matching(types={"f64", "i64", "u64"}, length=n0)
            if not vs:
                return ["  NOP"]
            s1 = rng.choice(vs)
            if subop == 3:  # vec_len
                dst = rng.randint(0, 15)
                st.r[dst] = RState("u64", False, False)
                return [f"  VEC_POINT.vec_len R{dst}, V{s1}"]
            if subop == 2:  # vec_cast
                dst = self._pick_dst(rng, profile, [s1])
                target_ty_code = rng.choice([0, 1, 4])  # i64, u64, f64
                ty_name = {0: "i64", 1: "u64", 4: "f64"}[target_ty_code]
                st.v[dst] = VState(ty_name, n0, dom=0)
                return [f"  VEC_POINT.vec_cast V{dst}, V{s1}, lo={target_ty_code}"]
            small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if not small_r:
                return ["  NOP"]
            idx_r = rng.choice(small_r)
            if subop == 0:  # vec_get
                dst = rng.randint(0, 15)
                st.r[dst] = RState(st.v[s1].ty, False, False)
                return [f"  VEC_POINT.vec_get R{dst}, V{s1}, R{idx_r}"]
            # vec_set
            val_r = st.r_with_types({st.v[s1].ty}, non_null_only=True)
            if not val_r:
                return ["  NOP"]
            return [f"  VEC_POINT.vec_set V{s1}, R{idx_r}, R{rng.choice(val_r)}"]

        if code == 0x52:  # VEC_REDUCE
            subop = rng.randint(0, 10)
            sname = self.by_code[0x52]["subops"][subop]
            dst = rng.randint(0, 15)
            if subop in (9, 10):
                vb = st.v_matching(types={"bool_3vl"}, length=n0)
                if not vb:
                    return ["  NOP"]
                st.r[dst] = RState("bool_3vl", False, False)
                return [f"  VEC_REDUCE.{sname} R{dst}, V{rng.choice(vb)}"]
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            res_ty = "u64" if subop in (3, 7, 8) else "f64"
            st.r[dst] = RState(res_ty, False, False)
            return [f"  VEC_REDUCE.{sname} R{dst}, V{rng.choice(vf)}"]

        if code == 0x53:  # TOP_K
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            subop = rng.randint(0, 3)
            sname = self.by_code[0x53]["subops"][subop]
            s1 = rng.choice(vf)
            dst = self._pick_dst(rng, profile, [s1])
            k = rng.randint(1, min(3, n0))
            st.v[dst] = VState("u64" if subop in (0, 1) else "f64", k)
            return [f"  TOP_K.{sname} V{dst}, V{s1}, k={k}"]

        if code == 0x54:  # INDEX_SCAN
            ir = st.r_with_types({"i64"}, non_null_only=True)
            if not ir:
                return ["  NOP"]
            subop = rng.randint(0, 6)
            sname = self.by_code[0x54]["subops"][subop]
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            r1 = rng.choice(ir)
            if subop in (1, 2):
                r2 = rng.choice(ir)
                return [f"  INDEX_SCAN.{sname} F{dst}, R{r1}, R{r2}, dom=0, attr=0"]
            return [f"  INDEX_SCAN.{sname} F{dst}, R{r1}, dom=0, attr=0"]

        if code == 0x55:  # MAP_KEYS_TO_DENSE
            subop = rng.randint(0, 4)
            if subop == 0:
                dst = rng.randint(0, 15)
                st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
                return [f"  MAP_KEYS_TO_DENSE.param_to_f F{dst}, dom=0, param=9"]
            if subop in (1, 4):
                vs = st.v_matching(types={"str"}, length=n0)
                if not vs:
                    return ["  NOP"]
                s1 = rng.choice(vs)
                dst = self._pick_dst(rng, profile, [s1])
                sname = self.by_code[0x55]["subops"][subop]
                st.v[dst] = VState("u64", n0, dom=0, is_node_ids_for_dom=0)
                return [f"  MAP_KEYS_TO_DENSE.{sname} V{dst}, V{s1}, dom=0"]
            if subop == 2:
                sr = st.r_with_types({"str"}, non_null_only=True)
                if not sr:
                    return ["  NOP"]
                s1 = rng.choice(sr)
                dst = self._pick_dst(rng, profile, [s1])
                st.r[dst] = RState("u64", False, False)
                return [f"  MAP_KEYS_TO_DENSE.scalar_to_r R{dst}, R{s1}, dom=0"]
            if not f0_list:
                return ["  NOP"]
            s1 = rng.choice(f0_list)
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=1, is_multilane=False, k_lanes=1)
            return [f"  MAP_KEYS_TO_DENSE.f_to_f F{dst}, F{s1}, dom=1"]

        if code == 0x56:  # MAP_DENSE_TO_KEYS
            subop = rng.randint(0, 2)
            if subop == 0:
                v_ids = st.v_matching(node_ids_dom=0)
                if not v_ids:
                    return ["  NOP"]
                s1 = rng.choice(v_ids)
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = VState("str", st.v[s1].length)
                return [f"  MAP_DENSE_TO_KEYS.vec V{dst}, V{s1}, dom=0"]
            if subop == 1:
                if not f0_list:
                    return ["  NOP"]
                s1 = rng.choice(f0_list)
                dst = rng.randint(0, 15)
                st.v[dst] = VState("str", n0)
                return [f"  MAP_DENSE_TO_KEYS.frontier_sorted V{dst}, F{s1}, dom=0"]
            small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
            if not small_r:
                return ["  NOP"]
            s1 = rng.choice(small_r)
            dst = self._pick_dst(rng, profile, [s1])
            st.r[dst] = RState("str", False, False)
            return [f"  MAP_DENSE_TO_KEYS.scalar R{dst}, R{s1}, dom=0"]

        if code == 0x57:  # COLLECT
            subop = rng.randint(0, 4)
            out_slot = rng.randint(0, 7)
            if subop in (0, 1):
                if not f0_list:
                    return ["  NOP"]
                sname = self.by_code[0x57]["subops"][subop]
                return [f"  COLLECT.{sname} F{rng.choice(f0_list)}, out={out_slot}"]
            if subop in (2, 4):
                vs = st.v_matching(types={"f64", "i64", "u64", "bool_3vl", "str"})
                if not vs:
                    return ["  NOP"]
                sname = self.by_code[0x57]["subops"][subop]
                return [f"  COLLECT.{sname} V{rng.choice(vs)}, out={out_slot}"]
            return [f"  COLLECT.scalar R{rng.randint(0, 15)}, out={out_slot}"]

        return ["  NOP"]

    # ---- Pillar 5 Instruction Emitter ----

    def _emit_pillar5_op(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker
    ) -> list[str]:
        code = rng.choice([0x60, 0x61, 0x63, 0x64, 0x65, 0x66, 0x67, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F])
        n0 = env.dom_cards[0]
        ml0 = st.f_matching(dom=0, multilane=True)
        f0_1b = st.f_matching(dom=0, multilane=False)

        if code == 0x60:  # LANE_INIT_SEEDS
            subop = rng.choice([0, 1])
            sname = self.by_code[0x60]["subops"][subop]
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=True, k_lanes=2)
            return [
                "  LOAD_PARAM.vec V14, param=11, type=u64",
                f"  LANE_INIT_SEEDS.{sname} F{dst}, V14, dom=0",
            ]

        if code == 0x61:  # LANE_INIT_GROUPS
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=True, k_lanes=2)
            return [
                "  LOAD_PARAM.vec V14, param=11, type=u64",
                "  LOAD_PARAM.vec V15, param=12, type=u64",
                f"  LANE_INIT_GROUPS.groups F{dst}, V14, V15, dom=0, k=0",
            ]

        if code == 0x63:  # LANE_WALK
            if not ml0:
                return ["  NOP"]
            subop = rng.randint(0, 9)
            sname = self.by_code[0x63]["subops"][subop]
            s1 = rng.choice(ml0)
            dst = self._pick_dst(rng, profile, [s1])
            st.f[dst] = copy.copy(st.f[s1])
            use_mask = (subop in (7, 8, 9)) and bool(f0_1b) and rng.random() < 0.3
            mod = ".mask" if use_mask else ""
            mask_arg = f", F{rng.choice(f0_1b)}" if use_mask else ""
            if subop in (7, 9):  # edge_masked
                return [
                    "  LOAD_PARAM.vec V15, param=13, type=u64",
                    f"  LANE_WALK.{sname}{mod} F{dst}, F{s1}, V15{mask_arg}, rel=0",
                ]
            return [f"  LANE_WALK.{sname}{mod} F{dst}, F{s1}{mask_arg}, rel=0"]

        if code == 0x64:  # LANE_BITWISE
            if not ml0:
                return ["  NOP"]
            subop = rng.randint(0, 3)
            sname = self.by_code[0x64]["subops"][subop]
            s1 = rng.choice(ml0)
            dst = self._pick_dst(rng, profile, [s1])
            if subop == 2:  # and_not_lanes
                same_k = [i for i in ml0 if st.f[i].k_lanes == st.f[s1].k_lanes]
                s2 = rng.choice(same_k)
                st.f[dst] = copy.copy(st.f[s1])
                return [f"  LANE_BITWISE.and_not_lanes F{dst}, F{s1}, F{s2}"]
            if subop == 3:  # extract_lane
                st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
                lane = rng.randint(0, max(0, st.f[s1].k_lanes - 1))
                return [f"  LANE_BITWISE.extract_lane F{dst}, F{s1}, {lane}"]
            st.f[dst] = copy.copy(st.f[s1])
            return [f"  LANE_BITWISE.{sname} F{dst}, F{s1}, {rng.choice([1, 2, 3])}"]

        if code == 0x65:  # LANE_FILTER_QUORUM
            if not ml0:
                return ["  NOP"]
            subop = rng.randint(0, 2)
            sname = self.by_code[0x65]["subops"][subop]
            s1 = rng.choice(ml0)
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            return [f"  LANE_FILTER_QUORUM.{sname} F{dst}, F{s1}, {rng.choice([1, 2])}"]

        if code == 0x66:  # LANE_REDUCE_AND
            if not ml0:
                return ["  NOP"]
            s1 = rng.choice(ml0)
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            return [f"  LANE_REDUCE_AND.reduce_and F{dst}, F{s1}"]

        if code == 0x67:  # LANE_REDUCE_ATTR
            if not ml0:
                return ["  NOP"]
            subop = rng.randint(0, 3)
            sname = self.by_code[0x67]["subops"][subop]
            s1 = rng.choice(ml0)
            dst = rng.randint(0, 15)
            if subop == 3:
                st.v[dst] = VState("u64", st.f[s1].k_lanes)
                return [f"  LANE_REDUCE_ATTR.count V{dst}, F{s1}"]
            vf = st.v_matching(types={"f64"}, length=n0)
            if not vf:
                return ["  NOP"]
            st.v[dst] = VState("f64", st.f[s1].k_lanes)
            return [f"  LANE_REDUCE_ATTR.{sname} V{dst}, F{s1}, V{rng.choice(vf)}"]

        if code == 0x69:  # VECTOR_ANN_SEARCH
            subop = rng.choice([0, 1])
            sname = self.by_code[0x69]["subops"][subop]
            dst = rng.randint(0, 15)
            st.v[dst] = VState("u64", 2)
            if subop == 0:
                return [
                    "  LOAD_PARAM.vec V14, param=10, type=f64",
                    "  LOAD_CONST.u32 R14, 2",
                    f"  VECTOR_ANN_SEARCH.ids V{dst}, V14, R14, dom=0, attr=3",
                ]
            return [
                "  LOAD_PARAM.vec V14, param=10, type=f64",
                "  LOAD_CONST.f32 R14, 0.5",
                f"  VECTOR_ANN_SEARCH.radius V{dst}, V14, R14, dom=0, attr=3",
            ]

        if code == 0x6A:  # VECTOR_RERANK
            subop = rng.randint(0, 7)
            sname = self.by_code[0x6A]["subops"][subop]
            v_ids = st.v_matching(node_ids_dom=0)
            if not v_ids:
                return ["  NOP"]
            s1 = rng.choice(v_ids)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = VState("f64", st.v[s1].length)
            if subop >= 4:  # pair_*
                same_len = [i for i in v_ids if st.v[i].length == st.v[s1].length]
                s2 = rng.choice(same_len)
                return [f"  VECTOR_RERANK.{sname} V{dst}, V{s1}, V{s2}, dom=0, attr=3"]
            return [
                "  LOAD_PARAM.vec V14, param=10, type=f64",
                f"  VECTOR_RERANK.{sname} V{dst}, V{s1}, V14, dom=0, attr=3",
            ]

        if code == 0x6B:  # LANE_STATE_INIT
            subop = rng.randint(0, 5)
            sname = self.by_code[0x6B]["subops"][subop]
            dst = rng.randint(0, 15)
            ty = "lane_state_f64" if subop in (2, 3) else "lane_state_f32"
            st.v[dst] = VState(ty, n0, dom=0, is_lane_state=True, k_lanes=2)
            if subop in (4, 5):
                if not ml0:
                    return ["  NOP"]
                s1 = rng.choice(ml0)
                if subop == 5:
                    return [
                        "  LOAD_CONST.f32 R14, 1.0",
                        f"  LANE_STATE_INIT.with_witness.scalar V{dst}, F{s1}, R14, dom=0, k=2",
                    ]
                return [f"  LANE_STATE_INIT.from_frontier_seeds V{dst}, F{s1}, dom=0, k=2"]
            return [f"  LANE_STATE_INIT.{sname} V{dst}, dom=0, k=2"]

        if code == 0x6C:  # LANE_STATE_WALK
            vls = st.v_matching(lane_state=True)
            if not vls:
                return ["  NOP"]
            subop = rng.randint(0, 9)
            sname = self.by_code[0x6C]["subops"][subop]
            s1 = rng.choice(vls)
            dst = self._pick_dst(rng, profile, [s1])
            st.v[dst] = copy.copy(st.v[s1])
            use_accum = rng.random() < 0.3
            use_mask = bool(f0_1b) and rng.random() < 0.3
            mods = (".accum" if use_accum else "") + (".mask" if use_mask else "")
            mask_arg = f", F{rng.choice(f0_1b)}" if use_mask else ""
            if use_accum and dst != s1:
                dst = s1
            return [f"  LANE_STATE_WALK.{sname}{mods} V{dst}, V{s1}{mask_arg}, rel=0, attr=0"]

        if code == 0x6D:  # LANE_STATE_OP
            vls = st.v_matching(lane_state=True)
            if not vls:
                return ["  NOP"]
            subop = rng.randint(0, 4)
            sname = self.by_code[0x6D]["subops"][subop]
            s1 = rng.choice(vls)
            if subop == 0:
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = copy.copy(st.v[s1])
                return [
                    "  LOAD_CONST.f32 R14, 2.0",
                    "  LOAD_CONST.f32 R15, 0.5",
                    f"  LANE_STATE_OP.project_affine V{dst}, V{s1}, R14, R15",
                ]
            if subop == 1:
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = copy.copy(st.v[s1])
                return [
                    "  LOAD_CONST.f32 R14, 10.0",
                    f"  LANE_STATE_OP.prune_cutoff V{dst}, V{s1}, R14",
                ]
            if subop == 2:
                dst = self._pick_dst(rng, profile, [s1])
                st.v[dst] = copy.copy(st.v[s1])
                return [
                    "  LOAD_CONST.u32 R14, 1",
                    f"  LANE_STATE_OP.prune_hub_quorum.scalar V{dst}, V{s1}, R14",
                ]
            if subop == 3:
                dst = rng.randint(0, 15)
                st.f[dst] = FState(dom=0, is_multilane=True, k_lanes=st.v[s1].k_lanes)
                return [f"  LANE_STATE_OP.to_lane_frontier F{dst}, V{s1}"]
            dst = rng.randint(0, 15)
            st.f[dst] = FState(dom=0, is_multilane=False, k_lanes=1)
            return [f"  LANE_STATE_OP.to_quorum_frontier F{dst}, V{s1}, 1"]

        if code == 0x6E:  # LANE_STATE_REDUCE
            vls = st.v_matching(lane_state=True)
            if not vls:
                return ["  NOP"]
            subop = rng.randint(0, 6)
            sname = self.by_code[0x6E]["subops"][subop]
            s1 = rng.choice(vls)
            dst = rng.randint(0, 15)
            if subop == 6:
                st.v[dst] = VState("u64", n0)
                return [
                    "  LOAD_CONST.u32 R14, 0",
                    f"  LANE_STATE_REDUCE.trace_path.scalar V{dst}, V{s1}, R14",
                ]
            st.v[dst] = VState("u64" if subop in (4, 5) else "f64", n0)
            return [f"  LANE_STATE_REDUCE.{sname} V{dst}, V{s1}"]

        if code == 0x6F:  # VECTOR_POOL
            subop = rng.randint(0, 3)
            sname = self.by_code[0x6F]["subops"][subop]
            dst = rng.randint(0, 15)
            if subop == 0:
                small_r = st.r_with_types({"i64", "u64"}, non_null_only=True, small_only=True)
                if not small_r:
                    return ["  NOP"]
                st.v[dst] = VState("f64", 3)
                return [f"  VECTOR_POOL.load_node V{dst}, R{rng.choice(small_r)}, dom=0, attr=3"]
            if subop in (1, 2):
                if not f0_1b:
                    return ["  NOP"]
                st.v[dst] = VState("f64", 3)
                return [f"  VECTOR_POOL.{sname} V{dst}, F{rng.choice(f0_1b)}, dom=0, attr=3"]
            v3 = st.v_matching(types={"f64"}, length=3)
            if not v3:
                return ["  NOP"]
            st.v[dst] = VState("f64", 3)
            return [f"  VECTOR_POOL.l2_norm V{dst}, V{rng.choice(v3)}, dom=0, attr=3"]

        return ["  NOP"]

    # ---- Control-Flow Block Emitter (Pillar 1) ----

    def _emit_controlflow_block(
        self, rng: random.Random, profile: str, env: FixtureEnv, st: V11RegTracker, blk_id: int
    ) -> tuple[list[str], list[str]]:
        """Emit bounded control-flow constructs (JMP, JCC all 10 subops, LOOP_DECR, CALL/RET)."""
        kind = rng.choice(["jcc", "loop", "call", "jmp"])
        body: list[str] = []
        subroutines: list[str] = []

        if kind == "jmp":
            lbl = f"jmp_t_{blk_id}"
            body.append(f"  JMP {lbl}")
            body.append("  NOP")
            body.append(f"{lbl}:")
            body.extend(self._emit_pillar1_op(rng, profile, env, st))
            return body, subroutines

        if kind == "jcc":
            subop = rng.randint(0, 9)
            sname = self.by_code[0x10]["subops"][subop]
            lbl = f"jcc_t_{blk_id}"
            if subop in (6, 7):
                r_idx = rng.choice(st.r_with_types({"bool_3vl", "null"}) or [11, 7])
                body.append(f"  JCC.{sname} R{r_idx}, {lbl}")
            elif subop in (8, 9):
                f_idx = rng.choice(st.f_matching() or [0])
                body.append(f"  JCC.{sname} F{f_idx}, {lbl}")
            else:
                body.append(f"  JCC.{sname} {lbl}")
            body.append("  SCALAR_INT_ALU.add.imm R12, R0, 1")
            st.r[12] = RState("i64", st.r[0].is_null, False)
            body.append(f"{lbl}:")
            return body, subroutines

        if kind == "loop":
            lbl = f"loop_h_{blk_id}"
            iters = rng.randint(1, 5)
            ctr = 13
            acc = 12
            body.append(f"  LOAD_CONST.i32 R{ctr}, {iters}")
            body.append(f"  LOAD_CONST.i32 R{acc}, 0")
            body.append(f"{lbl}:")
            body.append(f"  SCALAR_INT_ALU.add.imm R{acc}, R{acc}, 3")
            body.append(f"  LOOP_DECR R{ctr}, {lbl}")
            st.r[ctr] = RState("i64", False, True)
            st.r[acc] = RState("i64", False, False)
            return body, subroutines

        # CALL + RET
        fn_lbl = f"fn_{blk_id}"
        if rng.random() < 0.7:
            rs = rng.choice([0, 2, 4, 8])
            fs = rng.choice([0, 2, 4])
            vs = rng.choice([0, 2, 4])
            body.append(f"  CALL.stride {fn_lbl}, r_stride={rs}, f_stride={fs}, v_stride={vs}")
        else:
            body.append(f"  CALL.full16 {fn_lbl}")
        subroutines.append(f"{fn_lbl}:")
        subroutines.append("  LOAD_CONST.i32 R0, 7")
        subroutines.append("  SCALAR_INT_ALU.add.imm R0, R0, 5")
        subroutines.append("  RET")
        return body, subroutines

    # ---- Negative (Intentional Runtime Error) Emitter ----

    def _emit_negative_op(self, rng: random.Random, env: FixtureEnv, st: V11RegTracker) -> list[str]:
        scenarios: list[list[str]] = [
            ["  TRAP 42"],
            ["  THROW.imm 1234"],
            ["  THROW.reg R0"],
            ["  LOAD_CONST.null R14", "  ASSERT.true_r R14, code=9"],
            ["  LOAD_CONST.i32 R14, 0", "  ASSERT.true_r R14, code=10"],
            ["  LOAD_CONST.i32 R14, 1", "  ASSERT.is_null_r R14, code=11"],
            ["  LOAD_CONST.null R14", "  ASSERT.not_null_r R14, code=12"],
            ["  ASSERT.st_set code=13"],
            ["  LOAD_CONST.i32 R14, 10", "  LOAD_CONST.i32 R15, 0", "  SCALAR_INT_ALU.div_s R14, R14, R15"],
            ["  LOAD_CONST.i32 R14, 10", "  LOAD_CONST.i32 R15, 0", "  SCALAR_INT_ALU.rem_u R14, R14, R15"],
            ["  LOAD_CONST.i64.ext R14, 0x8000000000000000", "  LOAD_CONST.i32 R15, -1", "  SCALAR_INT_ALU.div_s R14, R14, R15"],
            ["  LOAD_CONST_POOL.vec V14, pool=99"],
            ["  LOAD_PARAM.scalar R14, param=99, type=i64"],
            ["  LOAD_PARAM.scalar R14, param=0, type=f64"],  # type mismatch
            ["  RELOAD.r R14, slot=15"],  # empty spill slot
            ["  SPILL.r R0, slot=14", "  RELOAD.f F14, slot=14"],  # cross-bank reload
            ["  RET"],  # stack underflow
            ["  LOAD_CONST.null R14", "  LOOP_DECR R14, 1"],  # null counter
            ["  LOAD_CONST.null R14", "  SET_MAX_DOP.reg R14"],
            ["  MOV.clear_f F14, dom=99"],  # OOB dom
        ]
        if 2 in self.allowed_pillars:
            scenarios.extend([
                ["  FSEED.single_imm F14, node=999, dom=0"],
                ["  SET_OP.union F14, F0, F2"],  # dom mismatch (0 vs 1)
                ["  INIT_DOMAIN_SPAN.all_span F14, dom=99"],
            ])
        if 3 in self.allowed_pillars:
            scenarios.extend([
                ["  WALK.fwd F14, F0, rel=99"],
                ["  WALK.fwd_fixpoint_plus F14, F0, rel=1"],  # bipartite fixpoint
                ["  GRAPH_KERNEL.cc_afforest V14, rel=1"],  # bipartite kernel
            ])
        if 4 in self.allowed_pillars:
            scenarios.extend([
                ["  LOAD_COLUMN.node_attr V14, dom=0, attr=99"],
                ["  LOAD_CONST.u32 R14, 999", "  VEC_POINT.vec_get R15, V0, R14"],
                ["  INDEX_SCAN.eq F14, R0, dom=0, attr=1"],  # missing index
                ["  VEC_INT_BITWISE.and V14, V0, V0"],  # bitwise on f64
            ])
        if 5 in self.allowed_pillars:
            scenarios.extend([
                ["  LANE_WALK.fwd_fixpoint_star F14, F4, rel=1"],  # bipartite lane fixpoint
                ["  VECTOR_ANN_SEARCH.ids V14, V5, R1, dom=0, attr=99"],
            ])
        return rng.choice(scenarios)

    # ---- Raw-Word Validator Mutator ----

    def generate_validator_case(
        self, seed: int, max_instrs: int, encode_word_fn: Any, encode_ext_fn: Any, decode_fn: Any, valid_subop_mask_fn: Any
    ) -> dict[str, Any]:
        rng = random.Random(seed)
        # Start from a small valid program for enabled pillars
        base = self.generate_program(seed ^ 0x13579BDF, "mixed", max(4, min(10, max_instrs)))
        words = list(base["words"])
        if not words:
            words = [encode_word_fn(0x00, 0, 0, 0, 0, 0, 0, 0)]

        n_muts = rng.randint(1, 2)
        mut_tags: list[str] = []
        for _ in range(n_muts):
            mut_kind = rng.choice([
                "unassigned_opcode",
                "invalid_subop",
                "disallowed_flag",
                "mbz_nonzero",
                "dangling_ext",
                "standalone_0xff",
                "bad_ext_meta24",
                "bad_jmp_target",
                "jmp_into_0xff",
                "spill_slot_oob",
                "scalar_alu_s3_mbz",
                "morsel_bad" if 3 in self.allowed_pillars else "invalid_subop",
            ])
            mut_tags.append(mut_kind)
            idx = rng.randint(0, len(words) - 1)

            if mut_kind == "unassigned_opcode":
                bad_op = rng.choice([0x15, 0x1F, 0x28, 0x2F, 0x32, 0x3C, 0x58, 0x62, 0x68, 0x70, 0xFE])
                words[idx] = encode_word_fn(bad_op, 0, 0, 0, 0, 0, 0, 0)
            elif mut_kind == "invalid_subop":
                op = rng.choice(self.active_ops)
                vmask = valid_subop_mask_fn(op)
                bad_subs = [s for s in range(16) if (vmask & (1 << s)) == 0]
                if bad_subs:
                    words[idx] = encode_word_fn(op["code"], 0, rng.choice(bad_subs), 0, 0, 0, 0, 0)
            elif mut_kind == "disallowed_flag":
                op = rng.choice(self.active_ops)
                bad_flags = [f for f in range(1, 16) if (f & (~op["allowed_flags"] & 0xF)) != 0]
                if bad_flags:
                    sub = next(iter(op["subops"].keys()))
                    words[idx] = encode_word_fn(op["code"], rng.choice(bad_flags), sub, 0, 0, 0, 0, 0)
            elif mut_kind == "mbz_nonzero":
                op_c = rng.choice([0x00, 0x01, 0x0F, 0x13])
                words[idx] = encode_word_fn(op_c, 0, 0, rng.randint(1, 15), 0, 0, 0, rng.choice([0, 1]))
            elif mut_kind == "dangling_ext":
                # LOAD_CONST.i64.ext without following 0xFF
                words.append(encode_word_fn(0x05, 0x8, 4, 0, 0, 0, 0, 0))
            elif mut_kind == "standalone_0xff":
                words.insert(idx, encode_ext_fn(0x12345678))
            elif mut_kind == "bad_ext_meta24":
                words.append(encode_word_fn(0x05, 0x8, 4, 0, 0, 0, 0, 0))
                words.append(0xFF | (rng.randint(1, 0xFF) << 8))
            elif mut_kind == "bad_jmp_target":
                rel = rng.choice([0, -100, 100, len(words) + 5])
                words[idx] = encode_word_fn(0x0F, 0, 0, 0, 0, 0, 0, rel & 0xFFFFFFFF)
            elif mut_kind == "jmp_into_0xff":
                # Insert LOAD_CONST.i64.ext + 0xFF, then JMP directly to the 0xFF word
                words = [
                    encode_word_fn(0x0F, 0, 0, 0, 0, 0, 0, 2),
                    encode_word_fn(0x05, 0x8, 4, 0, 0, 0, 0, 0),
                    encode_ext_fn(0),
                    encode_word_fn(0x00, 0, 0, 0, 0, 0, 0, 0),
                ]
            elif mut_kind == "spill_slot_oob":
                words[idx] = encode_word_fn(0x0A, 0, 0, 0, 0, 0, 0, rng.choice([16, 0x10000, 0x00010000]))
            elif mut_kind == "scalar_alu_s3_mbz":
                words[idx] = encode_word_fn(0x0C, 0, 0, 0, 0, 0, rng.randint(2, 15), 0)
            elif mut_kind == "morsel_bad":
                m_sub = rng.choice(["unclosed", "standalone_end", "commit_outside", "nested", "bad_end_off"])
                if m_sub == "unclosed":
                    words.append(encode_word_fn(0x36, 0, 0, 0, 0, 0, 0, 1 << 16))
                elif m_sub == "standalone_end":
                    words.insert(0, encode_word_fn(0x38, 0, 0, 0, 0, 0, 0, 1 << 16))
                elif m_sub == "commit_outside":
                    words.insert(0, encode_word_fn(0x37, 0, 8, 0, 0, 0, 0, 0))
                elif m_sub == "nested":
                    words = [
                        encode_word_fn(0x36, 0, 0, 0, 0, 0, 0, 3 << 16),
                        encode_word_fn(0x36, 0, 0, 0, 0, 0, 0, 1 << 16),
                        encode_word_fn(0x38, 0, 0, 0, 0, 0, 0, 1 << 16),
                        encode_word_fn(0x38, 0, 0, 0, 0, 0, 0, 3 << 16),
                        encode_word_fn(0x00, 0, 0, 0, 0, 0, 0, 0),
                    ]
                else:
                    words = [
                        encode_word_fn(0x36, 0, 0, 0, 0, 0, 0, 5 << 16),
                        encode_word_fn(0x38, 0, 0, 0, 0, 0, 0, 1 << 16),
                        encode_word_fn(0x00, 0, 0, 0, 0, 0, 0, 0),
                    ]

        return {
            "seed": seed,
            "profile": "validator",
            "is_raw_words": True,
            "mut_tags": mut_tags,
            "data_directives": base["data_directives"],
            "const_pool": base["const_pool"],
            "words": words,
            "source": None,
            "max_steps": 2000,
        }

    # ---- Main Assembled Program Generator ----

    def generate_program(self, seed: int, profile: str, max_instrs: int) -> dict[str, Any]:
        rng = random.Random(seed)
        env = self._build_fixture_env(rng, profile)
        st = V11RegTracker()

        code_lines = self._emit_preamble(rng, profile, env, st)
        subroutines: list[str] = []

        n_steps = rng.randint(max(3, max_instrs // 3), max_instrs)
        enabled_list = sorted(self.allowed_pillars)

        # Pillar weights per profile
        def pick_pillar() -> int:
            if profile == "morsel" and 3 in self.allowed_pillars:
                return 3 if rng.random() < 0.75 else rng.choice(enabled_list)
            if profile == "precision_3vl":
                if 4 in self.allowed_pillars and rng.random() < 0.7:
                    return 4
                return 1
            if profile == "multilane" and 5 in self.allowed_pillars:
                return 5 if rng.random() < 0.75 else rng.choice(enabled_list)
            if profile == "controlflow":
                return 1 if rng.random() < 0.65 else rng.choice(enabled_list)
            return rng.choice(enabled_list)

        neg_injected = False
        for step in range(n_steps):
            if profile == "negative" and not neg_injected and step >= n_steps // 2:
                code_lines.extend(self._emit_negative_op(rng, env, st))
                neg_injected = True
                continue

            if profile == "controlflow" and rng.random() < 0.45:
                blk, subs = self._emit_controlflow_block(rng, profile, env, st, step)
                code_lines.extend(blk)
                subroutines.extend(subs)
                continue

            p = pick_pillar()
            if p == 1:
                if rng.random() < 0.12:
                    blk, subs = self._emit_controlflow_block(rng, profile, env, st, step)
                    code_lines.extend(blk)
                    subroutines.extend(subs)
                else:
                    code_lines.extend(self._emit_pillar1_op(rng, profile, env, st))
            elif p == 2:
                code_lines.extend(self._emit_pillar2_op(rng, profile, env, st))
            elif p == 3:
                code_lines.extend(self._emit_pillar3_op(rng, profile, env, st, step))
            elif p == 4:
                code_lines.extend(self._emit_pillar4_op(rng, profile, env, st))
            elif p == 5:
                code_lines.extend(self._emit_pillar5_op(rng, profile, env, st))

        if profile == "negative" and not neg_injected:
            code_lines.extend(self._emit_negative_op(rng, env, st))

        code_lines.append("  HALT")
        code_lines.extend(subroutines)

        src_parts = [
            f"; V11 CONTRACT-GUIDED FUZZ (seed={seed}, profile={profile}, pillars={sorted(self.allowed_pillars)})",
            "",
            ".data",
            *env.data_lines,
            "",
            ".const_pool",
            *env.pool_lines,
            "",
            ".code",
            *code_lines,
            "",
        ]
        source = "\n".join(src_parts)
        prog = self.asm.assemble(source)

        max_steps = 10_000
        if profile == "boundary" and rng.random() < 0.12:
            max_steps = rng.choice([5, 12, 25])

        return {
            "seed": seed,
            "profile": profile,
            "is_raw_words": False,
            "mut_tags": [],
            "data_directives": prog.data_directives,
            "const_pool": prog.const_pool,
            "words": prog.words,
            "source": source,
            "max_steps": max_steps,
        }


# ---------------------------------------------------------------------------
# Corpus-Based Mutational & Feedback-Guided Fuzzer
# ---------------------------------------------------------------------------

class V11CorpusMutator:
    """Loads a seed corpus of .impas programs and evolves them via structured
    instruction/operand/data mutations + splicing with behavioral-signature feedback."""

    BOUNDARY_IMMS = (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 0xFFFF, 0x7FFF, 0xFFFFFFFF, 0x80000000)
    BOUNDARY_F64 = (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-17, -1e-17, 1e308, -1e308, None)
    BOUNDARY_I64 = (0, 1, -1, 2, -2, 63, 64, 9223372036854775807, -9223372036854775808, None)

    def __init__(
        self,
        isa: dict[str, Any],
        asm: Any,
        allowed_pillars: set[int],
        corpus_dirs: list[Path],
        max_corpus_size: int = 2048,
        save_dir: Path | None = None,
    ) -> None:
        self.isa = isa
        self.asm = asm
        self.allowed_pillars = allowed_pillars
        self.max_corpus_size = max_corpus_size
        self.save_dir = save_dir
        self.op_by_code: dict[int, dict[str, Any]] = {
            op["code"]: op for op in isa["opcodes"] if op["code"] != 0xFF
        }
        self.entries: list[dict[str, Any]] = []
        self.initial_count = 0
        self.seen_sigs: set[tuple[Any, ...]] = set()
        self.grown_count = 0

        for cdir in corpus_dirs:
            if not cdir.exists():
                continue
            for p in sorted(cdir.rglob("*.impas")):
                try:
                    src = p.read_text(encoding="utf-8")
                    if "{{RAW_WORDS:" in src:
                        continue
                    prog = asm.assemble(src)
                    if not prog.words:
                        continue
                    # Pillar filter: keep entry if all non-0xFF opcodes are in allowed_pillars
                    if allowed_pillars != {1, 2, 3, 4, 5}:
                        ok = True
                        for w in prog.words:
                            c = w & 0xFF
                            if c != 0xFF and c in self.op_by_code:
                                if self.op_by_code[c]["pillar"] not in allowed_pillars:
                                    ok = False
                                    break
                        if not ok:
                            continue
                    self.entries.append({
                        "name": p.name,
                        "words": list(prog.words),
                        "data_directives": prog.data_directives,
                        "const_pool": prog.const_pool,
                    })
                except Exception:
                    continue
        self.initial_count = len(self.entries)

    @staticmethod
    def _instr_spans(words: list[int]) -> list[tuple[int, int]]:
        """Partition words into (start_idx, length) logical instructions (1 or 2 words for FLAG_EXT)."""
        spans: list[tuple[int, int]] = []
        i = 0
        n = len(words)
        while i < n:
            flags = (words[i] >> 12) & 0xF
            if (flags & 0x8) != 0 and i + 1 < n and (words[i + 1] & 0xFF) == 0xFF:
                spans.append((i, 2))
                i += 2
            else:
                spans.append((i, 1))
                i += 1
        return spans

    @staticmethod
    def _encode(op: int, flags: int, subop: int, dst: int, s1: int, s2: int, s3: int, imm32: int) -> int:
        b0 = op & 0xFF
        b1 = ((flags & 0xF) << 4) | (subop & 0xF)
        b2 = ((dst & 0xF) << 4) | (s1 & 0xF)
        b3 = ((s2 & 0xF) << 4) | (s3 & 0xF)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | ((imm32 & 0xFFFFFFFF) << 32)

    @staticmethod
    def _decode(w: int) -> tuple[int, int, int, int, int, int, int, int]:
        op = w & 0xFF
        subop = (w >> 8) & 0xF
        flags = (w >> 12) & 0xF
        s1 = (w >> 16) & 0xF
        dst = (w >> 20) & 0xF
        s3 = (w >> 24) & 0xF
        s2 = (w >> 28) & 0xF
        imm32 = (w >> 32) & 0xFFFFFFFF
        return op, flags, subop, dst, s1, s2, s3, imm32

    def _mutate_data(self, rng: random.Random, data_dirs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = copy.deepcopy(data_dirs)
        candidates = [d for d in out if d.get("directive") in (".param", ".node_attr", ".edge_attr")]
        if not candidates:
            return out
        d = rng.choice(candidates)
        kv = d.get("kv", {})
        ty = str(kv.get("type", ""))
        if "values" in kv and isinstance(kv["values"], list) and kv["values"]:
            idx = rng.randrange(len(kv["values"]))
            if "f64" in ty or "f32" in ty:
                if not isinstance(kv["values"][idx], list):
                    kv["values"][idx] = rng.choice(self.BOUNDARY_F64)
            elif "i64" in ty or "u64" in ty:
                kv["values"][idx] = rng.choice(self.BOUNDARY_I64)
            elif "bool" in ty:
                kv["values"][idx] = rng.choice([True, False, None])
        elif "value" in kv:
            if "f64" in ty or "f32" in ty:
                kv["value"] = rng.choice(self.BOUNDARY_F64)
            elif "i64" in ty or "u64" in ty:
                kv["value"] = rng.choice(self.BOUNDARY_I64)
        return out

    def generate_mutant(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        if not self.entries:
            raise RuntimeError("Corpus is empty; cannot generate corpus mutant")
        base = rng.choice(self.entries)
        words = list(base["words"])
        data_dirs = base["data_directives"]
        const_pool = base["const_pool"]
        mut_tags: list[str] = [f"base:{base['name']}"]

        n_muts = rng.randint(1, 3)
        for _ in range(n_muts):
            spans = self._instr_spans(words)
            if not spans:
                break
            kind = rng.choice([
                "flip_subop",
                "flip_flags",
                "swap_regs",
                "mutate_imm",
                "duplicate_instr",
                "delete_instr",
                "swap_instrs",
                "splice_corpus",
                "mutate_data",
            ])
            mut_tags.append(kind)

            if kind == "mutate_data":
                data_dirs = self._mutate_data(rng, data_dirs)
                continue

            if kind == "splice_corpus" and len(self.entries) >= 2:
                other = rng.choice(self.entries)
                sp_a = spans
                sp_b = self._instr_spans(other["words"])
                if len(sp_a) >= 2 and len(sp_b) >= 2:
                    cut_a = rng.randint(1, len(sp_a) - 1)
                    cut_b = rng.randint(0, len(sp_b) - 1)
                    end_a = sp_a[cut_a - 1][0] + sp_a[cut_a - 1][1]
                    start_b = sp_b[cut_b][0]
                    words = words[:end_a] + other["words"][start_b:]
                    if len(words) > 160:
                        words = words[:159] + [0]
                continue

            si = rng.randrange(len(spans))
            pos, slen = spans[si]
            op, flags, subop, dst, s1, s2, s3, imm32 = self._decode(words[pos])
            meta = self.op_by_code.get(op)

            if kind == "flip_subop" and meta is not None:
                valid_subs = list(meta["subops"].keys())
                if valid_subs:
                    new_sub = rng.choice(valid_subs) if rng.random() < 0.9 else rng.randrange(16)
                    # Keep FLAG_EXT consistency for LOAD_CONST (0x05)
                    if op == 0x05 and ((new_sub >= 4) != ((flags & 0x8) != 0)):
                        continue
                    words[pos] = self._encode(op, flags, new_sub, dst, s1, s2, s3, imm32)
            elif kind == "flip_flags" and meta is not None:
                allowed = meta["allowed_flags"] & 0x7  # avoid breaking 2-word FLAG_EXT framing
                if allowed:
                    bit = rng.choice([b for b in (1, 2, 4) if allowed & b])
                    words[pos] = self._encode(op, flags ^ bit, subop, dst, s1, s2, s3, imm32)
            elif kind == "swap_regs":
                which = rng.randrange(4)
                new_reg = rng.choice([dst, s1, s2, s3, rng.randrange(16)])
                if which == 0:
                    dst = new_reg
                elif which == 1:
                    s1 = new_reg
                elif which == 2:
                    s2 = new_reg
                else:
                    s3 = new_reg
                words[pos] = self._encode(op, flags, subop, dst, s1, s2, s3, imm32)
            elif kind == "mutate_imm":
                if op not in (0x0F, 0x10, 0x11, 0x12, 0x35, 0x36, 0x38):
                    imm_new = rng.choice(self.BOUNDARY_IMMS)
                    words[pos] = self._encode(op, flags, subop, dst, s1, s2, s3, imm_new)
            elif kind == "duplicate_instr" and len(words) < 140:
                if op not in (0x00, 0x35, 0x36, 0x38):
                    words = words[:pos] + words[pos : pos + slen] + words[pos:]
            elif kind == "delete_instr" and len(spans) > 2:
                if op not in (0x00, 0x35, 0x36, 0x38):
                    words = words[:pos] + words[pos + slen :]
            elif kind == "swap_instrs" and si + 1 < len(spans):
                pos2, slen2 = spans[si + 1]
                op2 = words[pos2] & 0xFF
                if op not in (0x00, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x35, 0x36, 0x38) and op2 not in (
                    0x00, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x35, 0x36, 0x38
                ):
                    seg1 = words[pos : pos + slen]
                    seg2 = words[pos2 : pos2 + slen2]
                    words = words[:pos] + seg2 + seg1 + words[pos2 + slen2 :]

        return {
            "seed": seed,
            "profile": "corpus",
            "is_raw_words": True,
            "mut_tags": mut_tags,
            "data_directives": data_dirs,
            "const_pool": const_pool,
            "words": words,
            "source": None,
            "max_steps": 10_000,
        }

    def record_feedback(self, case: dict[str, Any], ref_res: dict[str, Any]) -> bool:
        """Promote mutant into the live corpus if it hits a new behavioral signature."""
        status = str(ref_res.get("status", "UNKNOWN"))
        flags_raw = ref_res.get("flags", {}).get("raw", 0) if isinstance(ref_res.get("flags"), dict) else 0
        words = case["words"]
        # Compact opcode bigram fingerprint (top 4 distinct adjacent opcode pairs)
        bigrams: set[tuple[int, int]] = set()
        prev_op = -1
        for w in words:
            op = w & 0xFF
            if op == 0xFF:
                continue
            if prev_op >= 0:
                bigrams.add((prev_op, op))
            prev_op = op
        sig = (status, flags_raw, len(ref_res.get("outputs", {})), frozenset(sorted(bigrams)[:6]))
        if sig in self.seen_sigs:
            return False
        self.seen_sigs.add(sig)
        if len(self.entries) < self.max_corpus_size and status != "IMPULSE_VM_ERR_VALIDATION_FAILED":
            entry = {
                "name": f"mut_s{case['seed']}",
                "words": list(words),
                "data_directives": case["data_directives"],
                "const_pool": case["const_pool"],
            }
            self.entries.append(entry)
            self.grown_count += 1
            if self.save_dir is not None:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                raw_hex = ", ".join(f"0x{w:016X}" for w in words)
                out_p = self.save_dir / f"corpus_seed{case['seed']}_{status}.impas"
                out_p.write_text(
                    f"; CORPUS ENTRY (seed={case['seed']}, status={status}, flags={flags_raw})\n"
                    f"; {{{{RAW_WORDS: {raw_hex}}}}}\n"
                    f"; Mutations: {', '.join(case.get('mut_tags', []))}\n",
                    encoding="utf-8",
                )
            return True
        return False


# ---------------------------------------------------------------------------
# Consensus Checking & Delta-Debugging Shrinker
# ---------------------------------------------------------------------------

def evaluate_case(
    case: dict[str, Any],
    clients: dict[str, BatchEngineClient],
    profiles: list[str],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Run fixture across all active engines & geometry profiles.

    Returns (divergence_reason_or_None, results[engine][profile]).
    """
    words: list[int] = case["words"]
    payload = {
        "data_directives": case["data_directives"],
        "const_pool": case["const_pool"],
        "words_hex": [f"0x{w:016X}" for w in words],
    }
    has_transcendental = any((w & 0xFF) in TRANSCENDENTAL_OPCODES for w in words)
    has_pinned_tiers = any(d.get("directive") == ".tier_thresholds" for d in case["data_directives"])
    run_profiles = ["P0"] if (has_pinned_tiers or case["is_raw_words"]) else profiles

    results: dict[str, dict[str, Any]] = {eng: {} for eng in clients}
    engine_names = list(clients.keys())

    for prof in run_profiles:
        for eng in engine_names:
            try:
                results[eng][prof] = clients[eng].run_fixture(payload, profile=prof, max_steps=case["max_steps"])
            except RuntimeError as exc:
                results[eng][prof] = {"status": "ENGINE_CRASH", "error": str(exc)}
                return f"[{prof}] {exc}", results

        # Cross-engine consensus for this profile
        if len(engine_names) > 1:
            ref_eng = engine_names[0]
            ref = results[ref_eng][prof]
            for eng in engine_names[1:]:
                got = results[eng][prof]
                if got.get("status") != ref.get("status"):
                    return (
                        f"[{prof}] STATUS mismatch: {ref_eng}={ref.get('status')} vs {eng}={got.get('status')}",
                        results,
                    )
                if got.get("pc") != ref.get("pc"):
                    return (
                        f"[{prof}] PC mismatch: {ref_eng}={ref.get('pc')} vs {eng}={got.get('pc')} "
                        f"(status={ref.get('status')})",
                        results,
                    )
                if ref.get("status") == "IMPULSE_VM_ERR_USER_THROW":
                    if got.get("user_throw_code") != ref.get("user_throw_code"):
                        return (
                            f"[{prof}] USER_THROW_CODE mismatch: {ref_eng}={ref.get('user_throw_code')} "
                            f"vs {eng}={got.get('user_throw_code')}",
                            results,
                        )
                if ref.get("status") == "IMPULSE_VM_OK":
                    if got.get("flags") != ref.get("flags"):
                        return (
                            f"[{prof}] FLAGS mismatch: {ref_eng}={ref.get('flags')} vs {eng}={got.get('flags')}",
                            results,
                        )
                if not has_transcendental:
                    h_ref = str(ref.get("logical_hash", "")).lower()
                    h_got = str(got.get("logical_hash", "")).lower()
                    if h_ref != h_got:
                        return (
                            f"[{prof}] ICTX v2 logical_hash mismatch: {ref_eng}={h_ref} vs {eng}={h_got} "
                            f"(status={ref.get('status')}, pc={ref.get('pc')})",
                            results,
                        )

    # Metamorphic P0..P3 invariance per engine (when program is positive & non-transcendental)
    if len(run_profiles) > 1 and not has_transcendental:
        for eng in engine_names:
            statuses = {results[eng][p].get("status") for p in run_profiles}
            if len(statuses) > 1:
                st_map = {p: results[eng][p].get("status") for p in run_profiles}
                return f"[{eng}] Metamorphic STATUS divergence across profiles: {st_map}", results
            if next(iter(statuses)) == "IMPULSE_VM_OK":
                hashes = {p: str(results[eng][p].get("logical_hash", "")).lower() for p in run_profiles}
                if len(set(hashes.values())) > 1:
                    return f"[{eng}] Metamorphic ICTX v2 logical_hash divergence: {hashes}", results

    return None, results


def shrink_case(
    case: dict[str, Any],
    clients: dict[str, BatchEngineClient],
    profiles: list[str],
    asm: Any,
) -> tuple[dict[str, Any], str]:
    """1-minimal delta-debugging shrinker for both raw-word validator and .impas cases."""
    best = copy.deepcopy(case)
    best_err, _ = evaluate_case(best, clients, profiles)
    if best_err is None:
        return best, ""

    if best["is_raw_words"]:
        # Shrink raw words list
        words = list(best["words"])
        n = 2
        while len(words) > 1:
            chunk = max(1, len(words) // n)
            reduced = False
            for start in range(0, len(words), chunk):
                cand_words = words[:start] + words[start + chunk :]
                if not cand_words:
                    continue
                cand = dict(best, words=cand_words)
                err, _ = evaluate_case(cand, clients, profiles)
                if err is not None:
                    words = cand_words
                    best = cand
                    best_err = err
                    n = max(2, n - 1)
                    reduced = True
                    break
            if not reduced:
                if n >= len(words):
                    break
                n = min(len(words), n * 2)
        return best, best_err

    # Shrink .code lines in .impas source
    src = best["source"]
    if not src or "\n.code\n" not in src:
        return best, best_err
    prefix, code_part = src.split("\n.code\n", 1)
    code_lines = [ln for ln in code_part.splitlines() if ln.strip()]
    if len(code_lines) <= 2:
        return best, best_err

    n = 2
    while len(code_lines) > 2:
        chunk = max(1, len(code_lines) // n)
        reduced = False
        for start in range(0, len(code_lines), chunk):
            cand_lines = code_lines[:start] + code_lines[start + chunk :]
            if not cand_lines:
                continue
            cand_src = prefix + "\n.code\n" + "\n".join(cand_lines) + "\n"
            try:
                prog = asm.assemble(cand_src)
            except Exception:
                continue
            cand = dict(best, source=cand_src, words=prog.words)
            err, _ = evaluate_case(cand, clients, profiles)
            if err is not None:
                code_lines = cand_lines
                best = cand
                best_err = err
                n = max(2, n - 1)
                reduced = True
                break
        if not reduced:
            if n >= len(code_lines):
                break
            n = min(len(code_lines), n * 2)

    return best, best_err


def dump_finding(
    dump_dir: Path,
    case: dict[str, Any],
    err: str,
    results: dict[str, dict[str, Any]],
) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    fname = f"finding_seed{case['seed']}_{case['profile']}.impas"
    out_path = dump_dir / fname

    header = [
        f"; FUZZ FINDING: {err}",
        f"; Seed: {case['seed']}, Profile: {case['profile']}",
    ]
    for eng, prof_map in results.items():
        for prof, r in prof_map.items():
            header.append(
                f";   {eng}[{prof}]: status={r.get('status')} pc={r.get('pc')} "
                f"logical_hash={r.get('logical_hash')} flags={r.get('flags')}"
            )

    if case["is_raw_words"]:
        raw_hex = ", ".join(f"0x{w:016X}" for w in case["words"])
        header.append(f"; {{{{RAW_WORDS: {raw_hex}}}}}")
        if case.get("mut_tags"):
            header.append(f"; Mutations: {', '.join(case['mut_tags'])}")
        out_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    else:
        out_path.write_text("\n".join(header) + "\n\n" + (case["source"] or ""), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Multiprocessing Worker
# ---------------------------------------------------------------------------

_W_FUZZER: V11ContractFuzzer | None = None
_W_CORPUS: V11CorpusMutator | None = None
_W_CLIENTS: dict[str, BatchEngineClient] = {}
_W_PROFILES: list[str] = []
_W_ASM: Any = None
_W_MODS: tuple[Any, Any] | None = None
_W_CONFIG: dict[str, Any] = {}


def _worker_init(cfg: dict[str, Any]) -> None:
    global _W_FUZZER, _W_CORPUS, _W_CLIENTS, _W_PROFILES, _W_ASM, _W_MODS, _W_CONFIG
    _W_CONFIG = cfg
    spec_dir = Path(cfg["spec_dir"])
    v11_isa_loader, impas_v11_assembler = _import_spec_modules(spec_dir)
    _W_MODS = (v11_isa_loader, impas_v11_assembler)
    isa = v11_isa_loader.load_isa(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    _W_ASM = impas_v11_assembler.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    _W_FUZZER = V11ContractFuzzer(isa, _W_ASM, set(cfg["pillars"]))
    corpus_dirs = [Path(d) for d in cfg.get("corpus_dirs", [])]
    if corpus_dirs:
        save_dir = Path(cfg["corpus_save_dir"]) if cfg.get("corpus_save_dir") else None
        _W_CORPUS = V11CorpusMutator(
            isa,
            _W_ASM,
            set(cfg["pillars"]),
            corpus_dirs,
            max_corpus_size=cfg.get("corpus_max_size", 2048),
            save_dir=save_dir,
        )
    else:
        _W_CORPUS = None
    _W_PROFILES = cfg["geom_profiles"]
    _W_CLIENTS = {name: BatchEngineClient(cmd, name) for name, cmd in cfg["engine_cmds"].items()}


def _worker_run_chunk(seeds: list[int]) -> dict[str, Any]:
    assert _W_FUZZER is not None and _W_MODS is not None and _W_ASM is not None
    v11_isa_loader, impas_v11_assembler = _W_MODS
    req_profile = _W_CONFIG["profile"]
    max_instrs = _W_CONFIG["max_instrs"]
    do_shrink = _W_CONFIG["shrink"]
    dump_dir = Path(_W_CONFIG["dump_dir"])

    status_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    asm_errors = 0
    op_cov: Counter[int] = Counter()
    subop_cov: Counter[tuple[int, int]] = Counter()
    findings: list[dict[str, Any]] = []
    corpus_grown = 0

    ref_engine = next(iter(_W_CLIENTS.keys()))

    for seed in seeds:
        if req_profile == "corpus":
            prof = "corpus"
        elif req_profile == "all" and _W_CORPUS is not None and (seed % 5 == 0):
            prof = "corpus"
        else:
            prof = _W_FUZZER.pick_profile(seed, req_profile)
        profile_counts[prof] += 1
        try:
            if prof == "corpus":
                assert _W_CORPUS is not None
                case = _W_CORPUS.generate_mutant(seed)
            elif prof == "validator":
                case = _W_FUZZER.generate_validator_case(
                    seed,
                    max_instrs,
                    impas_v11_assembler.encode_word,
                    impas_v11_assembler.encode_extension_word,
                    impas_v11_assembler.decode_word_fields,
                    v11_isa_loader.valid_subop_mask,
                )
            else:
                case = _W_FUZZER.generate_program(seed, prof, max_instrs)
        except Exception as exc:
            asm_errors += 1
            if len(findings) < 5:
                findings.append({"seed": seed, "profile": prof, "error": f"GEN/ASM ERROR: {exc}", "path": None})
            continue

        # Record coverage for non-validator programs (including corpus mutants)
        if prof != "validator":
            words = case["words"]
            i = 0
            while i < len(words):
                code, flags, subop, *_ = impas_v11_assembler.decode_word_fields(words[i])
                if code != 0xFF:
                    op_cov[code] += 1
                    subop_cov[(code, subop)] += 1
                i += 2 if (flags & impas_v11_assembler.FLAG_EXT) and i + 1 < len(words) else 1

        err, results = evaluate_case(case, _W_CLIENTS, _W_PROFILES)
        ref_res = results[ref_engine].get("P0") or next(iter(results[ref_engine].values()))
        status_counts[str(ref_res.get("status", "UNKNOWN"))] += 1
        if prof == "corpus" and _W_CORPUS is not None:
            if _W_CORPUS.record_feedback(case, ref_res):
                corpus_grown += 1

        if err is not None:
            if do_shrink:
                case, shrunk_err = shrink_case(case, _W_CLIENTS, _W_PROFILES, _W_ASM)
                if shrunk_err:
                    err = shrunk_err
                _, results = evaluate_case(case, _W_CLIENTS, _W_PROFILES)
            saved = dump_finding(dump_dir, case, err, results)
            findings.append({"seed": seed, "profile": prof, "error": err, "path": str(saved)})
            if len(findings) >= _W_CONFIG["max_errors"]:
                break

    return {
        "executed": sum(status_counts.values()),
        "asm_errors": asm_errors,
        "status_counts": dict(status_counts),
        "profile_counts": dict(profile_counts),
        "op_cov": dict(op_cov),
        "subop_cov": {f"{k[0]}:{k[1]}": v for k, v in subop_cov.items()},
        "findings": findings,
        "corpus_initial": _W_CORPUS.initial_count if _W_CORPUS is not None else 0,
        "corpus_grown": corpus_grown,
    }


def _close_worker_clients() -> None:
    for c in _W_CLIENTS.values():
        c.close()
    _W_CLIENTS.clear()


# ---------------------------------------------------------------------------
# CLI & Orchestration
# ---------------------------------------------------------------------------

def parse_pillars(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    out = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    for p in out:
        if p not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid pillar {p}; must be 1..5 or 'all'")
    return out


def build_engine_cmds(args: argparse.Namespace) -> dict[str, list[str]]:
    engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    cmds: dict[str, list[str]] = {}
    for eng in engines:
        if eng == "ocaml":
            bin_path = Path(args.ocaml_bin) if args.ocaml_bin else discover_default_ocaml_bin()
            if not bin_path.exists():
                raise FileNotFoundError(f"OCaml binary not found: {bin_path}")
            cmds["ocaml"] = [str(bin_path), "--vm-fixture-batch"]
        elif eng == "java":
            cmds["java"] = shlex.split(args.java_cmd) if args.java_cmd else discover_default_java_cmd()
        elif eng == "cpp_on":
            if not args.cpp_on_cmd:
                raise ValueError("--cpp-on-cmd is required when 'cpp_on' is in --engines")
            cmds["cpp_on"] = shlex.split(args.cpp_on_cmd)
        elif eng == "cpp_off":
            if not args.cpp_off_cmd:
                raise ValueError("--cpp-off-cmd is required when 'cpp_off' is in --engines")
            cmds["cpp_off"] = shlex.split(args.cpp_off_cmd)
        else:
            raise ValueError(f"Unknown engine {eng!r}; expected ocaml, java, cpp_on, or cpp_off")
    return cmds


def print_coverage_report(
    isa: dict[str, Any], pillars: list[int], op_cov: Counter[int], subop_cov: Counter[tuple[int, int]]
) -> None:
    target_ops = [op for op in isa["opcodes"] if op["code"] != 0xFF and op["pillar"] in pillars]
    tot_ops = len(target_ops)
    hit_ops = sum(1 for op in target_ops if op_cov[op["code"]] > 0)
    tot_subops = sum(len(op["subops"]) for op in target_ops)
    hit_subops = 0
    uncovered_ops: list[str] = []
    uncovered_subops: list[str] = []

    for op in target_ops:
        c = op["code"]
        if op_cov[c] == 0:
            uncovered_ops.append(f"0x{c:02X} {op['mnemonic']} (P{op['pillar']})")
        for sid, sname in sorted(op["subops"].items()):
            if subop_cov[(c, sid)] > 0:
                hit_subops += 1
            else:
                uncovered_subops.append(f"0x{c:02X} {op['mnemonic']}.{sname}")

    print(f"\n{BOLD}ISA Coverage Report (pillars={pillars}):{RESET}")
    print(f"  Opcodes : {hit_ops}/{tot_ops} ({100.0 * hit_ops / max(1, tot_ops):.1f}%)")
    print(f"  Subops  : {hit_subops}/{tot_subops} ({100.0 * hit_subops / max(1, tot_subops):.1f}%)")
    if uncovered_ops:
        print(f"  Uncovered opcodes ({len(uncovered_ops)}): {', '.join(uncovered_ops[:15])}")
    if uncovered_subops:
        preview = ", ".join(uncovered_subops[:20])
        more = f" (+{len(uncovered_subops) - 20} more)" if len(uncovered_subops) > 20 else ""
        print(f"  Uncovered subops ({len(uncovered_subops)}): {preview}{more}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ImpulseVM v0.11.0 Contract-Guided & Validator Differential Fuzzer"
    )
    parser.add_argument("--spec-dir", type=Path, default=None, help="Path to impulse-graph-spec")
    parser.add_argument("--pillars", type=str, default="all", help="Pillars to fuzz: 'all' or e.g. '1', '1,2'")
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=("all",) + ALL_PROFILES,
        help="Fuzz generation profile",
    )
    parser.add_argument("--engines", type=str, default="ocaml,java", help="Comma-separated engines (ocaml,java,cpp_on,cpp_off)")
    parser.add_argument("--ocaml-bin", type=Path, default=None, help="Path to impulse-ocaml binary")
    parser.add_argument("--java-cmd", type=str, default=None, help="Command line for Java VmFixtureBatchMain")
    parser.add_argument("--cpp-on-cmd", type=str, default=None, help="Command line for C++ (opt ON) batch runner")
    parser.add_argument("--cpp-off-cmd", type=str, default=None, help="Command line for C++ (opt OFF) batch runner")
    parser.add_argument("--profiles", type=str, default="P0", help="Geometry profiles (e.g. P0 or P0,P1,P2,P3)")
    parser.add_argument("--iterations", type=int, default=500, help="Number of fuzz iterations")
    parser.add_argument("--seed", type=int, default=42, help="Starting PRNG seed")
    parser.add_argument("--max-instrs", type=int, default=28, help="Maximum generated steps per program")
    parser.add_argument("--max-errors", type=int, default=10, help="Stop after this many divergences")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--chunk-size", type=int, default=50, help="Seeds per worker task chunk")
    parser.add_argument("--shrink", action="store_true", help="Delta-debug shrink any divergence")
    parser.add_argument("--coverage-report", action="store_true", help="Print ISA opcode/subop coverage report")
    parser.add_argument("--dump-dir", type=Path, default=TOOLS_DIR / "fuzz_findings_v11", help="Directory for finding repros")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        action="append",
        default=[],
        help="Seed corpus directory of .impas files (repeatable; defaults to <spec>/test-vectors/vm-impas when --profile corpus)",
    )
    parser.add_argument("--corpus-save-dir", type=Path, default=None, help="Directory to persist newly discovered corpus entries")
    parser.add_argument("--corpus-max-size", type=int, default=2048, help="Maximum in-memory corpus entries per worker")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    spec_dir = resolve_spec_dir(args.spec_dir)
    v11_isa_loader, _ = _import_spec_modules(spec_dir)
    isa = v11_isa_loader.load_isa(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
    pillars = parse_pillars(args.pillars)
    geom_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    engine_cmds = build_engine_cmds(args)

    corpus_dirs: list[str] = []
    for raw_d in args.corpus_dir:
        for part in raw_d.split(","):
            if part.strip():
                corpus_dirs.append(str(Path(part.strip()).resolve()))
    if args.profile == "corpus" and not corpus_dirs:
        corpus_dirs.append(str((spec_dir / "test-vectors" / "vm-impas").resolve()))

    print(
        f"{BOLD}ImpulseVM v0.11.0 Differential Fuzzer{RESET} | "
        f"engines={','.join(engine_cmds.keys())} | pillars={pillars} | "
        f"profile={args.profile} | geom={','.join(geom_profiles)} | "
        f"iters={args.iterations} | seed={args.seed} | workers={args.workers}"
        + (f" | corpus_dirs={len(corpus_dirs)}" if corpus_dirs else "")
    )

    cfg = {
        "spec_dir": str(spec_dir),
        "pillars": pillars,
        "profile": args.profile,
        "geom_profiles": geom_profiles,
        "engine_cmds": engine_cmds,
        "max_instrs": args.max_instrs,
        "max_errors": args.max_errors,
        "shrink": args.shrink,
        "dump_dir": str(args.dump_dir),
        "corpus_dirs": corpus_dirs,
        "corpus_save_dir": str(args.corpus_save_dir) if args.corpus_save_dir else None,
        "corpus_max_size": args.corpus_max_size,
    }

    all_seeds = [args.seed + i for i in range(args.iterations)]
    chunks = [all_seeds[i : i + args.chunk_size] for i in range(0, len(all_seeds), args.chunk_size)]

    t0 = time.perf_counter()
    total_exec = 0
    total_asm_err = 0
    total_corpus_grown = 0
    corpus_initial = 0
    status_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    op_cov: Counter[int] = Counter()
    subop_cov: Counter[tuple[int, int]] = Counter()
    findings: list[dict[str, Any]] = []

    if args.workers <= 1:
        _worker_init(cfg)
        try:
            for ch in chunks:
                res = _worker_run_chunk(ch)
                total_exec += res["executed"]
                total_asm_err += res["asm_errors"]
                corpus_initial = max(corpus_initial, res.get("corpus_initial", 0))
                total_corpus_grown += res.get("corpus_grown", 0)
                status_counts.update(res["status_counts"])
                profile_counts.update(res["profile_counts"])
                op_cov.update(res["op_cov"])
                for k, v in res["subop_cov"].items():
                    c_s, s_s = k.split(":")
                    subop_cov[(int(c_s), int(s_s))] += v
                findings.extend(res["findings"])
                if len(findings) >= args.max_errors:
                    break
        finally:
            _close_worker_clients()
    else:
        with mp.Pool(processes=args.workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            for res in pool.imap_unordered(_worker_run_chunk, chunks):
                total_exec += res["executed"]
                total_asm_err += res["asm_errors"]
                corpus_initial = max(corpus_initial, res.get("corpus_initial", 0))
                total_corpus_grown += res.get("corpus_grown", 0)
                status_counts.update(res["status_counts"])
                profile_counts.update(res["profile_counts"])
                op_cov.update(res["op_cov"])
                for k, v in res["subop_cov"].items():
                    c_s, s_s = k.split(":")
                    subop_cov[(int(c_s), int(s_s))] += v
                findings.extend(res["findings"])
                if len(findings) >= args.max_errors:
                    pool.terminate()
                    break

    dt = max(1e-6, time.perf_counter() - t0)
    rate = total_exec / dt

    print(
        f"\nCompleted {total_exec}/{args.iterations} fixtures in {dt:.2f}s "
        f"({rate:.1f} iters/s, asm_errors={total_asm_err}, findings={len(findings)})"
    )
    if corpus_dirs:
        print(f"Corpus: initial={corpus_initial} seeds, promoted=+{total_corpus_grown} new behavioral signatures")
    print(f"Profiles: {dict(sorted(profile_counts.items()))}")
    print(f"Statuses: {dict(status_counts.most_common())}")

    if args.coverage_report:
        print_coverage_report(isa, pillars, op_cov, subop_cov)

    if findings:
        print(f"\n{RED}{BOLD}DIVERGENCES FOUND ({len(findings)}):{RESET}")
        for f in findings[: args.max_errors]:
            print(f"  - seed={f['seed']} [{f['profile']}]: {f['error']}")
            if f.get("path"):
                print(f"    repro: {f['path']}")
        return 1

    print(f"\n{GREEN}{BOLD}PASS: 0 divergences across {total_exec} fixtures!{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
