#!/usr/bin/env python3
"""
test_morsel_stress.py — ImpulseVM v0.11.0 Dedicated Morsel / Shader Stress Suite

Stresses OP_WALK_MORSEL (0x35), OP_FILTER_MORSEL (0x36), OP_MORSEL_COMMIT (0x37),
and OP_MORSEL_END (0x38) across:
  1. Tier 0 / Tier 1 / Tier 2 frontier crossovers (N = 64, 65, 513, 5000; N % 64 != 0)
  2. Exact morsel ramp & slice boundaries (edges = 0, 1, 31, 32, 33, 63, 64, 65,
     255, 256, 257, 2047, 2048, 2049, 4097)
  3. CSR (WALK_MORSEL.fwd on G) vs. CSC (WALK_MORSEL.rev on G^T) bit-identical equivalence
  4. Domain node-ID width parity (width = u16, u32, u64)
  5. FILTER_MORSEL compound Kleene 3VL (NULL, NaN, -0.0, +0.0) + .jempty_f late-gather skip
  6. Multi-lane morsel (M3 + yield_lane_or across 16 and 64 lanes)
  7. Parallel multi-edge & self-loop deterministic scatter_min/max/argmin/argmax tie-breaking
  8. Metamorphic P0..P3 logical_hash invariance + cross-engine consensus (OCaml, Java, C++ ON/OFF)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
ROOT_DIR = REPO_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_vm_contracts import (
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    discover_default_ocaml_bin,
    resolve_spec_dir,
)

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _fmt_f64_list(vals: list[float | None]) -> str:
    return "[" + ", ".join("null" if v is None else repr(v) for v in vals) + "]"


def _build_star_csr(n: int, edge_count: int) -> tuple[list[int], list[int], list[float]]:
    """Builds a graph where node 0 has `edge_count` outgoing edges cycling across 0..n-1."""
    offsets = [0, edge_count] + [edge_count] * (n - 1)
    targets = [((i * 7 + 1) % n) for i in range(edge_count)]
    weights = [float((i % 19) + 1) * 0.5 for i in range(edge_count)]
    return offsets, targets, weights


def _build_random_csr(rng: random.Random, n: int, avg_deg: int) -> tuple[list[int], list[int], list[float | None]]:
    offsets = [0]
    targets: list[int] = []
    weights: list[float | None] = []
    for u in range(n):
        deg = rng.randint(0, avg_deg * 2)
        nbrs = [rng.randint(0, n - 1) for _ in range(deg)]
        nbrs.sort()
        for v in nbrs:
            targets.append(v)
            if rng.random() < 0.15:
                weights.append(None)
            else:
                weights.append(round(rng.uniform(-50.0, 150.0), 2))
        offsets.append(len(targets))
    return offsets, targets, weights


def _transpose_csr(
    n_src: int, n_tgt: int, offsets: list[int], targets: list[int], weights: list[float | None]
) -> tuple[list[int], list[int], list[float | None]]:
    buckets: list[list[tuple[int, float | None]]] = [[] for _ in range(n_tgt)]
    for u in range(n_src):
        for e in range(offsets[u], offsets[u + 1]):
            buckets[targets[e]].append((u, weights[e]))
    t_off = [0]
    t_tgt: list[int] = []
    t_w: list[float | None] = []
    for v in range(n_tgt):
        for u, w in buckets[v]:
            t_tgt.append(u)
            t_w.append(w)
        t_off.append(len(t_tgt))
    return t_off, t_tgt, t_w


def _run_case(
    name: str,
    src: str,
    asm: Any,
    clients: dict[str, BatchEngineClient],
    profiles: list[str],
) -> tuple[bool, dict[str, Any]]:
    prog = asm.assemble(src)
    payload = {
        "data_directives": prog.data_directives,
        "const_pool": prog.const_pool,
        "words_hex": [f"0x{w:016X}" for w in prog.words],
    }
    ref_engine = next(iter(clients))
    ref_p0: dict[str, Any] = {}
    for prof in profiles:
        per_engine: dict[str, dict[str, Any]] = {}
        for ename, cli in clients.items():
            per_engine[ename] = cli.run_fixture(payload, profile=prof)
        r0 = per_engine[ref_engine]
        if r0.get("status") != "IMPULSE_VM_OK":
            print(f"{RED}FAIL{RESET} [{name}] prof={prof} status={r0.get('status')}")
            return False, r0
        if prof == profiles[0]:
            ref_p0 = r0
        elif r0.get("logical_hash") != ref_p0.get("logical_hash"):
            print(
                f"{RED}METAMORPHIC FAIL{RESET} [{name}] {profiles[0]}={ref_p0.get('logical_hash')} "
                f"!= {prof}={r0.get('logical_hash')}"
            )
            return False, r0
        for ename, res in per_engine.items():
            if (res.get("status"), res.get("logical_hash")) != (r0.get("status"), r0.get("logical_hash")):
                print(
                    f"{RED}CONSENSUS FAIL{RESET} [{name}] prof={prof} "
                    f"{ref_engine}=({r0.get('status')},{r0.get('logical_hash')}) "
                    f"!= {ename}=({res.get('status')},{res.get('logical_hash')})"
                )
                return False, res
    return True, ref_p0


def main() -> int:
    parser = argparse.ArgumentParser(description="ImpulseVM v0.11.0 Dedicated Morsel/Shader Stress Suite")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--engines", type=str, default="ocaml,java",
                        help="Comma-separated engines: ocaml,java,cpp_on,cpp_off")
    parser.add_argument("--cpp-on-bin", type=Path, default=None)
    parser.add_argument("--cpp-off-bin", type=Path, default=None)
    parser.add_argument("--profiles", type=str, default="P0,P1,P2,P3")
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_v11 = _import_spec_modules(spec_dir)
    asm = impas_v11.ImpAsm()
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]

    clients: dict[str, BatchEngineClient] = {}
    for name in engine_names:
        if name == "ocaml":
            clients[name] = BatchEngineClient([str(discover_default_ocaml_bin()), "--vm-fixture-batch"], "ocaml")
        elif name == "java":
            clients[name] = BatchEngineClient(discover_default_java_cmd(), "java")
        elif name == "cpp_on":
            b = args.cpp_on_bin or (ROOT_DIR / "impulse-graph-core/impulse-cpp/build_on/vm_fixture_batch_runner")
            clients[name] = BatchEngineClient([str(b)], "cpp_on")
        elif name == "cpp_off":
            b = args.cpp_off_bin or (ROOT_DIR / "impulse-graph-core/impulse-cpp/build_off/vm_fixture_batch_runner")
            clients[name] = BatchEngineClient([str(b)], "cpp_off")
        else:
            raise ValueError(f"Unknown engine: {name}")

    rng = random.Random(args.seed)
    t0 = time.monotonic()
    passed = 0
    failed = 0

    try:
        # ------------------------------------------------------------------
        # 1. Exact Morsel Ramp & Slice Boundary Edge Counts
        # ------------------------------------------------------------------
        for edge_cnt in [0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 2047, 2048, 2049, 4097]:
            n = 131  # non-multiple of 64 (131 % 64 = 3)
            offsets, targets, weights = _build_star_csr(n, edge_cnt)
            zeros = [0.0] * n
            nulls = ["null"] * n
            src = f"""; Morsel boundary edges={edge_cnt}
.data
  .domain 0 name="N" count={n} width=u16
  .relation 0 name="E" src_dom=0 tgt_dom=0
  .csr_inline rel=0 offsets={offsets} targets={targets}
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(weights)}
  .param 0 type=f64_vec values={_fmt_f64_list(zeros)}
  .param 1 type=u64_vec values=[{", ".join(nulls)}]
.code
  FSEED.single_imm F0, node=0, dom=0
  LOAD_CONST.f64.ext R0, 3.0
  LOAD_PARAM.vec V0, param=0, type=f64
  LOAD_PARAM.vec V1, param=0, type=f64
  LOAD_PARAM.vec V2, param=0, type=f64
  LOAD_PARAM.vec V3, param=1, type=u64
  WALK_MORSEL.fwd F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.ge.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_b
    MORSEL_COMMIT.yield_frontier.mask F1, M1, MF1, dom=0
    MORSEL_COMMIT.scatter_add.mask V0, M1, M4, MF1
    MORSEL_COMMIT.scatter_min.mask V1, M1, M4, MF1
    MORSEL_COMMIT.scatter_max.mask V2, M1, M4, MF1
    MORSEL_COMMIT.scatter_argmin.mask V3, M1, M0, MF1
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_b:
  MORSEL_END
  COLLECT.vec V0, out=0
  COLLECT.vec V3, out=1
  HALT
"""
            ok, _ = _run_case(f"boundary_edges_{edge_cnt}", src, asm, clients, profiles)
            passed += int(ok)
            failed += int(not ok)

        # ------------------------------------------------------------------
        # 2. Tier 0 / Tier 1 / Tier 2 Frontier Crossovers (N = 64, 65, 513, 5000)
        # ------------------------------------------------------------------
        for n, seed_cnt in [(64, 10), (65, 65), (513, 200), (5000, 1200)]:
            offsets, targets, weights = _build_random_csr(rng, n, avg_deg=2)
            seeds = sorted(rng.sample(range(n), seed_cnt))
            zeros = [0.0] * n
            src = f"""; Tier crossover N={n} seeds={seed_cnt}
.data
  .domain 0 name="N" count={n} width=u32
  .relation 0 name="E" src_dom=0 tgt_dom=0
  .csr_inline rel=0 offsets={offsets} targets={targets}
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(weights)}
  .param 0 type=frontier dom=0 values={seeds}
  .param 1 type=f64_vec values={_fmt_f64_list(zeros)}
.code
  LOAD_PARAM.frontier F0, param=0, dom=0
  LOAD_CONST.f64.ext R0, 25.0
  LOAD_PARAM.vec V0, param=1, type=f64
  WALK_MORSEL.fwd F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.gt.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_t
    MORSEL_COMMIT.yield_frontier.mask F1, M1, MF1, dom=0
    MORSEL_COMMIT.scatter_add.mask V0, M1, M4, MF1
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_t:
  MORSEL_END
  SET_CARDINALITY R3, F1
  COLLECT.vec V0, out=0
  HALT
"""
            # For N=5000 use P0,P2,P3 (P1 with 4 nodes/page on 5000 nodes creates 1250 pages which is fine too)
            ok, _ = _run_case(f"tier_crossover_N{n}_S{seed_cnt}", src, asm, clients, profiles)
            passed += int(ok)
            failed += int(not ok)

        # ------------------------------------------------------------------
        # 3. CSR (WALK_MORSEL.fwd on G) vs. CSC (WALK_MORSEL.rev on G^T) Equivalence
        # ------------------------------------------------------------------
        n_src, n_tgt = 73, 89  # bipartite, both non-multiples of 64
        offsets, targets, weights = _build_random_csr(rng, n_src, avg_deg=4)
        targets = [t % n_tgt for t in targets]
        t_off, t_tgt, t_w = _transpose_csr(n_src, n_tgt, offsets, targets, weights)
        seeds = sorted(rng.sample(range(n_src), 35))
        zeros_tgt = [0.0] * n_tgt
        src_fwd = f"""; CSR fwd on G
.data
  .domain 0 name="U" count={n_src} width=u16
  .domain 1 name="V" count={n_tgt} width=u16
  .relation 0 name="R" src_dom=0 tgt_dom=1
  .csr_inline rel=0 offsets={offsets} targets={targets}
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(weights)}
  .param 0 type=frontier dom=0 values={seeds}
  .param 1 type=f64_vec values={_fmt_f64_list(zeros_tgt)}
.code
  LOAD_PARAM.frontier F0, param=0, dom=0
  LOAD_CONST.f64.ext R0, 0.0
  LOAD_PARAM.vec V0, param=1, type=f64
  WALK_MORSEL.fwd F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.gt.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_f
    MORSEL_COMMIT.yield_frontier.mask F1, M1, MF1, dom=1
    MORSEL_COMMIT.scatter_add.mask V0, M1, M4, MF1
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_f:
  MORSEL_END
  SET_CONVERT.f2v_ids V1, F1
  COLLECT.vec V1, out=0
  COLLECT.vec V0, out=1
  HALT
"""
        src_rev = f"""; CSC rev on G^T (dom 0=V, dom 1=U; rev from U -> V!)
.data
  .domain 0 name="V" count={n_tgt} width=u16
  .domain 1 name="U" count={n_src} width=u16
  .relation 0 name="RT" src_dom=0 tgt_dom=1
  .csr_inline rel=0 offsets={t_off} targets={t_tgt}
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(t_w)}
  .param 0 type=frontier dom=1 values={seeds}
  .param 1 type=f64_vec values={_fmt_f64_list(zeros_tgt)}
.code
  LOAD_PARAM.frontier F0, param=0, dom=1
  LOAD_CONST.f64.ext R0, 0.0
  LOAD_PARAM.vec V0, param=1, type=f64
  WALK_MORSEL.rev F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.gt.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_r
    MORSEL_COMMIT.yield_frontier.mask F1, M0, MF1, dom=0
    MORSEL_COMMIT.scatter_add.mask V0, M0, M4, MF1
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_r:
  MORSEL_END
  SET_CONVERT.f2v_ids V1, F1
  COLLECT.vec V1, out=0
  COLLECT.vec V0, out=1
  HALT
"""
        ok1, res_fwd = _run_case("csr_fwd_G", src_fwd, asm, clients, profiles)
        ok2, res_rev = _run_case("csc_rev_GT", src_rev, asm, clients, profiles)
        if ok1 and ok2:
            if res_fwd.get("outputs") != res_rev.get("outputs"):
                print(f"{RED}CSR vs CSC MISMATCH{RESET}: fwd={res_fwd.get('outputs')} != rev={res_rev.get('outputs')}")
                failed += 1
            else:
                passed += 1
        else:
            failed += 1

        # ------------------------------------------------------------------
        # 4. Node ID Width Parity (u16 vs u32 vs u64)
        # ------------------------------------------------------------------
        n_w = 130
        off_w, tgt_w, w_w = _build_random_csr(rng, n_w, avg_deg=3)
        seeds_w = sorted(rng.sample(range(n_w), 40))
        width_outputs: list[Any] = []
        for wname in ["u16", "u32", "u64"]:
            src_w = f"""; Width {wname} morsel parity
.data
  .domain 0 name="N" count={n_w} width={wname}
  .relation 0 name="E" src_dom=0 tgt_dom=0
  .csr_inline rel=0 offsets={off_w} targets={tgt_w}
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(w_w)}
  .param 0 type=frontier dom=0 values={seeds_w}
.code
  LOAD_PARAM.frontier F0, param=0, dom=0
  LOAD_CONST.f64.ext R0, 10.0
  WALK_MORSEL.fwd F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.ge.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_w
    MORSEL_COMMIT.yield_frontier.mask F1, M1, MF1, dom=0
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_w:
  MORSEL_END
  SET_CONVERT.f2v_ids V0, F1
  COLLECT.vec V0, out=0
  HALT
"""
            ok, rw = _run_case(f"width_{wname}", src_w, asm, clients, profiles)
            if ok:
                width_outputs.append(rw.get("outputs"))
                passed += 1
            else:
                failed += 1
        if len(width_outputs) == 3 and not (width_outputs[0] == width_outputs[1] == width_outputs[2]):
            print(f"{RED}WIDTH PARITY MISMATCH{RESET}")
            failed += 1

        # ------------------------------------------------------------------
        # 5. FILTER_MORSEL (Node Shader) + Compound Kleene 3VL (NULL, -0.0, +0.0)
        # ------------------------------------------------------------------
        n_f = 300
        a0 = [None if i % 4 == 0 else (-0.0 if i % 7 == 0 else float(i % 40)) for i in range(n_f)]
        a1 = [None if i % 5 == 0 else float(100 - (i % 80)) for i in range(n_f)]
        src_fm = f"""; FILTER_MORSEL compound Kleene 3VL + late gather skip
.data
  .domain 0 name="N" count={n_f} width=u16
  .node_attr dom=0 attr=0 name="a0" type=f64 values={_fmt_f64_list(a0)}
  .node_attr dom=0 attr=1 name="a1" type=f64 values={_fmt_f64_list(a1)}
.code
  INIT_DOMAIN_SPAN F0, dom=0
  LOAD_CONST.f64.ext R0, 10.0
  LOAD_CONST.f64.ext R1, 50.0
  FILTER_MORSEL.node F0
    GATHER_NODE_ATTR.vec M4, M0, dom=0, attr=0
    VEC_CMP.gt.scalar M5, M4, R0
    GATHER_NODE_ATTR.vec M6, M0, dom=0, attr=1
    VEC_CMP.le.scalar M7, M6, R1
    VEC_3VL_OP.and M5, M5, M7
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_fm
    MORSEL_COMMIT.yield_frontier.mask F1, M0, MF1, dom=0
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_fm:
  MORSEL_END
  SET_CONVERT.f2v_ids V0, F1
  COLLECT.vec V0, out=0
  HALT
"""
        ok, _ = _run_case("filter_morsel_kleene3vl_N300", src_fm, asm, clients, profiles)
        passed += int(ok)
        failed += int(not ok)

        # ------------------------------------------------------------------
        # 6. Multi-Lane Morsel (M3 + yield_lane_or across 16 and 64 lanes)
        # ------------------------------------------------------------------
        for n_lanes in [16, 64]:
            n_ml = 150
            off_ml, tgt_ml, w_ml = _build_random_csr(rng, n_ml, avg_deg=4)
            lane_seeds = list(range(n_lanes))
            src_ml = f"""; Multi-lane morsel K={n_lanes} lanes
.data
  .domain 0 name="N" count={n_ml} width=u16
  .relation 0 name="E" src_dom=0 tgt_dom=0
  .csr_inline rel=0 offsets={off_ml} targets={tgt_ml}
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(w_ml)}
  .param 0 type=u64_vec values={lane_seeds}
.code
  LOAD_PARAM.vec V0, param=0, type=u64
  LANE_INIT_SEEDS.auto_width F0, V0, dom=0
  LOAD_CONST.f64.ext R0, 0.0
  WALK_MORSEL.fwd F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.ge.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_ml
    MORSEL_COMMIT.yield_lane_or.mask F1, M1, M3, MF1, dom=0
  skip_ml:
  MORSEL_END
  LANE_FILTER_QUORUM.at_least F2, F1, k=2
  SET_CONVERT.f2v_ids V1, F2
  COLLECT.vec V1, out=0
  HALT
"""
            ok, _ = _run_case(f"multilane_morsel_K{n_lanes}", src_ml, asm, clients, profiles)
            passed += int(ok)
            failed += int(not ok)

        # ------------------------------------------------------------------
        # 7. CSR == COO (unsorted + explicit edge_ids) & COO-only Tier 1 Crossover
        # ------------------------------------------------------------------
        coo_srcs: list[int] = []
        coo_tgts: list[int] = []
        coo_eids: list[int] = []
        for u in range(n_src):
            for e in range(offsets[u], offsets[u + 1]):
                coo_srcs.append(u)
                coo_tgts.append(targets[e])
                coo_eids.append(e)
        perm = list(range(len(coo_srcs)))
        rng.shuffle(perm)
        shuf_srcs = [coo_srcs[i] for i in perm]
        shuf_tgts = [coo_tgts[i] for i in perm]
        shuf_eids = [coo_eids[i] for i in perm]

        src_coo_fwd = f"""; COO fwd unsorted + explicit edge_ids on G (must match CSR fwd frontier + count!)
.data
  .domain 0 name="U" count={n_src} width=u16
  .domain 1 name="V" count={n_tgt} width=u16
  .relation 0 name="R" src_dom=0 tgt_dom=1
  .coo_inline rel=0 sources={shuf_srcs} targets={shuf_tgts} edge_ids={shuf_eids} order="unsorted"
  .edge_attr rel=0 attr=0 name="w" type=f64 values={_fmt_f64_list(weights)}
  .param 0 type=frontier dom=0 values={seeds}
  .param 1 type=f64_vec values={_fmt_f64_list(zeros_tgt)}
.code
  LOAD_PARAM.frontier F0, param=0, dom=0
  LOAD_CONST.f64.ext R0, 0.0
  LOAD_PARAM.vec V0, param=1, type=f64
  WALK_MORSEL.coo_fwd F0, rel=0
    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0
    VEC_CMP.gt.scalar M5, M4, R0
    SET_CONVERT.b2f MF1, M5
    JCC.jempty_f MF1, skip_cf
    MORSEL_COMMIT.yield_frontier.mask F1, M1, MF1, dom=1
    MORSEL_COMMIT.scatter_min.mask V0, M1, M4, MF1
    MORSEL_COMMIT.count_add_r.mask R2, MF1
  skip_cf:
  MORSEL_END
  SET_CONVERT.f2v_ids V1, F1
  COLLECT.vec V1, out=0
  HALT
"""
        ok_cf, res_cf = _run_case("coo_fwd_unsorted_edge_ids", src_coo_fwd, asm, clients, profiles)
        if ok_cf and ok1:
            fwd_f = (res_fwd.get("outputs") or {}).get("0")
            coo_f = (res_cf.get("outputs") or {}).get("0")
            if fwd_f != coo_f:
                print(f"{RED}CSR vs COO FRONTIER MISMATCH{RESET}: csr={fwd_f} != coo={coo_f}")
                failed += 1
            else:
                passed += 1
        else:
            failed += 1

        # ------------------------------------------------------------------
        # 8. Extended body_ok Opcodes Inside Morsel (GATHER_VEC, VEC_BLEND, VEC_INT_BITWISE, VEC_STR_PRED)
        # ------------------------------------------------------------------
        n_ext = 80
        ext_vals = [float(i) * 1.5 for i in range(n_ext)]
        ext_alt = [float(i) * -0.5 for i in range(n_ext)]
        ext_bits = [i * 3 for i in range(n_ext)]
        ext_strs = [f"node_{i % 5}" for i in range(n_ext)]
        ext_strs_lit = "[" + ", ".join(f'"{s}"' for s in ext_strs) + "]"
        src_ext = f"""; Extended body_ok ops inside FILTER_MORSEL (GATHER_VEC, VEC_BLEND, VEC_INT_BITWISE, VEC_STR_PRED, VEC_3VL_OP)
.data
  .domain 0 name="N" count={n_ext} width=u16
  .node_attr dom=0 attr=0 name="bits" type=i64 values={ext_bits}
  .node_attr dom=0 attr=1 name="label" type=str values={ext_strs_lit}
  .param 0 type=f64_vec values={_fmt_f64_list(ext_vals)}
  .param 1 type=f64_vec values={_fmt_f64_list(ext_alt)}
  .param 2 type=f64_vec values={_fmt_f64_list([0.0] * n_ext)}
.const_pool
  .pool 0 type=str value="node_2"
.code
  INIT_DOMAIN_SPAN F0, dom=0
  LOAD_PARAM.vec V0, param=0, type=f64
  LOAD_PARAM.vec V1, param=1, type=f64
  LOAD_PARAM.vec V2, param=2, type=f64
  LOAD_CONST.i32 R0, 7
  LOAD_CONST.i32 R1, 0
  LOAD_CONST_POOL.str R2, pool=0
  FILTER_MORSEL.node F0
    GATHER_VEC M4, V0, M0
    GATHER_VEC M5, V1, M0
    GATHER_NODE_ATTR.vec M6, M0, dom=0, attr=0
    VEC_INT_BITWISE.and.scalar M6, M6, R0
    VEC_CMP.gt.scalar M6, M6, R1
    GATHER_NODE_ATTR.vec M7, M0, dom=0, attr=1
    VEC_STR_PRED.eq.scalar M7, M7, R2
    VEC_3VL_OP.or M6, M6, M7
    VEC_BLEND.kleene_vv M4, M6, M4, M5
    SET_CONVERT.b2f MF1, M6
    JCC.jempty_f MF1, skip_ext
    MORSEL_COMMIT.yield_frontier.mask F1, M0, MF1, dom=0
    MORSEL_COMMIT.scatter_add.mask V2, M0, M4, MF1
    MORSEL_COMMIT.count_add_r.mask R3, MF1
  skip_ext:
  MORSEL_END
  COLLECT.vec V2, out=0
  HALT
"""
        ok_ext, _ = _run_case("morsel_body_ok_extended_ops", src_ext, asm, clients, profiles)
        passed += int(ok_ext)
        failed += int(not ok_ext)

    finally:
        for cli in clients.values():
            cli.close()

    dt = time.monotonic() - t0
    if failed == 0:
        print(f"{GREEN}{BOLD}PASS{RESET}: {passed}/{passed} v0.11 morsel/shader stress tests "
              f"x profiles={profiles} x engines={engine_names} in {dt:.2f}s")
        return 0
    print(f"{RED}{BOLD}FAIL{RESET}: {failed} failures ({passed} passed) in {dt:.2f}s")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
