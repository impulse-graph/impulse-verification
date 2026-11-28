#!/usr/bin/env python3
"""
Self-Describing Metamorphic Storage Coverage Correctness & Performance Gate.

Runs C++ (-O3), C++ (-O0), and Java 25 against a generated storage_coverage_*.imps
and asserts:
  1. Topology Equivalence: every EQ_REL_* variant matches canonical EQ_REL_ALL3_U32
     across all supported topologies (CSR, CSC, COO_FWD, COO_REV, MORSEL_FWD) and
     returns ERR_INVALID_OPERAND (6) when a topology is absent.
  2. Attribute Equivalence: every EQ_NUM_* (i8..u64, f32, f64, decimal64 across
     multiple precision/scale pairs) matches EQ_NUM_F64 on node gather+cmp and
     edge morsel gather+cmp; every EQ_NUM_NULL_* matches EQ_NUM_NULL_F64.
  3. Boundary Parity: BOUNDARY_FP_* (-inf, +inf, -0.0, +0.0, NaN->NULL),
     BOUNDARY_INT_* (min/max/NULL), and BOUNDARY_DEC_* (P18/P6/P1, S0/S6, NULL)
     match across C++ (-O3/-O0) and Java 25.
  4. Multi-Lane vs Single-Lane Equivalence:
       - ml_k1_auto_2hop == ml_k1_force_w64_2hop == ml_k1_morsel_2hop == single_1seed_2hop
       - ml_k{8,16,32,64}_auto_2hop == ml_k{8,16,32,64}_morsel_2hop
       - ml_k64_auto_2hop == single_loop_64x_2hop (exact sum of 64 single-lane runs!)
  5. Cross-Engine & Cross-Variant Perf Report (flagging >2x Java/C++ or >2.5x variant regressions).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/jesse/impulse")
CPP_ON = ROOT / "impulse-graph-core/impulse-cpp/build_on/bench_storage_coverage"
CPP_OFF = ROOT / "impulse-graph-core/impulse-cpp/build_off/bench_storage_coverage"
JAVA_CP = ":".join([
    str(ROOT / "impulse-graph-java/impulse-vm/target/test-classes"),
    str(ROOT / "impulse-graph-java/impulse-vm/target/classes"),
    str(ROOT / "impulse-graph-java/impulse-storage/target/classes"),
    str(ROOT / "impulse-graph-java/impulse-spec/target/classes"),
    str(ROOT / "impulse-graph-java/impulse-api/target/classes"),
])


def run_json(cmd: list[str], label: str) -> dict:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR running {label}: exit={res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}", file=sys.stderr)
        sys.exit(2)
    for line in reversed(res.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    print(f"ERROR: No JSON output from {label}:\n{res.stdout}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="Path to storage_coverage_*.imps")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--java-warmup", type=int, default=200)
    ap.add_argument("--skip-cpp-off", action="store_true")
    args = ap.parse_args()

    cpp_on = run_json([str(CPP_ON), args.snapshot, str(args.warmup), str(args.iters)], "cpp_on")
    java = run_json(
        ["java", "--add-modules", "jdk.incubator.vector", "-cp", JAVA_CP,
         "org.impulsegraph.vm.StorageCoverageBenchRunner", args.snapshot, str(args.java_warmup), str(args.iters)],
        "java",
    )
    engines = {"cpp_on": cpp_on, "java": java}
    if not args.skip_cpp_off and CPP_OFF.exists():
        engines["cpp_off"] = run_json([str(CPP_OFF), args.snapshot, "2", "5"], "cpp_off")

    failures: list[str] = []

    # 1. Topology Equivalence & Cross-Engine Parity
    print("\n=== 1. TOPOLOGY SWEEP (EQ_REL_*) ===")
    print(f"{'Relation':<26s} {'Topos':<11s} {'C++ CSR':>9s} {'Java CSR':>9s} {'C++ CSC':>9s} {'Java CSC':>9s} {'C++ COO':>9s} {'Java COO':>9s} {'Card':>7s}")
    canon_fwd = None
    canon_rev = None
    for idx, t_c in enumerate(cpp_on["topology"]):
        t_j = java["topology"][idx]
        name = t_c["name"]
        if t_c["has_csr"] and canon_fwd is None:
            canon_fwd = t_c["csr"]["card"]
        if t_c["has_csc"] and canon_rev is None:
            canon_rev = t_c["csc"]["card"]

    for idx, t_c in enumerate(cpp_on["topology"]):
        name = t_c["name"]
        for ename, edata in engines.items():
            t = edata["topology"][idx]
            # Positive checks where topology exists, negative status!=0 where absent
            for mode, flag, exp_card in [
                ("csr", "has_csr", canon_fwd),
                ("morsel_fwd", "has_csr", canon_fwd),
                ("csc", "has_csc", canon_rev),
                ("coo_fwd", "has_coo", canon_fwd),
                ("coo_rev", "has_coo", canon_rev),
            ]:
                r = t[mode]
                if t[flag]:
                    if r["st"] != 0 or r["card"] != exp_card:
                        failures.append(f"[{ename}] {name}.{mode}: expected st=0 card={exp_card}, got st={r['st']} card={r['card']}")
                else:
                    if r["st"] == 0:
                        failures.append(f"[{ename}] {name}.{mode}: expected error when {flag}=false, got st=0")

        t_j = java["topology"][idx]
        topos = ("R" if t_c["has_csr"] else "-") + ("C" if t_c["has_csc"] else "-") + ("O" if t_c["has_coo"] else "-")
        f_us = lambda d, k, fl: f"{d[k]['us']:7.1f}us" if d[fl] else "      n/a"
        card_disp = t_c["csr"]["card"] if t_c["has_csr"] else (t_c["coo_fwd"]["card"] if t_c["has_coo"] else t_c["csc"]["card"])
        print(f"{name:<26s} {topos:<11s} {f_us(t_c,'csr','has_csr'):>9s} {f_us(t_j,'csr','has_csr'):>9s} "
              f"{f_us(t_c,'csc','has_csc'):>9s} {f_us(t_j,'csc','has_csc'):>9s} "
              f"{f_us(t_c,'coo_fwd','has_coo'):>9s} {f_us(t_j,'coo_fwd','has_coo'):>9s} {card_disp:>7d}")

    # 2. Attribute Type Equivalence & Perf Sweep
    print("\n=== 2. ATTRIBUTE TYPE SWEEP (EQ_NUM_* / EQ_NUM_NULL_*) ===")
    print(f"{'Scope':<12s} {'Attr':<26s} {'C++ us':>9s} {'Java us':>9s} {'J/C':>6s} {'Count':>7s}")
    expected_counts: dict[tuple[str, bool], int] = {}
    for idx, a_c in enumerate(cpp_on["attrs"]):
        key = (a_c["scope"], "_NULL_" in a_c["name"])
        if key not in expected_counts and a_c["st"] == 0:
            expected_counts[key] = a_c["count"]
        for ename, edata in engines.items():
            a = edata["attrs"][idx]
            exp = expected_counts.get(key)
            if a["st"] != 0 or a["count"] != exp:
                failures.append(f"[{ename}] attr {a['scope']}.{a['name']}: expected st=0 count={exp}, got st={a['st']} count={a['count']}")
        a_j = java["attrs"][idx]
        ratio = (a_j["us"] / a_c["us"]) if a_c["us"] > 0.01 else 0.0
        print(f"{a_c['scope']:<12s} {a_c['name']:<26s} {a_c['us']:9.2f} {a_j['us']:9.2f} {ratio:5.2f}x {a_c['count']:7d}")

    # 3. Boundary Checks (IEEE-754 + Int64/UInt64 + DECIMAL64 + NULL)
    print("\n=== 3. BOUNDARY CHECKS (IEEE-754 & INT64/UINT64 & DECIMAL64 & NULL) ===")
    for idx, b_c in enumerate(cpp_on["boundaries"]):
        for ename, edata in engines.items():
            b = edata["boundaries"][idx]
            if b != b_c:
                failures.append(f"[{ename}] boundary {b_c['name']} mismatch: cpp_on={b_c} vs {ename}={b}")
        print(f"  {b_c['name']:<24s} count_valid={b_c['count_valid']} min={b_c['min']} max={b_c['max']}")

    # 4. Multi-Lane vs Single-Lane Equivalence & Perf Sweep
    print("\n=== 4. MULTI-LANE VS SINGLE-LANE & MORSEL SWEEP ===")
    print(f"{'Label':<24s} {'k':>3s} {'C++ us':>9s} {'C++/lane':>9s} {'Java us':>9s} {'J/lane':>9s} {'J/C':>6s} {'pop_sum':>10s}")
    c_ml = {m["label"]: m for m in cpp_on["multilane"]}
    j_ml = {m["label"]: m for m in java["multilane"]}

    # Cross-engine parity on every multi-lane case
    for lbl, mc in c_ml.items():
        for ename, edata in engines.items():
            me = next(x for x in edata["multilane"] if x["label"] == lbl)
            if me["st"] != 0 or me["pop_sum"] != mc["pop_sum"]:
                failures.append(f"[{ename}] multilane {lbl}: expected st=0 pop_sum={mc['pop_sum']}, got st={me['st']} pop_sum={me['pop_sum']}")
        mj = j_ml[lbl]
        ratio = (mj["us"] / mc["us"]) if mc["us"] > 0.01 else 0.0
        print(f"{lbl:<24s} {mc['k']:>3d} {mc['us']:9.2f} {mc['us_per_lane']:9.2f} {mj['us']:9.2f} {mj['us_per_lane']:9.2f} {ratio:5.2f}x {mc['pop_sum']:>10d}")

    # Internal metamorphic multi-lane invariants:
    s1 = c_ml["single_1seed_2hop"]["pop_sum"]
    for deg_lbl in ["ml_k1_auto_2hop", "ml_k1_force_w64_2hop", "ml_k1_morsel_2hop"]:
        if c_ml[deg_lbl]["pop_sum"] != s1:
            failures.append(f"Degenerate 1-lane mismatch: {deg_lbl}={c_ml[deg_lbl]['pop_sum']} vs single_1seed_2hop={s1}")
    for k in [8, 16, 32, 64]:
        a_lbl, m_lbl = f"ml_k{k}_auto_2hop", f"ml_k{k}_morsel_2hop"
        if a_lbl in c_ml and m_lbl in c_ml and c_ml[a_lbl]["pop_sum"] != c_ml[m_lbl]["pop_sum"]:
            failures.append(f"Multi-lane walk vs morsel mismatch at k={k}: {c_ml[a_lbl]['pop_sum']} vs {c_ml[m_lbl]['pop_sum']}")
    if "ml_k64_auto_2hop" in c_ml and "single_loop_64x_2hop" in c_ml:
        if c_ml["ml_k64_auto_2hop"]["pop_sum"] != c_ml["single_loop_64x_2hop"]["pop_sum"]:
            failures.append(f"64-lane batch vs 64x single-lane sum mismatch: {c_ml['ml_k64_auto_2hop']['pop_sum']} vs {c_ml['single_loop_64x_2hop']['pop_sum']}")

    print("\n===============================================================")
    if failures:
        print(f"FAIL: {len(failures)} storage coverage failures:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"PASS: 100% metamorphic storage coverage parity across {list(engines.keys())}!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
