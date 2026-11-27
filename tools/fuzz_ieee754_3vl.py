#!/usr/bin/env python3
"""
Suite #5: Adversarial IEEE-754 & SQL 3-Valued Logic (3VL) Corner-Case Differential Fuzzer.

Fuzzes adversarial IEEE-754 edge values (-0.0, +0.0, 1e-308, -1e-308, 1e308, -1e308,
INT64_MIN, INT64_MAX, 0, -1, 1) and nullable vectors (mixed nulls, all-null, all-valid)
through:
  - VEC_CMP.{eq,ne,lt,le,gt,ge} (vec-vec and .scalar) + VEC_3VL_OP.{and,or,not,xor,is_null,is_not_null,is_true,is_false}
  - VEC_MATH_ARITH.{add,sub,mul,div,min,max} (vec-vec and .scalar)
  - VEC_REDUCE.{sum,min,max,count_valid,mean,l1_norm,argmin,argmax} + TOP_K.{desc_ids,asc_ids,desc_scores,asc_scores}
  - MORSEL_COMMIT.{scatter_add,scatter_min,scatter_max,scatter_argmin,scatter_argmax} on snapshot edge attributes
  - GATHER_NODE_ATTR.vec on snapshot nullable & boundary attribute columns
  - SPMV.{fwd_plus_times,rev_plus_times,fwd_min_plus,fwd_max_plus,fwd_max_times,fwd_min_max,fwd_argmin_plus}

Verifies 3-way differential parity (java == cpp_on == cpp_off) across profiles P0..P3.
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

INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807

ADV_F64_VALS = [
    "-0.0",
    "0.0",
    "1.0e-308",
    "-1.0e-308",
    "1.0e308",
    "-1.0e308",
    "1.0",
    "-1.0",
    "0.5",
    "-0.5",
    "1.0e-15",
    "1.0e15",
]

ADV_I64_VALS = [
    str(INT64_MIN),
    str(INT64_MAX),
    "0",
    "-1",
    "1",
    "-2",
    "2",
    "42",
    "-42",
]

CMP_OPS = ["eq", "ne", "lt", "le", "gt", "ge"]
ARITH_OPS = ["add", "sub", "mul", "div", "min", "max"]
REDUCE_NUM_OPS = ["sum", "min", "max", "count_valid", "mean", "l1_norm", "argmin", "argmax"]
TOPK_OPS = ["desc_ids", "asc_ids", "desc_scores", "asc_scores"]
THREE_VL_BIN_OPS = ["and", "or", "xor"]
THREE_VL_UN_OPS = ["not", "is_null", "is_not_null", "is_true", "is_false"]


def _gen_nullable_vec(rng: random.Random, pool: list[str], length: int, mode: str) -> list[str]:
    if mode == "all_null":
        return ["null"] * length
    if mode == "all_valid":
        return [rng.choice(pool) for _ in range(length)]
    # mixed
    out = [rng.choice(pool) if rng.random() > 0.35 else "null" for _ in range(length)]
    if length >= 2 and "null" not in out:
        out[rng.randrange(length)] = "null"
    return out


def _pick_vec_mode(rng: random.Random) -> str:
    return rng.choice(["mixed", "mixed", "all_valid", "all_null"])


def gen_vec_cmp_3vl(rng: random.Random) -> tuple[str, str]:
    is_f64 = rng.random() < 0.65
    pool = ADV_F64_VALS if is_f64 else ADV_I64_VALS
    vtype = "f64" if is_f64 else "i64"
    ptype = "f64_vec" if is_f64 else "i64_vec"
    length = rng.choice([1, 4, 8, 16, 63, 64, 65])

    lhs = _gen_nullable_vec(rng, pool, length, _pick_vec_mode(rng))
    rhs = _gen_nullable_vec(rng, pool, length, _pick_vec_mode(rng))
    use_scalar = rng.random() < 0.5
    scalar_val = rng.choice(pool)
    scalar_null = rng.random() < 0.2

    cmp1 = rng.choice(CMP_OPS)
    cmp2 = rng.choice(CMP_OPS)
    bin3vl = rng.choice(THREE_VL_BIN_OPS)
    un3vl = rng.choice(THREE_VL_UN_OPS)

    lines = [
        ".data",
        f"  .param 0 type={ptype} values=[{', '.join(lhs)}]",
        f"  .param 1 type={ptype} values=[{', '.join(rhs)}]",
    ]
    if use_scalar and not scalar_null:
        lines.append(f"  .param 2 type={vtype} value={scalar_val}")
    lines.append(".code")
    lines.append(f"  LOAD_PARAM.vec V0, param=0, type={vtype}")
    lines.append(f"  LOAD_PARAM.vec V1, param=1, type={vtype}")
    if use_scalar:
        if scalar_null:
            lines.append("  LOAD_CONST.null R0")
        else:
            lines.append(f"  LOAD_PARAM.scalar R0, param=2, type={vtype}")
        lines.append(f"  VEC_CMP.{cmp1}.scalar V2, V0, R0")
        lines.append(f"  VEC_CMP.{cmp2}.scalar V3, V1, R0")
    else:
        lines.append(f"  VEC_CMP.{cmp1} V2, V0, V1")
        lines.append(f"  VEC_CMP.{cmp2} V3, V1, V0")

    lines.extend([
        f"  VEC_3VL_OP.{bin3vl} V4, V2, V3",
        f"  VEC_3VL_OP.{un3vl} V5, V4",
        "  VEC_REDUCE.count_valid R1, V4",
        "  VEC_REDUCE.any_true R2, V4",
        "  VEC_REDUCE.all_true R3, V4",
        "  COLLECT.vec V2, out=0",
        "  COLLECT.vec V4, out=1",
        "  COLLECT.vec V5, out=2",
        "  COLLECT.scalar R1, out=3",
        "  COLLECT.scalar R2, out=4",
        "  COLLECT.scalar R3, out=5",
        "  HALT",
    ])
    return f"vec_cmp_3vl({vtype},{cmp1}/{cmp2},{bin3vl},{un3vl},len={length})", "\n".join(lines) + "\n"


def gen_vec_arith(rng: random.Random) -> tuple[str, str]:
    is_f64 = rng.random() < 0.70
    pool = ADV_F64_VALS if is_f64 else ADV_I64_VALS
    vtype = "f64" if is_f64 else "i64"
    ptype = "f64_vec" if is_f64 else "i64_vec"
    length = rng.choice([1, 4, 8, 16, 64, 65])

    lhs = _gen_nullable_vec(rng, pool, length, _pick_vec_mode(rng))
    rhs = _gen_nullable_vec(rng, pool, length, _pick_vec_mode(rng))
    op = rng.choice(ARITH_OPS)
    use_scalar = rng.random() < 0.45
    scalar_val = rng.choice(pool)
    scalar_null = rng.random() < 0.15

    lines = [
        ".data",
        f"  .param 0 type={ptype} values=[{', '.join(lhs)}]",
        f"  .param 1 type={ptype} values=[{', '.join(rhs)}]",
    ]
    if use_scalar and not scalar_null:
        lines.append(f"  .param 2 type={vtype} value={scalar_val}")
    lines.append(".code")
    lines.append(f"  LOAD_PARAM.vec V0, param=0, type={vtype}")
    lines.append(f"  LOAD_PARAM.vec V1, param=1, type={vtype}")
    if use_scalar:
        if scalar_null:
            lines.append("  LOAD_CONST.null R0")
        else:
            lines.append(f"  LOAD_PARAM.scalar R0, param=2, type={vtype}")
        lines.append(f"  VEC_MATH_ARITH.{op}.scalar V2, V0, R0")
    else:
        lines.append(f"  VEC_MATH_ARITH.{op} V2, V0, V1")

    lines.extend([
        "  VEC_REDUCE.sum R1, V2",
        "  VEC_REDUCE.min R2, V2",
        "  VEC_REDUCE.max R3, V2",
        "  VEC_REDUCE.count_valid R4, V2",
        "  COLLECT.vec V2, out=0",
        "  COLLECT.scalar R1, out=1",
        "  COLLECT.scalar R2, out=2",
        "  COLLECT.scalar R3, out=3",
        "  COLLECT.scalar R4, out=4",
        "  HALT",
    ])
    return f"vec_arith({vtype},{op},scalar={use_scalar},len={length})", "\n".join(lines) + "\n"


def gen_vec_reduce_topk(rng: random.Random) -> tuple[str, str]:
    is_f64 = rng.random() < 0.70
    pool = ADV_F64_VALS if is_f64 else ADV_I64_VALS
    vtype = "f64" if is_f64 else "i64"
    ptype = "f64_vec" if is_f64 else "i64_vec"
    length = rng.choice([1, 4, 8, 16, 32, 65])
    vals = _gen_nullable_vec(rng, pool, length, _pick_vec_mode(rng))
    k = rng.choice([1, 2, 4, min(8, length), length])
    tk1 = rng.choice(TOPK_OPS)
    tk2 = rng.choice(TOPK_OPS)

    lines = [
        ".data",
        f"  .param 0 type={ptype} values=[{', '.join(vals)}]",
        ".code",
        f"  LOAD_PARAM.vec V0, param=0, type={vtype}",
        "  VEC_REDUCE.sum R0, V0",
        "  VEC_REDUCE.min R1, V0",
        "  VEC_REDUCE.max R2, V0",
        "  VEC_REDUCE.count_valid R3, V0",
        "  VEC_REDUCE.argmin R4, V0",
        "  VEC_REDUCE.argmax R5, V0",
        f"  TOP_K.{tk1} V1, V0, k={k}",
        f"  TOP_K.{tk2} V2, V0, k={k}",
        "  COLLECT.scalar R0, out=0",
        "  COLLECT.scalar R1, out=1",
        "  COLLECT.scalar R2, out=2",
        "  COLLECT.scalar R3, out=3",
        "  COLLECT.scalar R4, out=4",
        "  COLLECT.scalar R5, out=5",
        "  COLLECT.vec V1, out=6",
        "  COLLECT.vec V2, out=7",
        "  HALT",
    ]
    return f"vec_reduce_topk({vtype},{tk1}/{tk2},k={k},len={length})", "\n".join(lines) + "\n"


def gen_morsel_scatter(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    rels_with_eattrs = [r for r in schema.relations.values() if r.edge_attrs and _valid_dirs_for_rel(r)]
    if not rels_with_eattrs:
        return gen_vec_arith(rng)
    rel = rng.choice(rels_with_eattrs)
    dsub = rng.choice(_valid_dirs_for_rel(rel))
    is_rev = dsub in ("rev", "coo_rev")
    in_dom = rel.tgt_dom if is_rev else rel.src_dom
    out_dom = rel.src_dom if is_rev else rel.tgt_dom
    id_reg = "M0" if is_rev else "M1"
    n_in = schema.dom_cards[in_dom]

    # Prefer NULL / BOUNDARY edge attributes
    adv_attrs = [a for a in rel.edge_attrs if ("NULL" in a.name or "BOUNDARY" in a.name)]
    attr = rng.choice(adv_attrs if (adv_attrs and rng.random() < 0.75) else rel.edge_attrs)

    seed_cnt = rng.choice([1, 4, 16, 64])
    seeds = sorted({rng.randint(0, n_in - 1) for _ in range(seed_cnt)})
    cmp_op = rng.choice(CMP_OPS)
    use_filter = rng.random() < 0.55

    if attr.is_float:
        scatter_mode = rng.choice(["scatter_add", "scatter_min", "scatter_max", "scatter_argmin", "scatter_argmax"])
        thresh_str = rng.choice(ADV_F64_VALS)
        ptype = "f64"
    else:
        scatter_mode = rng.choice(["scatter_argmin", "scatter_argmax"])
        thresh_str = rng.choice(ADV_I64_VALS)
        ptype = "i64"

    lines = [
        ".data",
        f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
        f"  .param 1 type={ptype} value={thresh_str}",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
        f"  LOAD_PARAM.scalar R5, param=1, type={ptype}",
    ]
    if scatter_mode in ("scatter_add", "scatter_min", "scatter_max"):
        lines.append(f"  VEC_INIT.dom_f64 V1, dom={out_dom}")
    else:
        lines.append(f"  VEC_INIT.dom_u64 V1, dom={out_dom}")

    lines.append(f"  WALK_MORSEL.{dsub} F0, rel={rel.rel_id}")
    lines.append(f"    GATHER_EDGE_ATTR.vec M4, M2, rel={rel.rel_id}, attr={attr.attr_id}")
    if use_filter:
        lines.append(f"    VEC_CMP.{cmp_op}.scalar M5, M4, R5")
        lines.append("    SET_CONVERT.b2f MF1, M5")
        if scatter_mode in ("scatter_add", "scatter_min", "scatter_max"):
            lines.append(f"    MORSEL_COMMIT.{scatter_mode}.mask V1, {id_reg}, M4, MF1")
        else:
            lines.append(f"    MORSEL_COMMIT.{scatter_mode}.mask V1, {id_reg}, M0, MF1")
    else:
        if scatter_mode in ("scatter_add", "scatter_min", "scatter_max"):
            lines.append(f"    MORSEL_COMMIT.{scatter_mode} V1, {id_reg}, M4")
        else:
            lines.append(f"    MORSEL_COMMIT.{scatter_mode} V1, {id_reg}, M0")
    lines.append("  MORSEL_END")
    lines.extend([
        "  VEC_REDUCE.count_valid R0, V1",
        "  VEC_REDUCE.sum R1, V1",
        "  VEC_REDUCE.min R2, V1",
        "  VEC_REDUCE.max R3, V1",
        "  COLLECT.scalar R0, out=0",
        "  COLLECT.scalar R1, out=1",
        "  COLLECT.scalar R2, out=2",
        "  COLLECT.scalar R3, out=3",
        "  HALT",
    ])
    return f"morsel_scatter({rel.name}.{dsub},{attr.name},{scatter_mode},filt={use_filter})", "\n".join(lines) + "\n"


def gen_snapshot_nullable_attrs(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    doms_with_attrs = [d for d, al in schema.node_attrs.items() if al]
    if not doms_with_attrs:
        return gen_vec_cmp_3vl(rng)
    dom_id = rng.choice(doms_with_attrs)
    n_dom = schema.dom_cards[dom_id]
    attrs = schema.node_attrs[dom_id]
    adv_attrs = [a for a in attrs if ("NULL" in a.name or "BOUNDARY" in a.name)]
    attr = rng.choice(adv_attrs if (adv_attrs and rng.random() < 0.80) else attrs)

    cnt = rng.choice([1, 8, 32, 64, 128])
    ids = sorted({rng.randint(0, n_dom - 1) for _ in range(cnt)})
    cmp_op = rng.choice(CMP_OPS)
    if attr.is_float:
        thresh_str = rng.choice(ADV_F64_VALS)
        ptype = "f64"
    else:
        thresh_str = rng.choice(ADV_I64_VALS)
        ptype = "i64"

    lines = [
        ".data",
        f"  .param 0 type=u64_vec values=[{', '.join(str(x) for x in ids)}]",
        f"  .param 1 type={ptype} value={thresh_str}",
        ".code",
        "  LOAD_PARAM.vec V0, param=0, type=u64",
        f"  LOAD_PARAM.scalar R5, param=1, type={ptype}",
        f"  GATHER_NODE_ATTR.vec V1, V0, dom={dom_id}, attr={attr.attr_id}",
        f"  VEC_CMP.{cmp_op}.scalar V2, V1, R5",
        "  VEC_3VL_OP.is_null V3, V1",
        "  VEC_3VL_OP.is_not_null V4, V1",
        "  VEC_REDUCE.count_valid R0, V1",
        "  VEC_REDUCE.sum R1, V1",
        "  VEC_REDUCE.min R2, V1",
        "  VEC_REDUCE.max R3, V1",
        "  VEC_REDUCE.any_true R4, V2",
        "  COLLECT.vec V2, out=0",
        "  COLLECT.vec V3, out=1",
        "  COLLECT.scalar R0, out=2",
        "  COLLECT.scalar R1, out=3",
        "  COLLECT.scalar R2, out=4",
        "  COLLECT.scalar R3, out=5",
        "  COLLECT.scalar R4, out=6",
        "  HALT",
    ]
    return f"snap_attr({schema.dom_names[dom_id]}.{attr.name},{cmp_op},n={len(ids)})", "\n".join(lines) + "\n"


def gen_spmv_adversarial(rng: random.Random, schema: SnapshotSchema) -> tuple[str, str]:
    rels = [
        r for r in schema.relations.values()
        if (r.has_csr or r.has_csc) and any(a.is_float for a in r.edge_attrs)
    ]
    if not rels:
        return gen_vec_arith(rng)
    rel = rng.choice(rels)
    f_attrs = [a for a in rel.edge_attrs if a.is_float]
    adv_attrs = [a for a in f_attrs if ("NULL" in a.name or "BOUNDARY" in a.name)]
    attr = rng.choice(adv_attrs if (adv_attrs and rng.random() < 0.75) else f_attrs)

    subops: list[tuple[str, int]] = []
    if rel.has_csr:
        for s in ("fwd_plus_times", "fwd_min_plus", "fwd_max_plus", "fwd_max_times", "fwd_min_max", "fwd_argmin_plus"):
            subops.append((s, rel.src_dom))
    if rel.has_csc:
        for s in ("rev_plus_times", "rev_min_plus"):
            subops.append((s, rel.tgt_dom))
    subop, in_dom = rng.choice(subops)
    n_in = schema.dom_cards[in_dom]
    seeds = sorted({rng.randint(0, n_in - 1) for _ in range(rng.choice([1, 8, 32, 64]))})

    # Build a full-domain f64 vector with adversarial values & nulls
    mode = rng.choice(["mixed", "all_valid", "all_null"])
    # Pick smaller subset of adversarial values for compact .param
    state_vals = _gen_nullable_vec(rng, ADV_F64_VALS, n_in, mode)

    lines = [
        ".data",
        f"  .param 0 type=frontier dom={in_dom} values=[{', '.join(str(x) for x in seeds)}]",
        f"  .param 1 type=f64_vec values=[{', '.join(state_vals)}]",
        ".code",
        f"  LOAD_PARAM.frontier F0, param=0, dom={in_dom}",
        "  LOAD_PARAM.vec V0, param=1, type=f64",
        f"  SPMV.{subop} V1, F0, V0, rel={rel.rel_id}, attr={attr.attr_id}",
        "  VEC_REDUCE.count_valid R0, V1",
        "  VEC_REDUCE.sum R1, V1",
        "  VEC_REDUCE.min R2, V1",
        "  VEC_REDUCE.max R3, V1",
        "  COLLECT.scalar R0, out=0",
        "  COLLECT.scalar R1, out=1",
        "  COLLECT.scalar R2, out=2",
        "  COLLECT.scalar R3, out=3",
        "  HALT",
    ]
    return f"spmv({rel.name}.{subop},{attr.name},mode={mode})", "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite #5: Adversarial IEEE-754 & SQL 3-Valued Logic Corner-Case Differential Fuzzer"
    )
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()

    snap_path = args.snapshot.resolve()
    lib_path = args.lib_path or _discover_default_lib()
    spec_dir = resolve_spec_dir(args.spec_dir)
    _, impas_asm = _import_spec_modules(spec_dir)
    asm = impas_asm.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")
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
        f"Suite #5 Adversarial IEEE-754 & 3VL Fuzzer | snapshot={snap_path.name} | "
        f"engines={','.join(clients.keys())} | profiles={','.join(profiles)} | "
        f"iters={args.iterations} | seed={args.seed}"
    )

    rng = random.Random(args.seed)
    families = [
        "vec_cmp_3vl",
        "vec_arith",
        "vec_reduce_topk",
        "morsel_scatter",
        "snap_attr",
        "spmv",
    ]
    fam_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    errors = 0
    t0 = time.perf_counter()

    try:
        for i in range(args.iterations):
            fam = families[i % len(families)]
            prof = profiles[i % len(profiles)]

            if fam == "vec_cmp_3vl":
                desc, impas = gen_vec_cmp_3vl(rng)
            elif fam == "vec_arith":
                desc, impas = gen_vec_arith(rng)
            elif fam == "vec_reduce_topk":
                desc, impas = gen_vec_reduce_topk(rng)
            elif fam == "morsel_scatter":
                desc, impas = gen_morsel_scatter(rng, schema)
            elif fam == "snap_attr":
                desc, impas = gen_snapshot_nullable_attrs(rng, schema)
            else:
                desc, impas = gen_spmv_adversarial(rng, schema)

            prog = asm.assemble(impas)
            fixture = {
                "words_hex": [f"0x{w:016x}" for w in prog.words],
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
            }
            results = {name: cli.run_fixture(fixture, prof, 500_000) for name, cli in clients.items()}
            err = compare_results(results)
            ref = next(iter(results.values()))
            fam_counts[fam] += 1
            status_counts[str(ref.get("status", "UNKNOWN"))] += 1

            if err is not None:
                errors += 1
                print(f"[DIFF #{errors}] iter={i} prof={prof} {desc}: {err}")
                print(impas[:1200] + ("...\n" if len(impas) > 1200 else ""))
                if errors >= args.max_errors:
                    break
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    done = sum(fam_counts.values())
    print(
        f"\nCompleted {done}/{args.iterations} iters in {dt:.2f}s ({done / dt:.1f} iters/s) | errors={errors}"
    )
    print(f"Families: {dict(fam_counts)}")
    print(f"Statuses: {dict(status_counts)}")
    if errors > 0:
        return 1
    print("PASS: 0 IEEE-754 / 3VL differential divergences!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
