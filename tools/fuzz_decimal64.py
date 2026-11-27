#!/usr/bin/env python3
"""
fuzz_decimal64.py — 6-Way Differential Fuzzer & Oracle for ImpulseVM v0.11.0 DECIMAL64.

Compares 6 independent implementations on every test case:
  1. Python `decimal.Decimal(prec=60, rounding=ROUND_HALF_UP)` normative oracle
  2. OCaml reference oracle (`impulse-ocaml --vm-fixture-batch`)
  3. Java 25 Interpreter (`VmFixtureBatchMain`)
  4. Java 25 JIT (`VmFixtureBatchMain --jit`)
  5. C++20 Kernel opt-ON (`build_on/vm_fixture_batch_runner`)
  6. C++20 Kernel opt-OFF (`build_off/vm_fixture_batch_runner`)

Verifies:
  - Exact 5-way VM consensus on `status`, `status_code`, `ICTX v2 logical_hash`,
    `r_valid_mask`, `r_types`, `r_regs`, and `outputs`.
  - Exact agreement with the Python `decimal.Decimal(prec=60, ROUND_HALF_UP)` oracle
    on exit status (`0`, `16`, `17`, `6`), scalar register outputs, and collected vectors.

Coverage:
  - Deterministic boundary matrix: 0, +-1, INT64_MIN, INT64_MIN+1, INT64_MAX,
    +-(10^P - 1) & +-10^P for P=1..18, all 8 sign octants x (.5 ties / 1-below / 1-above),
    huge divisors > 2^62, 128-bit non-overflowing intermediates, IEEE-754 double traps
    (0.5 - 2^-54, +-(2^53 - 1), +-2^53, +-(2^53 + 2), 9223372036854774784.0,
    9223372036854775808.0, +-0.0, NaN, +-Inf, subnormals 1e-320), invalid operand traps (6).
  - Metamorphic laws: commutativity (add_chk, mul_chk) and odd rounding symmetry
    ((-a)/d == -(a/d), (-a*b)/d == -(a*b/d)).
  - Randomized multi-op programs exercising:
    * 0x15 OP_SCALAR_DECIMAL_ALU (all 16 subops)
    * 0x58 OP_VEC_DECIMAL_ALU (all 16 subops, .mask / .scalar, masked-out & NULL non-trapping)
    * 0x52 OP_VEC_REDUCE.sum_i64_chk (128-bit cancellation, .mask, all-NULL, overflow trap)
    * 0x37 OP_MORSEL_COMMIT.scatter_add_chk + 0x58 inside WALK_MORSEL / FILTER_MORSEL
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
ROOT_DIR = REPO_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from fuzz_vm_contracts import (  # noqa: E402
    BatchEngineClient,
    _import_spec_modules,
    discover_default_java_cmd,
    discover_default_ocaml_bin,
)

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

STATUS_OK = 0
STATUS_INVALID_OPERAND = 6
STATUS_ARITHMETIC_OVERFLOW = 16
STATUS_DIV_BY_ZERO = 17

DEC_CTX = Context(prec=60, rounding=ROUND_HALF_UP)

POW10 = [10**i for i in range(19)]
POW10_F64 = [float(p) for p in POW10]

SCALAR_SUBOPS = [
    "add_chk",
    "sub_chk",
    "mul_chk",
    "mul_div_round",
    "div_round",
    "rescale",
    "bound_chk",
    "abs_chk",
    "neg_chk",
    "f64_to_dec",
    "dec_to_f64",
    "round_scale",
    "trunc_scale",
    "floor_scale",
    "ceil_scale",
    "div_safe",
]

VEC_SUBOPS = [
    "add_chk",
    "sub_chk",
    "mul_chk",
    "mul_div_r",
    "mul_div_v",
    "div_round",
    "div_safe",
    "rescale",
    "bound_chk",
    "abs_chk",
    "neg_chk",
    "f64_to_dec",
    "dec_to_f64",
    "round_scale",
    "trunc_scale",
    "mul_div_safe_r",
]


# ---------------------------------------------------------------------------
# Helpers: Formatting & Spec / Engine Discovery
# ---------------------------------------------------------------------------

def resolve_decimal_spec_dir(cli_spec_dir: Path | None = None) -> Path:
    candidates: list[Path] = []
    if cli_spec_dir is not None:
        candidates.append(cli_spec_dir)
    env_spec = os.environ.get("IMPULSE_SPEC_DIR")
    if env_spec:
        candidates.append(Path(env_spec))
    # Prefer the workspace's live impulse-graph-spec which has DECIMAL64 opcodes
    candidates.append(ROOT_DIR / "impulse-graph-spec")
    candidates.append(REPO_DIR / "spec")

    for c in candidates:
        isa = c / "v0.11-refactor" / "isa_v0_11.yaml"
        if isa.exists() and "OP_SCALAR_DECIMAL_ALU" in isa.read_text(encoding="utf-8"):
            return c.resolve()
    raise FileNotFoundError(
        "Could not locate v0.11 spec with OP_SCALAR_DECIMAL_ALU. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def discover_cpp_runner(mode: str, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    candidates = [
        ROOT_DIR / "impulse-graph-core" / "impulse-cpp" / f"build_{mode}" / "vm_fixture_batch_runner",
        ROOT_DIR / ".worktrees" / "v11-m3" / "impulse-graph-core" / "impulse-cpp" / f"build_{mode}" / "vm_fixture_batch_runner",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


def i64_hex(v: int) -> str:
    return f"0x{v & 0xFFFFFFFFFFFFFFFF:016X}"


def f64_to_u64_bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def f64_to_i64_bits(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


def f64_hex(v: float) -> str:
    return f"0x{f64_to_u64_bits(v):016X}"


def fmt_i64_vec(vals: list[int | None]) -> str:
    return "[" + ", ".join("null" if v is None else str(v) for v in vals) + "]"


def fmt_f64_vec(vals: list[float | None]) -> str:
    parts: list[str] = []
    for v in vals:
        if v is None:
            parts.append("null")
        elif math.isnan(v):
            parts.append('"nan"')
        elif math.isinf(v):
            parts.append('"inf"' if v > 0 else '"-inf"')
        else:
            parts.append(repr(v))
    return "[" + ", ".join(parts) + "]"


def _clean_f64(x: float) -> float | None:
    if math.isnan(x):
        return None
    if x == 0.0:
        return 0.0
    return x


# ---------------------------------------------------------------------------
# Normative Python `decimal.Decimal(prec=60, ROUND_HALF_UP)` Oracle
# ---------------------------------------------------------------------------

def _check_i64(v: int) -> tuple[int, int | None]:
    if INT64_MIN <= v <= INT64_MAX:
        return STATUS_OK, v
    return STATUS_ARITHMETIC_OVERFLOW, None


def py_add_chk(a: int, b: int) -> tuple[int, int | None]:
    return _check_i64(a + b)


def py_sub_chk(a: int, b: int) -> tuple[int, int | None]:
    return _check_i64(a - b)


def py_mul_chk(a: int, b: int) -> tuple[int, int | None]:
    return _check_i64(a * b)


def py_div_round(a: int, d: int) -> tuple[int, int | None]:
    if d == 0:
        return STATUS_DIV_BY_ZERO, None
    if a == 0:
        return STATUS_OK, 0
    q = DEC_CTX.divide(Decimal(a), Decimal(d))
    r = int(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP, context=DEC_CTX))
    return _check_i64(r)


def py_div_safe(a: int, d: int) -> tuple[int, int | None]:
    if d == 0:
        return STATUS_OK, None
    return py_div_round(a, d)


def py_mul_div_round(a: int, b: int, d: int) -> tuple[int, int | None]:
    if d == 0:
        return STATUS_DIV_BY_ZERO, None
    if a == 0 or b == 0:
        return STATUS_OK, 0
    prod = DEC_CTX.multiply(Decimal(a), Decimal(b))
    q = DEC_CTX.divide(prod, Decimal(d))
    r = int(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP, context=DEC_CTX))
    return _check_i64(r)


def py_mul_div_safe(a: int, b: int, d: int) -> tuple[int, int | None]:
    st, v = py_mul_div_round(a, b, d)
    if st != STATUS_OK:
        return STATUS_OK, None
    return STATUS_OK, v


def py_rescale(a: int, delta_s: int, direction: int) -> tuple[int, int | None]:
    if delta_s < 0 or delta_s > 18 or direction not in (0, 1):
        return STATUS_INVALID_OPERAND, None
    if direction == 0:
        return py_mul_chk(a, POW10[delta_s])
    return py_div_round(a, POW10[delta_s])


def py_bound_chk(a: int, prec: int) -> tuple[int, int | None]:
    if prec < 1 or prec > 18:
        return STATUS_INVALID_OPERAND, None
    lim = POW10[prec] - 1
    if abs(a) > lim:
        return STATUS_ARITHMETIC_OVERFLOW, None
    return STATUS_OK, a


def py_abs_chk(a: int) -> tuple[int, int | None]:
    if a == INT64_MIN:
        return STATUS_ARITHMETIC_OVERFLOW, None
    return STATUS_OK, abs(a)


def py_neg_chk(a: int) -> tuple[int, int | None]:
    if a == INT64_MIN:
        return STATUS_ARITHMETIC_OVERFLOW, None
    return STATUS_OK, -a


def _round_half_away_f64(x: float) -> float:
    if math.isnan(x) or math.isinf(x) or abs(x) >= 4503599627370496.0:
        return x
    a = abs(x)
    f = math.floor(a)
    r = f + 1.0 if (a - f) >= 0.5 else f
    return math.copysign(r, x)


def py_f64_to_dec(x: float, scale: int, prec: int) -> tuple[int, int | None]:
    if scale < 0 or scale > 18 or prec < 1 or prec > 18:
        return STATUS_INVALID_OPERAND, None
    if math.isnan(x):
        return STATUS_OK, None
    if math.isinf(x):
        return STATUS_ARITHMETIC_OVERFLOW, None
    scaled = x * POW10_F64[scale]
    if math.isnan(scaled) or math.isinf(scaled):
        return STATUS_ARITHMETIC_OVERFLOW, None
    r = _round_half_away_f64(scaled)
    if r >= 9223372036854775808.0 or r <= -9223372036854775808.0:
        return STATUS_ARITHMETIC_OVERFLOW, None
    v = int(r)
    return py_bound_chk(v, prec)


def py_dec_to_f64(a: int, scale: int) -> tuple[int, float | None]:
    if scale < 0 or scale > 18:
        return STATUS_INVALID_OPERAND, None
    # IEEE-754 double(int64(a)) / double(10^scale)
    res = float(a) / POW10_F64[scale]
    return STATUS_OK, res


def py_round_scale(a: int, delta_s: int) -> tuple[int, int | None]:
    if delta_s < 0 or delta_s > 18:
        return STATUS_INVALID_OPERAND, None
    return py_div_round(a, POW10[delta_s])


def py_trunc_scale(a: int, delta_s: int) -> tuple[int, int | None]:
    if delta_s < 0 or delta_s > 18:
        return STATUS_INVALID_OPERAND, None
    q = DEC_CTX.divide(Decimal(a), Decimal(POW10[delta_s]))
    return STATUS_OK, int(q.quantize(Decimal("1"), rounding=ROUND_DOWN, context=DEC_CTX))


def py_floor_scale(a: int, delta_s: int) -> tuple[int, int | None]:
    if delta_s < 0 or delta_s > 18:
        return STATUS_INVALID_OPERAND, None
    q = DEC_CTX.divide(Decimal(a), Decimal(POW10[delta_s]))
    return STATUS_OK, int(q.quantize(Decimal("1"), rounding=ROUND_FLOOR, context=DEC_CTX))


def py_ceil_scale(a: int, delta_s: int) -> tuple[int, int | None]:
    if delta_s < 0 or delta_s > 18:
        return STATUS_INVALID_OPERAND, None
    q = DEC_CTX.divide(Decimal(a), Decimal(POW10[delta_s]))
    return STATUS_OK, int(q.quantize(Decimal("1"), rounding=ROUND_CEILING, context=DEC_CTX))


def py_sum_i64_chk(vals: list[int | None], mask: set[int] | None = None) -> tuple[int, int | None]:
    total = 0
    cnt = 0
    for i, v in enumerate(vals):
        if mask is not None and i not in mask:
            continue
        if v is not None:
            total += v
            cnt += 1
    if cnt == 0:
        return STATUS_OK, None
    return _check_i64(total)


# ---------------------------------------------------------------------------
# Expected Case Descriptor & Oracle Comparison
# ---------------------------------------------------------------------------

@dataclass
class DecimalCase:
    name: str
    family: str
    impas: str
    expected_status: int
    expected_r: dict[int, tuple[str, int | float | None]] = field(default_factory=dict)
    expected_outputs: dict[str, tuple[str, list[Any]]] = field(default_factory=dict)
    op_count: int = 1
    profiles: list[str] = field(default_factory=lambda: ["P0"])


def _compare_f64_output_val(exp: float | None, got: Any) -> bool:
    if exp is None:
        return got is None
    if got is None:
        return False
    try:
        gf = float(got)
    except (TypeError, ValueError):
        return False
    if math.isnan(exp) and math.isnan(gf):
        return True
    if exp == 0.0 and gf == 0.0:
        return True
    return f64_to_u64_bits(exp) == f64_to_u64_bits(gf) or abs(exp - gf) == 0.0


def verify_against_python_oracle(case: DecimalCase, res: dict[str, Any]) -> str | None:
    got_sc = res.get("status_code")
    if got_sc != case.expected_status:
        return f"Python oracle status mismatch: expected {case.expected_status}, got {got_sc} ({res.get('status')})"
    if case.expected_status != STATUS_OK:
        return None

    r_regs = res.get("r_regs") or {}
    r_valid_mask = int(res.get("r_valid_mask", 0))
    r_types = res.get("r_types") or {}
    for reg_idx, (kind, exp_val) in case.expected_r.items():
        rkey = f"R{reg_idx}"
        is_valid = bool((r_valid_mask >> reg_idx) & 1)
        if exp_val is None:
            if is_valid:
                return f"Python oracle {rkey}: expected NULL (valid=0), got valid=1 val={r_regs.get(rkey)}"
            continue
        if not is_valid:
            return f"Python oracle {rkey}: expected {exp_val} ({kind}), got NULL (valid=0)"
        if kind == "i64":
            if int(r_types.get(rkey, -1)) != 1:
                return f"Python oracle {rkey}: expected type=1 (i64), got {r_types.get(rkey)}"
            if int(r_regs.get(rkey)) != int(exp_val):
                return f"Python oracle {rkey}: expected {exp_val}, got {r_regs.get(rkey)}"
        elif kind == "f64":
            if int(r_types.get(rkey, -1)) != 3:
                return f"Python oracle {rkey}: expected type=3 (f64), got {r_types.get(rkey)}"
            exp_bits = f64_to_i64_bits(float(exp_val))
            got_bits = int(r_regs.get(rkey))
            # Normalize -0.0 and +0.0 if needed
            if float(exp_val) == 0.0 and (got_bits == 0 or got_bits == INT64_MIN):
                continue
            if got_bits != exp_bits:
                return f"Python oracle {rkey} (f64): expected bits {exp_bits} ({exp_val!r}), got {got_bits}"

    outputs = res.get("outputs") or {}
    for out_key, (kind, exp_vec) in case.expected_outputs.items():
        if out_key not in outputs:
            return f"Python oracle missing COLLECT output {out_key}"
        got_vec = outputs[out_key]
        if not isinstance(got_vec, list) or len(got_vec) != len(exp_vec):
            return f"Python oracle output[{out_key}] len mismatch: expected {len(exp_vec)}, got {got_vec!r}"
        for idx, (ev, gv) in enumerate(zip(exp_vec, got_vec)):
            if kind == "i64":
                if ev is None:
                    if gv is not None:
                        return f"Python oracle output[{out_key}][{idx}]: expected null, got {gv!r}"
                else:
                    if gv is None or int(gv) != int(ev):
                        return f"Python oracle output[{out_key}][{idx}]: expected {ev}, got {gv!r}"
            elif kind == "f64":
                if not _compare_f64_output_val(ev, gv):
                    return f"Python oracle output[{out_key}][{idx}] (f64): expected {ev!r}, got {gv!r}"
    return None


def compare_vm_results(results: dict[str, dict[str, Any]]) -> str | None:
    names = list(results.keys())
    if len(names) <= 1:
        return None
    ref_name = names[0]
    ref = results[ref_name]
    ref_tuple = (
        ref.get("status"),
        ref.get("status_code"),
        ref.get("logical_hash"),
        ref.get("r_valid_mask"),
        ref.get("r_type_word"),
    )
    for name in names[1:]:
        cur = results[name]
        cur_tuple = (
            cur.get("status"),
            cur.get("status_code"),
            cur.get("logical_hash"),
            cur.get("r_valid_mask"),
            cur.get("r_type_word"),
        )
        if cur_tuple != ref_tuple:
            return (
                f"{ref_name}={ref_tuple} != {name}={cur_tuple} | "
                f"ref_r={ref.get('r_regs')} cur_r={cur.get('r_regs')} | "
                f"ref_out={ref.get('outputs')} cur_out={cur.get('outputs')}"
            )
        if ref.get("status_code") == STATUS_OK:
            # Verify valid R registers and outputs explicitly
            mask = int(ref.get("r_valid_mask", 0))
            for ri in range(16):
                if (mask >> ri) & 1:
                    rk = f"R{ri}"
                    if (ref.get("r_regs") or {}).get(rk) != (cur.get("r_regs") or {}).get(rk):
                        return f"{ref_name}.{rk}={(ref.get('r_regs') or {}).get(rk)} != {name}.{rk}={(cur.get('r_regs') or {}).get(rk)}"
            if ref.get("outputs") != cur.get("outputs"):
                return f"{ref_name}.outputs={ref.get('outputs')} != {name}.outputs={cur.get('outputs')}"
    return None


# ---------------------------------------------------------------------------
# Boundary Value Pools
# ---------------------------------------------------------------------------

def build_boundary_i64_pool() -> list[int]:
    vals: set[int] = {
        0, 1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 9, -9, 10, -10,
        INT64_MIN, INT64_MIN + 1, INT64_MIN + 2,
        INT64_MAX, INT64_MAX - 1, INT64_MAX - 2,
        (1 << 62), -(1 << 62),
        (1 << 62) + 1, -((1 << 62) + 1),
        5_000_000_000_000_000_000, -5_000_000_000_000_000_000,
        7_500_000_000_000_000_000, -7_500_000_000_000_000_000,
        7_499_999_999_999_999_999, -7_499_999_999_999_999_999,
        7_500_000_000_000_000_001, -7_500_000_000_000_000_001,
    }
    for p in range(1, 19):
        pw = POW10[p]
        vals.add(pw)
        vals.add(-pw)
        vals.add(pw - 1)
        vals.add(-(pw - 1))
        if pw + 1 <= INT64_MAX:
            vals.add(pw + 1)
            vals.add(-(pw + 1))
    return sorted(vals)


BOUNDARY_I64_POOL = build_boundary_i64_pool()

IEEE_F64_TRAPS: list[float] = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    0.5 - 2.0**-54,
    -0.5 + 2.0**-54,
    0.5 + 2.0**-53,
    -0.5 - 2.0**-53,
    1.5,
    -1.5,
    2.5,
    -2.5,
    float((1 << 53) - 1),
    -float((1 << 53) - 1),
    float(1 << 53),
    -float(1 << 53),
    float((1 << 53) + 2),
    -float((1 << 53) + 2),
    9223372036854774784.0,   # largest IEEE-754 double < 2^63
    -9223372036854774784.0,
    9223372036854775808.0,   # exact 2^63 (must trap 16)
    -9223372036854775808.0,  # exact -2^63 (must trap 16)
    1e-320,
    -1e-320,
    5e-324,
    -5e-324,
    1e18,
    -1e18,
    1e30,
    -1e30,
    float("nan"),
    float("inf"),
    float("-inf"),
]


# ---------------------------------------------------------------------------
# Phase 1: Deterministic Boundary Matrix Cases
# ---------------------------------------------------------------------------

def _emit_load_i64(reg: int, val: int | None) -> str:
    if val is None:
        return f"  LOAD_CONST.null R{reg}"
    if -(1 << 31) <= val <= (1 << 31) - 1:
        return f"  LOAD_CONST.i32 R{reg}, {val}"
    return f"  LOAD_CONST.i64.ext R{reg}, {i64_hex(val)}"


def _emit_load_f64(reg: int, val: float | None) -> str:
    if val is None:
        return f"  LOAD_CONST.null R{reg}"
    return f"  LOAD_CONST.f64.ext R{reg}, {f64_hex(val)}"


def build_deterministic_boundary_cases() -> list[DecimalCase]:
    cases: list[DecimalCase] = []

    # 1. All 8 sign octants x exact .5 tie / 1-below / 1-above for mul_div_round & div_round
    for sa in (1, -1):
        for sb in (1, -1):
            for sd in (1, -1):
                lines = [".code"]
                exp_r: dict[int, tuple[str, int | float | None]] = {}
                # Exact .5 tie: (sa*3 * sb*5) / (sd*2) = +-7.5
                a0, b0, d0 = sa * 3, sb * 5, sd * 2
                # 1-below .5: (sa*100 * sb*1 + sa*sb*49) / (sd*100) -> use direct a=149, b=1, d=100
                a1, b1, d1 = sa * 149, sb * 1, sd * 100
                # Exact .5: a=150, b=1, d=100
                a2, b2, d2 = sa * 150, sb * 1, sd * 100
                # 1-above .5: a=151, b=1, d=100
                a3, b3, d3 = sa * 151, sb * 1, sd * 100
                triples = [(a0, b0, d0), (a1, b1, d1), (a2, b2, d2), (a3, b3, d3)]
                for idx, (a, b, d) in enumerate(triples):
                    lines.append(_emit_load_i64(10, a))
                    lines.append(_emit_load_i64(11, b))
                    lines.append(_emit_load_i64(12, d))
                    lines.append(f"  SCALAR_DECIMAL_ALU.mul_div_round R{idx}, R10, R11, R12")
                    st, v = py_mul_div_round(a, b, d)
                    assert st == STATUS_OK
                    exp_r[idx] = ("i64", v)
                # div_round & div_safe on same signs
                for idx, (num, den) in enumerate([(sa * 149, sd * 100), (sa * 150, sd * 100), (sa * 151, sd * 100)], start=4):
                    lines.append(_emit_load_i64(10, num))
                    lines.append(_emit_load_i64(11, den))
                    lines.append(f"  SCALAR_DECIMAL_ALU.div_round R{idx}, R10, R11")
                    st, v = py_div_round(num, den)
                    assert st == STATUS_OK
                    exp_r[idx] = ("i64", v)
                lines.append("  HALT")
                cases.append(DecimalCase(
                    name=f"boundary_octant_{sa:+d}_{sb:+d}_{sd:+d}",
                    family="boundary_octants",
                    impas="\n".join(lines) + "\n",
                    expected_status=STATUS_OK,
                    expected_r=exp_r,
                    op_count=7,
                ))

    # 2. Precision bounds P=1..18: +-(10^P - 1) pass, +-10^P trap 16
    for p in range(1, 19):
        lim = POW10[p] - 1
        lines = [
            ".code",
            _emit_load_i64(0, lim),
            f"  SCALAR_DECIMAL_ALU.bound_chk R1, R0, lo={p}",
            _emit_load_i64(2, -lim),
            f"  SCALAR_DECIMAL_ALU.bound_chk R3, R2, lo={p}",
            "  HALT",
        ]
        cases.append(DecimalCase(
            name=f"boundary_prec_pass_P{p}",
            family="boundary_prec",
            impas="\n".join(lines) + "\n",
            expected_status=STATUS_OK,
            expected_r={1: ("i64", lim), 3: ("i64", -lim)},
            op_count=2,
        ))
        for sign in (1, -1):
            ov = sign * POW10[p]
            t_lines = [
                ".code",
                _emit_load_i64(0, ov),
                f"  SCALAR_DECIMAL_ALU.bound_chk R1, R0, lo={p}",
                "  HALT",
            ]
            cases.append(DecimalCase(
                name=f"boundary_prec_trap_P{p}_{sign:+d}",
                family="boundary_prec",
                impas="\n".join(t_lines) + "\n",
                expected_status=STATUS_ARITHMETIC_OVERFLOW,
                op_count=1,
            ))

    # 3. Huge divisors > 2^62 & 128-bit non-overflowing intermediates
    huge_d = 5_000_000_000_000_000_000
    huge_d_odd = 5_000_000_000_000_000_001
    huge_cases = [
        ("huge_tie_pos", 7_500_000_000_000_000_000, huge_d),
        ("huge_below_pos", 7_499_999_999_999_999_999, huge_d),
        ("huge_above_pos", 7_500_000_000_000_000_001, huge_d),
        ("huge_tie_neg", -7_500_000_000_000_000_000, huge_d),
        ("huge_below_neg", -7_499_999_999_999_999_999, huge_d),
        ("huge_odd_below", 7_500_000_000_000_000_001, huge_d_odd),
        ("huge_odd_above", 7_500_000_000_000_000_002, huge_d_odd),
        ("huge_int64min_div_int64max", INT64_MIN, INT64_MAX),
        ("huge_int64max_div_int64min", INT64_MAX, INT64_MIN),
        ("huge_int64min_div_int64min", INT64_MIN, INT64_MIN),
        ("huge_int64min_div_neg2", INT64_MIN, -2),
        ("huge_int64min_div_2", INT64_MIN, 2),
    ]
    lines = [".code"]
    exp_r = {}
    for idx, (_, num, den) in enumerate(huge_cases):
        lines.append(_emit_load_i64(14, num))
        lines.append(_emit_load_i64(15, den))
        lines.append(f"  SCALAR_DECIMAL_ALU.div_round R{idx}, R14, R15")
        st, v = py_div_round(num, den)
        assert st == STATUS_OK
        exp_r[idx] = ("i64", v)
    lines.append("  HALT")
    cases.append(DecimalCase(
        name="boundary_huge_divisors_gt_2p62",
        family="boundary_128bit",
        impas="\n".join(lines) + "\n",
        expected_status=STATUS_OK,
        expected_r=exp_r,
        op_count=len(huge_cases),
    ))

    # 128-bit non-overflowing intermediates for mul_div_round
    inter_triples = [
        (10**15, 10**15, 10**15),
        (10**18 - 1, 10**18 - 1, 10**18 - 1),
        (INT64_MAX, 2, 2),
        (INT64_MIN, -2, -2),
        (INT64_MIN, INT64_MAX, INT64_MAX),
        (INT64_MIN + 1, INT64_MAX, -INT64_MAX),
        (INT64_MAX, INT64_MAX, INT64_MAX),
        (INT64_MIN, INT64_MIN, INT64_MIN),
        (INT64_MIN + 1, INT64_MIN + 1, -(INT64_MIN + 1)),
        (10**12, 10**12, 3 * 10**11),
        (-10**12, 10**12, 3 * 10**11),
    ]
    lines = [".code"]
    exp_r = {}
    for idx, (a, b, d) in enumerate(inter_triples):
        lines.append(_emit_load_i64(13, a))
        lines.append(_emit_load_i64(14, b))
        lines.append(_emit_load_i64(15, d))
        lines.append(f"  SCALAR_DECIMAL_ALU.mul_div_round R{idx}, R13, R14, R15")
        st, v = py_mul_div_round(a, b, d)
        assert st == STATUS_OK
        exp_r[idx] = ("i64", v)
    lines.append("  HALT")
    cases.append(DecimalCase(
        name="boundary_128bit_intermediates",
        family="boundary_128bit",
        impas="\n".join(lines) + "\n",
        expected_status=STATUS_OK,
        expected_r=exp_r,
        op_count=len(inter_triples),
    ))

    # 4. IEEE-754 double traps for scalar f64_to_dec across scales & precisions
    for idx, x in enumerate(IEEE_F64_TRAPS):
        for scale, prec in [(0, 18), (2, 18), (6, 18), (0, 16)]:
            st, v = py_f64_to_dec(x, scale, prec)
            lines = [
                ".code",
                _emit_load_f64(0, x),
                f"  SCALAR_DECIMAL_ALU.f64_to_dec R1, R0, lo={scale}, hi={prec}",
                "  HALT",
            ]
            cases.append(DecimalCase(
                name=f"boundary_f64_to_dec_{idx}_s{scale}_p{prec}",
                family="boundary_ieee754",
                impas="\n".join(lines) + "\n",
                expected_status=st,
                expected_r={1: ("i64", v)} if st == STATUS_OK else {},
                op_count=1,
            ))

    # 5. Scalar & Vector invalid operand traps (status 6)
    bad_imm_cases = [
        ("bad_scalar_rescale_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.rescale R1, R0, lo=19, hi=0\n"),
        ("bad_scalar_rescale_hi2", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.rescale R1, R0, lo=2, hi=2\n"),
        ("bad_scalar_bound_lo0", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.bound_chk R1, R0, lo=0\n"),
        ("bad_scalar_bound_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.bound_chk R1, R0, lo=19\n"),
        ("bad_scalar_f64_to_dec_lo19", "  LOAD_CONST.f64.ext R0, 0x3FF0000000000000\n  SCALAR_DECIMAL_ALU.f64_to_dec R1, R0, lo=19, hi=10\n"),
        ("bad_scalar_f64_to_dec_hi0", "  LOAD_CONST.f64.ext R0, 0x3FF0000000000000\n  SCALAR_DECIMAL_ALU.f64_to_dec R1, R0, lo=2, hi=0\n"),
        ("bad_scalar_f64_to_dec_hi19", "  LOAD_CONST.f64.ext R0, 0x3FF0000000000000\n  SCALAR_DECIMAL_ALU.f64_to_dec R1, R0, lo=2, hi=19\n"),
        ("bad_scalar_dec_to_f64_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.dec_to_f64 R1, R0, lo=19\n"),
        ("bad_scalar_round_scale_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.round_scale R1, R0, lo=19\n"),
        ("bad_scalar_trunc_scale_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.trunc_scale R1, R0, lo=19\n"),
        ("bad_scalar_floor_scale_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.floor_scale R1, R0, lo=19\n"),
        ("bad_scalar_ceil_scale_lo19", "  LOAD_CONST.i32 R0, 1\n  SCALAR_DECIMAL_ALU.ceil_scale R1, R0, lo=19\n"),
    ]
    for name, body in bad_imm_cases:
        cases.append(DecimalCase(
            name=name,
            family="boundary_bad_imm",
            impas=f".code\n{body}  HALT\n",
            expected_status=STATUS_INVALID_OPERAND,
            op_count=1,
        ))

    # 6. Exact INT64_MIN / INT64_MAX overflow traps (status 16) and div0 traps (status 17)
    trap_cases = [
        ("trap_add_max_p1", INT64_MAX, 1, "add_chk", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_add_min_m1", INT64_MIN, -1, "add_chk", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_sub_min_p1", INT64_MIN, 1, "sub_chk", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_sub_max_m1", INT64_MAX, -1, "sub_chk", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_mul_min_neg1", INT64_MIN, -1, "mul_chk", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_div_round_min_neg1", INT64_MIN, -1, "div_round", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_div_safe_min_neg1", INT64_MIN, -1, "div_safe", STATUS_ARITHMETIC_OVERFLOW),
        ("trap_div_round_div0", 42, 0, "div_round", STATUS_DIV_BY_ZERO),
        ("trap_div_round_0_div0", 0, 0, "div_round", STATUS_DIV_BY_ZERO),
    ]
    for name, a, b, subop, exp_st in trap_cases:
        lines = [
            ".code",
            _emit_load_i64(0, a),
            _emit_load_i64(1, b),
            f"  SCALAR_DECIMAL_ALU.{subop} R2, R0, R1",
            "  HALT",
        ]
        cases.append(DecimalCase(
            name=name,
            family="boundary_traps",
            impas="\n".join(lines) + "\n",
            expected_status=exp_st,
            op_count=1,
        ))

    # abs_chk / neg_chk INT64_MIN
    for subop in ("abs_chk", "neg_chk"):
        lines = [
            ".code",
            _emit_load_i64(0, INT64_MIN),
            f"  SCALAR_DECIMAL_ALU.{subop} R1, R0",
            "  HALT",
        ]
        cases.append(DecimalCase(
            name=f"trap_{subop}_int64_min",
            family="boundary_traps",
            impas="\n".join(lines) + "\n",
            expected_status=STATUS_ARITHMETIC_OVERFLOW,
            op_count=1,
        ))

    # 7. Vector masked-out & NULL non-trapping matrix (length 65 crosses SIMD/word boundary)
    n_mask = 65
    active_ids = [0, 31, 32, 63, 64]
    a_vec: list[int | None] = [INT64_MAX if i % 2 == 0 else INT64_MIN for i in range(n_mask)]
    b_vec: list[int | None] = [1 if i % 2 == 0 else -1 for i in range(n_mask)]
    d_vec: list[int | None] = [0 if i % 2 == 0 else -1 for i in range(n_mask)]
    f_vec: list[float | None] = [float("inf") if i % 2 == 0 else float("-inf") for i in range(n_mask)]
    for i in active_ids:
        a_vec[i] = (i - 32) * 100
        b_vec[i] = 7
        d_vec[i] = 3
        f_vec[i] = (i - 32) * 0.25
    # Make lane 31 NULL in a_vec with lethal b/d to test NULL non-trapping
    a_vec[31] = None
    b_vec[31] = INT64_MAX
    d_vec[31] = 0
    f_vec[31] = None

    active_set = set(active_ids)
    exp_add = [py_add_chk(a_vec[i], b_vec[i])[1] if (i in active_set and a_vec[i] is not None and b_vec[i] is not None) else None for i in range(n_mask)]
    exp_div = [py_div_round(a_vec[i], d_vec[i])[1] if (i in active_set and a_vec[i] is not None and d_vec[i] is not None) else None for i in range(n_mask)]
    exp_f2d = [py_f64_to_dec(f_vec[i], 2, 10)[1] if (i in active_set and f_vec[i] is not None) else None for i in range(n_mask)]

    mask_src = f""".data
  .domain 0 name="N" count={n_mask} width=u16
  .param 0 type=frontier dom=0 values={active_ids}
  .param 1 type=i64_vec values={fmt_i64_vec(a_vec)}
  .param 2 type=i64_vec values={fmt_i64_vec(b_vec)}
  .param 3 type=i64_vec values={fmt_i64_vec(d_vec)}
  .param 4 type=f64_vec values={fmt_f64_vec(f_vec)}
.code
  LOAD_PARAM.frontier F0, param=0, dom=0
  LOAD_PARAM.vec V0, param=1, type=i64
  LOAD_PARAM.vec V1, param=2, type=i64
  LOAD_PARAM.vec V2, param=3, type=i64
  LOAD_PARAM.vec V3, param=4, type=f64
  VEC_DECIMAL_ALU.add_chk.mask V4, V0, V1, F0
  VEC_DECIMAL_ALU.div_round.mask V5, V0, V2, F0
  VEC_DECIMAL_ALU.f64_to_dec.mask V6, V3, F0, lo=2, hi=10
  COLLECT.vec V4, out=0
  COLLECT.vec V5, out=1
  COLLECT.vec V6, out=2
  HALT
"""
    cases.append(DecimalCase(
        name="boundary_vec_masked_65_no_trap",
        family="boundary_vec_mask",
        impas=mask_src,
        expected_status=STATUS_OK,
        expected_outputs={
            "0": ("i64", exp_add),
            "1": ("i64", exp_div),
            "2": ("i64", exp_f2d),
        },
        op_count=3,
    ))

    return cases


# ---------------------------------------------------------------------------
# Phase 2: Metamorphic Cases (Commutativity & Odd Rounding Symmetry)
# ---------------------------------------------------------------------------

def build_metamorphic_cases(rng: random.Random, count: int) -> list[DecimalCase]:
    cases: list[DecimalCase] = []
    for i in range(count):
        kind = i % 3
        if kind == 0:
            # Commutativity of add_chk and mul_chk (scalar + vector)
            a = rng.choice(BOUNDARY_I64_POOL) if rng.random() < 0.5 else rng.randint(-(1 << 30), 1 << 30)
            b = rng.choice(BOUNDARY_I64_POOL) if rng.random() < 0.5 else rng.randint(-(1 << 30), 1 << 30)
            subop = "add_chk" if (i % 2 == 0) else "mul_chk"
            fn = py_add_chk if subop == "add_chk" else py_mul_chk
            st1, v1 = fn(a, b)
            st2, v2 = fn(b, a)
            assert st1 == st2 and v1 == v2
            # Emit a single program that computes both (a op b) and (b op a) and verifies equality
            lines = [
                ".code",
                _emit_load_i64(0, a),
                _emit_load_i64(1, b),
                f"  SCALAR_DECIMAL_ALU.{subop} R2, R0, R1",
                f"  SCALAR_DECIMAL_ALU.{subop} R3, R1, R0",
                "  HALT",
            ]
            cases.append(DecimalCase(
                name=f"meta_comm_{subop}_{i}",
                family="metamorphic",
                impas="\n".join(lines) + "\n",
                expected_status=st1,
                expected_r={2: ("i64", v1), 3: ("i64", v2)} if st1 == STATUS_OK else {},
                op_count=2,
            ))
        elif kind == 1:
            # Odd rounding symmetry: div_round(-a, d) == -div_round(a, d) and div_round(a, -d) == -div_round(a, d)
            a = rng.choice(BOUNDARY_I64_POOL) if rng.random() < 0.5 else rng.randint(-(1 << 60), 1 << 60)
            d = rng.choice(BOUNDARY_I64_POOL) if rng.random() < 0.5 else rng.randint(-(1 << 60), 1 << 60)
            if a == INT64_MIN:
                a = INT64_MIN + 1
            if d in (0, INT64_MIN):
                d = 3
            st_pos, v_pos = py_div_round(a, d)
            st_na, v_na = py_div_round(-a, d)
            st_nd, v_nd = py_div_round(a, -d)
            if st_pos == STATUS_OK and v_pos is not None and v_pos != INT64_MIN:
                assert st_na == STATUS_OK and v_na == -v_pos
                assert st_nd == STATUS_OK and v_nd == -v_pos
                lines = [
                    ".code",
                    _emit_load_i64(0, a),
                    _emit_load_i64(1, d),
                    _emit_load_i64(2, -a),
                    _emit_load_i64(3, -d),
                    "  SCALAR_DECIMAL_ALU.div_round R4, R0, R1",
                    "  SCALAR_DECIMAL_ALU.div_round R5, R2, R1",
                    "  SCALAR_DECIMAL_ALU.div_round R6, R0, R3",
                    "  SCALAR_DECIMAL_ALU.neg_chk   R7, R4",
                    "  HALT",
                ]
                cases.append(DecimalCase(
                    name=f"meta_odd_sym_div_{i}",
                    family="metamorphic",
                    impas="\n".join(lines) + "\n",
                    expected_status=STATUS_OK,
                    expected_r={4: ("i64", v_pos), 5: ("i64", -v_pos), 6: ("i64", -v_pos), 7: ("i64", -v_pos)},
                    op_count=4,
                ))
        else:
            # Odd rounding symmetry for mul_div_round: (-a * b) / d == -((a * b) / d)
            a = rng.randint(-(1 << 40), 1 << 40)
            b = rng.randint(-(1 << 40), 1 << 40)
            d = rng.randint(1, 1 << 40) * rng.choice([-1, 1])
            st_p, v_p = py_mul_div_round(a, b, d)
            st_n, v_n = py_mul_div_round(-a, b, d)
            if st_p == STATUS_OK and v_p is not None and v_p != INT64_MIN:
                assert st_n == STATUS_OK and v_n == -v_p
                lines = [
                    ".code",
                    _emit_load_i64(0, a),
                    _emit_load_i64(1, -a),
                    _emit_load_i64(2, b),
                    _emit_load_i64(3, d),
                    "  SCALAR_DECIMAL_ALU.mul_div_round R4, R0, R2, R3",
                    "  SCALAR_DECIMAL_ALU.mul_div_round R5, R1, R2, R3",
                    "  SCALAR_DECIMAL_ALU.neg_chk       R6, R4",
                    "  HALT",
                ]
                cases.append(DecimalCase(
                    name=f"meta_odd_sym_muldiv_{i}",
                    family="metamorphic",
                    impas="\n".join(lines) + "\n",
                    expected_status=STATUS_OK,
                    expected_r={4: ("i64", v_p), 5: ("i64", -v_p), 6: ("i64", -v_p)},
                    op_count=3,
                ))
    return cases


# ---------------------------------------------------------------------------
# Phase 3: Randomized Multi-Op Program Generators
# ---------------------------------------------------------------------------

def _rand_i64(rng: random.Random, allow_null: bool = True, safe_scale: bool = False) -> int | None:
    if allow_null and rng.random() < 0.12:
        return None
    if safe_scale:
        r = rng.random()
        if r < 0.25:
            return rng.randint(-1000, 1000)
        if r < 0.65:
            return rng.randint(-(10**9), 10**9)
        return rng.randint(-(10**14), 10**14)
    r = rng.random()
    if r < 0.30:
        return rng.choice(BOUNDARY_I64_POOL)
    if r < 0.60:
        return rng.randint(-10_000, 10_000)
    if r < 0.85:
        return rng.randint(-(10**15), 10**15)
    return rng.randint(INT64_MIN, INT64_MAX)


def _rand_f64(rng: random.Random, allow_null: bool = True, safe_finite: bool = False) -> float | None:
    if allow_null and rng.random() < 0.12:
        return None
    if safe_finite:
        r = rng.random()
        if r < 0.35:
            return rng.choice([0.0, -0.0, 0.5, -0.5, 0.5 - 2.0**-54, -0.5 + 2.0**-54, 1.5, -1.5, 2.5, -2.5, 1e-320, -1e-320, float("nan")])
        return rng.uniform(-1e8, 1e8)
    r = rng.random()
    if r < 0.55:
        return rng.choice(IEEE_F64_TRAPS)
    return rng.uniform(-1e12, 1e12)


def gen_random_scalar_chain(rng: random.Random, idx: int) -> DecimalCase:
    """Multi-op 0x15 OP_SCALAR_DECIMAL_ALU program (10..14 ops) with Python oracle."""
    want_trap = rng.random() < 0.18
    num_ops = rng.randint(10, 14)
    lines = [".code"]
    exp_r: dict[int, tuple[str, int | float | None]] = {}
    exp_status = STATUS_OK
    op_count = 0

    # Use R0..R11 as destination slots; R12..R15 as operand scratch
    for step in range(num_ops):
        dst = step % 12
        subop = rng.choice(SCALAR_SUBOPS)
        safe = not want_trap

        if subop in ("add_chk", "sub_chk", "mul_chk", "div_round", "div_safe"):
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            b = _rand_i64(rng, allow_null=True, safe_scale=safe)
            if not safe and subop == "div_round" and rng.random() < 0.4:
                b = 0
            if safe and subop == "div_round" and b == 0:
                b = rng.choice([-7, -2, 2, 7, 10])
            if safe and subop == "mul_chk" and a is not None and b is not None:
                a = max(-10**9, min(10**9, a))
                b = max(-10**9, min(10**9, b))
            lines.append(_emit_load_i64(13, a))
            lines.append(_emit_load_i64(14, b))
            lines.append(f"  SCALAR_DECIMAL_ALU.{subop} R{dst}, R13, R14")
            op_count += 1
            if a is None or b is None:
                st, v = STATUS_OK, None
            else:
                fn = {
                    "add_chk": py_add_chk,
                    "sub_chk": py_sub_chk,
                    "mul_chk": py_mul_chk,
                    "div_round": py_div_round,
                    "div_safe": py_div_safe,
                }[subop]
                st, v = fn(a, b)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

        elif subop == "mul_div_round":
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            b = _rand_i64(rng, allow_null=True, safe_scale=safe)
            d = _rand_i64(rng, allow_null=True, safe_scale=safe)
            if safe and (d == 0 or d is None):
                d = rng.choice([-1000, -3, 2, 1000, 10**9])
            if safe and a is not None and b is not None and d is not None:
                # Ensure |a*b/d| fits comfortably
                if abs(a * b) // abs(d) > INT64_MAX // 2:
                    d = max(abs(a), abs(b), 1) * rng.choice([-1, 1])
            lines.append(_emit_load_i64(13, a))
            lines.append(_emit_load_i64(14, b))
            lines.append(_emit_load_i64(15, d))
            lines.append(f"  SCALAR_DECIMAL_ALU.mul_div_round R{dst}, R13, R14, R15")
            op_count += 1
            if a is None or b is None or d is None:
                st, v = STATUS_OK, None
            else:
                st, v = py_mul_div_round(a, b, d)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

        elif subop == "rescale":
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            ds = rng.randint(0, 18) if safe else rng.choice([0, 1, 2, 6, 18, 19])
            direction = rng.choice([0, 1]) if safe else rng.choice([0, 1, 2])
            if safe and direction == 0 and a is not None:
                lim = INT64_MAX // POW10[ds]
                if abs(a) > lim:
                    a = rng.randint(-lim, lim)
            lines.append(_emit_load_i64(13, a))
            lines.append(f"  SCALAR_DECIMAL_ALU.rescale R{dst}, R13, lo={ds}, hi={direction}")
            op_count += 1
            if ds > 18 or direction not in (0, 1):
                exp_status = STATUS_INVALID_OPERAND
                break
            if a is None:
                st, v = STATUS_OK, None
            else:
                st, v = py_rescale(a, ds, direction)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

        elif subop == "bound_chk":
            prec = rng.randint(1, 18) if safe else rng.choice([0, 1, 4, 18, 19])
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            if safe and a is not None:
                lim = POW10[prec] - 1
                if abs(a) > lim:
                    a = rng.randint(-lim, lim)
            lines.append(_emit_load_i64(13, a))
            lines.append(f"  SCALAR_DECIMAL_ALU.bound_chk R{dst}, R13, lo={prec}")
            op_count += 1
            if prec < 1 or prec > 18:
                exp_status = STATUS_INVALID_OPERAND
                break
            if a is None:
                st, v = STATUS_OK, None
            else:
                st, v = py_bound_chk(a, prec)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

        elif subop in ("abs_chk", "neg_chk"):
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            if safe and a == INT64_MIN:
                a = INT64_MIN + 1
            lines.append(_emit_load_i64(13, a))
            lines.append(f"  SCALAR_DECIMAL_ALU.{subop} R{dst}, R13")
            op_count += 1
            if a is None:
                st, v = STATUS_OK, None
            else:
                st, v = py_abs_chk(a) if subop == "abs_chk" else py_neg_chk(a)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

        elif subop == "f64_to_dec":
            scale = rng.randint(0, 6) if safe else rng.choice([0, 2, 6, 18, 19])
            prec = rng.randint(14, 18) if safe else rng.choice([0, 4, 10, 18, 19])
            x = _rand_f64(rng, allow_null=True, safe_finite=safe)
            lines.append(_emit_load_f64(13, x))
            lines.append(f"  SCALAR_DECIMAL_ALU.f64_to_dec R{dst}, R13, lo={scale}, hi={prec}")
            op_count += 1
            if scale > 18 or prec < 1 or prec > 18:
                exp_status = STATUS_INVALID_OPERAND
                break
            if x is None:
                st, v = STATUS_OK, None
            else:
                st, v = py_f64_to_dec(x, scale, prec)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

        elif subop == "dec_to_f64":
            scale = rng.randint(0, 18) if safe else rng.choice([0, 2, 6, 18, 19])
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            lines.append(_emit_load_i64(13, a))
            lines.append(f"  SCALAR_DECIMAL_ALU.dec_to_f64 R{dst}, R13, lo={scale}")
            op_count += 1
            if scale > 18:
                exp_status = STATUS_INVALID_OPERAND
                break
            if a is None:
                st, fv = STATUS_OK, None
            else:
                st, fv = py_dec_to_f64(a, scale)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("f64", fv)

        else:  # round_scale, trunc_scale, floor_scale, ceil_scale
            ds = rng.randint(0, 18) if safe else rng.choice([0, 1, 2, 6, 18, 19])
            a = _rand_i64(rng, allow_null=True, safe_scale=safe)
            lines.append(_emit_load_i64(13, a))
            lines.append(f"  SCALAR_DECIMAL_ALU.{subop} R{dst}, R13, lo={ds}")
            op_count += 1
            if ds > 18:
                exp_status = STATUS_INVALID_OPERAND
                break
            if a is None:
                st, v = STATUS_OK, None
            else:
                fn = {
                    "round_scale": py_round_scale,
                    "trunc_scale": py_trunc_scale,
                    "floor_scale": py_floor_scale,
                    "ceil_scale": py_ceil_scale,
                }[subop]
                st, v = fn(a, ds)
            if st != STATUS_OK:
                exp_status = st
                break
            exp_r[dst] = ("i64", v)

    lines.append("  HALT")
    return DecimalCase(
        name=f"rand_scalar_chain_{idx}",
        family="scalar_chain",
        impas="\n".join(lines) + "\n",
        expected_status=exp_status,
        expected_r=exp_r if exp_status == STATUS_OK else {},
        op_count=op_count,
    )


def gen_random_vec_batch(rng: random.Random, idx: int) -> DecimalCase:
    """Multi-op 0x58 OP_VEC_DECIMAL_ALU program (4..7 vec ops) with .mask/.scalar & Python oracle."""
    length = rng.choice([1, 3, 4, 7, 8, 15, 16, 17, 32, 63, 64, 65])
    want_trap = rng.random() < 0.16
    safe = not want_trap

    # Build base vectors & mask
    mask_ids = sorted(rng.sample(range(length), k=rng.randint(0, length)))
    mask_set = set(mask_ids)

    v0 = [_rand_i64(rng, allow_null=True, safe_scale=safe) for _ in range(length)]
    v1 = [_rand_i64(rng, allow_null=True, safe_scale=safe) for _ in range(length)]
    v2 = [_rand_i64(rng, allow_null=True, safe_scale=safe) for _ in range(length)]
    vf = [_rand_f64(rng, allow_null=True, safe_finite=safe) for _ in range(length)]
    if safe:
        # Clamp v0, v1 for safe mul/add; put lethal values in masked-out lanes!
        for i in range(length):
            if i not in mask_set:
                v0[i] = rng.choice([INT64_MAX, INT64_MIN, None])
                v1[i] = rng.choice([1, -1, 0])
                v2[i] = 0
                vf[i] = rng.choice([float("inf"), float("-inf"), 9223372036854775808.0])
            else:
                if v0[i] is not None:
                    v0[i] = max(-10**8, min(10**8, v0[i]))
                if v1[i] is not None:
                    v1[i] = max(-10**8, min(10**8, v1[i]))
                if v2[i] == 0:
                    v2[i] = rng.choice([-7, -2, 2, 5])

    r_rhs = _rand_i64(rng, allow_null=True, safe_scale=safe)
    r_div = _rand_i64(rng, allow_null=True, safe_scale=safe)
    if safe:
        if r_rhs is not None:
            r_rhs = max(-10**8, min(10**8, r_rhs))
        if r_div == 0:
            r_div = rng.choice([-11, -2, 2, 13])

    data_lines = [
        ".data",
        f'  .domain 0 name="N" count={length} width=u16',
        f"  .param 0 type=frontier dom=0 values={mask_ids}",
        f"  .param 1 type=i64_vec values={fmt_i64_vec(v0)}",
        f"  .param 2 type=i64_vec values={fmt_i64_vec(v1)}",
        f"  .param 3 type=i64_vec values={fmt_i64_vec(v2)}",
        f"  .param 4 type=f64_vec values={fmt_f64_vec(vf)}",
    ]
    code_lines = [
        ".code",
        "  LOAD_PARAM.frontier F0, param=0, dom=0",
        "  LOAD_PARAM.vec V0, param=1, type=i64",
        "  LOAD_PARAM.vec V1, param=2, type=i64",
        "  LOAD_PARAM.vec V2, param=3, type=i64",
        "  LOAD_PARAM.vec V3, param=4, type=f64",
        _emit_load_i64(0, r_rhs),
        _emit_load_i64(1, r_div),
    ]

    num_ops = rng.randint(4, 7)
    exp_outputs: dict[str, tuple[str, list[Any]]] = {}
    exp_status = STATUS_OK
    op_count = 0

    for step in range(num_ops):
        subop = rng.choice(VEC_SUBOPS)
        use_mask = rng.random() < 0.65 if subop not in ("mul_div_r", "mul_div_v", "mul_div_safe_r") else False
        # When safe, force mask on ops that read v0/v1/vf since unmasked lanes have lethal traps!
        if safe and subop not in ("mul_div_r", "mul_div_v", "mul_div_safe_r"):
            use_mask = True
        use_scalar = rng.random() < 0.45 if subop in ("add_chk", "sub_chk", "mul_chk", "div_round", "div_safe", "mul_div_r", "mul_div_safe_r") else False

        eff_mask = mask_set if use_mask else None
        out_vec: list[Any] = [None] * length
        out_kind = "f64" if subop == "dec_to_f64" else "i64"
        trapped_st = STATUS_OK

        if subop in ("add_chk", "sub_chk", "mul_chk", "div_round", "div_safe"):
            mods = (".mask" if use_mask else "") + (".scalar" if use_scalar else "")
            rhs_op = "R0" if use_scalar else "V1"
            if use_mask:
                code_lines.append(f"  VEC_DECIMAL_ALU.{subop}{mods} V4, V0, {rhs_op}, F0")
            else:
                code_lines.append(f"  VEC_DECIMAL_ALU.{subop}{mods} V4, V0, {rhs_op}")
            op_count += 1
            fn = {
                "add_chk": py_add_chk,
                "sub_chk": py_sub_chk,
                "mul_chk": py_mul_chk,
                "div_round": py_div_round,
                "div_safe": py_div_safe,
            }[subop]
            for i in range(length):
                if eff_mask is not None and i not in eff_mask:
                    continue
                a = v0[i]
                b = r_rhs if use_scalar else v1[i]
                if a is None or b is None:
                    continue
                st, val = fn(a, b)
                if st != STATUS_OK:
                    trapped_st = st
                    break
                out_vec[i] = val

        elif subop in ("mul_div_r", "mul_div_v", "mul_div_safe_r"):
            # If safe, use clean non-lethal lanes if we haven't masked
            if safe and len(mask_set) < length:
                # mul_div_* doesn't take .mask, so only run it if we use safe_r or clean lanes
                subop = "mul_div_safe_r"
            mods = ".scalar" if (use_scalar and subop != "mul_div_v") else ""
            rhs_op = "R0" if (use_scalar and subop != "mul_div_v") else "V1"
            div_op = "V2" if subop == "mul_div_v" else "R1"
            code_lines.append(f"  VEC_DECIMAL_ALU.{subop}{mods} V4, V0, {rhs_op}, {div_op}")
            op_count += 1
            for i in range(length):
                a = v0[i]
                b = r_rhs if (use_scalar and subop != "mul_div_v") else v1[i]
                d = v2[i] if subop == "mul_div_v" else r_div
                if a is None or b is None or d is None:
                    continue
                if subop == "mul_div_safe_r":
                    st, val = py_mul_div_safe(a, b, d)
                else:
                    st, val = py_mul_div_round(a, b, d)
                if st != STATUS_OK:
                    trapped_st = st
                    break
                out_vec[i] = val

        elif subop == "f64_to_dec":
            scale = rng.randint(0, 6) if safe else rng.choice([0, 2, 6, 18, 19])
            prec = rng.randint(14, 18) if safe else rng.choice([0, 4, 10, 18, 19])
            if use_mask:
                code_lines.append(f"  VEC_DECIMAL_ALU.f64_to_dec.mask V4, V3, F0, lo={scale}, hi={prec}")
            else:
                code_lines.append(f"  VEC_DECIMAL_ALU.f64_to_dec V4, V3, lo={scale}, hi={prec}")
            op_count += 1
            if scale > 18 or prec < 1 or prec > 18:
                trapped_st = STATUS_INVALID_OPERAND
            else:
                for i in range(length):
                    if eff_mask is not None and i not in eff_mask:
                        continue
                    x = vf[i]
                    if x is None:
                        continue
                    st, val = py_f64_to_dec(x, scale, prec)
                    if st != STATUS_OK:
                        trapped_st = st
                        break
                    out_vec[i] = val

        elif subop == "dec_to_f64":
            scale = rng.randint(0, 18) if safe else rng.choice([0, 2, 6, 18, 19])
            if use_mask:
                code_lines.append(f"  VEC_DECIMAL_ALU.dec_to_f64.mask V4, V0, F0, lo={scale}")
            else:
                code_lines.append(f"  VEC_DECIMAL_ALU.dec_to_f64 V4, V0, lo={scale}")
            op_count += 1
            if scale > 18:
                trapped_st = STATUS_INVALID_OPERAND
            else:
                for i in range(length):
                    if eff_mask is not None and i not in eff_mask:
                        continue
                    a = v0[i]
                    if a is None:
                        continue
                    _, fv = py_dec_to_f64(a, scale)
                    out_vec[i] = _clean_f64(fv) if fv is not None else None

        else:  # rescale, bound_chk, abs_chk, neg_chk, round_scale, trunc_scale
            if subop == "rescale":
                ds = rng.randint(0, 6) if safe else rng.choice([0, 2, 10, 18, 19])
                direction = rng.choice([0, 1]) if safe else rng.choice([0, 1, 2])
                imm_str = f", lo={ds}, hi={direction}"
                bad_imm = ds > 18 or direction not in (0, 1)
            elif subop == "bound_chk":
                prec = rng.randint(9, 18) if safe else rng.choice([0, 3, 18, 19])
                imm_str = f", lo={prec}"
                bad_imm = prec < 1 or prec > 18
            elif subop in ("round_scale", "trunc_scale"):
                ds = rng.randint(0, 18) if safe else rng.choice([0, 1, 6, 18, 19])
                imm_str = f", lo={ds}"
                bad_imm = ds > 18
            else:
                imm_str = ""
                bad_imm = False

            if use_mask:
                code_lines.append(f"  VEC_DECIMAL_ALU.{subop}.mask V4, V0, F0{imm_str}")
            else:
                code_lines.append(f"  VEC_DECIMAL_ALU.{subop} V4, V0{imm_str}")
            op_count += 1
            if bad_imm:
                trapped_st = STATUS_INVALID_OPERAND
            else:
                for i in range(length):
                    if eff_mask is not None and i not in eff_mask:
                        continue
                    a = v0[i]
                    if a is None:
                        continue
                    if subop == "rescale":
                        st, val = py_rescale(a, ds, direction)
                    elif subop == "bound_chk":
                        st, val = py_bound_chk(a, prec)
                    elif subop == "abs_chk":
                        st, val = py_abs_chk(a)
                    elif subop == "neg_chk":
                        st, val = py_neg_chk(a)
                    elif subop == "round_scale":
                        st, val = py_round_scale(a, ds)
                    else:
                        st, val = py_trunc_scale(a, ds)
                    if st != STATUS_OK:
                        trapped_st = st
                        break
                    out_vec[i] = val

        if trapped_st != STATUS_OK:
            exp_status = trapped_st
            break
        code_lines.append(f"  COLLECT.vec V4, out={step}")
        exp_outputs[str(step)] = (out_kind, out_vec)

    code_lines.append("  HALT")
    return DecimalCase(
        name=f"rand_vec_batch_{idx}_len{length}",
        family="vec_batch",
        impas="\n".join(data_lines + code_lines) + "\n",
        expected_status=exp_status,
        expected_outputs=exp_outputs if exp_status == STATUS_OK else {},
        op_count=op_count,
    )


def gen_random_sum_i64_chk(rng: random.Random, idx: int) -> DecimalCase:
    """Tests 0x52.sum_i64_chk: 128-bit intermediate excursions, .mask, all-NULL, overflow trap."""
    length = rng.choice([1, 3, 4, 8, 16, 33, 64, 65])
    mode = rng.choice(["cancel_pos", "cancel_neg", "random_safe", "all_null", "masked_skip", "overflow"])
    mask_ids = sorted(rng.sample(range(length), k=rng.randint(0, length)))
    use_mask = mode == "masked_skip" or rng.random() < 0.45

    if mode == "cancel_pos" and length >= 3:
        vals: list[int | None] = [INT64_MAX, 500_000, -500_000] + [rng.choice([None, 0]) for _ in range(length - 3)]
        rng.shuffle(vals)
        use_mask = False
    elif mode == "cancel_neg" and length >= 3:
        vals = [INT64_MIN, -500_000, 500_000] + [rng.choice([None, 0]) for _ in range(length - 3)]
        rng.shuffle(vals)
        use_mask = False
    elif mode == "all_null":
        vals = [None] * length
    elif mode == "masked_skip":
        vals = [INT64_MAX if i not in set(mask_ids) else rng.randint(-10_000, 10_000) for i in range(length)]
    elif mode == "overflow" and length >= 2:
        vals = [INT64_MAX, rng.randint(1, 1000)] + [_rand_i64(rng, allow_null=True, safe_scale=True) for _ in range(length - 2)]
        use_mask = False
    else:
        vals = [_rand_i64(rng, allow_null=True, safe_scale=True) for _ in range(length)]

    eff_mask = set(mask_ids) if use_mask else None
    exp_st, exp_sum = py_sum_i64_chk(vals, eff_mask)

    lines = [
        ".data",
        f'  .domain 0 name="N" count={length} width=u16',
        f"  .param 0 type=i64_vec values={fmt_i64_vec(vals)}",
        f"  .param 1 type=frontier dom=0 values={mask_ids}",
        ".code",
        "  LOAD_PARAM.vec V0, param=0, type=i64",
        "  LOAD_PARAM.frontier F0, param=1, dom=0",
        "  VEC_REDUCE.sum_i64_chk.mask R0, V0, F0" if use_mask else "  VEC_REDUCE.sum_i64_chk R0, V0",
        "  HALT",
    ]
    return DecimalCase(
        name=f"rand_sum_i64_chk_{idx}_{mode}",
        family="sum_i64_chk",
        impas="\n".join(lines) + "\n",
        expected_status=exp_st,
        expected_r={0: ("i64", exp_sum)} if exp_st == STATUS_OK else {},
        op_count=1,
    )


def gen_random_morsel_scatter(rng: random.Random, idx: int, profiles: list[str]) -> DecimalCase:
    """Tests 0x58 OP_VEC_DECIMAL_ALU + 0x37.scatter_add_chk inside WALK_MORSEL."""
    n_src = rng.choice([4, 8, 16, 33])
    n_tgt = rng.choice([4, 8, 15, 32])
    want_trap = rng.random() < 0.16
    use_accum = rng.random() < 0.5
    use_scalar_commit = rng.random() < 0.25

    offsets = [0]
    targets: list[int] = []
    edge_vals: list[int | None] = []
    for u in range(n_src):
        deg = rng.randint(0, 5)
        nbrs = sorted(rng.randint(0, n_tgt - 1) for _ in range(deg))
        for v in nbrs:
            targets.append(v)
            if rng.random() < 0.15:
                edge_vals.append(None)
            elif want_trap:
                edge_vals.append(rng.choice([INT64_MAX // 2, INT64_MAX, INT64_MIN, 100]))
            else:
                edge_vals.append(rng.randint(-10_000, 10_000))
        offsets.append(len(targets))

    seeds = sorted(rng.sample(range(n_src), k=rng.randint(1, n_src)))
    init_dst = [rng.randint(-500, 500) if use_accum else 0 for _ in range(n_tgt)]
    mul_factor = rng.choice([-10, -2, 1, 2, 10]) if not want_trap else rng.choice([2, 10, 100])
    commit_scalar = rng.randint(-1000, 1000) if not want_trap else INT64_MAX

    # Compute Python oracle result
    dst_acc: list[int | None] = list(init_dst) if use_accum else [0] * n_tgt
    exp_st = STATUS_OK
    seed_set = set(seeds)
    for u in seeds:
        for e in range(offsets[u], offsets[u + 1]):
            tgt = targets[e]
            if use_scalar_commit:
                v_add: int | None = commit_scalar
            else:
                ev = edge_vals[e]
                if ev is None:
                    v_add = None
                else:
                    st, mv = py_mul_chk(ev, mul_factor)
                    if st != STATUS_OK:
                        exp_st = st
                        break
                    v_add = mv
            if v_add is not None:
                prev = dst_acc[tgt] if dst_acc[tgt] is not None else 0
                st, nv = py_add_chk(prev, v_add)
                if st != STATUS_OK:
                    exp_st = st
                    break
                dst_acc[tgt] = nv
        if exp_st != STATUS_OK:
            break

    mods = (".accum" if use_accum else "") + (".scalar" if use_scalar_commit else "") + ".mask"
    lines = [
        ".data",
        f'  .domain 0 name="Src" count={n_src} width=u16',
        f'  .domain 1 name="Tgt" count={n_tgt} width=u16',
        '  .relation 0 name="E" src_dom=0 tgt_dom=1',
        f"  .csr_inline rel=0 offsets={offsets} targets={targets}",
        f'  .edge_attr rel=0 attr=0 name="amt" type=i64 values={fmt_i64_vec(edge_vals)}',
        f"  .param 0 type=frontier dom=0 values={seeds}",
        f"  .param 1 type=i64_vec values={fmt_i64_vec(init_dst)}",
        ".code",
        "  LOAD_PARAM.frontier F0, param=0, dom=0",
        "  LOAD_PARAM.vec V0, param=1, type=i64",
        _emit_load_i64(0, mul_factor),
        _emit_load_i64(1, commit_scalar),
        "  WALK_MORSEL.fwd F0, rel=0",
    ]
    if use_scalar_commit:
        lines.append(f"    MORSEL_COMMIT.scatter_add_chk{mods} V0, M1, R1, MF0")
    else:
        lines.append("    GATHER_EDGE_ATTR M4, M2, rel=0, attr=0")
        lines.append("    VEC_DECIMAL_ALU.mul_chk.scalar M5, M4, R0")
        lines.append(f"    MORSEL_COMMIT.scatter_add_chk{mods} V0, M1, M5, MF0")
    lines.extend([
        "  MORSEL_END",
        "  COLLECT.vec V0, out=0",
        "  HALT",
    ])
    # Test multi-profile invariance (P0..P3) on every 5th morsel case
    case_profiles = profiles if ((idx // 10) % 5 == 0) else [profiles[0]]
    return DecimalCase(
        name=f"rand_morsel_scatter_{idx}",
        family="morsel_scatter",
        impas="\n".join(lines) + "\n",
        expected_status=exp_st,
        expected_outputs={"0": ("i64", dst_acc)} if exp_st == STATUS_OK else {},
        op_count=2 if not use_scalar_commit else 1,
        profiles=case_profiles,
    )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="6-Way Differential Decimal64 Fuzzer (Python Decimal == OCaml == Java Interp == Java JIT == C++ ON == C++ OFF)"
    )
    parser.add_argument("--cases", type=int, default=2000, help="Total number of test programs to execute (default: 2000)")
    parser.add_argument("--seed", type=int, default=20261127, help="PRNG seed")
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument(
        "--engines",
        type=str,
        default="ocaml,java_interp,java_jit,cpp_on,cpp_off",
        help="Comma-separated VM engines (default: ocaml,java_interp,java_jit,cpp_on,cpp_off)",
    )
    parser.add_argument("--profiles", type=str, default="P0,P1,P2,P3")
    parser.add_argument("--cpp-on-bin", type=Path, default=None)
    parser.add_argument("--cpp-off-bin", type=Path, default=None)
    parser.add_argument("--max-errors", type=int, default=5)
    args = parser.parse_args()

    spec_dir = resolve_decimal_spec_dir(args.spec_dir)
    _, impas_mod = _import_spec_modules(spec_dir)
    asm = impas_mod.ImpAsmV11(spec_dir / "v0.11-refactor" / "isa_v0_11.yaml")

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    engine_names = [e.strip().lower() for e in args.engines.split(",") if e.strip()]

    clients: dict[str, BatchEngineClient] = {}
    for name in engine_names:
        if name == "ocaml":
            clients[name] = BatchEngineClient([str(discover_default_ocaml_bin()), "--vm-fixture-batch"], name)
        elif name == "java_interp":
            clients[name] = BatchEngineClient(discover_default_java_cmd(), name)
        elif name == "java_jit":
            clients[name] = BatchEngineClient(discover_default_java_cmd() + ["--jit"], name)
        elif name == "cpp_on":
            clients[name] = BatchEngineClient([str(discover_cpp_runner("on", args.cpp_on_bin))], name)
        elif name == "cpp_off":
            clients[name] = BatchEngineClient([str(discover_cpp_runner("off", args.cpp_off_bin))], name)
        else:
            raise ValueError(f"Unsupported engine: {name}")

    rng = random.Random(args.seed)

    # Assemble corpus: Phase 1 (Boundary) + Phase 2 (Metamorphic) + Phase 3 (Randomized Multi-Op)
    corpus: list[DecimalCase] = []
    boundary_cases = build_deterministic_boundary_cases()
    corpus.extend(boundary_cases)

    meta_count = min(150, max(30, args.cases // 20))
    meta_cases = build_metamorphic_cases(rng, meta_count)
    corpus.extend(meta_cases)

    remaining = max(0, args.cases - len(corpus))
    for i in range(remaining):
        r = i % 10
        if r in (0, 1, 2, 3):
            corpus.append(gen_random_scalar_chain(rng, i))
        elif r in (4, 5, 6, 7):
            corpus.append(gen_random_vec_batch(rng, i))
        elif r == 8:
            corpus.append(gen_random_sum_i64_chk(rng, i))
        else:
            corpus.append(gen_random_morsel_scatter(rng, i, profiles))

    print(
        f"6-Way DECIMAL64 Differential Fuzzer | cases={len(corpus)} | "
        f"engines=py_decimal,{','.join(clients.keys())} | profiles={','.join(profiles)} | seed={args.seed}"
    )

    fam_counts: Counter[str] = Counter()
    status_counts: Counter[int] = Counter()
    total_ops = 0
    total_runs = 0
    errors = 0
    t0 = time.perf_counter()

    try:
        for case_idx, case in enumerate(corpus):
            prog = asm.assemble(case.impas)
            payload = {
                "data_directives": prog.data_directives,
                "const_pool": prog.const_pool,
                "words_hex": [f"0x{w:016X}" for w in prog.words],
            }

            ref_p0_hash: str | None = None
            case_failed = False

            for prof in case.profiles:
                results = {ename: cli.run_fixture(payload, profile=prof, max_steps=500_000) for ename, cli in clients.items()}
                total_runs += len(results)

                # 1. 5-Way VM consensus
                vm_err = compare_vm_results(results)
                if vm_err is not None:
                    errors += 1
                    case_failed = True
                    print(f"[VM DIFF #{errors}] case={case_idx} ({case.name}) prof={prof}: {vm_err}")
                    print(case.impas[:1500])
                    break

                ref = next(iter(results.values()))

                # 2. Metamorphic profile invariance (P0..P3)
                # On OK runs, ICTX v2 logical_hash must be bit-identical across all profiles.
                # On trapping morsels, all engines must agree within each profile and trap with the same status_code.
                if ref.get("status_code") == STATUS_OK:
                    if ref_p0_hash is None:
                        ref_p0_hash = ref.get("logical_hash")
                    elif ref.get("logical_hash") != ref_p0_hash:
                        errors += 1
                        case_failed = True
                        print(
                            f"[PROFILE DIFF #{errors}] case={case_idx} ({case.name}) "
                            f"P0={ref_p0_hash} != {prof}={ref.get('logical_hash')}"
                        )
                        print(case.impas[:1500])
                        break

                # 3. 6th Oracle: Python decimal.Decimal(prec=60, ROUND_HALF_UP)
                py_err = verify_against_python_oracle(case, ref)
                if py_err is not None:
                    errors += 1
                    case_failed = True
                    print(f"[ORACLE DIFF #{errors}] case={case_idx} ({case.name}) prof={prof}: {py_err}")
                    print(case.impas[:1500])
                    break

            fam_counts[case.family] += 1
            status_counts[case.expected_status] += 1
            total_ops += case.op_count

            if case_failed and errors >= args.max_errors:
                break
    finally:
        for cli in clients.values():
            cli.close()

    dt = max(1e-6, time.perf_counter() - t0)
    done = sum(fam_counts.values())
    print(
        f"\nCompleted {done}/{len(corpus)} cases ({total_ops} decimal ops, {total_runs} VM executions) "
        f"in {dt:.2f}s ({done / dt:.1f} cases/s) | errors={errors}"
    )
    print(f"Families: {dict(fam_counts)}")
    print(f"Statuses: {dict(sorted(status_counts.items()))}")

    if errors > 0:
        return 1
    print("PASS: 0 divergences across 6-way oracle (Python Decimal == OCaml == Java Interp == Java JIT == C++ ON == C++ OFF)!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
