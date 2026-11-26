#!/usr/bin/env python3
"""
Cross-compiler & cross-architecture SIMD codegen audit gate for ImpulseVM v0.11.

Verifies that compiled object files contain actual hardware SIMD vector instructions
for the `impulse::vm::v11::simd::*` kernels across:
  - ARM64 (Apple Clang, Linux GCC/Clang, Windows MSVC): NEON `.2d` / `.4s` / `fadd.2d`, `fmul.2d`, `fdiv.2d`, `fsqrt.2d`, `fcmeq.2d`
  - x86_64 (Linux GCC/Clang, macOS cross-clang, Windows MSVC): AVX2/FMA `vaddpd`, `vsubpd`, `vmulpd`, `vdivpd`, `vsqrtpd`, `vcmppd`, `vmovmskpd`

Usage:
  python3 check_simd_codegen.py --obj <path/to/vm_dispatch.cpp.o> [--arch arm64|x86_64]
  python3 check_simd_codegen.py --cross-x64-compile <path/to/v11_simd_kernels.hpp>
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ARM64_VEC_FP_RE = re.compile(
    r"\b(fadd|fsub|fmul|fdiv|fsqrt|fneg|fabs|fcmeq|fcmgt|fcmge|frintm|frintp|fmla)\.(2d|4s)\b"
)
X64_AVX2_FP_RE = re.compile(
    r"\b(vaddpd|vsubpd|vmulpd|vdivpd|vsqrtpd|vcmppd|vcmpeqpd|vcmpgtpd|vcmpltpd|vcmplepd|vcmpgepd|vcmpordpd|vmovmskpd|vfmadd\w+pd|vroundpd)\b"
)


def disassemble(obj_path: str) -> str:
    if platform.system() == "Windows" and shutil.which("dumpbin"):
        return subprocess.check_output(["dumpbin", "/disasm", obj_path], text=True, errors="replace")
    objdump = shutil.which("llvm-objdump") or shutil.which("objdump")
    if not objdump:
        raise RuntimeError("Neither llvm-objdump nor objdump found on PATH")
    return subprocess.check_output([objdump, "-d", "--demangle", obj_path], text=True, errors="replace")


def audit_disasm(text: str, arch: str, min_vec_fp: int) -> int:
    pat = ARM64_VEC_FP_RE if arch == "arm64" else X64_AVX2_FP_RE
    cur = "<unknown>"
    counts = {}
    for line in text.splitlines():
        m = re.match(r"^[0-9a-fA-F]+ <(.+)>:$", line)
        if m:
            cur = m.group(1)
            continue
        if "simd::" in cur or "p3_run_morsel" in cur or "test_simd_" in cur:
            if pat.search(line):
                counts[cur] = counts.get(cur, 0) + 1
    total = sum(counts.values())
    print(f"=== SIMD Codegen Audit (arch={arch}) ===")
    for sym, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {sym[:72]:<72} {cnt:>4} vec-fp insns")
    print(f"Total SIMD FP instructions in audited kernels: {total} (min required: {min_vec_fp})")
    if total < min_vec_fp:
        print(f"FAIL: Expected >= {min_vec_fp} SIMD FP instructions for {arch}, got {total}", file=sys.stderr)
        return 1
    print("PASS: SIMD codegen gate satisfied.")
    return 0


def cross_x64_check(header_path: str) -> int:
    clang = shutil.which("clang++")
    if not clang:
        print("FAIL: clang++ not found", file=sys.stderr)
        return 1
    harness = f"""
#include "{header_path}"
extern "C" {{
void test_simd_arith_vec(uint8_t op, const double* a, const double* b, double* o, uint32_t n) {{
    impulse::vm::v11::simd::f64_arith_vec(op, a, b, o, n);
}}
void test_simd_arith_scalar(uint8_t op, const double* a, double b, double* o, uint32_t n) {{
    impulse::vm::v11::simd::f64_arith_scalar(op, a, b, o, n);
}}
void test_simd_arith_unary(uint8_t op, const double* a, double* o, uint32_t n) {{
    impulse::vm::v11::simd::f64_arith_unary(op, a, o, n);
}}
void test_simd_cmp(uint8_t op, const double* a, const double* b, bool sc, double bsc,
                   const uint64_t* av, const uint64_t* bv, uint64_t* ov, uint64_t* od, uint32_t n) {{
    impulse::vm::v11::simd::f64_cmp_words(op, a, b, sc, bsc, true, av, bv, ov, od, n);
}}
void test_simd_pack_valid(const double* o, const uint64_t* av, const uint64_t* bv, uint64_t* ov, uint32_t n) {{
    impulse::vm::v11::simd::f64_pack_valid_words(o, av, bv, true, true, ov, n);
}}
}}
"""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "simd_probe.cpp")
        obj = os.path.join(td, "simd_probe.o")
        with open(src, "w") as f:
            f.write(harness)
        cmd = [clang, "-std=c++20", "-O3", "-DNDEBUG", "-target", "x86_64-apple-macos", "-mavx2", "-mfma", "-c", src, "-o", obj]
        subprocess.check_call(cmd)
        dis = disassemble(obj)
        return audit_disasm(dis, "x86_64", min_vec_fp=30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", help="Compiled object file to disassemble and audit")
    ap.add_argument("--arch", choices=["arm64", "x86_64"], default="arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64")
    ap.add_argument("--min-vec-fp", type=int, default=80)
    ap.add_argument("--cross-x64-compile", help="Cross-compile v11_simd_kernels.hpp to x86_64 AVX2+FMA and audit")
    args = ap.parse_args()
    if args.cross_x64_compile:
        return cross_x64_check(args.cross_x64_compile)
    if not args.obj:
        ap.error("Provide --obj or --cross-x64-compile")
    return audit_disasm(disassemble(args.obj), args.arch, args.min_vec_fp)


if __name__ == "__main__":
    sys.exit(main())
