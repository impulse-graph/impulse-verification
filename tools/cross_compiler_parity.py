#!/usr/bin/env python3
"""
ImpulseVM Cross-Compiler & Optimizer Parity Validation Harness (v0.10 CEL & v0.11 ImpScheme).

In --isa v11 mode (default), compiles and executes every .impscm fixture in
impulse-graph-spec/test-fixtures/impscm-v11/ across 4 configurations:
  1. C++20 (build_on,  opt-ON)
  2. C++20 (build_off, opt-OFF)
  3. Java 25 (opt-ON)
  4. Java 25 (opt-OFF)
Asserting 100% semantic collected output identity, exact .impb byte parity between
C++ and Java (opt-ON), and identical static validation rejection on negative fixtures.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _lib_filename():
    if sys.platform == "darwin":
        return "libimpulse_graph.dylib"
    if sys.platform == "win32":
        return "impulse_graph.dll"
    return "libimpulse_graph.so"


def _find_cpp_lib(build_name: str) -> Path:
    env_dir = os.environ.get("IMPULSE_CPP_DIR")
    if env_dir:
        cand = Path(env_dir) / _lib_filename()
        if cand.exists():
            return cand
    root = Path(__file__).resolve().parent.parent.parent
    if root.parent.name == ".worktrees":
        root = root.parent.parent
    if root.parent.name == ".worktrees":
        root = root.parent.parent
    for base in (
        root / "impulse-graph-core" / "impulse-cpp" / build_name,
    ):
        cand = base / _lib_filename()
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate {_lib_filename()} in {build_name}")


class ParamBindingV11(ctypes.Structure):
    _fields_ = [
        ("param_id", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("domain_id", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("count", ctypes.c_uint32),
        ("scalar_bits", ctypes.c_uint64),
        ("data_ptr", ctypes.c_void_p),
        ("valid_bits", ctypes.c_void_p),
    ]


class ParamsV11(ctypes.Structure):
    _fields_ = [
        ("gas_limit", ctypes.c_uint64),
        ("geometry_override", ctypes.c_void_p),
        ("bindings", ctypes.POINTER(ParamBindingV11)),
        ("binding_count", ctypes.c_uint32),
    ]


class OutputViewV11(ctypes.Structure):
    _fields_ = [
        ("slot", ctypes.c_uint16),
        ("elem_tag", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("length", ctypes.c_uint32),
        ("i64_data", ctypes.POINTER(ctypes.c_int64)),
        ("f64_data", ctypes.POINTER(ctypes.c_double)),
        ("valid_bits", ctypes.POINTER(ctypes.c_uint64)),
        ("bool_data_bits", ctypes.POINTER(ctypes.c_uint64)),
    ]


def _build_cpp_snapshot(lib, snap_kind: str, tmp_dir: Path):
    path = str(tmp_dir / f"{snap_kind}.imps").encode("utf-8")
    lib.impulse_writer_create.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.impulse_writer_create.restype = ctypes.c_void_p
    lib.impulse_writer_add_domain.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint8, ctypes.c_char_p]
    lib.impulse_writer_add_domain.restype = ctypes.c_int
    lib.impulse_writer_set_domain_node_count.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint64]
    lib.impulse_writer_set_domain_node_count.restype = ctypes.c_int
    lib.impulse_writer_add_relation.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint8,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64,
    ]
    lib.impulse_writer_add_relation.restype = ctypes.c_int
    lib.impulse_writer_set_metadata.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.impulse_writer_set_metadata.restype = ctypes.c_int
    lib.impulse_writer_finalize.argtypes = [ctypes.c_void_p]
    lib.impulse_writer_finalize.restype = ctypes.c_int
    lib.impulse_writer_destroy.argtypes = [ctypes.c_void_p]
    lib.impulse_writer_destroy.restype = None
    lib.impulse_snapshot_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.impulse_snapshot_open.restype = ctypes.c_void_p

    w = lib.impulse_writer_create(path, 0)
    if snap_kind == "rebac":
        lib.impulse_writer_add_domain(w, 0, 0, b"User")
        lib.impulse_writer_set_domain_node_count(w, 0, 2)
        lib.impulse_writer_add_domain(w, 1, 0, b"Group")
        lib.impulse_writer_set_domain_node_count(w, 1, 3)
        lib.impulse_writer_add_domain(w, 2, 0, b"Resource")
        lib.impulse_writer_set_domain_node_count(w, 2, 2)
        m_off = (ctypes.c_uint32 * 3)(0, 1, 1)
        m_tgt = (ctypes.c_uint32 * 1)(0)
        lib.impulse_writer_add_relation(w, 0, 1, 1, 2, 1, 0, m_off, ctypes.sizeof(m_off), m_tgt, ctypes.sizeof(m_tgt))
        lib.impulse_writer_set_metadata(w, b"relation.0.name", b"MEMBER_OF")
        g_off = (ctypes.c_uint32 * 4)(0, 1, 2, 2)
        g_tgt = (ctypes.c_uint32 * 2)(1, 2)
        lib.impulse_writer_add_relation(w, 1, 1, 1, 3, 2, 0, g_off, ctypes.sizeof(g_off), g_tgt, ctypes.sizeof(g_tgt))
        lib.impulse_writer_set_metadata(w, b"relation.1.name", b"GROUP_INCLUDES")
        a_off = (ctypes.c_uint32 * 4)(0, 0, 0, 1)
        a_tgt = (ctypes.c_uint32 * 1)(1)
        lib.impulse_writer_add_relation(w, 1, 2, 1, 3, 1, 0, a_off, ctypes.sizeof(a_off), a_tgt, ctypes.sizeof(a_tgt))
        lib.impulse_writer_set_metadata(w, b"relation.2.name", b"CAN_ACCESS")
    else:
        lib.impulse_writer_add_domain(w, 0, 0, b"Node")
        lib.impulse_writer_set_domain_node_count(w, 0, 5)
        off = (ctypes.c_uint32 * 6)(0, 1, 2, 3, 4, 4)
        tgt = (ctypes.c_uint32 * 4)(1, 2, 3, 4)
        lib.impulse_writer_add_relation(w, 0, 0, 1, 5, 4, 0, off, ctypes.sizeof(off), tgt, ctypes.sizeof(tgt))
        lib.impulse_writer_set_metadata(w, b"relation.0.name", b"NEXT")

    assert lib.impulse_writer_finalize(w) == 0
    lib.impulse_writer_destroy(w)
    st = ctypes.c_int(0)
    snap = lib.impulse_snapshot_open(path, ctypes.byref(st))
    assert snap and st.value == 0
    return snap


def run_cpp_v11(lib_path: Path, fixture_path: Path, opt_level: int, snap_kind: str, param_vals: list[int], tmp_dir: Path):
    lib = ctypes.CDLL(str(lib_path))
    snap = _build_cpp_snapshot(lib, snap_kind, tmp_dir)

    lib.impulse_compile_impscm.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    lib.impulse_compile_impscm.restype = ctypes.c_int
    lib.impulse_vm_free_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.impulse_workspace_init.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    lib.impulse_workspace_init.restype = ctypes.c_uint32
    lib.impulse_vm_execute.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ParamsV11),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.impulse_vm_execute.restype = ctypes.c_uint32
    lib.impulse_vm_get_output_count.argtypes = [ctypes.c_void_p]
    lib.impulse_vm_get_output_count.restype = ctypes.c_uint32
    lib.impulse_vm_get_output.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(OutputViewV11)]
    lib.impulse_vm_get_output.restype = ctypes.c_int
    lib.impulse_snapshot_close.argtypes = [ctypes.c_void_p]

    src = fixture_path.read_text("utf-8").encode("utf-8")
    out_ptr = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t(0)
    err_buf = ctypes.create_string_buffer(512)
    rc = lib.impulse_compile_impscm(src, snap, opt_level, ctypes.byref(out_ptr), ctypes.byref(out_len), err_buf, 512)
    if rc != 0:
        lib.impulse_snapshot_close(snap)
        return {"status": "compile_error", "error": err_buf.value.decode("utf-8", errors="replace")}

    impb_bytes = bytes(out_ptr[:out_len.value])
    lib.impulse_vm_free_bytes(out_ptr)

    # 128-byte aligned workspace
    raw_buf = (ctypes.c_uint8 * (1024 * 1024 + 128))()
    base_addr = ctypes.addressof(raw_buf)
    aligned_addr = (base_addr + 127) & ~127
    ws = ctypes.c_void_p()
    assert lib.impulse_workspace_init(ctypes.c_void_p(aligned_addr), 1024 * 1024, ctypes.byref(ws)) == 0

    bindings = (ParamBindingV11 * max(1, len(param_vals)))()
    for i, val in enumerate(param_vals):
        bindings[i].param_id = i
        bindings[i].kind = 2  # IMPULSE_VM_PARAM_SCALAR_U64
        bindings[i].scalar_bits = val
    params = ParamsV11(gas_limit=100000, geometry_override=None,
                       bindings=bindings if param_vals else None, binding_count=len(param_vals))

    vm_st = lib.impulse_vm_execute(impb_bytes, len(impb_bytes), snap, ctypes.byref(params), ws, None, 0, None)
    n_out = lib.impulse_vm_get_output_count(ws)
    outputs = []
    for i in range(n_out):
        ov = OutputViewV11()
        assert lib.impulse_vm_get_output(ws, i, ctypes.byref(ov)) == 1
        outputs.append([int(ov.i64_data[k]) for k in range(ov.length)])

    lib.impulse_snapshot_close(snap)
    return {"status": "ok", "vm_status": int(vm_st), "impb_hex": impb_bytes.hex(), "outputs": outputs}


def run_java_v11(fixture_path: Path, opt_level: int, snap_kind: str, param_vals: list[int]):
    _r = Path(__file__).resolve().parent.parent.parent
    java_dir = (_r.parent.parent if _r.parent.name == ".worktrees" else _r) / "impulse-graph-java"
    cp = (
        "impulse-compiler/target/classes:"
        "impulse-vm/target/classes:"
        "impulse-api/target/classes:"
        "impulse-storage/target/classes"
    )
    cmd = [
        "java", "--add-modules", "jdk.incubator.vector", "-cp", cp,
        "org.impulsegraph.compiler.harness.CrossCompilerParityHarness",
        "--v11-impscm", str(fixture_path), str(opt_level), snap_kind,
        *[str(v) for v in param_vals],
    ]
    res = subprocess.run(cmd, cwd=java_dir, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Java harness failed: {res.stderr}\n{res.stdout}")
    for line in res.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON found in Java output: {res.stdout}")


def run_v11_suite() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    if root.parent.name == ".worktrees":
        root = root.parent.parent
    spec_dir = root / "impulse-graph-spec" / "test-fixtures" / "impscm-v11"
    cpp_on_lib = _find_cpp_lib("build_on")
    cpp_off_lib = _find_cpp_lib("build_off")

    cases = [
        ("01_rebac_reachability.impscm", "rebac", [0, 1], [[1]], None),
        ("02_two_hop_and_set_algebra.impscm", "chain", [0], [[2], [1]], None),
        ("03_fixpoint_reach_count.impscm", "chain", [1], [[4]], None),
        ("04_negative_domain_mismatch.impscm", "rebac", [], None, "DOMAIN_MISMATCH"),
        ("05_negative_mixed_int_float.impscm", "rebac", [], None, "TYPE_MISMATCH"),
        ("06_negative_non_endo_transitive.impscm", "rebac", [], None, "DOMAIN_MISMATCH"),
    ]

    print("=======================================================================")
    print(" ImpulseVM v0.11 Cross-Compiler & Optimizer Parity Suite (4-Way)")
    print("=======================================================================")
    passed = 0
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for fname, snap_kind, pvals, exp_outputs, exp_err in cases:
            fpath = spec_dir / fname
            c_on = run_cpp_v11(cpp_on_lib, fpath, 1, snap_kind, pvals, tmp_dir)
            c_off = run_cpp_v11(cpp_off_lib, fpath, 0, snap_kind, pvals, tmp_dir)
            j_on = run_java_v11(fpath, 1, snap_kind, pvals)
            j_off = run_java_v11(fpath, 0, snap_kind, pvals)

            if exp_err is not None:
                ok = all(r["status"] == "compile_error" and exp_err in r["error"] for r in (c_on, c_off, j_on, j_off))
                tag = "PASS" if ok else "FAIL"
                print(f"  [{tag}] {fname} (expect {exp_err})")
                if not ok:
                    print(f"         C++ ON={c_on} C++ OFF={c_off} Java ON={j_on} Java OFF={j_off}")
                else:
                    passed += 1
            else:
                outputs_match = (
                    c_on["status"] == "ok" and c_off["status"] == "ok"
                    and j_on["status"] == "ok" and j_off["status"] == "ok"
                    and c_on["vm_status"] == 0 and c_off["vm_status"] == 0
                    and j_on["vm_status"] == 0 and j_off["vm_status"] == 0
                    and c_on["outputs"] == exp_outputs
                    and c_off["outputs"] == exp_outputs
                    and j_on["outputs"] == exp_outputs
                    and j_off["outputs"] == exp_outputs
                )
                impb_match = (c_on["impb_hex"] == j_on["impb_hex"])
                ok = outputs_match and impb_match
                tag = "PASS" if ok else "FAIL"
                print(f"  [{tag}] {fname} outputs={c_on['outputs']} impb_exact={impb_match} ({len(c_on['impb_hex'])//2}B)")
                if not ok:
                    print(f"         C++ ON={c_on}\n         C++ OFF={c_off}\n         Java ON={j_on}\n         Java OFF={j_off}")
                else:
                    passed += 1

    print("=======================================================================")
    print(f"v0.11 Parity Result: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isa", choices=["v11"], default="v11")
    parser.parse_args()
    sys.exit(run_v11_suite())


if __name__ == "__main__":
    main()
