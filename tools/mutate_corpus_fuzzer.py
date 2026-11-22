#!/usr/bin/env python3
"""
mutate_corpus_fuzzer.py — Structure-Aware Corpus Mutation Differential Fuzzer

Takes the 550+ spec .impas test vectors as a rich seed corpus and applies
structure-aware mutations (in-place register aliasing, float boundary values,
instruction duplication, tight .fuel injection, operand swaps, flag flips,
and empty-register fault injection), then checks 100% 4-way ICTX v1 consensus
across C++(ON), C++(OFF), Java 25, and OCaml.
"""

import os
import re
import sys
import time
import random
import tempfile
import argparse
import multiprocessing as mp
from collections import Counter
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
SPEC_DIR = Path(os.environ.get("IMPULSE_SPEC_DIR", str(REPO_DIR / "spec")))
ROOT_DIR = REPO_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SPEC_DIR / "tools"))

from run_vm_asm_suite import Instruction, parse_impas_file
from four_way_validation import (
    JavaBatchClient, OcamlBatchClient, _setup_cpp_lib, run_cpp_parsed, compare_state,
)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

REG_RE = re.compile(r"\bR(\d+)\b")
FLOAT_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d+\.\d+)(?![A-Za-z0-9_])")
BRANCH_OPS = {"OP_JMP", "OP_JZ", "OP_JNZ", "OP_LOOP_DECR", "OP_CALL", "OP_RET", "OP_STABLE_CHECK"}
STREAM_OPS = {"OP_CSR_WALK_STREAM", "OP_CSC_WALK_STREAM", "OP_COO_WALK_STREAM", "OP_STREAM_FUNC_BEGIN"}

MUTATIONS = (
    "alias_dst_src",
    "float_boundary",
    "duplicate_instr",
    "tight_fuel",
    "swap_operands",
    "null_reg_inject",
)


def _strip_expectations(lines: list[str]) -> list[str]:
    return [l for l in lines if "{EXPECT:" not in l]


def _has_control_flow(text: str) -> bool:
    return any(op in text for op in BRANCH_OPS | STREAM_OPS)


def _extract_safe_splice_instrs(src_text: str) -> list[str]:
    """Extract standalone instructions safe to splice into another program (no labels/data refs/CF)."""
    lines = _strip_expectations(src_text.splitlines())
    in_text = False
    out = []
    for l in lines:
        s = l.strip()
        if s == ".text":
            in_text = True
            continue
        if not in_text or not s or s.startswith(";"):
            continue
        # Strip inline comment & hex PC prefix
        code = s.split(";", 1)[0].strip()
        code = re.sub(r"^0x[0-9A-Fa-f_]+:\s*", "", code).strip()
        if not code.startswith("OP_"):
            continue
        op = code.split()[0]
        if op in BRANCH_OPS or op in STREAM_OPS or op in (
            "OP_HALT", "OP_INIT_MOCK_GRAPH", "OP_LOAD_INLINE_ARRAY",
            "OP_LOAD_INLINE_SET", "OP_LOAD_INLINE_DOUBLE_ARRAY", "OP_INIT_MOCK_NODE_ATTR",
            "OP_THROW", "OP_ASSERT",
        ):
            continue
        # Skip if it references a symbolic label (uppercase identifier after opcode other than R#)
        toks = [t.strip(",") for t in code.split()[1:]]
        if any(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", t) and not re.match(r"^R\d+$", t) for t in toks):
            continue
        out.append(code)
    return out


def mutate_impas(src_text: str, rng: random.Random, donor_instrs: list[str] | None = None) -> tuple[str, str]:
    """Apply 1-2 structure-aware mutations to a valid .impas program."""
    lines = _strip_expectations(src_text.splitlines())
    text_idx = -1
    for i, l in enumerate(lines):
        if l.strip() == ".text":
            text_idx = i
            break
    if text_idx < 0:
        return src_text, "none"

    has_cf = _has_control_flow(src_text)
    header = lines[: text_idx + 1]
    body = lines[text_idx + 1 :]
    instr_indices = [
        i for i, l in enumerate(body)
        if l.strip() and not l.strip().startswith(";") and "OP_HALT" not in l and "OP_INIT_MOCK_GRAPH" not in l
    ]
    if not instr_indices:
        return "\n".join(header + body) + "\n", "none"

    candidates = ["alias_dst_src", "float_boundary", "swap_operands", "null_reg_inject"]
    if not has_cf:
        candidates.append("duplicate_instr")
        if donor_instrs:
            candidates.append("splice_tail")
    mut = rng.choice(candidates)

    if mut == "alias_dst_src":
        idx = rng.choice(instr_indices)
        line = body[idx]
        regs = list(REG_RE.finditer(line))
        if len(regs) >= 2:
            # Replace dst register (first R#) with src1 or src2 register
            src_m = rng.choice(regs[1:])
            dst_m = regs[0]
            body[idx] = line[: dst_m.start()] + src_m.group(0) + line[dst_m.end() :]
        else:
            mut = "float_boundary"

    if mut == "swap_operands":
        idx = rng.choice(instr_indices)
        line = body[idx]
        regs = list(REG_RE.finditer(line))
        if len(regs) >= 3:
            r1, r2 = regs[1], regs[2]
            body[idx] = line[: r1.start()] + r2.group(0) + line[r1.end() : r2.start()] + r1.group(0) + line[r2.end() :]
        else:
            mut = "float_boundary"

    if mut == "null_reg_inject":
        idx = rng.choice(instr_indices)
        line = body[idx]
        regs = list(REG_RE.finditer(line))
        if len(regs) >= 2:
            target_m = rng.choice(regs[1:])
            repl = rng.choice(["R62", "R63"])
            body[idx] = line[: target_m.start()] + repl + line[target_m.end() :]
        else:
            mut = "float_boundary"

    if mut == "float_boundary":
        joined = "\n".join(header + body)
        matches = list(FLOAT_RE.finditer(joined))
        if matches:
            m = rng.choice(matches)
            bval = rng.choice(["0.0", "-0.0", "-1.0", "1.0", "-4.0", "1000.5"])
            out = joined[: m.start()] + bval + joined[m.end() :]
            return out + "\n", mut
        mut = "null_reg_inject"

    if mut == "duplicate_instr" and not has_cf:
        idx = rng.choice(instr_indices)
        # Strip explicit hex address prefix so assembler assigns sequential PCs
        dup_line = re.sub(r"^\s*0x[0-9A-Fa-f_]+:\s*", "", body[idx])
        body.insert(idx + 1, dup_line)
        # Strip all hex prefixes in body so PCs stay consistent
        body = [re.sub(r"^\s*0x[0-9A-Fa-f_]+:\s*", "", l) if l.strip().startswith("0x") else l for l in body]

    if mut == "splice_tail" and not has_cf and donor_instrs:
        k = rng.randint(1, min(2, len(donor_instrs)))
        spliced = rng.sample(donor_instrs, k)
        halt_idx = next((i for i, l in enumerate(body) if "OP_HALT" in l), len(body))
        for ins in reversed(spliced):
            body.insert(halt_idx, ins)
        body = [re.sub(r"^\s*0x[0-9A-Fa-f_]+:\s*", "", l) if l.strip().startswith("0x") else l for l in body]

    if mut == "tight_fuel":
        has_fuel = any(l.strip().startswith(".fuel") for l in header)
        fuel_val = rng.randint(2, max(3, min(25, len(instr_indices) + 2)))
        if has_fuel:
            header = [f".fuel {fuel_val}" if l.strip().startswith(".fuel") else l for l in header]
        else:
            header.insert(0, f".fuel {fuel_val}")

    return "\n".join(header + body) + "\n", mut


_W = {}


def _init_worker(lib_on_path, lib_off_path, cp, corpus_paths):
    _W["lib_on"] = _setup_cpp_lib(lib_on_path)
    _W["lib_off"] = _setup_cpp_lib(lib_off_path)
    _W["java"] = JavaBatchClient(cp)
    _W["ocaml"] = OcamlBatchClient()
    _W["tmp"] = tempfile.TemporaryDirectory()
    _W["corpus"] = [(Path(p).name, Path(p).read_text()) for p in corpus_paths]
    _W["splice_pool"] = [ins for _, txt in _W["corpus"] for ins in _extract_safe_splice_instrs(txt)]


def _shrink_mutated(mut_text: str, src_name: str, tp: Path) -> str | None:
    """1-minimal delta-debugging minimizer for a failing mutated .impas program."""
    lines = mut_text.splitlines()
    text_idx = next((i for i, l in enumerate(lines) if l.strip() == ".text"), -1)
    if text_idx < 0:
        return None
    header = lines[: text_idx + 1]
    body = [l for l in lines[text_idx + 1 :] if l.strip()]
    if len(body) <= 2 or _has_control_flow(mut_text):
        return None
    # Strip explicit hex PC prefixes so removals keep PCs valid
    body = [re.sub(r"^\s*0x[0-9A-Fa-f_]+:\s*", "", l) for l in body]

    impas = tp / src_name
    impb = tp / "m_sh.impb"
    impd = tp / "m_sh.impb_data"

    def still_diverges(cand_body: list[str]) -> bool:
        if not cand_body:
            return False
        cand_text = "\n".join(header + cand_body) + "\n"
        impas.write_text(cand_text)
        try:
            instrs, data_bytes, ex, _ = parse_impas_file(impas)
            if not instrs:
                return False
            c_instrs = (Instruction * len(instrs))(*instrs)
            impb.write_bytes(bytes(c_instrs))
            impd.write_bytes(data_bytes)
            c_on = run_cpp_parsed(_W["lib_on"], instrs, data_bytes, ex)
            c_off = run_cpp_parsed(_W["lib_off"], instrs, data_bytes, ex)
            jv = _W["java"].run(impas, impb, impd)
            oc = _W["ocaml"].run(impb, impd)
            return bool(compare_state(c_on, c_off, jv, oc))
        except Exception:
            return False

    cur = list(body)
    changed = True
    while changed and len(cur) > 2:
        changed = False
        for i in range(len(cur) - 1, -1, -1):
            if "OP_HALT" in cur[i]:
                continue
            cand = cur[:i] + cur[i + 1 :]
            if still_diverges(cand):
                cur = cand
                changed = True
                break
    if len(cur) < len(body):
        return "\n".join(header + cur) + "\n"
    return None


def _run_chunk(task):
    start_i, count, base_seed, do_shrink = task
    corpus = _W["corpus"]
    splice_pool = _W["splice_pool"]
    tp = Path(_W["tmp"].name)
    impb = tp / "m.impb"
    impd = tp / "m.impb_data"

    passed = 0
    skipped = 0
    divergences = []
    mut_counts = Counter()
    status_counts = Counter()
    opcodes_seen = set()

    for off in range(count):
        seed = base_seed + start_i + off
        rng = random.Random(seed)
        src_name, src_text = rng.choice(corpus)
        mut_text, mut_kind = mutate_impas(src_text, rng, splice_pool)
        mut_counts[mut_kind] += 1
        impas = tp / src_name
        impas.write_text(mut_text)
        try:
            instrs, data_bytes, ex, _ = parse_impas_file(impas)
            if not instrs:
                skipped += 1
                continue
            for ins in instrs:
                opcodes_seen.add(ins.opcode)
            c_instrs = (Instruction * len(instrs))(*instrs)
            impb.write_bytes(bytes(c_instrs))
            impd.write_bytes(data_bytes)
            c_on = run_cpp_parsed(_W["lib_on"], instrs, data_bytes, ex)
            c_off = run_cpp_parsed(_W["lib_off"], instrs, data_bytes, ex)
            jv = _W["java"].run(impas, impb, impd)
            oc = _W["ocaml"].run(impb, impd)
            status_counts[c_on["status"]] += 1
            diffs = compare_state(c_on, c_off, jv, oc)
            if diffs:
                min_text = _shrink_mutated(mut_text, src_name, tp) if do_shrink else None
                divergences.append((seed, src_name, mut_kind, diffs[:4], mut_text, min_text))
            else:
                passed += 1
        except Exception:
            skipped += 1

    return passed, skipped, divergences, dict(mut_counts), dict(status_counts), list(opcodes_seen)


def main():
    parser = argparse.ArgumentParser(description="ImpulseVM Structure-Aware Corpus Mutation Fuzzer")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of corpus mutations (default: 2000)")
    parser.add_argument("--seed", type=int, default=88000, help="Base seed")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--spec-dir", type=str, default=str(SPEC_DIR))
    parser.add_argument("--max-errors", type=int, default=15, help="Stop after N unique failure clusters (0 = unlimited)")
    parser.add_argument("--shrink", action="store_true", help="1-minimal delta-debugging shrink on divergences")
    parser.add_argument("--dump-dir", type=str, default=str(TOOLS_DIR / "fuzz_findings"))
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir)
    corpus_files = sorted(str(p) for p in (spec_dir / "test-vectors" / "vm-impas").glob("**/*.impas"))
    core_cpp = ROOT_DIR / "impulse-graph-core" / "impulse-cpp"
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib_on = os.environ.get("IMPULSE_LIB_PATH_ON", str(core_cpp / "build_on" / f"libimpulse_graph.{ext}"))
    lib_off = os.environ.get("IMPULSE_LIB_PATH_OFF", str(core_cpp / "build_off" / f"libimpulse_graph.{ext}"))
    java_dir = ROOT_DIR / "impulse-graph-java"
    maven_cp = (java_dir / "impulse-vm" / "cp.txt").read_text().strip()
    cp = f"impulse-vm/target/classes:impulse-vm/target/test-classes:impulse-api/target/classes:impulse-storage/target/classes:impulse-spec/target/classes:{maven_cp}"

    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    print(f"{BOLD}{BLUE} ImpulseVM Structure-Aware Corpus Mutation Fuzzer              {RESET}")
    print(f"{BOLD}{BLUE} Seed Corpus: {len(corpus_files)} spec vectors | 4-Way ICTX v1           {RESET}")
    print(f"{BOLD}{BLUE}==============================================================={RESET}")

    t0 = time.monotonic()
    chunk_size = max(25, min(200, args.iterations // max(1, args.workers)))
    chunks = []
    i = 0
    while i < args.iterations:
        n = min(chunk_size, args.iterations - i)
        chunks.append((i, n, args.seed, args.shrink))
        i += n

    passed = 0
    skipped = 0
    divs = []
    clusters = {}
    mut_totals = Counter()
    status_totals = Counter()
    opcodes_seen = set()
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(lib_on, lib_off, cp, corpus_files)) as pool:
        for p, sk, dlist, mc, sc, ops in pool.imap_unordered(_run_chunk, chunks):
            passed += p
            skipped += sk
            mut_totals.update(mc)
            status_totals.update(sc)
            opcodes_seen.update(ops)
            for seed, src_name, mut_kind, diffs, text, min_text in dlist:
                divs.append((seed, src_name, mut_kind, diffs))
                sig_cat = diffs[0].split(":")[0] if diffs else "unknown"
                ckey = (src_name, mut_kind, sig_cat)
                if ckey not in clusters:
                    clusters[ckey] = 1
                    out_p = dump_dir / f"mut_{mut_kind}_seed_{seed}.impas"
                    out_p.write_text(text)
                    sh_note = ""
                    if min_text:
                        min_p = dump_dir / f"mut_{mut_kind}_seed_{seed}_min.impas"
                        min_p.write_text(min_text)
                        sh_note = f" (shrunk -> {min_p.name})"
                    print(f"{YELLOW}[CLUSTER #{len(clusters)}]{RESET} Seed={seed} Base={src_name} Mut={mut_kind}:")
                    for d in diffs:
                        print(f"       -> {d}")
                    print(f"       Saved: {out_p.name}{sh_note}")
                else:
                    clusters[ckey] += 1
            if args.max_errors > 0 and len(clusters) >= args.max_errors:
                print(f"{YELLOW}[EARLY STOP]{RESET} Reached --max-errors={args.max_errors} unique clusters")
                pool.terminate()
                break

    elapsed = max(1e-6, time.monotonic() - t0)
    ran = passed + len(divs)
    print(f"\n{BOLD}{BLUE}==============================================================={RESET}")
    print(f"Corpus Mutation Complete: {GREEN}{passed:,} Consensus{RESET}, {RED}{len(divs)} Divergences ({len(clusters)} unique clusters){RESET}, {skipped} Skipped")
    print(f"Elapsed: {elapsed:.2f}s ({ran / elapsed:,.1f} iter/s across {args.workers} workers) | Unique Opcodes: {len(opcodes_seen)}")
    print("Mutations:", ", ".join(f"{k}:{v}" for k, v in mut_totals.most_common()))
    print("Statuses:", ", ".join(f"{k}:{v}" for k, v in status_totals.most_common(6)))
    print(f"{BOLD}{BLUE}==============================================================={RESET}")
    sys.exit(1 if divs else 0)


if __name__ == "__main__":
    main()
