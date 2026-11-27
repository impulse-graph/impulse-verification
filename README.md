# Impulse Graph Verification Suite (`impulse-verification`)

Dedicated cross-engine polyglot verification, differential fuzzing, and compiler parity repository for the **Impulse Graph Engine** ecosystem.

## Dependency Architecture

`impulse-verification` sits above the Layer 1 engine repositories in the ecosystem dependency graph:

```text
impulse-graph-spec (pinned at spec/)
  ├── impulse-graph-core (C++20 / Rust / CLI)
  ├── impulse-graph-java (Java 21 LTS / Java 25 FFM & Vector API)
  └── impulse-ocaml      (OCaml Normative Reference Oracle)
        └── impulse-verification (Four-Way Polyglot Validation & Differential Fuzzing)
```

## Unified Verification Runner & `.imps` Knowledge Graph (`verify.py`)

All test suites, fuzzers, sanitizers, coverage gates, benchmarks, and post-release checks across the ecosystem are formally cataloged in [`catalog.toml`](catalog.toml) and invoked through [`verify.py`](verify.py).

### Named Gamuts

| Gamut | Meaning | Typical Time |
| :--- | :--- | :--- |
| **`veryquick`** | Fast deterministic edit-loop sanity (zero fuzzing) | ~30s |
| **`standard`** | **Default *"run all tests"***: deterministic unit/parity/contracts + bounded-seed metamorphic & differential fuzzing | ~3–5m |
| **`hardened`** | Pre-merge safety sign-off: `standard` + FFI bindings + ASan/UBSan/TSan + LLVM MC/DC + JaCoCo + corpus mutation | ~10–20m |
| **`full-gamut`** | **Pre-release sign-off** (`docs/RELEASE_CHECKLIST.md`): `hardened` at 20k+ fuzz budgets + SDK install + perf regression gate | ~30–90m |
| **`soak`** | Overnight endurance: high-iteration fuzzing, morsel stress, Pitest mutation testing | Hours |
| **`bench`** | Dedicated performance benchmarks (`impulse-benchmarks`), excluded from default correctness runs | Varies |
| **`post-release`** | Post-publish website strict MkDocs build & registry checks | ~5m |

### Common Commands

```bash
# Check sibling repo HEADs, build artifacts, and optional SCIP indexers
python3 verify.py doctor

# Canonical "run all the tests" (writes run ledger + syncs datasets/verification.imps)
python3 verify.py run --gamut standard

# Pre-release sign-off
python3 verify.py run --gamut full-gamut

# Answer "are all tests passing?" with exact PASS/FAIL/SKIP + cross-repo commit staleness
python3 verify.py status

# Slice by orthogonal pillar or repo, or override a sibling repo with an active worktree
python3 verify.py run --pillar fuzz,metamorphic --intensity quick
python3 verify.py run --gamut veryquick --override java=../.worktrees/my-fix/impulse-graph-java

# Index static call chains (SCIP-first + fallback) & query datasets/verification.imps natively
python3 verify.py index
python3 verify.py opcode OP_INDEX_SCAN
python3 verify.py callers execIndexScan --depth 3
python3 verify.py where impulse_stmt_execute
python3 verify.py impacted --diff
python3 verify.py gaps
python3 verify.py audit-invariants
```

## Tools & Suites

| Tool | Description |
| :--- | :--- |
| `verify.py` + `catalog.toml` | Unified verification CLI, declarative suite catalog, `.imps` dogfooding sync (`datasets/verification.imps`), and SCIP-first static call-chain / opcode provenance queries. |
| `tools/run_quality_suite.py` | Legacy 4-stage VM quality gate (superseded by `verify.py run --gamut standard`). |
| `tools/fuzz_vm_contracts.py` | Contract-guided differential VM bytecode fuzzer generating type-valid `impOps` instruction streams and asserting 4-way consensus. |
| `tools/generate_random_imps.py` | Deterministic, seeded, versioned (`.imps`) schema and dataset generator synthesizing random domains, relations, and typed/nullable attributes into `manifest.json` + CSVs and compiling/validating `.imps` snapshots via `impulse-graph`. |
| `tools/generate_fixtures.py` | Canonical topology snapshot fixture generator (`g0_empty.imps` .. `g7_bipartite_rebac.imps`). |
| `tools/cross_compiler_parity.py` | Cross-compiler bytecode parity runner comparing C++ and Java CEL / traversal / ImpScheme compilation outputs. |
| `java-cross-engine/` | Relocated Java JNA/CLI cross-engine verification tests (`CrossEngineVerificationTest.java`, `SnapshotBuilderCliValidationTest.java`). |

## Generating Seeded Random `.imps` Schemas & Snapshots

```bash
# Generate a populated random schema + dataset for seed 42 (generator version 1)
python3 tools/generate_random_imps.py --seed 42 --gen-version 1

# Generate a schema-only (0-row) snapshot with identical schema structure for seed 42
python3 tools/generate_random_imps.py --seed 42 --gen-version 1 --schema-only

# Generate a batch of 20 seeded snapshots
python3 tools/generate_random_imps.py --seed 100 --count 20
```

## Running Four-Way Polyglot Validation

```bash
python3 tools/four_way_validation.py
```

By default, `four_way_validation.py` auto-discovers sibling repositories in `../impulse-graph-core`, `../impulse-graph-java`, and `../impulse-ocaml` (or respects `IMPULSE_LIB_PATH_ON`, `IMPULSE_LIB_PATH_OFF`, `IMPULSE_CORE_DIR`, `IMPULSE_JAVA_DIR`).
