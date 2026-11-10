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

## Tools & Suites

| Tool | Description |
| :--- | :--- |
| `tools/four_way_validation.py` | Primary four-way polyglot validation suite verifying 100% bitwise/state-identical outputs (`status`, `PC`, `ZF`/`ST` flags, `R0`..`R63`, and `ICTX` v1 context hash) across C++ (`ON`), C++ (`OFF`), Java 25, and OCaml across all `.impas` test vectors in `spec/test-vectors/vm-impas/`. |
| `tools/fuzz_vm_contracts.py` | Contract-guided differential VM bytecode fuzzer generating type-valid `impOps` instruction streams and asserting 4-way consensus. |
| `tools/generate_random_imps.py` | Deterministic, seeded, versioned (`.imps`) schema and dataset generator synthesizing random domains (`D1`..`Dk`), relations (`R_D1_D2_1`..), and typed/nullable attributes (`D1_A1_INT`, `D1_A2_NULL_INT`, `R_D1_D2_1_A1_VEC4_FLOAT`, etc.) into `manifest.json` + CSVs and compiling/validating `.imps` snapshots via `impulse-graph`. |
| `tools/generate_fixtures.py` | Canonical topology snapshot fixture generator (`g0_empty.imps` .. `g7_bipartite_rebac.imps`). |
| `tools/cross_compiler_parity.py` | Cross-compiler bytecode parity runner comparing C++ and Java CEL / traversal compilation outputs. |
| `tools/three_way_validation.py` | Legacy three-way validation runner. |
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
