# Impulse Graph Engine — Full-Gamut Verification & Release Checklist

This document formalizes two things:
1. **The "Full Gamut" Pre-Release Verification Suite** — the exact set of tests, sanitizers, fuzzers, and end-to-end runners required to sign off on a release candidate.
2. **The Repeatable Release Process Checklist** — step-by-step instructions from milestone triage and branching through version bumping, CI publishing, release notes, and post-publish verification.

Related: [WEBSITE_CODE_SAMPLE_VERIFICATION.md](WEBSITE_CODE_SAMPLE_VERIFICATION.md)

---

## Part 1 — Formal Definition of the "Full Gamut" Test Suite

All tiers below are formally cataloged in [`catalog.toml`](../catalog.toml) and invoked via the unified runner:

```bash
# Run the complete pre-release gamut (Tier 1..4 + 20k fuzz budgets + perf regression gate)
python3 verify.py run --gamut full-gamut

# Verify exact PASS/FAIL/SKIP status and confirm 0 stale/dirty repos before tagging
python3 verify.py status --gamut full-gamut
```

Run all applicable tiers against the exact `release/0.X.x` branch (with freshly rebuilt `build_on`, `build_off`, `~/.m2` snapshots, and local wheels/binaries) before tagging.

### Tier 1 — Native Core & Polyglot Bindings (`impulse-graph-core`)

| Check | Command | Pass Criterion |
| :--- | :--- | :--- |
| **C++20 Assertions + `ctest`** | `cmake -S impulse-cpp -B impulse-cpp/build-rel -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIMPULSE_ENABLE_ASSERTIONS=ON && cmake --build impulse-cpp/build-rel -j && ctest --test-dir impulse-cpp/build-rel --output-on-failure` | All non-deferred tests pass (note: `test_opcode_contracts` `0xA0..0xBF` stream shaders tracked in `v0.11 #67`). |
| **C++20 ASan + Fatal UBSan** | `cmake -S impulse-cpp -B impulse-cpp/build-san -DCMAKE_BUILD_TYPE=Debug -DIMPULSE_SANITIZE=ON -DIMPULSE_ENABLE_ASSERTIONS=ON && cmake --build impulse-cpp/build-san -j && ctest --test-dir impulse-cpp/build-san --output-on-failure` | 0 ASan/LSan/UBSan findings. |
| **C++20 LLVM MC/DC Coverage** | `cmake -S impulse-cpp -B impulse-cpp/build-mcdc -DCMAKE_BUILD_TYPE=Debug -DIMPULSE_ENABLE_MCDC_COVERAGE=ON -DIMPULSE_COVERAGE_TEST=ON && cmake --build impulse-cpp/build-mcdc -j` + `llvm-profdata merge` + `llvm-cov report --show-mcdc-summary` | All modified C-ABI/index/VM branches covered and passing. |
| **C-ABI Zero-Alloc & Zero-Stdio** | `ctest --test-dir impulse-cpp/build-rel -R "test_v0_10_1_remediation\|test_c_statement_api" --output-on-failure` | 0 heap allocations on hot-path `impulse_stmt_execute()` / `OP_CB` / `OP_92` / `OP_93`; 0 bytes written to `stderr`. |
| **CMake SDK Install Surface** | `cmake --install impulse-cpp/build-rel --prefix /tmp/impulse-sdk-check` | All public headers, `ImpulseGraphConfig*.cmake`, and `impulse-graph.pc` installed cleanly. |
| **Rust Crate & CLI** | `cargo fmt --check` + `cargo test --lib` in `impulse-rust/`; `cargo check --tests` / `cargo test` in `impulse-cli/` | 100% pass, 0 formatting diffs. |
| **Python Binding** | `cd impulse-python && python3 setup.py build_ext --inplace && PYTHONPATH=. python3 -m pytest tests/ -v` | 100% pass (0 failures); `impulse_graph.__version__` matches target release. |
| **Go / .NET / Node Bindings** | CI matrix (`go test -race ./...` in `impulse-go/`, `dotnet test` in `impulse-dotnet/`, `npm test` in `impulse-node/`) | 100% green in GitHub Actions CI. |

### Tier 2 — JVM Core & Polyglot DSLs (`impulse-graph-java`)

| Check | Command | Pass Criterion |
| :--- | :--- | :--- |
| **Full 11-Module Build, Spotless & Tests** | `cd impulse-graph-java && mvn clean install -DfailIfNoTests=false` | All 11 modules (`impulse-spec`, `impulse-api`, `impulse-storage`, `impulse-vm`, `impulse-compiler`, `impulse-builder`, `impulse-kotlin`, `impulse-scala`, `impulse-clojure`, `impulse-samples`) build, pass `spotless:check`, and pass all JUnit 5 + jqwik tests with 0 failures. |

### Tier 3 — Four-Way Polyglot Parity & Differential Fuzzing (`impulse-verification`)

Rebuild both C++ dylibs from `release/0.X.x` first:
```bash
cmake --build impulse-graph-core/impulse-cpp/build_on  --target impulse_graph -j
cmake --build impulse-graph-core/impulse-cpp/build_off --target impulse_graph -j
```

| Check | Command | Pass Criterion |
| :--- | :--- | :--- |
| **Four-Way Polyglot Spec Compliance** | `python3 verify.py run --id parity.four_way_spec` (`impulse-graph-spec/tools/run_vm_asm_suite.py --check-coverage`) | **100% PASS & 100% opcode/subop coverage** (0 mismatches, 0 skipped across C++ ON, C++ OFF, Java 25, OCaml Oracle). |
| **Contract-Guided Differential Fuzzer** | `IMPULSE_LIB_PATH_ON=.../build_on/libimpulse_graph.dylib IMPULSE_LIB_PATH_OFF=.../build_off/libimpulse_graph.dylib python3 tools/fuzz_vm_contracts.py --iterations 20000 --seed <seed>` | **20,000 / 20,000 consensus matches**, 0 divergences, 0 errors. *(Use `--iterations 500` for PR smoke checks; `--iterations 20000` for release sign-off.)* |
| **50-Seed Snapshot Roundtrip** | `python3 tools/test_seeded_roundtrip_50.py` | **50 / 50 PASS** (multi-domain CSV → `.imps` → inspect roundtrip identical). |

### Tier 4 — Minimal REST API, Memory Leak Soak & Performance Regression Gates

| Check | Command | Pass Criterion |
| :--- | :--- | :--- |
| **JVM / Kotlin REST E2E** | `cd impulse-verification/hetionet-rest && mvn test -Dimpulse.version=<snapshot-or-release-ver>` | `HetionetRestE2eTest` + `PowergridFluentE2eTest` 100% pass (`/health`, `/stats`, `/query` Cypher JSON & text/plain). |
| **Memory Leakage & Hot-Path Allocation Gate (C++ & Java)** | C++: ASan/LSan (`-DIMPULSE_SANITIZE=ON`) + `test_v0_10_1_remediation` allocation counter (`0` heap allocs on repeated `impulse_stmt_execute`, `OP_CB`, `OP_92`, `OP_93`) + `impulse_exec()` loop.<br>Java: `V0101VmMemorySafetyTest` (no immortal `GLOBAL_POOL_ARENA`/`COMPILER_ARENA` leaks) + sustained REST soak (`python3 hetionet-rest/stress_test.py`) monitoring JVM heap/non-heap RSS via `GET /stats`. | **0** ASan/LSan leaks, **0** hot-path heap allocations on prepared statements, and flat steady-state RSS / off-heap memory (no unbounded monotonic growth) across sustained soak. |
| **Performance Regression Gate (`impulse-benchmarks`)** | Run core microbenchmarks (`bench_opcode_dispatch`, `BfsVmJmhBenchmark`, and representative `impulse-benchmarks` workloads) on physical hardware and compare against the prior release baseline (`AGENTS.md` §3.7 & §3.13). | No unexplained latency or throughput regression (`> 5%` outside measurement noise); warmed-up Java 25 remains within **`<= 2x` of C++** (`> 5x` gap is a mandatory blocker requiring profiling before release). |

### Tier 5 — Post-Publish Registry & Documentation Verification

| Check | Command / Action | Pass Criterion |
| :--- | :--- | :--- |
| **Registry Live Checks** | Verify HTTP 200 on PyPI (`pypi.org/pypi/impulse-graph/<ver>/json`), Maven Central (`repo1.maven.org/maven2/org/impulsegraph/impulse-api/<ver>/`), crates.io (`crates.io/api/v1/crates/impulse-graph`) | All published registries serve `<ver>` cleanly. |
| **Website Build & Code Samples** | `docker run --rm -v $PWD:/docs squidfunk/mkdocs-material build --strict` + Docker snippet verification ([WEBSITE_CODE_SAMPLE_VERIFICATION.md](WEBSITE_CODE_SAMPLE_VERIFICATION.md)) | Strict MkDocs build passes (0 warnings/errors); Python & Java doc snippets pass against released `<ver>` artifacts. |

---

## Part 2 — End-to-End Release Process Checklist

### Phase 1 — Milestone Triage & Branching
- [ ] Audit open issues in `impulse-graph-core`, `impulse-graph-java`, `impulse-graph-spec`, and `impulse-ocaml`; ensure every issue for the release is assigned to milestone `vX.Y.Z` and all remaining issues are moved to the next milestone.
- [ ] Confirm whether `.imps` format or `impOps` semantics changed:
  - **Patch (`vX.Y.Z`)**: No format/ABI/4-way-semantic changes; `impulse-graph-spec` and `impulse-ocaml` do not need lockstep tags.
  - **Minor (`vX.Y.0`)**: Tag `impulse-graph-spec` first and bump `spec/` submodules across `core`, `java`, `ocaml`, and `verification`.
- [ ] **Release Branch Hygiene (Strict One-Way Flow `main → release/X.Y.x`)**:
  - `release/X.Y.x` is a **long-lived branch** cut once from `origin/main` at the `X.Y.0` branch point (and should be protected against auto-deletion).
  - **NEVER open a PR with `release/X.Y.x` as the head branch into `main`**. Squash-merging `release/X.Y.x → main` rewrites commit SHAs (diverging `release/X.Y.x` from `main`) and causes GitHub's "auto-delete head branches" setting to delete `origin/release/X.Y.x`.
  - Land all bug fixes and features on `main` first via short-lived `fix/*` or `feat/*` PRs (`fix/* → main`).
  - Backport release fixes to `release/X.Y.x` via cherry-pick onto a short-lived PR branch (`backport/X.Y.Z-... → release/X.Y.x`), never the reverse.
- [ ] Confirm all GitHub Actions CI checks are green on every PR and on `release/X.Y.x`.

### Phase 2 — Version Bumping Matrix
Bump release versions **strictly on `release/X.Y.x`** via a short-lived PR (`chore/bump-X.Y.Z → release/X.Y.x`). Do **not** merge patch release version bumps back into `main`; `main` stays on the next minor dev/snapshot version (`0.(Y+1).0-SNAPSHOT`).

Verify **every** version file on `release/X.Y.x` before tagging (no hardcoded stale versions):

#### `impulse-graph-core`
- [ ] `impulse-cpp/CMakeLists.txt` (`project(impulse_graph VERSION X.Y.Z ...)`)
- [ ] `impulse-cpp/include/impulse_graph.h` (`IMPULSE_VERSION_MAJOR`, `IMPULSE_VERSION_MINOR`, `IMPULSE_VERSION_PACKED`)
- [ ] `impulse-rust/Cargo.toml` (`version = "X.Y.Z"`)
- [ ] `impulse-cli/Cargo.toml` (`version = "X.Y.Z"` + `impulse-graph` dependency version)
- [ ] `impulse-python/pyproject.toml` (`version = "X.Y.Z"`)
- [ ] `impulse-python/impulse_graph/__init__.py` (uses `importlib.metadata.version("impulse-graph")` with fallback `"X.Y.Z"`)
- [ ] `impulse-dotnet/ImpulseGraph.csproj` (`<Version>X.Y.Z</Version>`)
- [ ] `impulse-node/package.json` (`"version": "X.Y.Z"`)

#### `impulse-graph-java`
- [ ] Root `pom.xml` and all child module `pom.xml`s (`mvn versions:set -DnewVersion=X.Y.Z -DgenerateBackupPoms=false`)
- [ ] `impulse-gradle-plugin` build version (if released)

### Phase 3 — Full-Gamut Pre-Release Sign-Off
- [ ] **Tier 1** (`impulse-graph-core`): C++ assertions `ctest`, ASan+UBSan, MC/DC, CMake install header check, Rust `cargo test`, Python `pytest`, CI Go/.NET/Node green.
- [ ] **Tier 2** (`impulse-graph-java`): Full 11-module `mvn clean install` + `spotless:check` green.
- [ ] **Tier 3** (`impulse-verification`): `four_way_validation.py` (**531/531**), `fuzz_vm_contracts.py` (**20,000/20,000**, record seed), `test_seeded_roundtrip_50.py` (**50/50**).
- [ ] **Tier 4** (`impulse-verification/hetionet-rest` & `impulse-benchmarks`): REST E2E (`mvn test`) green; **Memory Leakage Gate** passed (0 ASan/LSan leaks, 0 hot-path prepared-statement allocs, flat steady-state RSS under `stress_test.py` soak); **Performance Regression Gate** passed (no unexplained `> 5%` regression vs. prior release, warmed-up Java `<= 2x` C++).

### Phase 4 — Tagging, 100% CI Release & Registry Publishing
*(Per `AGENTS.md` §3.8, all release binaries and packages MUST be built strictly in GitHub Actions CI — never locally.)*
- [ ] Verify `origin/release/X.Y.x` is green in CI and contains the exact bumped `X.Y.Z` commit.
- [ ] Create and push annotated tag `vX.Y.Z` directly on `origin/release/X.Y.x` in `impulse-graph-core` (and `impulse-graph-java` if releasing Java). Keep `origin/release/X.Y.x` alive for future `X.Y.(Z+1)` patches.
- [ ] Verify `release.yml` succeeds and uploads all platform binaries, SDK archives, `SHA256SUMS`, Cosign/Sigstore bundles (`SHA256SUMS.sigstore.json`), and SLSA provenance attestations.
- [ ] Verify **PyPI** (`pypi.yml` / Trusted Publishing) publishes `impulse-graph==X.Y.Z`.
- [ ] Verify **crates.io** (`crates.yml`) publishes `impulse-graph = "X.Y.Z"`.
- [ ] Verify **Maven Central** (`maven-publish.yml`) stages and publishes `org.impulsegraph:*:X.Y.Z` (including `-sources.jar` and `-javadoc.jar` for Kotlin/Scala/Clojure).

### Phase 5 — Release Notes, Documentation & Website
- [ ] Draft and publish GitHub Release notes on `vX.Y.Z` (grouped by C-ABI/Core, VM, Bindings, Packaging, with issue/PR links and fuzzer/validation sign-off stats).
- [ ] Update `impulse-website` (`status.md`, installation version pins, announcements/changelog).
- [ ] Run `mkdocs build --strict` and execute Tier 5 Docker snippet verification ([WEBSITE_CODE_SAMPLE_VERIFICATION.md](WEBSITE_CODE_SAMPLE_VERIFICATION.md)) against the live `X.Y.Z` packages.

### Phase 6 — Post-Release Hygiene
- [ ] Close milestone `vX.Y.Z` in GitHub after all issues are closed.
- [ ] Ensure `main` contains all `vX.Y.Z` fixes (already true if PRs landed on `main` first and were cherry-picked onto `release/X.Y.x`).
- [ ] Bump `main` snapshot/dev version if starting a new minor cycle (e.g., `0.11.0-SNAPSHOT`).
