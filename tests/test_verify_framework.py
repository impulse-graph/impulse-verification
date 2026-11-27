"""Self-test suite for catalog.toml, verify.py, coverage parsers, SCIP decoder, and .imps graph queries."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

VERIFICATION_DIR = Path(__file__).resolve().parent.parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from verify_lib.catalog import build_suite_command, load_catalog, resolve_workspace
from verify_lib.coverage import (
    parse_jacoco_xml,
    parse_llvm_cov_json,
    parse_opcode_coverage_output,
    parse_pitest_xml,
)
from verify_lib.scip_ingest import (
    ROLE_DEFINITION,
    ScipDocument,
    ScipIndex,
    ScipOccurrence,
    ScipRelationship,
    ScipSymbolInfo,
    decode_scip_bytes,
    encode_scip_index,
    scip_index_to_ir,
)


class TestCatalogIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.cat = load_catalog(VERIFICATION_DIR / "catalog.toml")
        self.ws = resolve_workspace(VERIFICATION_DIR, self.cat.repo_names)

    def test_gamuts_defined_and_expand(self) -> None:
        for expected in ("veryquick", "standard", "hardened", "full-gamut", "soak", "bench", "post-release"):
            self.assertIn(expected, self.cat.gamuts)
            suites = self.cat.resolve_gamut_suites(expected)
            self.assertGreater(len(suites), 0, f"Gamut {expected} must include at least one suite")

        vq_ids = {s.id for s, _ in self.cat.resolve_gamut_suites("veryquick")}
        std_ids = {s.id for s, _ in self.cat.resolve_gamut_suites("standard")}
        hard_ids = {s.id for s, _ in self.cat.resolve_gamut_suites("hardened")}
        fg_ids = {s.id for s, _ in self.cat.resolve_gamut_suites("full-gamut")}
        self.assertTrue(vq_ids.issubset(std_ids))
        self.assertTrue(std_ids.issubset(hard_ids))
        self.assertTrue(hard_ids.issubset(fg_ids))
        # Benchmarks must NOT be silently included in standard
        self.assertFalse(any(s.pillar == "perf" for s, _ in self.cat.resolve_gamut_suites("standard")))

    def test_all_verification_scripts_exist(self) -> None:
        for sid, s in self.cat.suites.items():
            if s.repo == "verification":
                cmd, cwd = build_suite_command(s, "standard", self.ws, seed=1, workers=2)
                for tok in cmd:
                    if tok.startswith("tools/") and tok.endswith(".py"):
                        script_path = cwd / tok
                        self.assertTrue(script_path.is_file(), f"Suite {sid} references missing script {script_path}")

    def test_override_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = resolve_workspace(VERIFICATION_DIR, self.cat.repo_names, overrides={"java": td})
            self.assertEqual(ws.repos["java"], Path(td).resolve())


class TestCoverageParsers(unittest.TestCase):
    def test_opcode_coverage_parser(self) -> None:
        sample = "Covered opcodes: OP_INDEX_SCAN (0x54), OP_CSR_WALK 0x10, OP_LANE_WALK"
        ops = parse_opcode_coverage_output(sample)
        self.assertIn("OP_INDEX_SCAN", ops)
        self.assertIn("OP_CSR_WALK", ops)
        self.assertIn("OP_LANE_WALK", ops)
        self.assertIn("0X54", ops)

    def test_llvm_cov_parser(self) -> None:
        payload = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/repo/impulse-cpp/src/vm.cpp",
                            "summary": {
                                "lines": {"percent": 91.5},
                                "branches": {"percent": 82.0},
                                "mcdc": {"percent": 76.4},
                            },
                        }
                    ]
                }
            ]
        }
        rows = parse_llvm_cov_json(payload)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["mcdc_pct"], 76.4)
        self.assertEqual(rows[0]["tool"], "llvm-cov")

    def test_jacoco_xml_parser(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <report name="impulse-vm">
          <package name="org/impulsegraph/vm">
            <sourcefile name="ImpulseVm.java">
              <counter type="LINE" missed="10" covered="90"/>
              <counter type="BRANCH" missed="5" covered="15"/>
            </sourcefile>
          </package>
        </report>"""
        rows = parse_jacoco_xml(xml, module_prefix="impulse-vm")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file_path"], "impulse-vm/src/main/java/org/impulsegraph/vm/ImpulseVm.java")
        self.assertAlmostEqual(rows[0]["line_pct"], 90.0)
        self.assertAlmostEqual(rows[0]["branch_pct"], 75.0)

    def test_pitest_xml_parser(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <mutations>
          <mutation status="KILLED"><mutatedClass>org.impulsegraph.vm.ImpulseVm</mutatedClass><sourceFile>ImpulseVm.java</sourceFile></mutation>
          <mutation status="SURVIVED"><mutatedClass>org.impulsegraph.vm.ImpulseVm</mutatedClass><sourceFile>ImpulseVm.java</sourceFile></mutation>
        </mutations>"""
        rows = parse_pitest_xml(xml)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mutants_killed"], 1)
        self.assertEqual(rows[0]["mutants_total"], 2)


class TestScipWireDecoder(unittest.TestCase):
    def test_roundtrip_and_call_graph_derivation(self) -> None:
        idx = ScipIndex(
            project_root="file:///workspace/impulse-graph-java",
            documents=[
                ScipDocument(
                    relative_path="impulse-vm/src/main/java/org/impulsegraph/vm/ImpulseVm.java",
                    language="java",
                    symbols=[
                        ScipSymbolInfo(
                            symbol="scip-java maven . ImpulseVm#execute().",
                            kind=26,
                            display_name="execute",
                            relationships=[
                                ScipRelationship(
                                    symbol="scip-java maven . VmEngine#execute().",
                                    is_implementation=True,
                                )
                            ],
                        ),
                        ScipSymbolInfo(
                            symbol="scip-java maven . ImpulseVm#execIndexScan().",
                            kind=26,
                            display_name="execIndexScan",
                        ),
                    ],
                    occurrences=[
                        ScipOccurrence(
                            range=[10, 4, 11],
                            symbol="scip-java maven . ImpulseVm#execute().",
                            symbol_roles=ROLE_DEFINITION,
                            enclosing_range=[10, 0, 25, 1],
                        ),
                        ScipOccurrence(
                            range=[15, 8, 21],
                            symbol="scip-java maven . ImpulseVm#execIndexScan().",
                            symbol_roles=0,
                        ),
                        ScipOccurrence(
                            range=[30, 4, 17],
                            symbol="scip-java maven . ImpulseVm#execIndexScan().",
                            symbol_roles=ROLE_DEFINITION,
                            enclosing_range=[30, 0, 40, 1],
                        ),
                    ],
                )
            ],
        )
        wire = encode_scip_index(idx)
        decoded = decode_scip_bytes(wire)
        self.assertEqual(decoded.project_root, idx.project_root)
        self.assertEqual(len(decoded.documents), 1)

        ir = scip_index_to_ir(decoded, repo_name="java")
        self.assertEqual(len(ir["symbols"]), 2)
        self.assertIn(
            (
                "java:java:scip-java maven . ImpulseVm#execute().",
                "java:java:scip-java maven . ImpulseVm#execIndexScan().",
            ),
            ir["calls"],
        )
        self.assertIn(
            (
                "java:java:scip-java maven . ImpulseVm#execute().",
                "java:java:scip-java maven . VmEngine#execute().",
            ),
            ir["implements_or_extends"],
        )


if __name__ == "__main__":
    unittest.main()
