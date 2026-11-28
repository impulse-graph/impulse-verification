#!/usr/bin/env python3
"""Cross-Language .imps Metadata Introspection Parity Verification Suite.

Builds a v0.11 .imps snapshot with domains, domain/relation DECIMAL64 attributes,
secondary index, and UTF-8 Section 6 custom metadata, then introspects it across
C++/Python, Node.js, and Rust/CLI and asserts exact field-by-field parity.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "impulse-graph-core" / "impulse-python"
NODE_DIR = ROOT / "impulse-graph-core" / "impulse-node"
CLI_BIN = ROOT / "impulse-graph-core" / "impulse-cli" / "target" / "debug" / "impulse-graph"

sys.path.insert(0, str(PY_DIR))
from impulse_graph import Snapshot, Writer  # noqa: E402


def build_parity_snapshot(path: Path) -> str:
    w = Writer(str(path))
    w.add_domain(0, 0x03, "Account")
    w.set_domain_node_count(0, 3)
    w.add_domain_attribute_decimal64(
        0, "balance", 10, 2, struct.pack("<3q", 12345, -99999, 0)
    )
    w.add_domain(1, 0x03, "Merchant")
    w.set_domain_node_count(1, 2)
    w.add_relation(
        0,
        1,
        0,
        3,
        2,
        0,
        struct.pack("<4I", 0, 1, 2, 2),
        struct.pack("<2I", 0, 1),
    )
    w.add_attribute_decimal64(0, "amount", 10, 2, struct.pack("<2q", 500, 1500))
    w.add_index(0, 0xFFFF, 0, 1, "idx_acct_balance", bytes(64), 0xCAFEBABE)
    w.set_metadata("owner", "parity-suite")
    w.set_metadata("env", "verification")
    w.finalize()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_python(path: Path) -> dict:
    with Snapshot(str(path)) as snap:
        hdr = snap.header()
        doms = []
        for i in range(snap.domain_count()):
            d = snap.get_domain(i)
            attrs = [
                snap.get_domain_attribute_info(i, a)
                for a in range(snap.get_domain_attribute_count(i))
            ]
            doms.append(
                {
                    "domain_id": d["domain_id"],
                    "name": d["name"],
                    "node_count": d["node_count"],
                    "attr_count": d["attr_count"],
                    "attributes": [
                        {
                            "name": a["name"],
                            "type_code": a["type_code"],
                            "base_type_code": a["base_type_code"],
                            "is_nullable": a["is_nullable"],
                            "data_bytes": a["data_bytes"],
                        }
                        for a in attrs
                    ],
                }
            )
        rels = []
        for i in range(snap.relation_count()):
            r = snap.get_relation(i)
            attrs = [snap.get_attribute_info(i, a) for a in range(r["attr_count"])]
            rels.append(
                {
                    "relation_id": r["relation_id"],
                    "src_domain_id": r["src_domain_id"],
                    "tgt_domain_id": r["tgt_domain_id"],
                    "node_count": r["node_count"],
                    "edge_count": r["edge_count"],
                    "src_node_id_width": r["src_node_id_width"],
                    "tgt_node_id_width": r["tgt_node_id_width"],
                    "has_csr": r["has_csr"],
                    "has_csc": r["has_csc"],
                    "has_coo": r["has_coo"],
                    "is_virtual": r["is_virtual"],
                    "attributes": [
                        {
                            "name": a["name"],
                            "type_code": a["type_code"],
                            "base_type_code": a["base_type_code"],
                            "is_nullable": a["is_nullable"],
                            "data_bytes": a["data_bytes"],
                        }
                        for a in attrs
                    ],
                }
            )
        idxs = [
            {
                "index_id": idx["index_id"],
                "domain_id": idx["domain_id"],
                "relation_id": idx["relation_id"],
                "name": idx["name"],
                "data_bytes": idx["data_bytes"],
                "payload_feature_mask": idx["payload_feature_mask"],
            }
            for idx in snap.indices()
        ]
        return {
            "magic": hdr["magic"],
            "version": hdr["version"],
            "data_offset": hdr["data_offset"],
            "domain_count": hdr["domain_count"],
            "relation_count": hdr["relation_count"],
            "index_count": hdr["index_count"],
            "file_size": snap.file_size(),
            "sha256": snap.sha256(),
            "domains": doms,
            "relations": rels,
            "indices": idxs,
            "metadata": snap.metadata(),
        }


def collect_node(path: Path) -> dict:
    script = r"""
const impulse = require('./index.js');
const snap = new impulse.Snapshot(process.argv[1]);
const hdr = snap.header();
const doms = snap.domains().map((d, i) => {
  const attrs = [];
  for (let a = 0; a < snap.getDomainAttributeCount(i); a++) {
    const ai = snap.getDomainAttributeInfo(i, a);
    attrs.push({
      name: ai.name,
      type_code: ai.typeCode,
      base_type_code: ai.baseTypeCode,
      is_nullable: ai.isNullable,
      data_bytes: Number(ai.dataBytes),
    });
  }
  return {
    domain_id: d.domainId,
    name: d.name,
    node_count: Number(d.nodeCount),
    attr_count: d.attrCount,
    attributes: attrs,
  };
});
const rels = snap.relations().map((r, i) => {
  const attrs = [];
  for (let a = 0; a < r.attrCount; a++) {
    const ai = snap.getAttributeInfo(i, a);
    attrs.push({
      name: ai.name,
      type_code: ai.typeCode,
      base_type_code: ai.baseTypeCode,
      is_nullable: ai.isNullable,
      data_bytes: Number(ai.dataBytes),
    });
  }
  return {
    relation_id: r.relationId,
    src_domain_id: r.srcDomainId,
    tgt_domain_id: r.tgtDomainId,
    node_count: Number(r.nodeCount),
    edge_count: Number(r.edgeCount),
    src_node_id_width: r.srcNodeIdWidth,
    tgt_node_id_width: r.tgtNodeIdWidth,
    has_csr: r.hasCsr,
    has_csc: r.hasCsc,
    has_coo: r.hasCoo,
    is_virtual: r.isVirtual,
    attributes: attrs,
  };
});
const idxs = snap.indices().map((idx) => ({
  index_id: idx.indexId,
  domain_id: idx.domainId,
  relation_id: idx.relationId,
  name: idx.name,
  data_bytes: Number(idx.dataBytes),
  payload_feature_mask: Number(idx.payloadFeatureMask),
}));
const out = {
  magic: hdr.magic,
  version: hdr.version,
  data_offset: hdr.dataOffset,
  domain_count: hdr.domainCount,
  relation_count: hdr.relationCount,
  index_count: hdr.indexCount,
  file_size: Number(snap.fileSize()),
  sha256: snap.sha256(),
  domains: doms,
  relations: rels,
  indices: idxs,
  metadata: snap.getAllMetadata(),
};
snap.close();
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", script, str(path)],
        cwd=str(NODE_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def collect_rust_cli(path: Path) -> dict:
    if not CLI_BIN.exists():
        subprocess.run(
            ["cargo", "build", "--manifest-path", str(ROOT / "impulse-graph-core" / "impulse-cli" / "Cargo.toml")],
            check=True,
            capture_output=True,
        )
    proc = subprocess.run(
        [str(CLI_BIN), "inspect", str(path), "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    raw = json.loads(proc.stdout)
    ver_str = raw["header"]["version"]
    version_num = int(ver_str.split("(0x")[1].rstrip(")"), 16)
    return {
        "magic": int(raw["header"]["magic"], 16),
        "version": version_num,
        "data_offset": raw["header"]["data_offset"],
        "domain_count": raw["header"]["domain_count"],
        "relation_count": raw["header"]["relation_count"],
        "index_count": raw["header"]["index_count"],
        "file_size": raw["size_bytes"],
        "sha256": raw["sha256"],
        "domains": [
            {
                "domain_id": d["domain_id"],
                "name": d["name"],
                "node_count": d["node_count"],
                "attr_count": d["attr_count"],
                "attributes": [
                    {
                        "name": a["name"],
                        "type_code": a["type_code"],
                        "base_type_code": a["base_type_code"],
                        "is_nullable": a["is_nullable"],
                        "data_bytes": a["data_bytes"],
                    }
                    for a in d["attributes"]
                ],
            }
            for d in raw["domains"]
        ],
        "relations": [
            {
                "relation_id": r["relation_id"],
                "src_domain_id": r["src_domain_id"],
                "tgt_domain_id": r["tgt_domain_id"],
                "node_count": r["node_count"],
                "edge_count": r["edge_count"],
                "src_node_id_width": r["src_node_id_width"],
                "tgt_node_id_width": r["tgt_node_id_width"],
                "has_csr": r["has_csr"],
                "has_csc": r["has_csc"],
                "has_coo": r["has_coo"],
                "is_virtual": r["is_virtual"],
                "attributes": [
                    {
                        "name": a["name"],
                        "type_code": a["type_code"],
                        "base_type_code": a["base_type_code"],
                        "is_nullable": a["is_nullable"],
                        "data_bytes": a["data_bytes"],
                    }
                    for a in r["attributes"]
                ],
            }
            for r in raw["relations"]
        ],
        "indices": [
            {
                "index_id": idx["index_id"],
                "domain_id": idx["domain_id"],
                "relation_id": idx["relation_id"],
                "name": idx["name"],
                "data_bytes": idx["data_bytes"],
                "payload_feature_mask": idx["payload_feature_mask"],
            }
            for idx in raw["indices"]
        ],
        "metadata": raw["metadata"],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="impulse_meta_parity_") as td:
        snap_path = Path(td) / "parity_v11.imps"
        expected_sha = build_parity_snapshot(snap_path)

        py_out = collect_python(snap_path)
        node_out = collect_node(snap_path)
        cli_out = collect_rust_cli(snap_path)

        assert py_out["sha256"] == expected_sha, f"Python SHA mismatch: {py_out['sha256']}"
        assert node_out == py_out, f"Node != Python:\nNODE={json.dumps(node_out, indent=2)}\nPY={json.dumps(py_out, indent=2)}"
        assert cli_out == py_out, f"Rust/CLI != Python:\nCLI={json.dumps(cli_out, indent=2)}\nPY={json.dumps(py_out, indent=2)}"

        print(f"PASS: Cross-language metadata parity verified (C++/Python == Node.js == Rust/CLI, sha256={expected_sha[:16]}...)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
