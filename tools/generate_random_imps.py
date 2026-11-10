#!/usr/bin/env python3
"""
Deterministic, Seeded Random .imps Schema & Dataset Generator for Impulse Graph Engine.
Synthesizes random domains, relations, and domain/relation attributes (with random types,
vector dimensions, and nullability), outputs manifest.json + CSV files + schema_metadata.json,
and compiles/validates the .imps snapshot using the official impulse-graph CLI (Rule 3.17).
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AttrTypeSpec:
    """Immutable specification of an attribute type in a generator version pool."""
    tag: str                # Suffix tag used in deterministic attribute naming (e.g. INT, VEC4_FLOAT)
    manifest_type: str      # Base type string in manifest.json (e.g. int32, float32, string)
    base_type_code: int     # Spec v0.9.0 base type code (0x01..0x0C)
    dimension: int = 1      # 1 for scalar, >1 for fixed-size vector

    def manifest_type_for(self, nullable: bool) -> str:
        return f"nullable_{self.manifest_type}" if nullable else self.manifest_type

    def expected_type_code(self, nullable: bool) -> int:
        return (self.base_type_code | 0x80) if nullable else self.base_type_code


@dataclass(frozen=True)
class GeneratorVersionSpec:
    """
    Immutable generator version definition.
    Pinning a version guarantees that historical seeds reproduce the exact same
    schema and data even when newer versions introduce additional types (e.g. F16).
    """
    version: int
    description: str
    domain_key_types: Tuple[str, ...]
    attr_types: Tuple[AttrTypeSpec, ...]


# ============================================================================
# Immutable Generator Version Registry
# NEVER mutate existing version entries; add new integer keys (2, 3, ...) when
# expanding the type or feature pool in the future.
# ============================================================================
GENERATOR_VERSIONS: Dict[int, GeneratorVersionSpec] = {
    1: GeneratorVersionSpec(
        version=1,
        description="Initial v1 pool: scalar (INT, LONG, FLOAT, DOUBLE, STR, I8, I16) and vector (VEC4_FLOAT, VEC8_FLOAT) types",
        domain_key_types=(
            "int32",
            "int64",
            "string",
            "uuid",
            "int16",
            "int8",
        ),
        attr_types=(
            AttrTypeSpec(tag="INT",        manifest_type="int32",   base_type_code=0x03, dimension=1),
            AttrTypeSpec(tag="LONG",       manifest_type="int64",   base_type_code=0x04, dimension=1),
            AttrTypeSpec(tag="FLOAT",      manifest_type="float32", base_type_code=0x06, dimension=1),
            AttrTypeSpec(tag="DOUBLE",     manifest_type="float64", base_type_code=0x07, dimension=1),
            AttrTypeSpec(tag="STR",        manifest_type="string",  base_type_code=0x0B, dimension=1),
            AttrTypeSpec(tag="I8",         manifest_type="int8",    base_type_code=0x01, dimension=1),
            AttrTypeSpec(tag="I16",        manifest_type="int16",   base_type_code=0x02, dimension=1),
            AttrTypeSpec(tag="VEC4_FLOAT", manifest_type="float32", base_type_code=0x06, dimension=4),
            AttrTypeSpec(tag="VEC8_FLOAT", manifest_type="float32", base_type_code=0x06, dimension=8),
        ),
    ),
}

LATEST_GENERATOR_VERSION = max(GENERATOR_VERSIONS.keys())


@dataclass
class GeneratorConfig:
    min_domains: int = 1
    max_domains: int = 6
    min_relations: int = 1
    max_relations: int = 10
    min_domain_attrs: int = 0
    max_domain_attrs: int = 5
    min_rel_attrs: int = 0
    max_rel_attrs: int = 5
    null_attr_prob: float = 0.40
    csc_prob: float = 0.50
    schema_only: bool = False
    min_nodes: int = 5
    max_nodes: int = 50
    min_edges: int = 1
    max_edges: int = 100
    empty_rel_prob: float = 0.15
    null_cell_prob: float = 0.25


@dataclass
class AttributeSchema:
    name: str
    column: str
    type_tag: str
    manifest_type: str
    base_type_code: int
    expected_type_code: int
    nullable: bool
    dimension: int = 1


@dataclass
class DomainSchema:
    id: int
    num: int
    name: str
    key_type: str
    file: str
    attributes: List[AttributeSchema] = field(default_factory=list)


@dataclass
class RelationSchema:
    index: int
    name: str
    src_domain: int
    tgt_domain: int
    src_domain_name: str
    tgt_domain_name: str
    include_csc: bool
    file: str
    attributes: List[AttributeSchema] = field(default_factory=list)


@dataclass
class GeneratedSchema:
    seed: int
    gen_version: int
    spec_version: str
    domains: List[DomainSchema]
    relations: List[RelationSchema]


def derive_stream_seed(seed: int, gen_version: int, stream_name: str) -> int:
    """
    Deterministically derives an integer PRNG seed for an isolated stream
    ('schema' vs 'data') so schema generation is 100% invariant to data population flags.
    """
    payload = f"impulse-random-imps:v={gen_version}:seed={seed}:stream={stream_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def find_impulse_cli() -> Path:
    env_cli = os.environ.get("IMPULSE_CLI")
    if env_cli and os.path.isfile(env_cli) and os.access(env_cli, os.X_OK):
        return Path(env_cli).resolve()

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root.parent / "impulse-graph-core" / "impulse-cli" / "target" / "release" / "impulse-graph",
        repo_root.parent / "impulse-graph-core" / "impulse-cli" / "target" / "debug" / "impulse-graph",
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()

    which_cli = shutil.which("impulse-graph")
    if which_cli:
        return Path(which_cli).resolve()

    raise RuntimeError(
        "Could not find 'impulse-graph' binary in IMPULSE_CLI, ../impulse-graph-core/impulse-cli/target/, or PATH."
    )


def _generate_attributes(
    rng: random.Random,
    prefix: str,
    count: int,
    vspec: GeneratorVersionSpec,
    null_attr_prob: float,
) -> List[AttributeSchema]:
    attrs: List[AttributeSchema] = []
    for a_idx in range(1, count + 1):
        tspec: AttrTypeSpec = rng.choice(vspec.attr_types)
        is_nullable = rng.random() < null_attr_prob
        if is_nullable:
            attr_name = f"{prefix}_A{a_idx}_NULL_{tspec.tag}"
        else:
            attr_name = f"{prefix}_A{a_idx}_{tspec.tag}"

        attrs.append(
            AttributeSchema(
                name=attr_name,
                column=attr_name,
                type_tag=tspec.tag,
                manifest_type=tspec.manifest_type_for(is_nullable),
                base_type_code=tspec.base_type_code,
                expected_type_code=tspec.expected_type_code(is_nullable),
                nullable=is_nullable,
                dimension=tspec.dimension,
            )
        )
    return attrs


def generate_schema(seed: int, gen_version: int, config: GeneratorConfig) -> GeneratedSchema:
    if gen_version not in GENERATOR_VERSIONS:
        supported = ", ".join(str(k) for k in sorted(GENERATOR_VERSIONS.keys()))
        raise ValueError(f"Unsupported --gen-version {gen_version}. Supported versions: {supported}")

    vspec = GENERATOR_VERSIONS[gen_version]
    schema_seed = derive_stream_seed(seed, gen_version, "schema")
    rng = random.Random(schema_seed)

    num_domains = rng.randint(config.min_domains, config.max_domains)
    domains: List[DomainSchema] = []

    for d_id in range(num_domains):
        d_num = d_id + 1
        d_name = f"D{d_num}"
        key_type = rng.choice(vspec.domain_key_types)
        num_attrs = rng.randint(config.min_domain_attrs, config.max_domain_attrs)
        d_attrs = _generate_attributes(rng, d_name, num_attrs, vspec, config.null_attr_prob)

        domains.append(
            DomainSchema(
                id=d_id,
                num=d_num,
                name=d_name,
                key_type=key_type,
                file=f"{d_name}.csv",
                attributes=d_attrs,
            )
        )

    num_relations = rng.randint(config.min_relations, config.max_relations)
    relations: List[RelationSchema] = []

    for r_idx in range(1, num_relations + 1):
        src_dom = rng.choice(domains)
        tgt_dom = rng.choice(domains)
        r_name = f"R_{src_dom.name}_{tgt_dom.name}_{r_idx}"
        include_csc = rng.random() < config.csc_prob
        num_attrs = rng.randint(config.min_rel_attrs, config.max_rel_attrs)
        r_attrs = _generate_attributes(rng, r_name, num_attrs, vspec, config.null_attr_prob)

        relations.append(
            RelationSchema(
                index=r_idx,
                name=r_name,
                src_domain=src_dom.id,
                tgt_domain=tgt_dom.id,
                src_domain_name=src_dom.name,
                tgt_domain_name=tgt_dom.name,
                include_csc=include_csc,
                file=f"{r_name}.csv",
                attributes=r_attrs,
            )
        )

    return GeneratedSchema(
        seed=seed,
        gen_version=gen_version,
        spec_version="0.9.0",
        domains=domains,
        relations=relations,
    )


def _format_domain_key(dom: DomainSchema, node_idx: int) -> str:
    kt = dom.key_type.lower()
    if kt in ("int8", "int16", "int32", "int64"):
        return str(node_idx)
    if kt == "uuid":
        return f"00000000-0000-4000-{dom.num:04x}-{node_idx:012x}"
    return f"{dom.name}_N{node_idx}"


def _synthesize_attr_value(rng: random.Random, attr: AttributeSchema, row_idx: int) -> str:
    tag = attr.type_tag
    if attr.dimension > 1:
        vals = [f"{rng.uniform(-10.0, 10.0):.4f}" for _ in range(attr.dimension)]
        return "[" + ";".join(vals) + "]"
    if tag == "I8":
        return str(rng.randint(-120, 120))
    if tag == "I16":
        return str(rng.randint(-30000, 30000))
    if tag == "INT":
        return str(rng.randint(-100000, 100000))
    if tag == "LONG":
        return str(rng.randint(-10_000_000_000, 10_000_000_000))
    if tag == "FLOAT":
        return f"{rng.uniform(-1000.0, 1000.0):.4f}"
    if tag == "DOUBLE":
        return f"{rng.uniform(-100000.0, 100000.0):.8f}"
    if tag == "STR":
        return f"{attr.name}_v{row_idx}_{rng.randint(100, 999)}"
    return str(rng.randint(0, 100))


def write_dataset_files(
    schema: GeneratedSchema,
    config: GeneratorConfig,
    out_dir: Path,
) -> dict:
    """
    Writes manifest.json, domain CSVs, relation CSVs, and schema_metadata.json
    into out_dir. Returns the metadata dictionary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_seed = derive_stream_seed(schema.seed, schema.gen_version, "data")
    rng = random.Random(data_seed)

    domain_keys: Dict[int, List[str]] = {}
    domain_stats: Dict[str, dict] = {}

    # 1. Synthesize Domain Node CSV files
    for dom in schema.domains:
        if config.schema_only:
            node_count = 0
        else:
            node_count = rng.randint(config.min_nodes, config.max_nodes)

        keys = [_format_domain_key(dom, i) for i in range(node_count)]
        domain_keys[dom.id] = keys

        csv_path = out_dir / dom.file
        # Always include at least 2 columns ('id', '_domain') so delimiter auto-detection
        # in impulse-cli reliably detects comma-delimited CSV even when attributes is empty.
        headers = ["id", "_domain"] + [a.column for a in dom.attributes]
        null_counts = {a.name: 0 for a in dom.attributes}

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(headers)
            for row_idx, key in enumerate(keys):
                row = [key, dom.name]
                for attr in dom.attributes:
                    if attr.nullable and rng.random() < config.null_cell_prob:
                        row.append("")
                        null_counts[attr.name] += 1
                    else:
                        row.append(_synthesize_attr_value(rng, attr, row_idx))
                writer.writerow(row)

        domain_stats[dom.name] = {
            "node_count": node_count,
            "null_counts": null_counts,
        }

    # 2. Synthesize Relation Edge CSV files
    relation_stats: Dict[str, dict] = {}
    for rel in schema.relations:
        src_keys = domain_keys[rel.src_domain]
        tgt_keys = domain_keys[rel.tgt_domain]

        if config.schema_only or not src_keys or not tgt_keys:
            edge_count = 0
        elif rng.random() < config.empty_rel_prob:
            edge_count = 0
        else:
            max_possible = len(src_keys) * len(tgt_keys)
            upper = min(config.max_edges, max_possible)
            lower = min(config.min_edges, upper)
            edge_count = rng.randint(lower, upper) if upper > 0 else 0

        # Draw distinct directed (src, tgt) pairs deterministically
        edges: List[Tuple[str, str]] = []
        if edge_count > 0:
            max_possible = len(src_keys) * len(tgt_keys)
            if edge_count >= max_possible // 2 and max_possible <= 10_000:
                all_pairs = [(s, t) for s in src_keys for t in tgt_keys]
                edges = rng.sample(all_pairs, edge_count)
            else:
                seen = set()
                while len(edges) < edge_count:
                    pair = (rng.choice(src_keys), rng.choice(tgt_keys))
                    if pair not in seen:
                        seen.add(pair)
                        edges.append(pair)

        csv_path = out_dir / rel.file
        headers = ["src", "tgt"] + [a.column for a in rel.attributes]
        null_counts = {a.name: 0 for a in rel.attributes}

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(headers)
            for row_idx, (s_key, t_key) in enumerate(edges):
                row = [s_key, t_key]
                for attr in rel.attributes:
                    if attr.nullable and rng.random() < config.null_cell_prob:
                        row.append("")
                        null_counts[attr.name] += 1
                    else:
                        row.append(_synthesize_attr_value(rng, attr, row_idx))
                writer.writerow(row)

        relation_stats[rel.name] = {
            "edge_count": len(edges),
            "null_counts": null_counts,
        }

    # 3. Write manifest.json for `impulse-graph build`
    manifest_domains = []
    for dom in schema.domains:
        d_entry = {
            "id": dom.id,
            "name": dom.name,
            "key_type": dom.key_type,
            "file": dom.file,
            "format": "csv",
            "id_column": "id",
        }
        if dom.attributes:
            d_entry["attributes"] = [
                {
                    "name": a.name,
                    "column": a.column,
                    "type": a.manifest_type,
                    **({"dimension": a.dimension} if a.dimension > 1 else {}),
                }
                for a in dom.attributes
            ]
        manifest_domains.append(d_entry)

    manifest_relations = []
    for rel in schema.relations:
        r_entry = {
            "name": rel.name,
            "src_domain": rel.src_domain,
            "tgt_domain": rel.tgt_domain,
            "encoding": "raw_uint32",
            "include_csc": rel.include_csc,
            "file": rel.file,
            "format": "csv",
            "src_column": "src",
            "tgt_column": "tgt",
        }
        if rel.attributes:
            r_entry["attributes"] = [
                {
                    "name": a.name,
                    "column": a.column,
                    "type": a.manifest_type,
                    **({"dimension": a.dimension} if a.dimension > 1 else {}),
                }
                for a in rel.attributes
            ]
        manifest_relations.append(r_entry)

    manifest = {
        "version": schema.spec_version,
        "domains": manifest_domains,
        "relations": manifest_relations,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # 4. Write schema_metadata.json sidecar with full reproducibility & verification metadata
    metadata = {
        "seed": schema.seed,
        "gen_version": schema.gen_version,
        "spec_version": schema.spec_version,
        "schema_only": config.schema_only,
        "config": dataclasses.asdict(config),
        "schema": dataclasses.asdict(schema),
        "stats": {
            "domains": domain_stats,
            "relations": relation_stats,
        },
    }
    meta_path = out_dir / "schema_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_and_validate_imps(
    cli: Path,
    manifest_path: Path,
    imps_path: Path,
    validate: bool = True,
) -> None:
    subprocess.run(
        [str(cli), "build", "-m", str(manifest_path), "-o", str(imps_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if validate:
        res = subprocess.run(
            [
                str(cli),
                "snapshot",
                "validate",
                "--strict-alignment",
                "--deep-audit",
                str(imps_path),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Snapshot validation failed for {imps_path}:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )


def generate_one(
    seed: int,
    gen_version: int,
    config: GeneratorConfig,
    out_dir: Path,
    imps_name: Optional[str] = None,
    no_build: bool = False,
    cli: Optional[Path] = None,
) -> Tuple[GeneratedSchema, Path, Optional[Path]]:
    schema = generate_schema(seed, gen_version, config)
    meta = write_dataset_files(schema, config, out_dir)
    manifest_path = out_dir / "manifest.json"

    imps_path: Optional[Path] = None
    if not no_build:
        if cli is None:
            cli = find_impulse_cli()
        fname = imps_name or f"seed_{seed}_v{gen_version}.imps"
        imps_path = out_dir / fname
        build_and_validate_imps(cli, manifest_path, imps_path, validate=True)

    total_dom_attrs = sum(len(d.attributes) for d in schema.domains)
    total_rel_attrs = sum(len(r.attributes) for r in schema.relations)
    total_nodes = sum(s["node_count"] for s in meta["stats"]["domains"].values())
    total_edges = sum(s["edge_count"] for s in meta["stats"]["relations"].values())
    empty_rels = sum(1 for s in meta["stats"]["relations"].values() if s["edge_count"] == 0)

    status_tag = "[BUILT+VALIDATED]" if imps_path else "[MANIFEST+CSV]"
    print(
        f"{status_tag} seed={seed} v={gen_version} | "
        f"domains={len(schema.domains)} (attrs={total_dom_attrs}, nodes={total_nodes}) | "
        f"relations={len(schema.relations)} (attrs={total_rel_attrs}, edges={total_edges}, empty_rels={empty_rels}) | "
        f"dir={out_dir}"
    )
    return schema, manifest_path, imps_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic, seeded .imps schema and dataset generator for Impulse Graph Engine."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base deterministic PRNG seed (default: 42)")
    parser.add_argument(
        "--gen-version",
        type=int,
        default=LATEST_GENERATOR_VERSION,
        choices=sorted(GENERATOR_VERSIONS.keys()),
        help=f"Generator version pool to pin for reproducibility (default: {LATEST_GENERATOR_VERSION})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of sequential seeds (seed .. seed + count - 1) to generate (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: datasets/random_imps/seed_<S>_v<V> or datasets/random_imps for --count > 1)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Generate 0-row CSVs (schema-only) while preserving exact schema parity for the given --seed",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Only emit manifest.json, CSVs, and schema_metadata.json without invoking impulse-graph build",
    )
    parser.add_argument("--min-domains", type=int, default=1, help="Minimum domains per schema (default: 1)")
    parser.add_argument("--max-domains", type=int, default=6, help="Maximum domains per schema (default: 6)")
    parser.add_argument("--min-relations", type=int, default=1, help="Minimum relations per schema (default: 1)")
    parser.add_argument("--max-relations", type=int, default=10, help="Maximum relations per schema (default: 10)")
    parser.add_argument("--min-domain-attrs", type=int, default=0, help="Min attributes per domain (default: 0)")
    parser.add_argument("--max-domain-attrs", type=int, default=5, help="Max attributes per domain (default: 5)")
    parser.add_argument("--min-rel-attrs", type=int, default=0, help="Min attributes per relation (default: 0)")
    parser.add_argument("--max-rel-attrs", type=int, default=5, help="Max attributes per relation (default: 5)")
    parser.add_argument("--null-attr-prob", type=float, default=0.40, help="Probability an attribute is nullable (default: 0.40)")
    parser.add_argument("--csc-prob", type=float, default=0.50, help="Probability a relation includes CSC transpose (default: 0.50)")
    parser.add_argument("--min-nodes", type=int, default=5, help="Min nodes per domain in populated mode (default: 5)")
    parser.add_argument("--max-nodes", type=int, default=50, help="Max nodes per domain in populated mode (default: 50)")
    parser.add_argument("--min-edges", type=int, default=1, help="Min edges for non-empty relation in populated mode (default: 1)")
    parser.add_argument("--max-edges", type=int, default=100, help="Max edges per relation in populated mode (default: 100)")
    parser.add_argument("--empty-rel-prob", type=float, default=0.15, help="Probability a relation has 0 rows in populated mode (default: 0.15)")
    parser.add_argument("--null-cell-prob", type=float, default=0.25, help="Probability a cell in a nullable column is NULL (default: 0.25)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    base_default_dir = repo_root / "datasets" / "random_imps"

    config = GeneratorConfig(
        min_domains=args.min_domains,
        max_domains=args.max_domains,
        min_relations=args.min_relations,
        max_relations=args.max_relations,
        min_domain_attrs=args.min_domain_attrs,
        max_domain_attrs=args.max_domain_attrs,
        min_rel_attrs=args.min_rel_attrs,
        max_rel_attrs=args.max_rel_attrs,
        null_attr_prob=args.null_attr_prob,
        csc_prob=args.csc_prob,
        schema_only=args.schema_only,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        min_edges=args.min_edges,
        max_edges=args.max_edges,
        empty_rel_prob=args.empty_rel_prob,
        null_cell_prob=args.null_cell_prob,
    )

    cli: Optional[Path] = None
    if not args.no_build:
        cli = find_impulse_cli()
        print(f"Using official Impulse CLI: {cli}")

    if args.count == 1:
        target_dir = args.out_dir or (base_default_dir / f"seed_{args.seed}_v{args.gen_version}")
        generate_one(
            seed=args.seed,
            gen_version=args.gen_version,
            config=config,
            out_dir=target_dir,
            no_build=args.no_build,
            cli=cli,
        )
    else:
        parent_dir = args.out_dir or base_default_dir
        for offset in range(args.count):
            cur_seed = args.seed + offset
            sub_dir = parent_dir / f"seed_{cur_seed}_v{args.gen_version}"
            generate_one(
                seed=cur_seed,
                gen_version=args.gen_version,
                config=config,
                out_dir=sub_dir,
                no_build=args.no_build,
                cli=cli,
            )


if __name__ == "__main__":
    main()
