"""Declarative catalog loader, validator, workspace resolver, and gamut selector."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_DETERMINISM = frozenset({"deterministic", "seeded", "stochastic", "benchmark"})
VALID_COSTS = frozenset({"fast", "medium", "slow", "soak"})
VALID_INTENSITIES = frozenset({"quick", "standard", "deep", "soak"})


@dataclass(frozen=True)
class PillarSpec:
    name: str
    description: str


@dataclass(frozen=True)
class GamutSpec:
    name: str
    description: str
    target_time_sec: int
    intensity: str
    extends: tuple[str, ...]
    suites: tuple[str, ...]


@dataclass(frozen=True)
class PrereqSpec:
    name: str
    kind: str
    description: str
    hint: str
    command: str | None = None
    module: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class SuiteSpec:
    id: str
    title: str
    pillar: str
    repo: str
    determinism: str
    cost: str
    requires: tuple[str, ...]
    cwd: str
    cmd: tuple[str, ...]
    paths: tuple[str, ...] = ()
    profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    coverage_parser: str | None = None


@dataclass
class WorkspacePaths:
    root: Path
    repos: dict[str, Path]

    def resolve_template(self, text: str, *, seed: int, workers: int) -> str:
        mapping: dict[str, str] = {
            "root": str(self.root),
            "python": sys.executable,
            "seed": str(seed),
            "workers": str(workers),
        }
        for name, path in self.repos.items():
            mapping[name] = str(path)
        return text.format(**mapping)

    def detect_repo_for_cwd(self, cwd: Path) -> tuple[str | None, Path | None]:
        """Return (repo_key, repo_root) if cwd lies inside a known repo or worktree."""
        resolved = cwd.resolve()
        for key, path in self.repos.items():
            try:
                resolved.relative_to(path.resolve())
                return key, path.resolve()
            except ValueError:
                continue
        # Also detect worktrees whose basename matches a repo directory name
        for parent in (resolved, *resolved.parents):
            for key, path in self.repos.items():
                if parent.name == path.name and (parent / ".git").exists():
                    return key, parent
        return None, None


@dataclass
class Catalog:
    schema_version: int
    default_gamut: str
    default_seed: int
    repo_names: dict[str, str]
    pillars: dict[str, PillarSpec]
    gamuts: dict[str, GamutSpec]
    prereqs: dict[str, PrereqSpec]
    suites: dict[str, SuiteSpec]
    catalog_path: Path

    def resolve_gamut_suites(self, gamut_name: str) -> list[tuple[SuiteSpec, str]]:
        """Expand gamut inheritance in topological order, returning [(SuiteSpec, intensity)]."""
        if gamut_name not in self.gamuts:
            raise KeyError(f"Unknown gamut '{gamut_name}'. Known: {', '.join(sorted(self.gamuts))}")

        seen_gamuts: set[str] = set()
        ordered_ids: list[str] = []
        seen_suites: set[str] = set()

        def _visit(name: str) -> None:
            if name in seen_gamuts:
                raise ValueError(f"Cycle detected in gamut inheritance at '{name}'")
            seen_gamuts.add(name)
            g = self.gamuts[name]
            for parent in g.extends:
                _visit(parent)
            for sid in g.suites:
                if sid not in seen_suites:
                    seen_suites.add(sid)
                    ordered_ids.append(sid)

        _visit(gamut_name)
        intensity = self.gamuts[gamut_name].intensity
        return [(self.suites[sid], intensity) for sid in ordered_ids]

    def select_suites(
        self,
        *,
        gamut: str | None = None,
        pillars: list[str] | None = None,
        repos: list[str] | None = None,
        ids: list[str] | None = None,
        intensity_override: str | None = None,
    ) -> list[tuple[SuiteSpec, str]]:
        """Resolve suites matching gamut/id/pillar/repo filters."""
        if ids:
            missing = [sid for sid in ids if sid not in self.suites]
            if missing:
                raise KeyError(f"Unknown suite id(s): {', '.join(missing)}")
            intensity = intensity_override or "standard"
            base = [(self.suites[sid], intensity) for sid in ids]
        elif gamut:
            base = self.resolve_gamut_suites(gamut)
            if intensity_override:
                base = [(s, intensity_override) for s, _ in base]
        elif pillars or repos:
            intensity = intensity_override or "standard"
            base = [(s, intensity) for s in self.suites.values()]
        else:
            base = self.resolve_gamut_suites(self.default_gamut)
            if intensity_override:
                base = [(s, intensity_override) for s, _ in base]

        if pillars:
            pset = set(pillars)
            unknown = pset - set(self.pillars)
            if unknown:
                raise KeyError(f"Unknown pillar(s): {', '.join(sorted(unknown))}")
            base = [(s, i) for s, i in base if s.pillar in pset]
        if repos:
            rset = set(repos)
            unknown = rset - set(self.repo_names)
            if unknown:
                raise KeyError(f"Unknown repo(s): {', '.join(sorted(unknown))}")
            base = [(s, i) for s, i in base if s.repo in rset]
        return base


def resolve_workspace(
    verification_dir: Path,
    repo_names: dict[str, str],
    overrides: dict[str, str] | None = None,
) -> WorkspacePaths:
    """Resolve sibling repository paths, supporting worktrees and --override repo=path."""
    vdir = verification_dir.resolve()
    # If running inside .worktrees/<name>/impulse-verification, walk up to workspace root
    root = vdir.parent
    if root.parent.name == ".worktrees":
        root = root.parent.parent

    repos: dict[str, Path] = {}
    for key, dirname in repo_names.items():
        if key == "verification":
            repos[key] = vdir
        else:
            env_key = f"IMPULSE_{key.upper()}_DIR"
            if env_key in os.environ and os.environ[env_key]:
                repos[key] = Path(os.environ[env_key]).expanduser().resolve()
            else:
                repos[key] = (root / dirname).resolve()

    if overrides:
        for key, raw_path in overrides.items():
            if key not in repo_names:
                raise KeyError(f"Unknown repo override '{key}'. Known: {', '.join(sorted(repo_names))}")
            repos[key] = Path(raw_path).expanduser().resolve()

    return WorkspacePaths(root=root, repos=repos)


def load_catalog(catalog_path: Path | None = None) -> Catalog:
    if catalog_path is None:
        catalog_path = Path(__file__).resolve().parent.parent / "catalog.toml"
    with catalog_path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    meta = raw.get("meta", {})
    repo_names = dict(raw.get("repos", {}))
    pillars = {
        name: PillarSpec(name=name, description=str(cfg.get("description", "")))
        for name, cfg in raw.get("pillars", {}).items()
    }
    prereqs = {
        name: PrereqSpec(
            name=name,
            kind=str(cfg["kind"]),
            description=str(cfg.get("description", "")),
            hint=str(cfg.get("hint", "")),
            command=cfg.get("command"),
            module=cfg.get("module"),
            path=cfg.get("path"),
        )
        for name, cfg in raw.get("prereqs", {}).items()
    }

    suites: dict[str, SuiteSpec] = {}
    for sid, cfg in raw.get("suites", {}).items():
        pillar = str(cfg["pillar"])
        repo = str(cfg["repo"])
        det = str(cfg["determinism"])
        cost = str(cfg["cost"])
        if pillar not in pillars:
            raise ValueError(f"Suite '{sid}' references unknown pillar '{pillar}'")
        if repo not in repo_names:
            raise ValueError(f"Suite '{sid}' references unknown repo '{repo}'")
        if det not in VALID_DETERMINISM:
            raise ValueError(f"Suite '{sid}' has invalid determinism '{det}'")
        if cost not in VALID_COSTS:
            raise ValueError(f"Suite '{sid}' has invalid cost '{cost}'")
        reqs = tuple(str(r) for r in cfg.get("requires", []))
        for req in reqs:
            if req not in prereqs:
                raise ValueError(f"Suite '{sid}' references unknown prereq '{req}'")
        profiles = {
            pname: tuple(str(x) for x in args)
            for pname, args in cfg.get("profiles", {}).items()
        }
        for pname in profiles:
            if pname not in VALID_INTENSITIES:
                raise ValueError(f"Suite '{sid}' has invalid profile '{pname}'")
        suites[sid] = SuiteSpec(
            id=sid,
            title=str(cfg["title"]),
            pillar=pillar,
            repo=repo,
            determinism=det,
            cost=cost,
            requires=reqs,
            cwd=str(cfg["cwd"]),
            cmd=tuple(str(x) for x in cfg["cmd"]),
            paths=tuple(str(p) for p in cfg.get("paths", [])),
            profiles=profiles,
            coverage_parser=cfg.get("coverage_parser"),
        )

    gamuts: dict[str, GamutSpec] = {}
    for gname, cfg in raw.get("gamuts", {}).items():
        intensity = str(cfg.get("intensity", "standard"))
        if intensity not in VALID_INTENSITIES:
            raise ValueError(f"Gamut '{gname}' has invalid intensity '{intensity}'")
        g_suites = tuple(str(s) for s in cfg.get("suites", []))
        for sid in g_suites:
            if sid not in suites:
                raise ValueError(f"Gamut '{gname}' references unknown suite '{sid}'")
        gamuts[gname] = GamutSpec(
            name=gname,
            description=str(cfg.get("description", "")),
            target_time_sec=int(cfg.get("target_time_sec", 60)),
            intensity=intensity,
            extends=tuple(str(p) for p in cfg.get("extends", [])),
            suites=g_suites,
        )

    for gname, g in gamuts.items():
        for parent in g.extends:
            if parent not in gamuts:
                raise ValueError(f"Gamut '{gname}' extends unknown gamut '{parent}'")

    cat = Catalog(
        schema_version=int(meta.get("schema_version", 1)),
        default_gamut=str(meta.get("default_gamut", "standard")),
        default_seed=int(meta.get("default_seed", 20261122)),
        repo_names=repo_names,
        pillars=pillars,
        gamuts=gamuts,
        prereqs=prereqs,
        suites=suites,
        catalog_path=catalog_path.resolve(),
    )
    # Validate all gamuts expand without cycles
    for gname in gamuts:
        cat.resolve_gamut_suites(gname)
    return cat


def build_suite_command(
    suite: SuiteSpec,
    intensity: str,
    ws: WorkspacePaths,
    *,
    seed: int,
    workers: int,
) -> tuple[list[str], Path]:
    cwd = Path(ws.resolve_template(suite.cwd, seed=seed, workers=workers))
    cmd = [ws.resolve_template(arg, seed=seed, workers=workers) for arg in suite.cmd]
    extra = suite.profiles.get(intensity, ())
    cmd.extend(ws.resolve_template(arg, seed=seed, workers=workers) for arg in extra)
    return cmd, cwd
