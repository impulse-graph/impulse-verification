"""Prerequisite probing, binary discovery, and sibling repo git provenance."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .catalog import Catalog, PrereqSpec, WorkspacePaths


@dataclass(frozen=True)
class PrereqResult:
    name: str
    ok: bool
    optional: bool
    detail: str
    hint: str


@dataclass(frozen=True)
class RepoGitState:
    name: str
    path: str
    exists: bool
    head_sha: str
    short_sha: str
    branch: str
    dirty: bool


def _lib_ext() -> str:
    return "dylib" if sys.platform == "darwin" else "so"


def find_cpp_lib(ws: WorkspacePaths, variant: str) -> Path | None:
    env_key = f"IMPULSE_LIB_PATH_{variant.upper()}"
    if env_key in os.environ and Path(os.environ[env_key]).is_file():
        return Path(os.environ[env_key]).resolve()
    ext = _lib_ext()
    cand = ws.repos["core"] / "impulse-cpp" / f"build_{variant}" / f"libimpulse_graph.{ext}"
    if cand.is_file():
        return cand.resolve()
    return None


def find_impulse_cli(ws: WorkspacePaths) -> Path | None:
    env_cli = os.environ.get("IMPULSE_CLI")
    if env_cli and Path(env_cli).is_file() and os.access(env_cli, os.X_OK):
        return Path(env_cli).resolve()
    for profile in ("release", "debug"):
        cand = ws.repos["core"] / "impulse-cli" / "target" / profile / "impulse-graph"
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    found = shutil.which("impulse-graph")
    return Path(found).resolve() if found else None


def find_impulse_python_path(ws: WorkspacePaths) -> Path | None:
    py_dir = ws.repos["core"] / "impulse-python"
    pkg_dir = py_dir / "impulse_graph"
    if not pkg_dir.is_dir():
        return None
    if any(pkg_dir.glob("_impulse_native*.so")) or any(pkg_dir.glob("_impulse_native*.pyd")):
        return py_dir.resolve()
    return None


def probe_prereq(spec: PrereqSpec, ws: WorkspacePaths) -> PrereqResult:
    optional = spec.kind == "optional_command"
    if spec.kind in ("command", "optional_command"):
        assert spec.command is not None
        found = shutil.which(spec.command)
        return PrereqResult(
            name=spec.name,
            ok=found is not None,
            optional=optional,
            detail=found or "not found on PATH",
            hint=spec.hint,
        )
    if spec.kind == "library":
        variant = "on" if spec.name == "build_on" else "off"
        found_lib = find_cpp_lib(ws, variant)
        return PrereqResult(
            name=spec.name,
            ok=found_lib is not None,
            optional=False,
            detail=str(found_lib) if found_lib else f"missing build_{variant}/libimpulse_graph.{_lib_ext()}",
            hint=spec.hint,
        )
    if spec.kind == "binary" and spec.name == "impulse_cli":
        cli = find_impulse_cli(ws)
        return PrereqResult(
            name=spec.name,
            ok=cli is not None,
            optional=False,
            detail=str(cli) if cli else "impulse-graph binary not found",
            hint=spec.hint,
        )
    if spec.kind == "python_binding" and spec.name == "impulse_python":
        py_dir = find_impulse_python_path(ws)
        return PrereqResult(
            name=spec.name,
            ok=py_dir is not None,
            optional=False,
            detail=str(py_dir) if py_dir else "native _impulse_native extension not built",
            hint=spec.hint,
        )
    if spec.kind == "python_module":
        assert spec.module is not None
        found_mod = importlib.util.find_spec(spec.module) is not None
        return PrereqResult(
            name=spec.name,
            ok=found_mod,
            optional=False,
            detail=f"python module '{spec.module}' available" if found_mod else f"missing python module '{spec.module}'",
            hint=spec.hint,
        )
    if spec.kind == "path":
        assert spec.path is not None
        resolved = Path(ws.resolve_template(spec.path, seed=0, workers=1))
        ok = resolved.exists()
        return PrereqResult(
            name=spec.name,
            ok=ok,
            optional=False,
            detail=str(resolved) if ok else f"missing path {resolved}",
            hint=spec.hint,
        )
    return PrereqResult(
        name=spec.name,
        ok=False,
        optional=optional,
        detail=f"unsupported prereq kind '{spec.kind}'",
        hint=spec.hint,
    )


def probe_all_prereqs(cat: Catalog, ws: WorkspacePaths) -> dict[str, PrereqResult]:
    return {name: probe_prereq(spec, ws) for name, spec in cat.prereqs.items()}


def _git_output(repo_dir: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def collect_repo_git_states(ws: WorkspacePaths) -> dict[str, RepoGitState]:
    states: dict[str, RepoGitState] = {}
    for name, path in ws.repos.items():
        if not path.is_dir():
            states[name] = RepoGitState(
                name=name,
                path=str(path),
                exists=False,
                head_sha="missing",
                short_sha="missing",
                branch="missing",
                dirty=False,
            )
            continue
        sha = _git_output(path, "rev-parse", "HEAD") or "untracked"
        branch = _git_output(path, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
        porcelain = _git_output(path, "status", "--porcelain")
        dirty = bool(porcelain) if porcelain is not None else False
        states[name] = RepoGitState(
            name=name,
            path=str(path),
            exists=True,
            head_sha=sha,
            short_sha=sha[:10] if len(sha) >= 10 else sha,
            branch=branch,
            dirty=dirty,
        )
    return states


def build_execution_env(ws: WorkspacePaths) -> dict[str, str]:
    env = os.environ.copy()
    lib_on = find_cpp_lib(ws, "on")
    lib_off = find_cpp_lib(ws, "off")
    if lib_on:
        env.setdefault("IMPULSE_LIB_PATH_ON", str(lib_on))
    if lib_off:
        env.setdefault("IMPULSE_LIB_PATH_OFF", str(lib_off))
    cli = find_impulse_cli(ws)
    if cli:
        env.setdefault("IMPULSE_CLI", str(cli))
    env.setdefault("IMPULSE_SPEC_DIR", str(ws.repos["spec"]))
    env.setdefault("IMPULSE_CORE_DIR", str(ws.repos["core"]))
    env.setdefault("IMPULSE_JAVA_DIR", str(ws.repos["java"]))
    py_dir = find_impulse_python_path(ws)
    if py_dir:
        existing = env.get("PYTHONPATH", "")
        parts = [str(py_dir)] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env
