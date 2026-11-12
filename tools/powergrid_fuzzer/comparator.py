"""
Multi-Engine Differential Result Comparator & Reproducer Persistence.

Enforces:
  - 100% bitwise/structural identity on `nodes`, `keys`, `count`, and `lane_masks`.
  - `1e-4` relative / `1e-5` absolute tolerance on floating-point `scalar` and `top_k` `scores`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def floats_close(a: float, b: float, rel_tol: float = 1e-4, abs_tol: float = 1e-5) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _normalize_lanes(raw_lanes: Any) -> List[List[int]]:
    out: List[List[int]] = []
    for pair in raw_lanes or []:
        out.append([int(pair[0]), int(pair[1])])
    return out


def compare_two_results(
    ref_name: str,
    ref: Dict[str, Any],
    cand_name: str,
    cand: Dict[str, Any],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-5,
) -> List[str]:
    diffs: List[str] = []

    ref_status = ref.get("status")
    cand_status = cand.get("status")
    if ref_status != cand_status:
        diffs.append(
            f"status mismatch: {ref_name}={ref_status!r} vs {cand_name}={cand_status!r} "
            f"(detail={cand.get('error') or cand.get('reason')})"
        )
        return diffs

    if cand_status != "ok":
        return diffs

    rtype = ref.get("result_type")
    ctype = cand.get("result_type")
    if rtype != ctype:
        diffs.append(f"result_type mismatch: {ref_name}={rtype!r} vs {cand_name}={ctype!r}")
        return diffs

    if rtype == "nodes":
        rn = [int(x) for x in ref.get("nodes", [])]
        cn = [int(x) for x in cand.get("nodes", [])]
        if rn != cn:
            diffs.append(
                f"nodes mismatch ({ref_name} len={len(rn)} vs {cand_name} len={len(cn)}): "
                f"{ref_name[:12]}={rn[:10]} vs {cand_name[:12]}={cn[:10]}"
            )

    elif rtype == "keys":
        rk = [str(x) for x in ref.get("keys", [])]
        ck = [str(x) for x in cand.get("keys", [])]
        if rk != ck:
            diffs.append(
                f"keys mismatch ({ref_name} len={len(rk)} vs {cand_name} len={len(ck)}): "
                f"{ref_name}={rk[:8]} vs {cand_name}={ck[:8]}"
            )

    elif rtype == "count":
        rc = int(ref.get("count", -1))
        cc = int(cand.get("count", -2))
        if rc != cc:
            diffs.append(f"count mismatch: {ref_name}={rc} vs {cand_name}={cc}")

    elif rtype == "scalar":
        rs = ref.get("scalar")
        cs = cand.get("scalar")
        if (rs is None) != (cs is None):
            diffs.append(f"scalar nullability mismatch: {ref_name}={rs!r} vs {cand_name}={cs!r}")
        elif rs is not None and cs is not None:
            if not floats_close(float(rs), float(cs), rel_tol=rel_tol, abs_tol=abs_tol):
                diffs.append(
                    f"scalar mismatch: {ref_name}={float(rs):.8g} vs {cand_name}={float(cs):.8g} "
                    f"(abs_diff={abs(float(rs) - float(cs)):.3e})"
                )

    elif rtype == "lane_masks":
        rl = _normalize_lanes(ref.get("lanes"))
        cl = _normalize_lanes(cand.get("lanes"))
        if rl != cl:
            diffs.append(
                f"lane_masks mismatch ({ref_name} len={len(rl)} vs {cand_name} len={len(cl)}): "
                f"{ref_name}={rl[:8]} vs {cand_name}={cl[:8]}"
            )

    elif rtype == "top_k":
        rn = [int(x) for x in ref.get("nodes", [])]
        cn = [int(x) for x in cand.get("nodes", [])]
        if rn != cn:
            diffs.append(f"top_k nodes mismatch: {ref_name}={rn} vs {cand_name}={cn}")
        rscores = [float(x) for x in ref.get("scores", [])]
        cscores = [float(x) for x in cand.get("scores", [])]
        if len(rscores) != len(cscores):
            diffs.append(f"top_k scores length mismatch: {ref_name}={len(rscores)} vs {cand_name}={len(cscores)}")
        else:
            for idx, (sa, sb) in enumerate(zip(rscores, cscores)):
                if not floats_close(sa, sb, rel_tol=rel_tol, abs_tol=abs_tol):
                    diffs.append(
                        f"top_k score[{idx}] mismatch: {ref_name}={sa:.6f} vs {cand_name}={sb:.6f}"
                    )
                    break

    else:
        diffs.append(f"Unknown result_type '{rtype}'")

    return diffs


def compare_engine_outputs(
    per_engine: Dict[str, Dict[str, Any]],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-5,
) -> Tuple[List[str], List[str]]:
    """
    Compares all engine outputs against the primary reference engine (`sqlite` if present, else first).
    Returns (hard_mismatches, unsupported_notices).
    """
    if not per_engine:
        return ["No engine outputs provided"], []

    engine_names = list(per_engine.keys())
    ref_name = "sqlite" if "sqlite" in per_engine else engine_names[0]
    ref_out = per_engine[ref_name]

    mismatches: List[str] = []
    unsupported: List[str] = []

    if ref_out.get("status") != "ok":
        mismatches.append(f"Reference engine '{ref_name}' failed: {ref_out}")
        return mismatches, unsupported

    for eng in engine_names:
        if eng == ref_name:
            continue
        cand_out = per_engine[eng]
        if cand_out.get("status") == "unsupported":
            unsupported.append(f"{eng}: {cand_out.get('reason', 'unsupported')}")
            continue
        mismatches.extend(
            compare_two_results(ref_name, ref_out, eng, cand_out, rel_tol=rel_tol, abs_tol=abs_tol)
        )

    return mismatches, unsupported


def save_reproducer(
    fuzz_runs_dir: Path,
    case_name: str,
    query: Dict[str, Any],
    sql_query: str,
    per_engine_outputs: Dict[str, Dict[str, Any]],
    mismatches: List[str],
) -> Path:
    qid = str(query["id"])
    repro_dir = fuzz_runs_dir / f"repro_{case_name}_{qid}"
    repro_dir.mkdir(parents=True, exist_ok=True)

    (repro_dir / "query.json").write_text(json.dumps(query, indent=2) + "\n", encoding="utf-8")
    (repro_dir / "query.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
    (repro_dir / "oracle.sql").write_text(sql_query + "\n", encoding="utf-8")
    (repro_dir / "outputs.json").write_text(
        json.dumps(per_engine_outputs, indent=2) + "\n", encoding="utf-8"
    )
    diff_lines = [
        f"Case   : {case_name}",
        f"Query  : {qid} ({query.get('family')})",
        f"ImpScm : {query.get('impscm')}",
        "",
        "Mismatches:",
        *[f"  - {m}" for m in mismatches],
    ]
    (repro_dir / "diff.txt").write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
    return repro_dir
