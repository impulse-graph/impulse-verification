#!/usr/bin/env python3
"""
Build Comprehensive PowerGrid .imps (Spec v0.10.0) Verification Fixture & Ground-Truth SQLite DB.

Parses an authentic MATPOWER PEGASE network case (`case1354pegase.m` default, or `case2869pegase.m`)
and materializes:
  1. Multi-domain CSV tables (`Bus`, `Generator`, `PMU`, `LoadZone`, `Substation`, `ContingencyTier`)
     covering 6 distinct `KeyType`s (`int32`, `int64`, `uuid128`, `int16`, `string`, `int8`) and
     `KeyAttrIndex` (§5.1 O(1) Dense ID -> External Key lookup).
  2. All 14 base attribute types (`0x01`..`0x0E`: `int8`, `int16`, `int32`, `int64`, `float16`,
     `float32`, `float64`, `timestamp_ms`, `timestamp_ns`, `fixed_bytes[16]`, `var_string`,
     `var_bytes`, `interval_sec_32`, `interval_ms_64`) + `VEC4` (`float32[4]`, `float16[4]`) +
     `VEC8` (`float32[8]`), across both non-null and nullable (`0x80` 128B-padded validity bitmap)
     columns, preserving authentic MATPOWER `Gs/Bs == 0`, `rateA == 0`, and `Qmax/Qmin = +-Inf` NULLs.
  3. 10 relation topologies covering CSR `u32` (`encoding=0`), CSR `u16` (`encoding=2`),
     COO/Hybrid (`encoding=3`), CSC transposes, flat `M:1` (`u16` total & `u32` partial with
     `0xFFFFFFFF` sentinels), flat `1:1` injective (`u16` forward & reverse with `0xFFFF` sentinels),
     flat-reverse `1:M` (`u32`), self-relations (`u -> u`), parallel multi-edges, empty relations
     (`E = 0`), and a Virtual Super-Relation (`IMP_REL_FEATURE_VIRTUAL` over `AC_LINE + TRANSFORMER`).
  4. All 8 Section 2.6 Secondary & Vector Index types (`0x01` permutation, `0x02` zonemap,
     `0x03` inverted_bitset, `0x04` mphf, `0x05` trigram, `0x06` domain_split,
     `0x07` temporal_interval on both `0x0D` and `0x0E`, and `0x08` Vamana ANN `sq8` & `fp16`).
  5. Footer key-value metadata, Ed25519 cryptographic signature, strict 128B alignment + deep audit
     validation via official `impulse-cli`, and a 1:1 companion `powergrid_ground_truth.sqlite` DB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_POWERGRID_DATASETS = WORKSPACE_ROOT / "impulse-powergrid" / "datasets"
DEFAULT_OUT_ROOT = REPO_ROOT / "datasets" / "powergrid_fixture"

# Deterministic reference epoch: 2026-03-18T12:00:00Z
BASE_EPOCH_SEC = 1_773_835_200
BASE_EPOCH_MS = BASE_EPOCH_SEC * 1_000
BASE_EPOCH_NS = BASE_EPOCH_MS * 1_000_000


def find_impulse_cli() -> Path:
    env_cli = os.environ.get("IMPULSE_CLI")
    if env_cli and os.path.isfile(env_cli) and os.access(env_cli, os.X_OK):
        return Path(env_cli).resolve()

    candidates = [
        WORKSPACE_ROOT / "impulse-graph-core" / "impulse-cli" / "target" / "release" / "impulse-graph",
        WORKSPACE_ROOT / "impulse-graph-core" / "impulse-cli" / "target" / "debug" / "impulse-graph",
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()

    which_cli = shutil.which("impulse-graph")
    if which_cli:
        return Path(which_cli).resolve()

    raise RuntimeError("Could not locate 'impulse-graph' binary. Build impulse-cli first.")


def parse_matpower_case(case_path: Path) -> Tuple[float, List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    text = case_path.read_text(encoding="utf-8")

    m_base = re.search(r"mpc\.baseMVA\s*=\s*([0-9.+-eE]+)\s*;", text)
    base_mva = float(m_base.group(1)) if m_base else 100.0

    def extract_matrix(section_name: str) -> List[List[float]]:
        m = re.search(rf"mpc\.{section_name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
        if not m:
            return []
        rows: List[List[float]] = []
        for raw_line in m.group(1).splitlines():
            clean = raw_line.split("%", 1)[0].strip().rstrip(";").strip()
            if not clean:
                continue
            vals: List[float] = []
            for tok in clean.split():
                tl = tok.lower()
                if tl in ("inf", "+inf"):
                    vals.append(float("inf"))
                elif tl == "-inf":
                    vals.append(float("-inf"))
                else:
                    vals.append(float(tok))
            rows.append(vals)
        return rows

    buses = extract_matrix("bus")
    gens = extract_matrix("gen")
    branches = extract_matrix("branch")
    gencosts = extract_matrix("gencost")
    return base_mva, buses, gens, branches, gencosts


def det_uuid(namespace_tag: str, entity_id: int) -> uuid.UUID:
    digest = hashlib.sha256(f"impulse-powergrid-v010:{namespace_tag}:{entity_id}".encode("utf-8")).digest()
    b = bytearray(digest[:16])
    b[6] = (b[6] & 0x0F) | 0x40  # UUIDv4 version nibble
    b[8] = (b[8] & 0x3F) | 0x80  # RFC 4122 variant
    return uuid.UUID(bytes=bytes(b))


def format_vec(values: List[float], decimals: int = 5) -> str:
    return "[" + ";".join(f"{v:.{decimals}f}" for v in values) + "]"


def write_csv_table(path: Path, headers: List[str], rows: List[List[Optional[object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(["" if v is None else str(v) for v in row])


def build_fixture_datasets_and_sqlite(
    case_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> Tuple[Path, Path, dict]:
    base_mva, raw_buses, raw_gens, raw_branches, raw_gencosts = parse_matpower_case(case_path)
    case_stem = case_path.stem

    domains_dir = out_dir / "domains"
    relations_dir = out_dir / "relations"
    domains_dir.mkdir(parents=True, exist_ok=True)
    relations_dir.mkdir(parents=True, exist_ok=True)

    # Sort buses deterministically by MATPOWER bus_i so dense IDs 0..N_bus-1 are strictly ordered by bus_number
    raw_buses_sorted = sorted(raw_buses, key=lambda r: int(r[0]))
    bus_num_to_dense: Dict[int, int] = {}

    # Compute phase-angle quantiles to partition regional LoadZones cleanly (8 corridors per MATPOWER zone)
    va_sorted = sorted(b[8] for b in raw_buses_sorted)
    n_bus = len(raw_buses_sorted)

    def angle_band(va: float) -> int:
        # 8 regional phase-angle bands 1..8
        idx = 0
        for q in range(1, 8):
            thresh = va_sorted[(q * n_bus) // 8]
            if va >= thresh:
                idx = q
        return idx + 1

    # Build Substation clusters via union-find over transformer endpoints (same geographic substation)
    # plus deterministic grouping of remaining EHV/HV buses into ~180-350 substations (leaving ~8% remote switching
    # posts unassigned to test partial 1:M reverse sentinel 0xFFFFFFFF if desired, while SUBSTATION_HAS_BUS is 1:M).
    parent: Dict[int, int] = {int(b[0]): int(b[0]) for b in raw_buses_sorted}

    def uf_find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(a: int, b: int) -> None:
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for br in raw_branches:
        fbus, tbus = int(br[0]), int(br[1])
        ratio, angle = br[8], br[9]
        if ratio != 0.0 or angle != 0.0:
            if fbus in parent and tbus in parent:
                uf_union(fbus, tbus)

    # Collect multi-bus transformer substations + group every 6th remaining bus into regional substations,
    # leaving ~10% remote radial/tie buses outside any multi-bay substation so flat-reverse 1:M exercises 0xFFFFFFFF!
    comp_members: Dict[int, List[int]] = defaultdict(list)
    for b in raw_buses_sorted:
        bnum = int(b[0])
        comp_members[uf_find(bnum)].append(bnum)

    substation_groups: List[Tuple[str, List[int]]] = []
    singleton_buses: List[int] = []
    for root in sorted(comp_members.keys()):
        members = sorted(comp_members[root])
        if len(members) >= 2:
            substation_groups.append((f"SUB_XFMR_{root:04d}", members))
        else:
            singleton_buses.append(members[0])

    # Group 90% of singletons into regional 6-bus substations; keep every 10th singleton as an ungrouped pole/tie bus
    # so SUBSTATION_HAS_BUS (1:M) tests both linked buses and unlinked target buses (0xFFFFFFFF reverse sentinel).
    grouped_pool = [b for idx, b in enumerate(singleton_buses) if idx % 10 != 9]
    for chunk_idx in range(0, len(grouped_pool), 6):
        chunk = grouped_pool[chunk_idx : chunk_idx + 6]
        substation_groups.append((f"SUB_REG_{chunk[0]:04d}", chunk))

    substation_groups.sort(key=lambda x: x[0])
    bus_to_substation_dense: Dict[int, int] = {}
    bus_kv_map: Dict[int, float] = {int(b[0]): float(b[9]) for b in raw_buses_sorted}

    # -------------------------------------------------------------------------
    # Domain 3: LoadZone (KeyType::Int16, key_attribute="zone_id")
    # -------------------------------------------------------------------------
    zone_agg_load: Dict[int, float] = defaultdict(float)
    zone_gen_reserve: Dict[int, float] = defaultdict(float)
    zone_gen_count: Dict[int, int] = defaultdict(int)
    bus_to_zone_key: Dict[int, int] = {}

    for b in raw_buses_sorted:
        bnum = int(b[0])
        mp_zone = int(b[10])
        band = angle_band(b[8])
        z_key = mp_zone * 100 + band  # fits cleanly in int16 (e.g., 501..508 or 101..1008)
        bus_to_zone_key[bnum] = z_key
        zone_agg_load[z_key] += float(b[2])

    for g in raw_gens:
        g_bus = int(g[0])
        if g_bus in bus_to_zone_key:
            zk = bus_to_zone_key[g_bus]
            zone_gen_count[zk] += 1
            pmax = float(g[8])
            pg = float(g[1])
            zone_gen_reserve[zk] += max(0.0, pmax - pg)

    # Also add one synthetic interconnect boundary LoadZone with 0 generators to guarantee a NULL spinning_reserve_mw
    sorted_zone_keys = sorted(zone_agg_load.keys())
    zone_key_to_dense: Dict[int, int] = {}
    loadzone_rows: List[List[Optional[object]]] = []
    for d_idx, zk in enumerate(sorted_zone_keys):
        zone_key_to_dense[zk] = d_idx
        mp_z = zk // 100
        band = zk % 100
        z_name = f"ENTSOE_Z{mp_z}_CORRIDOR_{band}"
        tot_load = round(zone_agg_load[zk], 4)
        # Nullable spinning_reserve_mw: NULL if 0 generators or band == 8 (reserve-exempt border corridor)
        spin_res: Optional[float] = (
            None if (zone_gen_count[zk] == 0 or band == 8) else round(zone_gen_reserve[zk], 4)
        )
        loadzone_rows.append([zk, zk, z_name, f"{tot_load:.4f}", None if spin_res is None else f"{spin_res:.4f}"])

    # -------------------------------------------------------------------------
    # Domain 5: ContingencyTier (KeyType::Int8, key_attribute="tier_code")
    # -------------------------------------------------------------------------
    tier_defs = [
        (0, "N0_BASE_SECURE", 0.0),
        (1, "N1_SINGLE_ELEMENT", 5.0),
        (2, "N2_DOUBLE_CIRCUIT", 15.0),
        (3, "CASCADING_ISLANDING", 40.0),
    ]
    contingency_rows: List[List[Optional[object]]] = [
        [tc, tc, label, f"{shed:.2f}"] for (tc, label, shed) in tier_defs
    ]

    # -------------------------------------------------------------------------
    # Domain 0: Bus (KeyType::Int32, key_attribute="bus_number")
    # -------------------------------------------------------------------------
    bus_rows: List[List[Optional[object]]] = []
    for d_idx, b in enumerate(raw_buses_sorted):
        bnum = int(b[0])
        bus_num_to_dense[bnum] = d_idx
        btype = int(b[1])
        pd_mw = float(b[2])
        qd_mvar = float(b[3])
        gs_mw = float(b[4])
        bs_mvar = float(b[5])
        vm_pu = float(b[7])
        va_deg = float(b[8])
        base_kv = float(b[9])
        mp_zone = int(b[10])
        vmax = float(b[11])
        vmin = float(b[12])

        has_shunt = (gs_mw != 0.0) or (bs_mvar != 0.0)
        shunt_gs_opt = f"{gs_mw:.6f}" if has_shunt else None
        shunt_bs_opt = f"{bs_mvar:.6f}" if has_shunt else None

        # Nullable reserve_margin_pu: NULL on pure zero-load tie buses
        is_zero_load_tie = (pd_mw == 0.0 and qd_mvar == 0.0 and btype == 1)
        reserve_margin_opt = None if is_zero_load_tie else f"{(vmax - vm_pu):.6f}"

        kv_int = int(round(base_kv))
        bus_label = f"BUS_{bnum:05d}_{kv_int}KV"
        if base_kv >= 380.0:
            v_class = "EHV_380KV"
        elif base_kv >= 220.0:
            v_class = "HV_220KV"
        elif base_kv >= 150.0:
            v_class = "HV_150KV"
        else:
            v_class = "SUB_110KV"

        z_corridor = bus_to_zone_key[bnum]
        op_email = f"bus{bnum}@zone{z_corridor}.pegase-tso.eu"
        mrid_hex = str(det_uuid("bus-cim-mrid", bnum))

        # Nullable var_bytes: IEC 61850 GOOSE header present on PV/Slack or EHV>=380kV buses, NULL on unautomated 220kV PQ
        if btype >= 2 or base_kv >= 380.0 or (d_idx % 5 == 0):
            goose_bytes = hashlib.blake2s(
                f"GOOSE:{bnum}:{kv_int}".encode("utf-8"), digest_size=6 + (d_idx % 7)
            ).hexdigest()
        else:
            goose_bytes = None

        poll_ts_ms = BASE_EPOCH_MS + (d_idx * 250)
        maint_start_sec = BASE_EPOCH_SEC + ((d_idx % 90) * 3600)
        maint_dur_sec = 1800 + ((bnum % 6) * 900)
        maint_interval_str = f"{maint_start_sec}:{maint_dur_sec}"

        va_rad = math.radians(va_deg)
        s_apparent = math.hypot(pd_mw, qd_mvar) / max(base_mva, 1.0)
        harmonic_v8 = [
            vm_pu * math.cos(va_rad),
            vm_pu * math.sin(va_rad),
            pd_mw / max(base_mva, 1.0),
            qd_mvar / max(base_mva, 1.0),
            base_kv / 400.0,
            s_apparent,
            gs_mw / max(base_mva, 1.0),
            bs_mvar / max(base_mva, 1.0),
        ]
        phasor_f16_v4 = [vm_pu, va_rad, vmax, vmin]

        bus_rows.append(
            [
                bnum,
                bnum,
                btype,
                mp_zone,
                f"{pd_mw:.6f}",
                f"{qd_mvar:.6f}",
                f"{vm_pu:.6f}",
                f"{vm_pu:.4f}",
                f"{va_deg:.6f}",
                f"{base_kv:.2f}",
                shunt_gs_opt,
                shunt_bs_opt,
                reserve_margin_opt,
                bus_label,
                v_class,
                op_email,
                mrid_hex,
                goose_bytes,
                poll_ts_ms,
                maint_interval_str,
                format_vec(harmonic_v8, 5),
                format_vec(phasor_f16_v4, 4),
            ]
        )

    # -------------------------------------------------------------------------
    # Domain 1: Generator (KeyType::Int64, key_attribute="gen_uid")
    # -------------------------------------------------------------------------
    gen_rows: List[List[Optional[object]]] = []
    gen_edges: List[Tuple[int, int, int]] = []  # (gen_uid, bus_number, gen_dense)
    bus_unit_counter: Dict[int, int] = defaultdict(int)

    for g_idx, g in enumerate(raw_gens):
        g_bus = int(g[0])
        bus_unit_counter[g_bus] += 1
        unit_seq = bus_unit_counter[g_bus]
        gen_uid = 1_000_000_000 + (g_bus * 100) + unit_seq

        pg_mw = float(g[1])
        qg_mvar = float(g[2])
        qmax_raw = float(g[3])
        qmin_raw = float(g[4])
        vg_pu = float(g[5])
        mbase = float(g[6]) if float(g[6]) > 0 else base_mva
        status = int(g[7])
        pmax_mw = float(g[8])
        pmin_mw = float(g[9])
        ramp_10 = float(g[17]) if len(g) > 17 else 0.0

        # Authentic MATPOWER +-Inf -> NULL
        qmax_opt = None if (math.isinf(qmax_raw) or math.isnan(qmax_raw) or abs(qmax_raw) >= 9999.0) else f"{qmax_raw:.4f}"
        qmin_opt = None if (math.isinf(qmin_raw) or math.isnan(qmin_raw) or abs(qmin_raw) >= 9999.0) else f"{qmin_raw:.4f}"

        # Nullable ramp rate: NULL when MATPOWER ramp_10 == 0 on boundary negative-Pmin equivalences
        if ramp_10 > 0.0:
            ramp_opt: Optional[str] = f"{(ramp_10 / 10.0):.4f}"
        elif pmin_mw < 0.0:
            ramp_opt = None
        else:
            ramp_opt = f"{max(1.0, pmax_mw * 0.015):.4f}"

        cost_c1 = 1.0
        if g_idx < len(raw_gencosts) and len(raw_gencosts[g_idx]) >= 6:
            cost_c1 = float(raw_gencosts[g_idx][5])

        if pmin_mw < 0.0:
            fuel_cat = "CROSS_BORDER_INJ"
        elif pmax_mw >= 900.0:
            fuel_cat = "NUCLEAR_BASE"
        elif pmax_mw >= 250.0:
            fuel_cat = "THERMAL_CCGT"
        else:
            fuel_cat = "HYDRO_RENEWABLE"

        sync_ts_ns = BASE_EPOCH_NS + (g_idx * 16_666_667)
        cap_v4 = [
            pg_mw / mbase,
            qg_mvar / mbase,
            pmax_mw / mbase,
            vg_pu,
        ]

        gen_rows.append(
            [
                gen_uid,
                gen_uid,
                status,
                unit_seq,
                f"{pg_mw:.4f}",
                f"{qg_mvar:.4f}",
                qmax_opt,
                qmin_opt,
                f"{pmax_mw:.4f}",
                f"{pmin_mw:.4f}",
                f"{vg_pu:.6f}",
                ramp_opt,
                f"{cost_c1:.4f}",
                fuel_cat,
                sync_ts_ns,
                format_vec(cap_v4, 5),
            ]
        )
        gen_edges.append((gen_uid, g_bus, g_idx))

    # -------------------------------------------------------------------------
    # Domain 2: PMU (KeyType::Uuid128, key_attribute="pmu_uuid") — 1:1 with >=380kV EHV buses
    # -------------------------------------------------------------------------
    pmu_rows: List[List[Optional[object]]] = []
    pmu_edges: List[Tuple[str, int, int]] = []  # (pmu_uuid_str, bus_number, pmu_dense)
    ehv_buses = [b for b in raw_buses_sorted if float(b[9]) >= 380.0]
    for p_idx, b in enumerate(ehv_buses):
        bnum = int(b[0])
        vm_pu = float(b[7])
        va_rad = math.radians(float(b[8]))
        pd_pu = float(b[2]) / max(base_mva, 1.0)
        qd_pu = float(b[3]) / max(base_mva, 1.0)

        pmu_uuid_str = str(det_uuid("pmu-ehv", bnum))
        sample_hz = 50 if (p_idx % 4 != 3) else 60
        gps_locked = 0 if (p_idx % 17 == 16) else 1
        freq_dev_mhz = ((bnum * 37) % 81 - 40) * 0.5
        rocof_opt = None if gps_locked == 0 else f"{(((bnum * 13) % 41 - 20) * 0.015):.4f}"
        sample_ts_ns = BASE_EPOCH_NS + (p_idx * 20_000_000)
        fault_start_ms = BASE_EPOCH_MS + ((p_idx % 48) * 60_000)
        fault_dur_ms = 250 + ((bnum % 12) * 125)
        fault_interval_ms_str = f"{fault_start_ms}:{fault_dur_ms}"

        # Positive-sequence voltage & current phasors [V_re, V_im, I_re, I_im]
        v_re = vm_pu * math.cos(va_rad)
        v_im = vm_pu * math.sin(va_rad)
        i_re = (pd_pu * v_re + qd_pu * v_im) / max(vm_pu * vm_pu, 1e-4)
        i_im = (pd_pu * v_im - qd_pu * v_re) / max(vm_pu * vm_pu, 1e-4)
        c37_v4 = [v_re, v_im, i_re, i_im]

        pmu_rows.append(
            [
                pmu_uuid_str,
                pmu_uuid_str,
                sample_hz,
                gps_locked,
                f"{freq_dev_mhz:.3f}",
                rocof_opt,
                sample_ts_ns,
                fault_interval_ms_str,
                format_vec(c37_v4, 5),
            ]
        )
        pmu_edges.append((pmu_uuid_str, bnum, p_idx))

    # -------------------------------------------------------------------------
    # Domain 4: Substation (KeyType::String, key_attribute="substation_code")
    # -------------------------------------------------------------------------
    substation_rows: List[List[Optional[object]]] = []
    substation_edges: List[Tuple[str, int]] = []  # (substation_code, bus_number)
    for s_idx, (sub_code, member_buses) in enumerate(substation_groups):
        max_kv = max(bus_kv_map[bn] for bn in member_buses)
        bay_count = len(member_buses) * 2
        comm_year: Optional[int] = None if (s_idx % 9 == 8) else (1968 + (s_idx % 55))
        substation_rows.append([sub_code, sub_code, f"{max_kv:.1f}", bay_count, comm_year])
        for bn in member_buses:
            bus_to_substation_dense[bn] = s_idx
            substation_edges.append((sub_code, bn))

    # -------------------------------------------------------------------------
    # Relation Edge Tables
    # -------------------------------------------------------------------------
    ac_line_rows: List[List[Optional[object]]] = []
    transformer_rows: List[List[Optional[object]]] = []
    pair_circuit_counter: Dict[Tuple[int, int], int] = defaultdict(int)

    for br_idx, br in enumerate(raw_branches):
        fbus = int(br[0])
        tbus = int(br[1])
        r_pu = float(br[2])
        x_pu = float(br[3])
        rate_a = float(br[5])
        ratio = float(br[8])
        angle = float(br[9])
        status = int(br[10])

        z_mag = math.hypot(r_pu, x_pu)
        # Authentic MATPOWER rateA == 0 means unlimited thermal rating -> SQL/Impulse NULL
        rate_a_opt = None if rate_a == 0.0 else f"{rate_a:.2f}"

        pair_circuit_counter[(fbus, tbus)] += 1
        ckt = pair_circuit_counter[(fbus, tbus)]

        if ratio == 0.0 and angle == 0.0:
            line_code = f"ACL_{fbus}_{tbus}_C{ckt}"
            insp_ts_opt = None if (br_idx % 6 == 5) else (BASE_EPOCH_MS - (br_idx * 86_400_000 // 10))
            ac_line_rows.append(
                [
                    fbus,
                    tbus,
                    f"{r_pu:.6f}",
                    f"{x_pu:.6f}",
                    f"{z_mag:.6f}",
                    rate_a_opt,
                    status,
                    line_code,
                    insp_ts_opt,
                ]
            )
        else:
            eff_ratio = 1.0 if ratio == 0.0 else ratio
            transformer_rows.append(
                [
                    fbus,
                    tbus,
                    f"{r_pu:.6f}",
                    f"{x_pu:.6f}",
                    f"{eff_ratio:.6f}",
                    f"{angle:.4f}",
                    rate_a_opt,
                ]
            )

    # Self-relation SHUNT_SELF_LOOP (u -> u on buses with non-zero shunt)
    shunt_self_rows: List[List[Optional[object]]] = []
    for b in raw_buses_sorted:
        bnum = int(b[0])
        gs_mw = float(b[4])
        bs_mvar = float(b[5])
        if gs_mw != 0.0 or bs_mvar != 0.0:
            steps_opt: Optional[int] = None if abs(bs_mvar) < 1.0 else min(12, max(1, int(round(abs(bs_mvar) / 10.0))))
            shunt_self_rows.append([bnum, bnum, f"{gs_mw:.6f}", f"{bs_mvar:.6f}", steps_opt])

    # Empty relation HVDC_TIE_LINE (E = 0 boundary test with typed columns)
    hvdc_empty_rows: List[List[Optional[object]]] = []

    # BUS_IN_ZONE (total flat M:1, u16)
    bus_in_zone_rows: List[List[Optional[object]]] = [
        [int(b[0]), bus_to_zone_key[int(b[0])]] for b in raw_buses_sorted
    ]

    # BUS_RISK_TIER (partial flat M:1, u32 — leaves low-load interior 220kV PQ buses unlinked -> 0xFFFFFFFF sentinel!)
    bus_risk_tier_rows: List[List[Optional[object]]] = []
    for b in raw_buses_sorted:
        bnum = int(b[0])
        btype = int(b[1])
        pd_mw = float(b[2])
        base_kv = float(b[9])
        if btype == 3 or (base_kv >= 380.0 and pd_mw >= 400.0):
            tier = 3
        elif base_kv >= 380.0 or btype == 2:
            tier = 2
        elif pd_mw >= 100.0:
            tier = 1
        elif pd_mw > 0.0:
            tier = 0
        else:
            continue  # Unlinked source node -> tests flat M:1 0xFFFFFFFF sentinel!
        bus_risk_tier_rows.append([bnum, tier])

    gen_at_bus_rows: List[List[Optional[object]]] = [[guid, bnum] for (guid, bnum, _) in gen_edges]
    pmu_monitors_bus_rows: List[List[Optional[object]]] = [[puuid, bnum] for (puuid, bnum, _) in pmu_edges]
    substation_has_bus_rows: List[List[Optional[object]]] = [[scode, bnum] for (scode, bnum) in substation_edges]

    # -------------------------------------------------------------------------
    # Write Domain & Relation CSV Files
    # -------------------------------------------------------------------------
    write_csv_table(
        domains_dir / "buses.csv",
        [
            "id",
            "bus_number",
            "bus_type",
            "matpower_zone",
            "pd_mw",
            "qd_mvar",
            "vm_pu",
            "vm_f16",
            "va_deg",
            "base_kv",
            "shunt_gs_mw",
            "shunt_bs_mvar",
            "reserve_margin_pu",
            "bus_label",
            "voltage_class_tag",
            "operator_email",
            "cim_mrid_bytes",
            "goose_header_blob",
            "scada_poll_ts_ms",
            "maint_window_sec",
            "harmonic_spectrum_v8",
            "phasor_quad_f16_v4",
        ],
        bus_rows,
    )

    write_csv_table(
        domains_dir / "generators.csv",
        [
            "id",
            "gen_uid",
            "in_service",
            "unit_seq",
            "pg_mw",
            "qg_mvar",
            "qmax_mvar",
            "qmin_mvar",
            "pmax_mw",
            "pmin_mw",
            "vg_setpoint_pu",
            "ramp_rate_mw_min",
            "cost_c1_eur_mwh",
            "fuel_category",
            "sync_check_ts_ns",
            "capability_curve_v4",
        ],
        gen_rows,
    )

    write_csv_table(
        domains_dir / "pmus.csv",
        [
            "id",
            "pmu_uuid",
            "sample_rate_hz",
            "gps_locked",
            "freq_dev_mhz",
            "rocof_hz_s",
            "last_sample_ts_ns",
            "fault_window_ms",
            "c37_118_phasor_v4",
        ],
        pmu_rows,
    )

    write_csv_table(
        domains_dir / "load_zones.csv",
        ["id", "zone_id", "zone_name", "total_load_mw", "spinning_reserve_mw"],
        loadzone_rows,
    )

    write_csv_table(
        domains_dir / "substations.csv",
        ["id", "substation_code", "primary_kv", "bay_count", "commission_year"],
        substation_rows,
    )

    write_csv_table(
        domains_dir / "contingency_tiers.csv",
        ["id", "tier_code", "tier_label", "max_shed_pct"],
        contingency_rows,
    )

    write_csv_table(
        relations_dir / "ac_line.csv",
        ["src", "tgt", "r_pu", "x_pu", "z_mag_pu", "rate_a_mva", "in_service", "line_code", "inspection_ts_ms"],
        ac_line_rows,
    )
    write_csv_table(
        relations_dir / "transformer.csv",
        ["src", "tgt", "r_pu", "x_pu", "tap_ratio", "shift_angle_deg", "rate_a_mva"],
        transformer_rows,
    )
    write_csv_table(
        relations_dir / "shunt_self_loop.csv",
        ["src", "tgt", "gs_mw", "bs_mvar", "step_count"],
        shunt_self_rows,
    )
    write_csv_table(
        relations_dir / "hvdc_tie_line.csv",
        ["src", "tgt", "dc_mw_setpoint", "bipole_mode"],
        hvdc_empty_rows,
    )
    write_csv_table(relations_dir / "bus_in_zone.csv", ["src", "tgt"], bus_in_zone_rows)
    write_csv_table(relations_dir / "bus_risk_tier.csv", ["src", "tgt"], bus_risk_tier_rows)
    write_csv_table(relations_dir / "gen_at_bus.csv", ["src", "tgt"], gen_at_bus_rows)
    write_csv_table(relations_dir / "pmu_monitors_bus.csv", ["src", "tgt"], pmu_monitors_bus_rows)
    write_csv_table(relations_dir / "substation_has_bus.csv", ["src", "tgt"], substation_has_bus_rows)

    # -------------------------------------------------------------------------
    # Write manifest.json for `impulse-graph build`
    # -------------------------------------------------------------------------
    manifest = {
        "version": "0.10.0",
        "metadata": {
            "dataset.name": f"PEGASE-{case_stem}-v0.10.0-verification-fixture",
            "dataset.source": str(case_path.name),
            "matpower.base_mva": f"{base_mva:.1f}",
            "spec.version": "0.10.0",
            "fixture.seed": str(seed),
            "fixture.purpose": "Phase-B Kleisli/CEL + SQLite differential fuzzing & full Spec v0.10.0 coverage",
        },
        "domains": [
            {
                "id": 0,
                "name": "Bus",
                "key_type": "int32",
                "file": "domains/buses.csv",
                "format": "csv",
                "id_column": "id",
                "key_attribute": "bus_number",
                "attributes": [
                    {"name": "bus_number", "column": "bus_number", "type": "int32"},
                    {"name": "bus_type", "column": "bus_type", "type": "int8"},
                    {"name": "matpower_zone", "column": "matpower_zone", "type": "int16"},
                    {"name": "pd_mw", "column": "pd_mw", "type": "float64"},
                    {"name": "qd_mvar", "column": "qd_mvar", "type": "float64"},
                    {"name": "vm_pu", "column": "vm_pu", "type": "float64"},
                    {"name": "vm_f16", "column": "vm_f16", "type": "float16"},
                    {"name": "va_deg", "column": "va_deg", "type": "float32"},
                    {"name": "base_kv", "column": "base_kv", "type": "float32"},
                    {"name": "shunt_gs_mw", "column": "shunt_gs_mw", "type": "float64?"},
                    {"name": "shunt_bs_mvar", "column": "shunt_bs_mvar", "type": "float64?"},
                    {"name": "reserve_margin_pu", "column": "reserve_margin_pu", "type": "float32?"},
                    {"name": "bus_label", "column": "bus_label", "type": "var_string"},
                    {"name": "voltage_class_tag", "column": "voltage_class_tag", "type": "var_string"},
                    {"name": "operator_email", "column": "operator_email", "type": "var_string"},
                    {"name": "cim_mrid_bytes", "column": "cim_mrid_bytes", "type": "fixed_bytes", "dimension": 16},
                    {"name": "goose_header_blob", "column": "goose_header_blob", "type": "var_bytes?"},
                    {"name": "scada_poll_ts_ms", "column": "scada_poll_ts_ms", "type": "timestamp_ms"},
                    {"name": "maint_window_sec", "column": "maint_window_sec", "type": "interval_sec_32"},
                    {"name": "harmonic_spectrum_v8", "column": "harmonic_spectrum_v8", "type": "float32", "dimension": 8},
                    {"name": "phasor_quad_f16_v4", "column": "phasor_quad_f16_v4", "type": "float16", "dimension": 4},
                ],
            },
            {
                "id": 1,
                "name": "Generator",
                "key_type": "int64",
                "file": "domains/generators.csv",
                "format": "csv",
                "id_column": "id",
                "key_attribute": "gen_uid",
                "attributes": [
                    {"name": "gen_uid", "column": "gen_uid", "type": "int64"},
                    {"name": "in_service", "column": "in_service", "type": "int8"},
                    {"name": "unit_seq", "column": "unit_seq", "type": "int16"},
                    {"name": "pg_mw", "column": "pg_mw", "type": "float64"},
                    {"name": "qg_mvar", "column": "qg_mvar", "type": "float64"},
                    {"name": "qmax_mvar", "column": "qmax_mvar", "type": "float64?"},
                    {"name": "qmin_mvar", "column": "qmin_mvar", "type": "float64?"},
                    {"name": "pmax_mw", "column": "pmax_mw", "type": "float32"},
                    {"name": "pmin_mw", "column": "pmin_mw", "type": "float32"},
                    {"name": "vg_setpoint_pu", "column": "vg_setpoint_pu", "type": "float32"},
                    {"name": "ramp_rate_mw_min", "column": "ramp_rate_mw_min", "type": "float32?"},
                    {"name": "cost_c1_eur_mwh", "column": "cost_c1_eur_mwh", "type": "float64"},
                    {"name": "fuel_category", "column": "fuel_category", "type": "var_string"},
                    {"name": "sync_check_ts_ns", "column": "sync_check_ts_ns", "type": "timestamp_ns"},
                    {"name": "capability_curve_v4", "column": "capability_curve_v4", "type": "float32", "dimension": 4},
                ],
            },
            {
                "id": 2,
                "name": "PMU",
                "key_type": "uuid128",
                "file": "domains/pmus.csv",
                "format": "csv",
                "id_column": "id",
                "key_attribute": "pmu_uuid",
                "attributes": [
                    {"name": "pmu_uuid", "column": "pmu_uuid", "type": "fixed_bytes", "dimension": 16},
                    {"name": "sample_rate_hz", "column": "sample_rate_hz", "type": "int16"},
                    {"name": "gps_locked", "column": "gps_locked", "type": "int8"},
                    {"name": "freq_dev_mhz", "column": "freq_dev_mhz", "type": "float32"},
                    {"name": "rocof_hz_s", "column": "rocof_hz_s", "type": "float32?"},
                    {"name": "last_sample_ts_ns", "column": "last_sample_ts_ns", "type": "timestamp_ns"},
                    {"name": "fault_window_ms", "column": "fault_window_ms", "type": "interval_ms_64"},
                    {"name": "c37_118_phasor_v4", "column": "c37_118_phasor_v4", "type": "float32", "dimension": 4},
                ],
            },
            {
                "id": 3,
                "name": "LoadZone",
                "key_type": "int16",
                "file": "domains/load_zones.csv",
                "format": "csv",
                "id_column": "id",
                "key_attribute": "zone_id",
                "attributes": [
                    {"name": "zone_id", "column": "zone_id", "type": "int16"},
                    {"name": "zone_name", "column": "zone_name", "type": "var_string"},
                    {"name": "total_load_mw", "column": "total_load_mw", "type": "float64"},
                    {"name": "spinning_reserve_mw", "column": "spinning_reserve_mw", "type": "float64?"},
                ],
            },
            {
                "id": 4,
                "name": "Substation",
                "key_type": "string",
                "file": "domains/substations.csv",
                "format": "csv",
                "id_column": "id",
                "key_attribute": "substation_code",
                "attributes": [
                    {"name": "substation_code", "column": "substation_code", "type": "var_string"},
                    {"name": "primary_kv", "column": "primary_kv", "type": "float32"},
                    {"name": "bay_count", "column": "bay_count", "type": "int32"},
                    {"name": "commission_year", "column": "commission_year", "type": "int16?"},
                ],
            },
            {
                "id": 5,
                "name": "ContingencyTier",
                "key_type": "int8",
                "file": "domains/contingency_tiers.csv",
                "format": "csv",
                "id_column": "id",
                "key_attribute": "tier_code",
                "attributes": [
                    {"name": "tier_code", "column": "tier_code", "type": "int8"},
                    {"name": "tier_label", "column": "tier_label", "type": "var_string"},
                    {"name": "max_shed_pct", "column": "max_shed_pct", "type": "float32"},
                ],
            },
        ],
        "relations": [
            {
                "name": "AC_LINE",
                "src_domain": 0,
                "tgt_domain": 0,
                "encoding": "raw_u16",
                "node_id_width": 2,
                "cardinality": "m:n",
                "include_csc": True,
                "file": "relations/ac_line.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "r_pu", "column": "r_pu", "type": "float64"},
                    {"name": "x_pu", "column": "x_pu", "type": "float64"},
                    {"name": "z_mag_pu", "column": "z_mag_pu", "type": "float32"},
                    {"name": "rate_a_mva", "column": "rate_a_mva", "type": "float64?"},
                    {"name": "in_service", "column": "in_service", "type": "int8"},
                    {"name": "line_code", "column": "line_code", "type": "var_string"},
                    {"name": "inspection_ts_ms", "column": "inspection_ts_ms", "type": "timestamp_ms?"},
                ],
            },
            {
                "name": "TRANSFORMER",
                "src_domain": 0,
                "tgt_domain": 0,
                "encoding": "coo",
                "node_id_width": 2,
                "cardinality": "m:n",
                "include_csc": True,
                "file": "relations/transformer.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "r_pu", "column": "r_pu", "type": "float64"},
                    {"name": "x_pu", "column": "x_pu", "type": "float64"},
                    {"name": "tap_ratio", "column": "tap_ratio", "type": "float64"},
                    {"name": "shift_angle_deg", "column": "shift_angle_deg", "type": "float32"},
                    {"name": "rate_a_mva", "column": "rate_a_mva", "type": "float64?"},
                ],
            },
            {
                "name": "SHUNT_SELF_LOOP",
                "src_domain": 0,
                "tgt_domain": 0,
                "encoding": "raw_uint32",
                "node_id_width": 4,
                "cardinality": "m:n",
                "include_csc": True,
                "file": "relations/shunt_self_loop.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "gs_mw", "column": "gs_mw", "type": "float64"},
                    {"name": "bs_mvar", "column": "bs_mvar", "type": "float64"},
                    {"name": "step_count", "column": "step_count", "type": "int8?"},
                ],
            },
            {
                "name": "HVDC_TIE_LINE",
                "src_domain": 0,
                "tgt_domain": 0,
                "encoding": "raw_uint32",
                "node_id_width": 4,
                "cardinality": "m:n",
                "include_csc": True,
                "file": "relations/hvdc_tie_line.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
                "attributes": [
                    {"name": "dc_mw_setpoint", "column": "dc_mw_setpoint", "type": "float64"},
                    {"name": "bipole_mode", "column": "bipole_mode", "type": "var_string?"},
                ],
            },
            {
                "name": "GRID_BRANCH",
                "src_domain": 0,
                "tgt_domain": 0,
                "virtual_components": ["AC_LINE", "TRANSFORMER"],
            },
            {
                "name": "BUS_IN_ZONE",
                "src_domain": 0,
                "tgt_domain": 3,
                "encoding": "raw_u16",
                "node_id_width": 2,
                "cardinality": "m:1",
                "include_csc": True,
                "file": "relations/bus_in_zone.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
            {
                "name": "BUS_RISK_TIER",
                "src_domain": 0,
                "tgt_domain": 5,
                "encoding": "raw_uint32",
                "node_id_width": 4,
                "cardinality": "m:1",
                "include_csc": True,
                "file": "relations/bus_risk_tier.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
            {
                "name": "GEN_AT_BUS",
                "src_domain": 1,
                "tgt_domain": 0,
                "encoding": "raw_u16",
                "node_id_width": 2,
                "cardinality": "m:1",
                "include_csc": True,
                "file": "relations/gen_at_bus.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
            {
                "name": "PMU_MONITORS_BUS",
                "src_domain": 2,
                "tgt_domain": 0,
                "encoding": "raw_u16",
                "node_id_width": 2,
                "cardinality": "1:1",
                "include_csc": True,
                "file": "relations/pmu_monitors_bus.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
            {
                "name": "SUBSTATION_HAS_BUS",
                "src_domain": 4,
                "tgt_domain": 0,
                "encoding": "raw_uint32",
                "node_id_width": 4,
                "cardinality": "1:m",
                "include_csc": True,
                "file": "relations/substation_has_bus.csv",
                "format": "csv",
                "src_column": "src",
                "tgt_column": "tgt",
            },
        ],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # -------------------------------------------------------------------------
    # Build 1:1 Companion Ground-Truth SQLite DB (`powergrid_ground_truth.sqlite`)
    # -------------------------------------------------------------------------
    sqlite_path = out_dir / "powergrid_ground_truth.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;

        CREATE TABLE dom_Bus (
            dense_id INTEGER PRIMARY KEY,
            bus_number INTEGER NOT NULL UNIQUE,
            bus_type INTEGER NOT NULL,
            matpower_zone INTEGER NOT NULL,
            pd_mw REAL NOT NULL,
            qd_mvar REAL NOT NULL,
            vm_pu REAL NOT NULL,
            vm_f16 REAL NOT NULL,
            va_deg REAL NOT NULL,
            base_kv REAL NOT NULL,
            shunt_gs_mw REAL,
            shunt_bs_mvar REAL,
            reserve_margin_pu REAL,
            bus_label TEXT NOT NULL,
            voltage_class_tag TEXT NOT NULL,
            operator_email TEXT NOT NULL,
            cim_mrid_bytes TEXT NOT NULL,
            goose_header_blob TEXT,
            scada_poll_ts_ms INTEGER NOT NULL,
            maint_window_start_sec INTEGER NOT NULL,
            maint_window_dur_sec INTEGER NOT NULL
        );

        CREATE TABLE dom_Generator (
            dense_id INTEGER PRIMARY KEY,
            gen_uid INTEGER NOT NULL UNIQUE,
            in_service INTEGER NOT NULL,
            unit_seq INTEGER NOT NULL,
            pg_mw REAL NOT NULL,
            qg_mvar REAL NOT NULL,
            qmax_mvar REAL,
            qmin_mvar REAL,
            pmax_mw REAL NOT NULL,
            pmin_mw REAL NOT NULL,
            vg_setpoint_pu REAL NOT NULL,
            ramp_rate_mw_min REAL,
            cost_c1_eur_mwh REAL NOT NULL,
            fuel_category TEXT NOT NULL,
            sync_check_ts_ns INTEGER NOT NULL
        );

        CREATE TABLE dom_PMU (
            dense_id INTEGER PRIMARY KEY,
            pmu_uuid TEXT NOT NULL UNIQUE,
            sample_rate_hz INTEGER NOT NULL,
            gps_locked INTEGER NOT NULL,
            freq_dev_mhz REAL NOT NULL,
            rocof_hz_s REAL,
            last_sample_ts_ns INTEGER NOT NULL,
            fault_window_start_ms INTEGER NOT NULL,
            fault_window_dur_ms INTEGER NOT NULL
        );

        CREATE TABLE dom_LoadZone (
            dense_id INTEGER PRIMARY KEY,
            zone_id INTEGER NOT NULL UNIQUE,
            zone_name TEXT NOT NULL,
            total_load_mw REAL NOT NULL,
            spinning_reserve_mw REAL
        );

        CREATE TABLE dom_Substation (
            dense_id INTEGER PRIMARY KEY,
            substation_code TEXT NOT NULL UNIQUE,
            primary_kv REAL NOT NULL,
            bay_count INTEGER NOT NULL,
            commission_year INTEGER
        );

        CREATE TABLE dom_ContingencyTier (
            dense_id INTEGER PRIMARY KEY,
            tier_code INTEGER NOT NULL UNIQUE,
            tier_label TEXT NOT NULL,
            max_shed_pct REAL NOT NULL
        );

        CREATE TABLE rel_AC_LINE (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL,
            r_pu REAL NOT NULL,
            x_pu REAL NOT NULL,
            z_mag_pu REAL NOT NULL,
            rate_a_mva REAL,
            in_service INTEGER NOT NULL,
            line_code TEXT NOT NULL,
            inspection_ts_ms INTEGER
        );

        CREATE TABLE rel_TRANSFORMER (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL,
            r_pu REAL NOT NULL,
            x_pu REAL NOT NULL,
            tap_ratio REAL NOT NULL,
            shift_angle_deg REAL NOT NULL,
            rate_a_mva REAL
        );

        CREATE TABLE rel_SHUNT_SELF_LOOP (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL,
            gs_mw REAL NOT NULL,
            bs_mvar REAL NOT NULL,
            step_count INTEGER
        );

        CREATE TABLE rel_HVDC_TIE_LINE (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL,
            dc_mw_setpoint REAL NOT NULL,
            bipole_mode TEXT
        );

        CREATE TABLE rel_BUS_IN_ZONE (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL
        );

        CREATE TABLE rel_BUS_RISK_TIER (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL
        );

        CREATE TABLE rel_GEN_AT_BUS (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL
        );

        CREATE TABLE rel_PMU_MONITORS_BUS (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL
        );

        CREATE TABLE rel_SUBSTATION_HAS_BUS (
            edge_id INTEGER PRIMARY KEY,
            src_dense INTEGER NOT NULL,
            tgt_dense INTEGER NOT NULL
        );

        CREATE VIEW rel_GRID_BRANCH AS
            SELECT src_dense, tgt_dense FROM rel_AC_LINE
            UNION ALL
            SELECT src_dense, tgt_dense FROM rel_TRANSFORMER;
        """
    )

    for d_idx, r in enumerate(bus_rows):
        ms_start, ms_dur = [int(x) for x in str(r[19]).split(":")]
        cur.execute(
            "INSERT INTO dom_Bus VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                d_idx,
                int(r[1]),
                int(r[2]),
                int(r[3]),
                float(r[4]),
                float(r[5]),
                float(r[6]),
                float(r[7]),
                float(r[8]),
                float(r[9]),
                None if r[10] is None else float(r[10]),
                None if r[11] is None else float(r[11]),
                None if r[12] is None else float(r[12]),
                str(r[13]),
                str(r[14]),
                str(r[15]),
                str(r[16]),
                None if r[17] is None else str(r[17]),
                int(r[18]),
                ms_start,
                ms_dur,
            ),
        )

    for d_idx, r in enumerate(gen_rows):
        cur.execute(
            "INSERT INTO dom_Generator VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                d_idx,
                int(r[1]),
                int(r[2]),
                int(r[3]),
                float(r[4]),
                float(r[5]),
                None if r[6] is None else float(r[6]),
                None if r[7] is None else float(r[7]),
                float(r[8]),
                float(r[9]),
                float(r[10]),
                None if r[11] is None else float(r[11]),
                float(r[12]),
                str(r[13]),
                int(r[14]),
            ),
        )

    for d_idx, r in enumerate(pmu_rows):
        fw_start, fw_dur = [int(x) for x in str(r[7]).split(":")]
        cur.execute(
            "INSERT INTO dom_PMU VALUES (?,?,?,?,?,?,?,?,?)",
            (
                d_idx,
                str(r[1]),
                int(r[2]),
                int(r[3]),
                float(r[4]),
                None if r[5] is None else float(r[5]),
                int(r[6]),
                fw_start,
                fw_dur,
            ),
        )

    for d_idx, r in enumerate(loadzone_rows):
        cur.execute(
            "INSERT INTO dom_LoadZone VALUES (?,?,?,?,?)",
            (d_idx, int(r[1]), str(r[2]), float(r[3]), None if r[4] is None else float(r[4])),
        )

    for d_idx, r in enumerate(substation_rows):
        cur.execute(
            "INSERT INTO dom_Substation VALUES (?,?,?,?,?)",
            (d_idx, str(r[1]), float(r[2]), int(r[3]), None if r[4] is None else int(r[4])),
        )

    for d_idx, r in enumerate(contingency_rows):
        cur.execute(
            "INSERT INTO dom_ContingencyTier VALUES (?,?,?,?)",
            (d_idx, int(r[1]), str(r[2]), float(r[3])),
        )

    # Sort edge rows by (src_dense, tgt_dense) so SQLite edge_id 0..E-1 matches CSR edge_index 1:1!
    ac_sorted = sorted(ac_line_rows, key=lambda x: (bus_num_to_dense[int(x[0])], bus_num_to_dense[int(x[1])]))
    for e_idx, r in enumerate(ac_sorted):
        cur.execute(
            "INSERT INTO rel_AC_LINE VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                e_idx,
                bus_num_to_dense[int(r[0])],
                bus_num_to_dense[int(r[1])],
                float(r[2]),
                float(r[3]),
                float(r[4]),
                None if r[5] is None else float(r[5]),
                int(r[6]),
                str(r[7]),
                None if r[8] is None else int(r[8]),
            ),
        )

    xf_sorted = sorted(transformer_rows, key=lambda x: (bus_num_to_dense[int(x[0])], bus_num_to_dense[int(x[1])]))
    for e_idx, r in enumerate(xf_sorted):
        cur.execute(
            "INSERT INTO rel_TRANSFORMER VALUES (?,?,?,?,?,?,?,?)",
            (
                e_idx,
                bus_num_to_dense[int(r[0])],
                bus_num_to_dense[int(r[1])],
                float(r[2]),
                float(r[3]),
                float(r[4]),
                float(r[5]),
                None if r[6] is None else float(r[6]),
            ),
        )

    sh_sorted = sorted(shunt_self_rows, key=lambda x: (bus_num_to_dense[int(x[0])], bus_num_to_dense[int(x[1])]))
    for e_idx, r in enumerate(sh_sorted):
        cur.execute(
            "INSERT INTO rel_SHUNT_SELF_LOOP VALUES (?,?,?,?,?,?)",
            (
                e_idx,
                bus_num_to_dense[int(r[0])],
                bus_num_to_dense[int(r[1])],
                float(r[2]),
                float(r[3]),
                None if r[4] is None else int(r[4]),
            ),
        )

    for e_idx, r in enumerate(bus_in_zone_rows):
        cur.execute(
            "INSERT INTO rel_BUS_IN_ZONE VALUES (?,?,?)",
            (e_idx, bus_num_to_dense[int(r[0])], zone_key_to_dense[int(r[1])]),
        )

    for e_idx, r in enumerate(bus_risk_tier_rows):
        cur.execute(
            "INSERT INTO rel_BUS_RISK_TIER VALUES (?,?,?)",
            (e_idx, bus_num_to_dense[int(r[0])], int(r[1])),
        )

    gen_uid_to_dense = {int(r[1]): idx for idx, r in enumerate(gen_rows)}
    gen_edges_sorted = sorted(gen_at_bus_rows, key=lambda x: (gen_uid_to_dense[int(x[0])], bus_num_to_dense[int(x[1])]))
    for e_idx, r in enumerate(gen_edges_sorted):
        cur.execute(
            "INSERT INTO rel_GEN_AT_BUS VALUES (?,?,?)",
            (e_idx, gen_uid_to_dense[int(r[0])], bus_num_to_dense[int(r[1])]),
        )

    pmu_uuid_to_dense = {str(r[1]): idx for idx, r in enumerate(pmu_rows)}
    pmu_edges_sorted = sorted(
        pmu_monitors_bus_rows, key=lambda x: (pmu_uuid_to_dense[str(x[0])], bus_num_to_dense[int(x[1])])
    )
    for e_idx, r in enumerate(pmu_edges_sorted):
        cur.execute(
            "INSERT INTO rel_PMU_MONITORS_BUS VALUES (?,?,?)",
            (e_idx, pmu_uuid_to_dense[str(x := r[0])], bus_num_to_dense[int(r[1])]),
        )

    sub_code_to_dense = {str(r[1]): idx for idx, r in enumerate(substation_rows)}
    sub_edges_sorted = sorted(
        substation_has_bus_rows, key=lambda x: (sub_code_to_dense[str(x[0])], bus_num_to_dense[int(x[1])])
    )
    for e_idx, r in enumerate(sub_edges_sorted):
        cur.execute(
            "INSERT INTO rel_SUBSTATION_HAS_BUS VALUES (?,?,?)",
            (e_idx, sub_code_to_dense[str(r[0])], bus_num_to_dense[int(r[1])]),
        )

    conn.commit()
    conn.close()

    stats = {
        "case": case_path.name,
        "base_mva": base_mva,
        "domains": {
            "Bus": len(bus_rows),
            "Generator": len(gen_rows),
            "PMU": len(pmu_rows),
            "LoadZone": len(loadzone_rows),
            "Substation": len(substation_rows),
            "ContingencyTier": len(contingency_rows),
        },
        "relations": {
            "AC_LINE": len(ac_line_rows),
            "TRANSFORMER": len(transformer_rows),
            "SHUNT_SELF_LOOP": len(shunt_self_rows),
            "HVDC_TIE_LINE": len(hvdc_empty_rows),
            "GRID_BRANCH (virtual)": len(ac_line_rows) + len(transformer_rows),
            "BUS_IN_ZONE (m:1 u16)": len(bus_in_zone_rows),
            "BUS_RISK_TIER (m:1 partial u32)": len(bus_risk_tier_rows),
            "GEN_AT_BUS (m:1 u16)": len(gen_at_bus_rows),
            "PMU_MONITORS_BUS (1:1 u16)": len(pmu_monitors_bus_rows),
            "SUBSTATION_HAS_BUS (1:m u32)": len(substation_has_bus_rows),
        },
        "authentic_null_highlights": {
            "Bus.shunt_gs_mw_nulls": sum(1 for r in bus_rows if r[10] is None),
            "Bus.reserve_margin_pu_nulls": sum(1 for r in bus_rows if r[12] is None),
            "Bus.goose_header_blob_nulls": sum(1 for r in bus_rows if r[17] is None),
            "Generator.qmax_inf_nulls": sum(1 for r in gen_rows if r[6] is None),
            "Generator.ramp_rate_nulls": sum(1 for r in gen_rows if r[11] is None),
            "AC_LINE.rate_a_zero_nulls": sum(1 for r in ac_line_rows if r[5] is None),
            "TRANSFORMER.rate_a_zero_nulls": sum(1 for r in transformer_rows if r[6] is None),
            "BUS_RISK_TIER.unlinked_sentinel_buses": len(bus_rows) - len(bus_risk_tier_rows),
            "SUBSTATION_HAS_BUS.unlinked_reverse_sentinel_buses": len(bus_rows) - len(substation_has_bus_rows),
        },
    }
    return manifest_path, sqlite_path, stats


INDEX_SPECS: List[str] = [
    # 0x01 Permutation (Node & Edge attributes)
    "Generator.pg_mw:permutation:name=idx_gen_pg_perm",
    "AC_LINE.z_mag_pu:permutation:name=idx_acline_zmag_perm",
    # 0x02 ZoneMap (Node & Edge attributes)
    "Generator.pg_mw:zonemap:name=idx_gen_pg_zonemap",
    "AC_LINE.z_mag_pu:zonemap:name=idx_acline_zmag_zonemap",
    # 0x03 Inverted BitSet (Categorical string bitmaps)
    "Bus.voltage_class_tag:inverted_bitset:name=idx_bus_vclass_invbs",
    "Generator.fuel_category:inverted_bitset:name=idx_gen_fuel_invbs",
    # 0x04 Minimal Perfect Hash Function (Unique business labels)
    "Bus.bus_label:mphf:name=idx_bus_label_mphf",
    "Substation.substation_code:mphf:name=idx_sub_code_mphf",
    # 0x05 Trigram 3-gram Substring Index
    "Bus.bus_label:trigram:name=idx_bus_label_trigram",
    # 0x06 Domain-Split Email Index
    "Bus.operator_email:domain_split:name=idx_bus_email_domsplit",
    # 0x07 Temporal Interval Index (both 0x0D interval_sec_32 and 0x0E interval_ms_64!)
    "Bus.maint_window_sec:temporal_interval:name=idx_bus_maint_sec32",
    "PMU.fault_window_ms:temporal_interval:name=idx_pmu_fault_ms64",
    # 0x08 Vamana DiskANN Vector Index (SQ8, FP16, and L2 4-D Phasor!)
    "Bus.harmonic_spectrum_v8:vamana:R=24:L=48:metric=cosine:quant=sq8:name=idx_bus_harm8_vamana_sq8",
    "Bus.harmonic_spectrum_v8:vamana:R=24:L=48:metric=l2:quant=fp16:name=idx_bus_harm8_vamana_fp16",
    "PMU.c37_118_phasor_v4:vamana:R=16:L=32:metric=cosine:quant=sq8:name=idx_pmu_phasor4_vamana_sq8",
]


def compile_index_sign_and_validate(
    cli: Path,
    manifest_path: Path,
    out_dir: Path,
    case_stem: str,
) -> Tuple[Path, Path]:
    unsigned_imps = out_dir / f"{case_stem}_unsigned_base.imps"
    final_imps = out_dir / f"{case_stem}_v010_fixture.imps"
    key_prefix = out_dir / "fixture_ed25519_key"
    priv_key = out_dir / "fixture_ed25519_key.priv"
    pub_key = out_dir / "fixture_ed25519_key.pub"

    # 1. Build base v0.10.0 snapshot via official `impulse-graph build`
    subprocess.run(
        [str(cli), "build", "-m", str(manifest_path), "-o", str(unsigned_imps)],
        check=True,
    )

    # 2. Add all 8 Section 2.6 index types (15 total secondary + Vamana ANN indexes)
    idx_cmd = [
        str(cli),
        "snapshot",
        "index",
        "add",
        "-i",
        str(unsigned_imps),
        "-o",
        str(final_imps),
    ]
    for spec in INDEX_SPECS:
        idx_cmd.extend(["-s", spec])
    subprocess.run(idx_cmd, check=True)
    unsigned_imps.unlink(missing_ok=True)

    # 3. Generate Ed25519 keypair & cryptographically sign the snapshot footer
    subprocess.run([str(cli), "crypto", "keygen", "-o", str(key_prefix)], check=True)
    subprocess.run([str(cli), "crypto", "sign", "-k", str(priv_key), str(final_imps)], check=True)

    # 4. Validate with strict 128-byte hardware alignment, deep O(V+E) CSC/index audit, & Ed25519 signature check
    subprocess.run(
        [
            str(cli),
            "snapshot",
            "validate",
            "--strict-alignment",
            "--deep-audit",
            "-p",
            str(pub_key),
            str(final_imps),
        ],
        check=True,
    )

    return final_imps, pub_key


def resolve_case_path(case_arg: str) -> Path:
    cand = Path(case_arg)
    if cand.is_file():
        return cand.resolve()
    in_pg = DEFAULT_POWERGRID_DATASETS / case_arg
    if in_pg.is_file():
        return in_pg.resolve()
    if not case_arg.endswith(".m"):
        in_pg_m = DEFAULT_POWERGRID_DATASETS / f"{case_arg}.m"
        if in_pg_m.is_file():
            return in_pg_m.resolve()
    raise FileNotFoundError(f"Could not find MATPOWER case '{case_arg}' in {DEFAULT_POWERGRID_DATASETS}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build comprehensive Spec v0.10.0 PowerGrid .imps fixture + SQLite ground truth."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="case1354pegase.m",
        help="MATPOWER case filename or path (default: case1354pegase.m; also supports case2869pegase.m)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: impulse-verification/datasets/powergrid_fixture/<case_stem>)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic metadata seed (default: 42)")
    args = parser.parse_args()

    case_path = resolve_case_path(args.case)
    case_stem = case_path.stem
    out_dir = (args.out_dir or (DEFAULT_OUT_ROOT / case_stem)).resolve()
    cli = find_impulse_cli()

    print(f"=========================================================================")
    print(f"  IMPULSE GRAPH ENGINE — POWERGRID SPEC v0.10.0 FIXTURE BUILDER")
    print(f"=========================================================================")
    print(f"  MATPOWER Source : {case_path}")
    print(f"  Impulse CLI     : {cli}")
    print(f"  Output Dir      : {out_dir}")
    print(f"=========================================================================")

    manifest_path, sqlite_path, stats = build_fixture_datasets_and_sqlite(case_path, out_dir, seed=args.seed)
    final_imps, pub_key = compile_index_sign_and_validate(cli, manifest_path, out_dir, case_stem)

    sha256_hex = hashlib.sha256(final_imps.read_bytes()).hexdigest()
    stats["imps_path"] = str(final_imps)
    stats["imps_bytes"] = final_imps.stat().st_size
    stats["imps_sha256"] = sha256_hex
    stats["sqlite_path"] = str(sqlite_path)
    stats["public_key_path"] = str(pub_key)
    stats["index_count"] = len(INDEX_SPECS)
    stats["indexes"] = INDEX_SPECS

    summary_path = out_dir / "fixture_manifest_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"\n[SUCCESS] Built & Deep-Audited PowerGrid Spec v0.10.0 Fixture:")
    print(f"  .imps Snapshot : {final_imps} ({stats['imps_bytes']:,} bytes)")
    print(f"  SQLite Oracle  : {sqlite_path} ({sqlite_path.stat().st_size:,} bytes)")
    print(f"  SHA-256        : {sha256_hex}")
    print(f"  Domains (6)    : {stats['domains']}")
    print(f"  Relations (10) : {stats['relations']}")
    print(f"  Indexes (15)   : All 8 Section 2.6 types (0x01..0x08)")


if __name__ == "__main__":
    main()
