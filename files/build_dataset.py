"""
build_dataset.py
================
Builds a single merged, cleaned, leakage-controlled tabular dataset for
depression-risk auditing from raw NHANES public files (CDC / NCHS).

What it does
------------
1. Downloads the required component files (.XPT) for each survey cycle from
   wwwn.cdc.gov, caching them locally so re-runs are free.
2. Merges components within each cycle on the respondent id SEQN.
3. Pools cycles into one table with a `cycle` column (this is what enables the
   temporal cohort-shift experiments later).
4. Builds the PHQ-9 total and the binary MDD label (PHQ-9 >= 10), handling the
   NHANES "refused / don't know" sentinel codes correctly.
5. Engineers a compact, defensible feature set and the protected-attribute
   columns used for fairness auditing (sex, race, age band, income band).
6. Enforces leakage control by construction: the nine PHQ items, the
   functional-impairment item, and antidepressant medications are NEVER used as
   features. A guard asserts this before writing.
7. Writes two CSVs: a full one (everything, for transparency) and a
   model-ready one (features + label + protected attrs + survey-design vars).

Data source
-----------
National Health and Nutrition Examination Survey (NHANES), National Center for
Health Statistics, CDC. Public-use files, no ethics approval required.
Cite the survey and the specific cycles used.

Usage
-----
    python build_dataset.py                 # default: 2005-2006 .. 2017-2018
    python build_dataset.py --cycles 2013-2014 2015-2016 2017-2018
    python build_dataset.py --outdir ./data --min-phq-items 9

Then upload data/nhanes_depression_modelready.csv to Kaggle (plus this script
and data/nhanes_depression_full.csv for reproducibility).

Note on the label threshold: PHQ-9 >= 10 is the standard screening cut for
major depressive disorder (~88% sensitivity/specificity). A second cut at >= 15
(moderate-to-severe) is written as an extra column for sensitivity analysis.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Cycle -> (first-year folder on CDC site, cycle-suffix letter)
CYCLES: dict[str, tuple[str, str]] = {
    "2005-2006": ("2005", "D"),
    "2007-2008": ("2007", "E"),
    "2009-2010": ("2009", "F"),
    "2011-2012": ("2011", "G"),
    "2013-2014": ("2013", "H"),
    "2015-2016": ("2015", "I"),
    "2017-2018": ("2017", "J"),
}

# Component base name -> variables we want from it.
# Only variables actually present in a given cycle are kept, so a name that
# does not exist in an older/newer cycle is silently skipped (it just becomes a
# missing feature for that cycle, which the imputer handles later).
COMPONENTS: dict[str, list[str]] = {
    # Demographics: covariates AND protected attributes AND survey design vars.
    "DEMO": [
        "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3",
        "INDFMPIR", "DMDEDUC2", "SDMVPSU", "SDMVSTRA", "WTMEC2YR",
    ],
    # Depression screener: LABEL SOURCE ONLY. Never used as features.
    "DPQ": [
        "SEQN", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050",
        "DPQ060", "DPQ070", "DPQ080", "DPQ090", "DPQ100",
    ],
    # Body measures.
    "BMX": ["SEQN", "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT"],
    # Blood pressure (mercury sphygmomanometer, 2005-2016).
    "BPX": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
            "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    # Blood pressure (oscillometric, 2017-2018 only).
    "BPXO": ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3",
             "BPXODI1", "BPXODI2", "BPXODI3"],
    # Complete blood count -> WBC, Hgb, platelets, neutrophil/lymphocyte %.
    "CBC": ["SEQN", "LBXWBCSI", "LBXHGB", "LBXPLTSI", "LBXNEPCT", "LBXLYPCT"],
    # Glycohaemoglobin (HbA1c).
    "GHB": ["SEQN", "LBXGH"],
    # Fasting plasma glucose (fasting subsample -> expect missingness).
    "GLU": ["SEQN", "LBXGLU"],
    # Lipids.
    "HDL": ["SEQN", "LBDHDD"],
    "TCHOL": ["SEQN", "LBXTC"],
    "TRIGLY": ["SEQN", "LBXTR", "LBDLDL"],
    # Smoking.
    "SMQ": ["SEQN", "SMQ020", "SMQ040"],
    # Sleep hours (name changes across cycles: SLD010H pre-2015, SLD012 from 2015).
    "SLQ": ["SEQN", "SLD010H", "SLD012"],
    # Self-reported conditions / history.
    "DIQ": ["SEQN", "DIQ010"],
    "BPQ": ["SEQN", "BPQ020", "BPQ080"],
    "MCQ": ["SEQN", "MCQ160C", "MCQ160F", "MCQ220"],
    "HUQ": ["SEQN", "HUQ010"],
    # Sedentary time (available 2007+).
    "PAQ": ["SEQN", "PAD680"],
}

# BPXO only exists in this cycle; do not attempt to download it elsewhere.
COMPONENT_ONLY_IN = {"BPXO": {"2017-2018"}}

# The nine PHQ items + impairment item. Used to build the label, then dropped.
# The leakage guard asserts none of these survive into the feature matrix.
PHQ_ITEMS = [f"DPQ0{n}0" for n in range(1, 10)]  # DPQ010..DPQ090
LEAKAGE_EXCLUDE = set(PHQ_ITEMS) | {"DPQ100"}

BASE_URLS = [
    # Primary (current CDC layout).
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{comp}_{sfx}.XPT",
    # Legacy fallbacks (in case a file still sits at the old path).
    "https://wwwn.cdc.gov/Nchs/Nhanes/{year}-{year2}/{comp}_{sfx}.XPT",
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{comp}_{sfx}.xpt",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (research data pipeline; contact: you@example.com)"}

MDD_THRESHOLD = 10       # PHQ-9 >= 10  -> MDD proxy (primary label)
MODSEVERE_THRESHOLD = 15  # PHQ-9 >= 15 -> moderate-to-severe (sensitivity)

# Sentinel codes meaning refused / don't know for the questionnaire items below.
DPQ_SENTINELS = {7, 9}


# --------------------------------------------------------------------------- #
# Download + read
# --------------------------------------------------------------------------- #

def download_xpt(component: str, cycle: str, cache_dir: Path,
                 retries: int = 3, timeout: int = 60) -> Path | None:
    """Download one component .XPT for one cycle, with caching and URL fallback.
    Returns the local path, or None if the file could not be fetched."""
    year, sfx = CYCLES[cycle]
    year2 = str(int(year) + 1)
    dest = cache_dir / cycle / f"{component}_{sfx}.XPT"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        u.format(year=year, year2=year2, comp=component, sfx=sfx)
        for u in BASE_URLS
    ]
    for url in candidates:
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=HEADERS, timeout=timeout)
                if r.status_code == 200 and r.content[:2] == b"HE":  # XPT header
                    dest.write_bytes(r.content)
                    return dest
                if r.status_code == 404:
                    break  # try next candidate URL
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
        # try next candidate URL
    return None


def read_xpt(path: Path, wanted: list[str]) -> pd.DataFrame | None:
    """Read an XPT file, keep only the wanted columns that are present,
    and coerce SEQN to a nullable integer."""
    try:
        df = pd.read_sas(path, format="xport", encoding="latin-1")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not read {path.name}: {exc}", file=sys.stderr)
        return None
    df.columns = [c.strip().upper() for c in df.columns]
    keep = [c for c in wanted if c in df.columns]
    if "SEQN" not in keep:
        return None
    df = df[keep].copy()
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# Per-cycle assembly
# --------------------------------------------------------------------------- #

def load_cycle(cycle: str, cache_dir: Path) -> pd.DataFrame | None:
    """Download + merge every configured component for one cycle."""
    print(f"[{cycle}]")
    frames: dict[str, pd.DataFrame] = {}
    for comp, wanted in COMPONENTS.items():
        if comp in COMPONENT_ONLY_IN and cycle not in COMPONENT_ONLY_IN[comp]:
            continue
        path = download_xpt(comp, cycle, cache_dir)
        if path is None:
            print(f"  - {comp}: not available for this cycle (skipped)")
            continue
        df = read_xpt(path, wanted)
        if df is None or df.empty:
            print(f"  - {comp}: empty / unreadable (skipped)")
            continue
        # A respondent can appear once per component; collapse just in case.
        df = df.drop_duplicates(subset="SEQN")
        frames[comp] = df
        print(f"  - {comp}: {df.shape[0]} rows, {df.shape[1]-1} vars")

    if "DEMO" not in frames or "DPQ" not in frames:
        print(f"  ! missing DEMO or DPQ for {cycle}; cannot use this cycle.")
        return None

    merged = frames.pop("DEMO")
    for comp, df in frames.items():
        merged = merged.merge(df, on="SEQN", how="left")
    merged.insert(1, "cycle", cycle)
    return merged


# --------------------------------------------------------------------------- #
# Cleaning / label / feature engineering  (pure functions -> unit-testable)
# --------------------------------------------------------------------------- #

def _blank_sentinels(s: pd.Series, sentinels: set) -> pd.Series:
    return s.where(~s.isin(sentinels), other=np.nan)


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column if present, else an all-NaN column (defensive:
    lets feature engineering run even when a variable is absent in a cycle)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _coalesce(*series: pd.Series) -> pd.Series:
    out = series[0].copy()
    for s in series[1:]:
        out = out.where(out.notna(), s)
    return out


def build_label(df: pd.DataFrame, min_items: int = 9) -> pd.DataFrame:
    """Compute PHQ-9 total and MDD labels. 7/9 are refused/don't-know -> NaN.
    Rows with fewer than `min_items` valid items are dropped (no valid label)."""
    items = pd.DataFrame(index=df.index)
    for it in PHQ_ITEMS:
        col = _series(df, it)
        items[it] = _blank_sentinels(col, DPQ_SENTINELS)

    valid_count = items.notna().sum(axis=1)
    total = items.sum(axis=1, min_count=min_items)  # NaN unless >= min_items present
    df = df.copy()
    df["phq9_valid_items"] = valid_count
    df["phq9_total"] = total
    df["mdd"] = (df["phq9_total"] >= MDD_THRESHOLD).astype("Int64")
    df["mdd_modsevere"] = (df["phq9_total"] >= MODSEVERE_THRESHOLD).astype("Int64")
    # Drop rows without a computable label.
    df = df[df["phq9_total"].notna()].copy()
    df["mdd"] = df["mdd"].astype(int)
    df["mdd_modsevere"] = df["mdd_modsevere"].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the compact feature set + protected attributes. Defensive against
    variables that are missing in a given cycle."""
    out = pd.DataFrame(index=df.index)
    out["SEQN"] = df["SEQN"].values
    out["cycle"] = df["cycle"].values

    # --- demographics / covariates ---
    out["age"] = _series(df, "RIDAGEYR")
    out["sex"] = _series(df, "RIAGENDR").map({1: "Male", 2: "Female"})
    race_map = {1: "MexicanAmerican", 2: "OtherHispanic", 3: "NH_White",
                4: "NH_Black", 5: "Other_Multi"}
    out["race"] = _series(df, "RIDRETH1").map(race_map)
    out["income_pir"] = _series(df, "INDFMPIR")
    out["education"] = _blank_sentinels(_series(df, "DMDEDUC2"), {7, 9})

    # --- body ---
    out["bmi"] = _series(df, "BMXBMI")
    out["waist"] = _series(df, "BMXWAIST")

    # --- blood pressure: average available readings, coalesce BPX / BPXO ---
    sys_bpx = pd.concat([_series(df, f"BPXSY{i}") for i in range(1, 5)], axis=1).mean(axis=1)
    dia_bpx = pd.concat([_series(df, f"BPXDI{i}") for i in range(1, 5)], axis=1).mean(axis=1)
    sys_bpxo = pd.concat([_series(df, f"BPXOSY{i}") for i in range(1, 4)], axis=1).mean(axis=1)
    dia_bpxo = pd.concat([_series(df, f"BPXODI{i}") for i in range(1, 4)], axis=1).mean(axis=1)
    out["sbp"] = _coalesce(sys_bpx, sys_bpxo)
    out["dbp"] = _coalesce(dia_bpx, dia_bpxo)

    # --- labs ---
    out["wbc"] = _series(df, "LBXWBCSI")
    out["haemoglobin"] = _series(df, "LBXHGB")
    out["platelets"] = _series(df, "LBXPLTSI")
    neut = _series(df, "LBXNEPCT")
    lymph = _series(df, "LBXLYPCT")
    out["neutrophil_pct"] = neut
    out["lymphocyte_pct"] = lymph
    out["nlr"] = neut / lymph.replace(0, np.nan)  # neutrophil-to-lymphocyte ratio
    out["hba1c"] = _series(df, "LBXGH")
    out["glucose_fasting"] = _series(df, "LBXGLU")
    out["hdl"] = _series(df, "LBDHDD")
    out["total_chol"] = _series(df, "LBXTC")
    out["triglyceride"] = _series(df, "LBXTR")
    out["ldl"] = _series(df, "LBDLDL")

    # --- behaviours / conditions ---
    smq020 = _series(df, "SMQ020")   # smoked >=100 cigs (1 yes, 2 no)
    smq040 = _series(df, "SMQ040")   # now smoke (1 every day, 2 some days, 3 not at all)
    current_smoker = pd.Series(np.nan, index=df.index)
    current_smoker = current_smoker.where(~smq020.eq(2), 0.0)            # never -> 0
    current_smoker = current_smoker.where(~smq040.isin([1, 2]), 1.0)     # current -> 1
    current_smoker = current_smoker.where(~smq040.eq(3), 0.0)            # former -> 0
    out["current_smoker"] = current_smoker

    out["sleep_hours"] = _coalesce(
        _blank_sentinels(_series(df, "SLD010H"), {77, 99}),
        _blank_sentinels(_series(df, "SLD012"), {77, 99}),
    )
    out["sedentary_min"] = _blank_sentinels(_series(df, "PAD680"), {7777, 9999})

    out["diabetes"] = _series(df, "DIQ010").map({1: 1, 2: 0}).astype("float")
    out["high_bp_hx"] = _series(df, "BPQ020").map({1: 1, 2: 0}).astype("float")
    out["high_chol_hx"] = _series(df, "BPQ080").map({1: 1, 2: 0}).astype("float")
    out["chd_hx"] = _series(df, "MCQ160C").map({1: 1, 2: 0}).astype("float")
    out["stroke_hx"] = _series(df, "MCQ160F").map({1: 1, 2: 0}).astype("float")
    out["cancer_hx"] = _series(df, "MCQ220").map({1: 1, 2: 0}).astype("float")
    out["gen_health"] = _blank_sentinels(_series(df, "HUQ010"), {7, 9})

    # --- protected attributes (bands) for fairness auditing ---
    out["age_band"] = pd.cut(out["age"], bins=[17, 34, 49, 64, 200],
                             labels=["18-34", "35-49", "50-64", "65+"])
    out["income_band"] = pd.cut(out["income_pir"], bins=[-0.01, 1.3, 3.5, 5.01],
                                labels=["low", "mid", "high"])

    # --- survey design (for anyone doing weighted analysis) ---
    out["sdmvpsu"] = _series(df, "SDMVPSU")
    out["sdmvstra"] = _series(df, "SDMVSTRA")
    out["wtmec2yr"] = _series(df, "WTMEC2YR")

    # --- labels carried through ---
    for c in ["phq9_total", "phq9_valid_items", "mdd", "mdd_modsevere"]:
        out[c] = df[c].values
    return out


# --------------------------------------------------------------------------- #
# Guards + orchestration
# --------------------------------------------------------------------------- #

def leakage_guard(feature_cols: list[str]) -> None:
    bad = LEAKAGE_EXCLUDE.intersection(feature_cols)
    if bad:
        raise AssertionError(f"Leakage: PHQ/med items in features: {sorted(bad)}")


def add_pooled_weight(df: pd.DataFrame, n_cycles: int) -> pd.DataFrame:
    """NHANES analytic guideline: when pooling k 2-year cycles, divide the
    2-year MEC weight by k."""
    df = df.copy()
    df["wt_pooled"] = df["wtmec2yr"] / n_cycles
    return df


def summarise(df: pd.DataFrame) -> None:
    print("\n================ SUMMARY ================")
    print(f"rows: {len(df):,}   cols: {df.shape[1]}")
    print(f"cycles: {sorted(df['cycle'].unique())}")
    pos = df["mdd"].mean()
    print(f"MDD (PHQ-9>=10) positive rate: {pos:.3%}")
    print(f"mod-severe (>=15) positive rate: {df['mdd_modsevere'].mean():.3%}")
    print("\nper-cycle positive rate (this is the shift signal):")
    print(df.groupby("cycle")["mdd"].mean().round(4).to_string())
    print("\nmissingness (top 12 features):")
    feat = df.drop(columns=[c for c in df.columns if c.startswith("mdd")
                            or c in {"SEQN", "cycle", "phq9_total",
                                     "phq9_valid_items"}])
    miss = feat.isna().mean().sort_values(ascending=False).head(12)
    print((miss * 100).round(1).astype(str).add(" %").to_string())
    print("=========================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", nargs="+", default=list(CYCLES.keys()),
                    help="which cycles to build (default: all configured)")
    ap.add_argument("--outdir", default="./data", help="output + cache directory")
    ap.add_argument("--min-phq-items", type=int, default=9,
                    help="min valid PHQ-9 items required to score a row (default 9)")
    ap.add_argument("--min-age", type=int, default=18)
    args = ap.parse_args()

    for c in args.cycles:
        if c not in CYCLES:
            ap.error(f"unknown cycle {c!r}; known: {list(CYCLES)}")

    outdir = Path(args.outdir)
    cache = outdir / "raw"
    cache.mkdir(parents=True, exist_ok=True)

    per_cycle = []
    for cycle in args.cycles:
        merged = load_cycle(cycle, cache)
        if merged is None:
            continue
        merged = merged[_series(merged, "RIDAGEYR") >= args.min_age].copy()
        labelled = build_label(merged, min_items=args.min_phq_items)
        feats = engineer_features(labelled)
        per_cycle.append(feats)

    if not per_cycle:
        sys.exit("No cycles could be built. Check your network / CDC availability.")

    full = pd.concat(per_cycle, ignore_index=True)
    full = add_pooled_weight(full, n_cycles=len(per_cycle))

    # model-ready view: drop id + raw phq total from the FEATURE role, but keep
    # them as columns so nothing is lost; the leakage guard checks feature names.
    non_features = {"SEQN", "cycle", "phq9_total", "phq9_valid_items",
                    "mdd", "mdd_modsevere", "sdmvpsu", "sdmvstra",
                    "wtmec2yr", "wt_pooled"}
    feature_cols = [c for c in full.columns if c not in non_features]
    leakage_guard(feature_cols)

    outdir.mkdir(parents=True, exist_ok=True)
    full_path = outdir / "nhanes_depression_full.csv"
    model_path = outdir / "nhanes_depression_modelready.csv"
    full.to_csv(full_path, index=False)
    full.to_csv(model_path, index=False)  # same columns; kept as the named artefact

    summarise(full)
    print(f"wrote {full_path}")
    print(f"wrote {model_path}")
    print(f"feature columns ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()