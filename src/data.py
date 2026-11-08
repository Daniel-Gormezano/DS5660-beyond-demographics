"""Data preparation for the CMPS 2020 psychological-distress project.

Reads the ICPSR public-use file (2020_data.tsv), keeps the core (baseline)
White / Latino / Black / Asian sample, recodes the variables named in the
proposal, and writes a clean analysis table plus a fixed, stratified
70 / 15 / 15 train / validation / test split.

Variable map (CMPS 2020 codebook, ICPSR 39096)
------------------------------------------------
Outcomes (Q411_Q416 battery, "Over the last 2 weeks, how often...")
    depressive   Q411_Q416R2  Feeling down, depressed, or hopeless
    anxiety      Q411_Q416R3  Feeling nervous, anxious, or on edge
    uncertainty  Q411_Q416R6  Uncertainty makes me uneasy, anxious or stressed
    Survey codes 1-5 are recoded to the standard PHQ/GAD 0-3 scale:
        1 Not at all -> 0, 2 Several days -> 1,
        3 More than half the days -> 2, 4 Almost every day / 5 Every day -> 3

Demographic & socioeconomic inputs
    age          2021 - S5 (birth year), clipped to 18-90
    gender       S3B   (Man, Woman, Another identity)
    education    S13   (6 ordered levels)
    income       Q813  (12 ordered brackets; 99 "prefer not to say" -> missing flag)
    party        Q21   (Republican, Democrat, Independent, Other party)
    race         MAIN_ETH_QUOTA (White, Latino, Black, Asian)

Immigration-related stressors
    deport_worry Q492   Worry that someone you know could be detained or deported (0-3)
    policy_stress Q504R8 Stress caused by U.S. immigration policy in the past year (1-10)
    ice_discrim  Q281R3 Discrimination in dealings with immigration officers, past 4 years (0/1)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SEED = 2026

OUTCOMES = {
    "depressive": "Q411_Q416R2",
    "anxiety": "Q411_Q416R3",
    "uncertainty": "Q411_Q416R6",
}
OUTCOME_LEVELS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]

RACE = {1: "White", 2: "Latino", 3: "Black", 4: "Asian"}
GENDER = {1: "Man", 2: "Woman", 3: "Another identity", 4: "Another identity"}
EDUCATION = {  # S13 -> ordered 0-5
    1: 0, 2: 0,  # less than high school
    3: 1,        # high school / GED
    4: 2,        # some college
    5: 3,        # associate's
    6: 4,        # bachelor's
    7: 5,        # post-graduate
}
EDUCATION_LABELS = ["Less than high school", "High school / GED", "Some college",
                    "Associate's degree", "Bachelor's degree", "Graduate degree"]
INCOME_LABELS = ["Under $20k", "$20k–30k", "$30k–40k", "$40k–50k", "$50k–60k", "$60k–70k",
                 "$70k–80k", "$80k–90k", "$90k–100k", "$100k–150k", "$150k–200k", "$200k+"]
PARTY = {1: "Republican", 2: "Democrat", 3: "Independent", 4: "Other party"}
DEPORT_WORRY = {4: 0, 3: 1, 2: 2, 1: 3}  # Not at all, Not much, Some, A lot
DEPORT_LABELS = ["Not at all", "Not much", "Some", "A lot"]

RAW_COLS = ["BASELINE", "MAIN_ETH_QUOTA", "S5", "S3B", "S13", "Q813", "Q21",
            "Q492", "Q504R8", "Q281R3", *OUTCOMES.values()]


def to_level(code: int) -> int:
    """Survey frequency code 1-5 -> 0-3 (4 and 5 merged)."""
    return min(int(code), 4) - 1


def load_clean(tsv_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(tsv_path, sep="\t", usecols=RAW_COLS, low_memory=False)
    n_total = len(raw)
    df = raw[(raw.BASELINE == 1) & (raw.MAIN_ETH_QUOTA.isin(RACE))].copy()

    out = pd.DataFrame(index=df.index)
    out["race"] = df.MAIN_ETH_QUOTA.map(RACE)
    out["age"] = (2021 - df.S5).clip(18, 90)
    out["gender"] = df.S3B.map(GENDER)
    out["education"] = df.S13.map(EDUCATION)
    out["income_missing"] = (df.Q813 == 99).astype(int)
    out["income"] = df.Q813.where(df.Q813 != 99).astype(float) - 1  # 0-11, NaN if refused
    out["party"] = df.Q21.map(PARTY)
    out["deport_worry"] = df.Q492.map(DEPORT_WORRY)
    out["policy_stress"] = df.Q504R8.astype(float)
    out["ice_discrim"] = df.Q281R3.astype(int)
    for name, col in OUTCOMES.items():
        out[name] = df[col].map(to_level)

    assert out.drop(columns="income").notna().all().all(), "unexpected missing values"
    out.attrs["n_total"] = n_total
    return out.reset_index(drop=True)


def split(df: pd.DataFrame):
    """70/15/15 split stratified by race x depressive-symptom level."""
    strata = df.race + "_" + df.depressive.astype(str)
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=SEED)
    strata_rest = rest.race + "_" + rest.depressive.astype(str)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata_rest, random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main(tsv_path: str):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_clean(tsv_path)
    train, val, test = split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(PROCESSED / f"{name}.csv", index=False)

    dist = {o: (df[o].value_counts(normalize=True).sort_index() * 100).round(1).tolist() for o in OUTCOMES}
    summary = {
        "n_raw": int(df.attrs["n_total"]),
        "n_analytic": len(df),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "by_race": df.race.value_counts().to_dict(),
        "outcome_distribution_pct": dist,
        "income_missing_pct": round(df.income_missing.mean() * 100, 1),
    }
    (PROCESSED / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "2020_data.tsv"))
