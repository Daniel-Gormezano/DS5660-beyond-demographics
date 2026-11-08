"""Feature preprocessing shared by training and the Streamlit app.

The same JSON parameters (fit on the training split only) are used at
inference time, so the app transforms inputs exactly as during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "gender": ["Man", "Woman", "Another identity"],
    "party": ["Republican", "Democrat", "Independent", "Other party"],
    "race": ["White", "Latino", "Black", "Asian"],
}
DEMO_NUMERIC = ["age", "education", "income"]
IMM_NUMERIC = ["deport_worry", "policy_stress"]
IMM_BINARY = ["ice_discrim"]

FEATURE_SETS = ("demographic", "immigration")


class Preprocessor:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    # ---------- fitting ----------
    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        num = DEMO_NUMERIC + IMM_NUMERIC
        income_median = float(train.income.median())
        filled = train.assign(income=train.income.fillna(income_median))
        self.params = {
            "income_median": income_median,
            "mean": {c: float(filled[c].mean()) for c in num},
            "std": {c: float(filled[c].std()) for c in num},
            "categories": CATEGORIES,
        }
        return self

    # ---------- transforming ----------
    def feature_names(self, feature_set: str) -> list[str]:
        names = list(DEMO_NUMERIC) + ["income_missing"]
        for col, cats in CATEGORIES.items():
            names += [f"{col}={c}" for c in cats]
        if feature_set == "immigration":
            names += IMM_NUMERIC + IMM_BINARY
        return names

    def transform(self, df: pd.DataFrame, feature_set: str) -> np.ndarray:
        p = self.params
        df = df.copy()
        df["income_missing"] = df["income"].isna().astype(int) if "income_missing" not in df else df["income_missing"]
        df["income"] = df["income"].fillna(p["income_median"])
        cols = []
        for c in DEMO_NUMERIC:
            cols.append((df[c].astype(float) - p["mean"][c]) / p["std"][c])
        cols.append(df["income_missing"].astype(float))
        for col, cats in CATEGORIES.items():
            for cat in cats:
                cols.append((df[col] == cat).astype(float))
        if feature_set == "immigration":
            for c in IMM_NUMERIC:
                cols.append((df[c].astype(float) - p["mean"][c]) / p["std"][c])
            for c in IMM_BINARY:
                cols.append(df[c].astype(float))
        return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols])

    # ---------- persistence ----------
    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.params, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        return cls(json.loads(Path(path).read_text()))
