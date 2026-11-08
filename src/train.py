"""Train and evaluate every model in the proposal.

Models (each trained on two feature sets: demographic-only and immigration-aware)
    1. Multinomial logistic regression   (baseline, one per outcome)
    2. Random Forest                     (baseline, one per outcome)
    3. Separate neural networks          (one single-task MLP per outcome)
    4. Multi-task neural network         (proposed: shared layers + 3 heads)
    5. Multi-task ensemble               (deployed: average of the 5 seed-trained multi-task nets)

Protocol
    * fixed stratified 70/15/15 split (src/data.py)
    * hyperparameters chosen on the validation set only
    * best configuration re-trained with 5 random seeds -> mean +/- SD on the test set
    * class-weighted losses for imbalanced outcomes; Macro F1 is the primary metric

Run:  python -m src.train
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn

from .data import OUTCOME_LEVELS, PROCESSED, ROOT
from .features import FEATURE_SETS, Preprocessor
from .models import N_CLASSES, TASKS, MultiTaskMLP, SingleTaskMLP

torch.set_num_threads(2)
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
GROUPS = ["White", "Latino", "Black", "Asian"]

NN_GRID = {
    "hidden": [(64, 32), (128, 64), (256, 128)],
    "dropout": [0.2, 0.4],
    "lr": [1e-3, 3e-3],
}
LR_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}
RF_GRID = {"max_depth": [8, 16, None], "min_samples_leaf": [5, 20]}


# --------------------------------------------------------------------------- metrics
def evaluate(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(N_CLASSES))),
        "mae": float(np.abs(pred - y).mean()),
    }


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=N_CLASSES)
    return torch.tensor(len(y) / (N_CLASSES * counts), dtype=torch.float32)


# --------------------------------------------------------------------------- neural nets
def predict_proba(model: nn.Module, X: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X))
    return {t: torch.softmax(v, 1).numpy() for t, v in out.items()}


def train_nn(model: nn.Module, Xtr, Ytr: dict, Xva, Yva: dict, lr: float, seed: int,
             max_epochs=200, patience=15, batch=256, weight_decay=1e-4):
    """Mini-batch AdamW with class-weighted cross-entropy summed over the model's tasks.
    Early stopping on mean validation Macro F1 across tasks."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tasks = model.tasks
    losses = {t: nn.CrossEntropyLoss(weight=class_weights(Ytr[t])) for t in tasks}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = {t: torch.from_numpy(np.array(Ytr[t])).long() for t in tasks}

    best, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    for epoch in range(max_epochs):
        model.train()
        idx = rng.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(idx), batch):
            b = torch.from_numpy(idx[i:i + batch])
            out = model(Xtr_t[b])
            loss = sum(losses[t](out[t], Ytr_t[t][b]) for t in tasks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        proba = predict_proba(model, Xva)
        score = float(np.mean([f1_score(Yva[t], proba[t].argmax(1), average="macro") for t in tasks]))
        history.append({"epoch": epoch + 1, "train_loss": total / len(idx), "val_macro_f1": score})
        if score > best + 1e-4:
            best, best_epoch, wait = score, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, best_epoch, history


def nn_configs():
    keys = list(NN_GRID)
    for vals in itertools.product(*NN_GRID.values()):
        yield dict(zip(keys, vals))


# --------------------------------------------------------------------------- main experiment
def main():
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    prep = Preprocessor().fit(train)
    prep.save(MODELS_DIR / "preprocess.json")

    Y = {name: {t: part[t].to_numpy() for t in TASKS} for name, part in
         [("train", train), ("val", val), ("test", test)]}
    groups_test = test.race.to_numpy()

    rows, group_rows, tuning, deployed, test_probas = [], [], {}, {}, {}

    def record(model_name, fs, seed, proba_by_task):
        for t in TASKS:
            m = evaluate(Y["test"][t], proba_by_task[t])
            rows.append({"model": model_name, "features": fs, "seed": seed, "outcome": t, **m})
            for g in GROUPS:
                mask = groups_test == g
                group_rows.append({"model": model_name, "features": fs, "seed": seed, "outcome": t, "group": g,
                                   "macro_f1": f1_score(Y["test"][t][mask], proba_by_task[t][mask].argmax(1), average="macro")})

    for fs in FEATURE_SETS:
        X = {name: prep.transform(part, fs) for name, part in [("train", train), ("val", val), ("test", test)]}
        print(f"\n=== feature set: {fs}  ({X['train'].shape[1]} inputs)")

        # ---- 1. multinomial logistic regression
        best_c = {}
        for t in TASKS:
            scores = {}
            for C in LR_GRID["C"]:
                lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
                lr.fit(X["train"], Y["train"][t])
                scores[C] = f1_score(Y["val"][t], lr.predict(X["val"]), average="macro")
            best_c[t] = max(scores, key=scores.get)
        tuning[f"logreg/{fs}"] = best_c
        proba = {}
        for t in TASKS:
            lr = LogisticRegression(C=best_c[t], class_weight="balanced", max_iter=3000).fit(X["train"], Y["train"][t])
            proba[t] = lr.predict_proba(X["test"])
        for seed in SEEDS:  # deterministic; repeated so every model has 5 rows
            record("Logistic regression", fs, seed, proba)
        print(f"  logreg done  C={best_c}")

        # ---- 2. random forest
        best_rf = {}
        for t in TASKS:
            scores = {}
            for depth, leaf in itertools.product(RF_GRID["max_depth"], RF_GRID["min_samples_leaf"]):
                rf = RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                                            class_weight="balanced_subsample", n_jobs=2, random_state=0)
                rf.fit(X["train"], Y["train"][t])
                scores[(depth, leaf)] = f1_score(Y["val"][t], rf.predict(X["val"]), average="macro")
            best_rf[t] = max(scores, key=scores.get)
        tuning[f"rf/{fs}"] = {t: {"max_depth": d, "min_samples_leaf": l} for t, (d, l) in best_rf.items()}
        for seed in SEEDS:
            proba = {}
            for t in TASKS:
                d, l = best_rf[t]
                rf = RandomForestClassifier(n_estimators=300, max_depth=d, min_samples_leaf=l,
                                            class_weight="balanced_subsample", n_jobs=2, random_state=seed)
                proba[t] = rf.fit(X["train"], Y["train"][t]).predict_proba(X["test"])
            record("Random forest", fs, seed, proba)
        print(f"  random forest done  {best_rf}")

        # ---- 3. separate neural networks (tune per outcome)
        best_sep = {}
        for t in TASKS:
            scores = []
            for cfg in nn_configs():
                m = SingleTaskMLP(X["train"].shape[1], t, cfg["hidden"], cfg["dropout"])
                _, s, _, _ = train_nn(m, X["train"], Y["train"], X["val"], Y["val"], cfg["lr"], seed=0)
                scores.append((s, cfg))
            best_sep[t] = max(scores, key=lambda z: z[0])[1]
        tuning[f"separate_nn/{fs}"] = best_sep
        for seed in SEEDS:
            proba = {}
            for t in TASKS:
                cfg = best_sep[t]
                m = SingleTaskMLP(X["train"].shape[1], t, cfg["hidden"], cfg["dropout"])
                m, _, _, _ = train_nn(m, X["train"], Y["train"], X["val"], Y["val"], cfg["lr"], seed=seed)
                proba[t] = predict_proba(m, X["test"])[t]
            record("Separate neural nets", fs, seed, proba)
        print(f"  separate NNs done  {best_sep}")

        # ---- 4. multi-task neural network (proposed)
        scores = []
        for cfg in nn_configs():
            m = MultiTaskMLP(X["train"].shape[1], cfg["hidden"], cfg["dropout"])
            _, s, _, _ = train_nn(m, X["train"], Y["train"], X["val"], Y["val"], cfg["lr"], seed=0)
            scores.append((s, cfg))
        cfg = max(scores, key=lambda z: z[0])[1]
        tuning[f"multitask_nn/{fs}"] = cfg
        members, calibs, histories = [], [], []
        for seed in SEEDS:
            m = MultiTaskMLP(X["train"].shape[1], cfg["hidden"], cfg["dropout"])
            m, val_score, best_epoch, hist = train_nn(m, X["train"], Y["train"], X["val"], Y["val"], cfg["lr"], seed=seed)
            record("Multi-task neural net", fs, seed, predict_proba(m, X["test"]))
            torch.save(m.state_dict(), MODELS_DIR / f"multitask_{fs}_seed{seed}.pt")
            members.append(m)
            calibs.append(calibrate(m, X["val"], Y["val"], Y["train"]))
            histories.append({"seed": seed, "best_epoch": best_epoch, "val_macro_f1": val_score, "history": hist})

        # ---- 5. deployed model: ensemble of the 5 seed-trained multi-task networks
        proba = ensemble_proba(members, X["test"])
        for seed in SEEDS:  # deterministic given the members; repeated so every model has 5 rows
            record("Multi-task ensemble", fs, seed, proba)
        deployed[fs] = {"config": {**cfg, "hidden": list(cfg["hidden"])}, "seeds": SEEDS,
                        "in_dim": int(X["train"].shape[1]),
                        "n_params_per_network": int(sum(p.numel() for p in members[0].parameters())),
                        "feature_names": prep.feature_names(fs), "members": histories,
                        "calibration": calibs,
                        "calibration_check_test": calibration_check(
                            ensemble_calibrated(members, calibs, X["test"]), Y["test"])}
        test_probas[fs] = proba
        print(f"  multi-task NN + ensemble done  cfg={cfg}")

        if fs == "immigration":
            deployed[fs]["permutation_importance"] = permutation_importance(
                lambda Z: ensemble_proba(members, Z), X["test"], Y["test"], prep)

    # ---------------------------------------------------------------- summaries
    res = pd.DataFrame(rows)
    grp = pd.DataFrame(group_rows)
    res.to_csv(RESULTS_DIR / "test_metrics_all_seeds.csv", index=False)
    grp.to_csv(RESULTS_DIR / "test_macro_f1_by_group_all_seeds.csv", index=False)
    metric_cols = ["macro_f1", "weighted_f1", "balanced_acc", "precision", "recall", "roc_auc", "mae"]
    summary = res.groupby(["model", "features", "outcome"])[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(RESULTS_DIR / "test_metrics_summary.csv")
    overall = res.groupby(["model", "features", "seed"])[metric_cols].mean().groupby(["model", "features"]).agg(["mean", "std"]).round(4)
    overall.to_csv(RESULTS_DIR / "test_metrics_overall.csv")
    group_summary = grp.groupby(["model", "features", "group"]).macro_f1.mean().unstack("group").round(4)
    group_summary.to_csv(RESULTS_DIR / "test_macro_f1_by_group.csv")

    # deployed-model confusion matrices + paired bootstrap of the immigration gain
    conf = {fs: {t: confusion_matrix(Y["test"][t], test_probas[fs][t].argmax(1), labels=range(N_CLASSES)).tolist()
                 for t in TASKS} for fs in FEATURE_SETS}
    boot = bootstrap_gain(Y["test"], test_probas["demographic"], test_probas["immigration"])

    out = {
        "data": json.loads((PROCESSED / "summary.json").read_text()),
        "outcome_levels": OUTCOME_LEVELS,
        "tuning": json.loads(json.dumps(tuning, default=str)),
        "deployed": deployed,
        "overall": tidy(overall),
        "by_outcome": tidy(summary),
        "by_group": group_summary.reset_index().to_dict(orient="records"),
        "confusion_deployed": conf,
        "bootstrap_gain_deployed": boot,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=2, default=float))
    card = {fs: {k: deployed[fs][k] for k in ("config", "in_dim", "feature_names", "calibration", "seeds")}
            for fs in FEATURE_SETS}
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print("\n", overall[["macro_f1"]])
    print(f"\nfinished in {out['runtime_minutes']} min")


def tidy(frame: pd.DataFrame) -> list[dict]:
    flat = frame.copy()
    flat.columns = [f"{a}_{b}" for a, b in flat.columns]
    return flat.reset_index().to_dict(orient="records")


def ensemble_proba(members, X) -> dict[str, np.ndarray]:
    """Average of the members' (class-weighted) probabilities - used for classification metrics."""
    ps = [predict_proba(m, X) for m in members]
    return {t: np.mean([p[t] for p in ps], axis=0) for t in TASKS}


def ensemble_calibrated(members, calibs, X) -> dict[str, np.ndarray]:
    """Average of the members' calibrated probabilities - what the app displays."""
    out = {t: [] for t in TASKS}
    for m, cal in zip(members, calibs):
        z = logits(m, X)
        for t in TASKS:
            out[t].append(calibrated_proba(z[t], np.array(cal[t]["log_w"]), cal[t]["T"]))
    return {t: np.mean(v, axis=0) for t, v in out.items()}


def permutation_importance(proba_fn, X_test, Y_test, prep, repeats=30, seed=0):
    """Drop in mean Macro F1 (over the 3 outcomes) when one input variable is shuffled."""
    rng = np.random.default_rng(seed)
    names = prep.feature_names("immigration")
    groups = {
        "Age": ["age"], "Education": ["education"], "Income": ["income", "income_missing"],
        "Gender": [n for n in names if n.startswith("gender=")],
        "Party": [n for n in names if n.startswith("party=")],
        "Race / ethnicity": [n for n in names if n.startswith("race=")],
        "Deportation worry": ["deport_worry"], "Immigration-policy stress": ["policy_stress"],
        "Discrimination by immigration officers": ["ice_discrim"],
    }

    def score(X):
        p = proba_fn(X)
        return np.mean([f1_score(Y_test[t], p[t].argmax(1), average="macro") for t in TASKS])

    base = score(X_test)
    out = {}
    for label, cols in groups.items():
        idx = [names.index(c) for c in cols]
        drops = []
        for _ in range(repeats):
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xp[perm][:, idx]
            drops.append(base - score(Xp))
        out[label] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return out


def logits(model, X):
    model.eval()
    with torch.no_grad():
        return {t: v.numpy() for t, v in model(torch.from_numpy(X)).items()}


def calibrated_proba(z: np.ndarray, log_w: np.ndarray, T: float) -> np.ndarray:
    """Undo the class weighting (prior correction), then temperature-scale."""
    a = (z - log_w) / T
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def calibrate(model, X_val, Y_val, Y_train):
    """Class weights push predictions toward rare levels (good for Macro F1, bad for
    'chance' estimates). For the app's probabilities we subtract log(class weight)
    from each logit and fit one temperature per outcome on the validation set."""
    z = logits(model, X_val)
    out = {}
    for t in model.tasks:
        log_w = np.log(class_weights(Y_train[t]).numpy())
        grid = np.linspace(0.5, 3.0, 51)
        nll = [-np.log(calibrated_proba(z[t], log_w, T)[np.arange(len(Y_val[t])), Y_val[t]] + 1e-9).mean() for T in grid]
        out[t] = {"log_w": log_w.tolist(), "T": float(grid[int(np.argmin(nll))])}
    return out


def calibration_check(p_by_task, Y, n_bins=10):
    """Test-set check on P(level >= 2, i.e. 'more than half the days' or more)."""
    out = {}
    for t in TASKS:
        p = p_by_task[t][:, 2:].sum(1)
        y = (Y[t] >= 2).astype(float)
        bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
        ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(n_bins) if (bins == b).any())
        out[t] = {"mean_predicted": float(p.mean()), "observed_rate": float(y.mean()), "ece": float(ece),
                  "auc_frequent": float(roc_auc_score(y, p))}
    return out


def bootstrap_gain(Y_test, p_demo, p_imm, n=2000, seed=0):
    """Paired bootstrap of the test-set Macro F1 gain (immigration-aware - demographic-only)."""
    rng = np.random.default_rng(seed)
    n_obs = len(Y_test[TASKS[0]])
    out = {}
    for t in TASKS + ("mean",):
        diffs = []
        for _ in range(n):
            i = rng.integers(0, n_obs, n_obs)
            tasks = TASKS if t == "mean" else (t,)
            d = np.mean([f1_score(Y_test[k][i], p_imm[k][i].argmax(1), average="macro")
                         - f1_score(Y_test[k][i], p_demo[k][i].argmax(1), average="macro") for k in tasks])
            diffs.append(d)
        diffs = np.array(diffs)
        out[t] = {"gain": float(np.mean(diffs)), "ci_low": float(np.percentile(diffs, 2.5)),
                  "ci_high": float(np.percentile(diffs, 97.5)), "p_gain_le_0": float((diffs <= 0).mean())}
    return out


if __name__ == "__main__":
    main()
