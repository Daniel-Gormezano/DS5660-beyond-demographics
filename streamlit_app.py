"""Beyond Demographics - interactive demo.

Pipeline: user profile -> same preprocessing as training -> two trained multi-task
PyTorch ensembles (demographics-only and immigration-aware, 5 networks each) ->
calibrated probabilities for three distress outcomes, shown side by side.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from app_theme import INK, MUTED, ROSA, TEAL, css
from src.features import Preprocessor
from src.models import MultiTaskMLP, TASKS

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

st.set_page_config(page_title="Beyond Demographics", page_icon="🌼", layout="wide")
st.html(css())

OUTCOME_UI = {
    "depressive": ("Depressive Symptoms", "Feeling down, depressed, or hopeless", "#3D6E8F",
                   '<path d="M7 15a5 5 0 0 1 .5-10A6 6 0 0 1 19 7a4 4 0 0 1-1 8z" fill="white"/>'
                   '<path d="M9 18l-1 3M13 18l-1 3M17 18l-1 3" stroke="white" stroke-width="1.8" stroke-linecap="round"/>'),
    "anxiety": ("Anxiety", "Feeling nervous, anxious, or on edge", "#7B4B7A",
                '<path d="M3 12h4l2-5 4 10 2-5h6" stroke="white" stroke-width="2" fill="none" '
                'stroke-linecap="round" stroke-linejoin="round"/>'),
    "uncertainty": ("Uncertainty Stress", "Uncertainty makes me uneasy, anxious, or stressed", "#B07A12",
                    '<circle cx="12" cy="12" r="9" stroke="white" stroke-width="2" fill="none"/>'
                    '<path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .8-1 1.5v.4" stroke="white" stroke-width="2" '
                    'fill="none" stroke-linecap="round"/><circle cx="12" cy="17" r="1.2" fill="white"/>'),
}
LEVELS_SHORT = ["Not At All", "Several Days", "Most Days", "Daily"]
LEVELS_FULL = ["Not at all", "Several days", "More than half the days", "Nearly every day"]

RACES = ["White", "Latino", "Black", "Asian"]
GENDERS = ["Woman", "Man", "Another identity"]
PARTIES = ["Democrat", "Republican", "Independent", "Other party"]
EDUCATION = ["Less than high school", "High school / GED", "Some college",
             "Associate's degree", "Bachelor's degree", "Graduate degree"]
INCOME = ["Under $20k", "$20k–30k", "$30k–40k", "$40k–50k", "$50k–60k", "$60k–70k", "$70k–80k",
          "$80k–90k", "$90k–100k", "$100k–150k", "$150k–200k", "$200k+", "Prefer not to say"]
WORRY = ["Not at all", "Not much", "Some", "A lot"]

DEMO_LABEL = "Demographics-Only Multi-Task Ensemble"
IMM_LABEL = "Demographics + Immigration Experiences Multi-Task Ensemble"


# ----------------------------------------------------------------------------- model loading
@st.cache_resource
def load_models():
    prep = Preprocessor.load(MODELS / "preprocess.json")
    card = json.loads((MODELS / "model_card.json").read_text())
    nets = {}
    for fs, info in card.items():
        cfg = info["config"]
        nets[fs] = []
        for seed in info["seeds"]:  # ensemble of the 5 seed-trained networks
            net = MultiTaskMLP(info["in_dim"], tuple(cfg["hidden"]), cfg["dropout"])
            net.load_state_dict(torch.load(MODELS / f"multitask_{fs}_seed{seed}.pt", map_location="cpu"))
            net.eval()
            nets[fs].append(net)
    return prep, card, nets


@st.cache_data
def load_results():
    return json.loads((RESULTS / "results.json").read_text())


def predict(profile: dict) -> dict[str, dict[str, np.ndarray]]:
    """Real inference with both trained ensembles; returns calibrated probabilities
    (the average of the 5 networks' calibrated probabilities)."""
    prep, card, nets = load_models()
    row = pd.DataFrame([profile])
    out = {}
    for fs, members in nets.items():
        x = torch.from_numpy(prep.transform(row, fs))
        probs = {t: [] for t in TASKS}
        for net, cal in zip(members, card[fs]["calibration"]):
            with torch.no_grad():
                logits = net(x)
            for t in TASKS:
                z = (logits[t].numpy()[0] - np.array(cal[t]["log_w"])) / cal[t]["T"]
                p = np.exp(z - z.max())
                probs[t].append(p / p.sum())
        out[fs] = {t: np.mean(v, axis=0) for t, v in probs.items()}
    return out


# ----------------------------------------------------------------------------- rendering helpers
def svg_uri(inner: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">{inner}</svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def section(title: str, color: str):
    st.html(f'<div class="section-title"><span class="dot" style="background:{color}"></span>{title}</div>')


def chart_title(title: str, note: str = ""):
    note_html = f'<div class="chart-note">{note}</div>' if note else ""
    st.html(f'<div class="section-title">{title}</div>{note_html}')


def outcome_card(task: str, p_demo: np.ndarray, p_imm: np.ndarray) -> str:
    name, item, color, icon = OUTCOME_UI[task]
    most_demo, most_imm = p_demo[2:].sum() * 100, p_imm[2:].sum() * 100
    d = most_imm - most_demo
    d_color = ROSA if d > 0.5 else TEAL if d < -0.5 else MUTED
    d_bg = "#FCE4EE" if d > 0.5 else "#DDF2F4" if d < -0.5 else "#EFE7DD"
    d_txt = "No Change" if abs(d) < 0.5 else f"{'+' if d > 0 else '−'}{abs(d):.0f} pts"

    peak = max(p_demo.max(), p_imm.max())
    cols = ""
    for i, lab in enumerate(LEVELS_SHORT):
        bars = ""
        for p, c, who in ((p_demo[i], TEAL, DEMO_LABEL), (p_imm[i], ROSA, IMM_LABEL)):
            h = max(p / peak * 72, 2)
            bars += (f'<div class="col" title="{who}: {p * 100:.0f}% chance of answering &quot;{LEVELS_FULL[i]}&quot;">'
                     f'<div class="pct" style="color:{c}">{p * 100:.0f}%</div>'
                     f'<div class="bar" style="height:{h:.1f}px;background:{c}"></div></div>')
        cols += f'<div class="grp"><div class="pair">{bars}</div><div class="lab">{lab}</div></div>'
    return f"""
<div class="ocard">
  <div class="top">
    <div class="icon" style="background:{color}"><img alt="" src="{svg_uri(icon)}"></div>
    <div><div class="name">{name}</div><div class="item">Survey question: “{item}”</div></div>
  </div>
  <div class="body">
    <div>
      <div class="klabel">Chance Of Feeling This Most Days Or Daily</div>
      <div class="knums">
        <span class="knum" style="color:{TEAL}">{most_demo:.0f}%</span>
        <span class="arrow">→</span>
        <span class="knum" style="color:{ROSA}">{most_imm:.0f}%</span>
        <span class="delta" style="color:{d_color};background:{d_bg}">{d_txt}</span>
      </div>
    </div>
    <div>
      <div class="klabel">Predicted Chance Of Each Answer</div>
      <div class="bars">{cols}</div>
      <div class="xtitle">Answer (Over The Last 2 Weeks)</div>
    </div>
  </div>
</div>"""


LEGEND = (f'<div class="legend"><span><span class="sw" style="background:{TEAL}"></span>{DEMO_LABEL}</span>'
          f'<span><span class="sw" style="background:{ROSA}"></span>{IMM_LABEL}</span></div>'
          '<div class="legend-note">Both are predictions for the person described on the left. Each ensemble averages '
          '5 copies of the same multi-task neural network, each trained from a different random starting point. '
          'The percentages for each outcome add up to 100%.</div>')


def style_chart(chart):
    return (chart.configure(background="#FFFFFF", font="Nunito, Helvetica, Arial, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False, labelColor=MUTED, titleColor=INK, labelFontSize=12, titleFontSize=13,
                            titleFontWeight=700, titlePadding=10)
            .configure_legend(labelColor=INK, labelFontSize=12, symbolType="square"))


# ----------------------------------------------------------------------------- page
st.html('<div class="hero"><h1>Beyond Demographics</h1>'
        '<p>Do Immigration Experiences Change Predicted Psychological Distress?</p></div>')

tab_explore, tab_results = st.tabs(["Explore", "Model Results"])

with tab_explore:
    left, right = st.columns([1, 1.55], gap="large")

    with left:
        with st.container(border=True, key="profile_card"):
            section("Profile", TEAL)
            race = st.segmented_control("Race / Ethnicity", RACES, default="Latino", key="race")
            age = st.slider("Age", 18, 90, 35)
            gender = st.segmented_control("Gender", GENDERS, default="Woman", key="gender")
            edu = st.selectbox("Education", EDUCATION, index=2)
            inc = st.selectbox("Household Income", INCOME, index=2)
            party = st.segmented_control("Political Party", PARTIES, default="Democrat", key="party",
                                         format_func=lambda x: "Other" if x == "Other party" else x)

        with st.container(border=True, key="imm_card"):
            section("Immigration Experiences", ROSA)
            worry = st.segmented_control("Worry That Someone You Know Could Be Deported", WORRY,
                                         default="Some", key="worry")
            stress = st.slider("Stress From U.S. Immigration Policy This Past Year (1 = None, 10 = Extreme)",
                               1, 10, 8)
            discrim = st.segmented_control("Treated Unfairly By Immigration Officers (Past 4 Years)",
                                           ["No", "Yes"], default="Yes", key="discrim")

    profile = {
        "race": race or "Latino",
        "age": age,
        "gender": gender or "Woman",
        "education": EDUCATION.index(edu),
        "income": None if inc == "Prefer not to say" else float(INCOME.index(inc)),
        "income_missing": int(inc == "Prefer not to say"),
        "party": party or "Democrat",
        "deport_worry": WORRY.index(worry or "Not at all"),
        "policy_stress": float(stress),
        "ice_discrim": int((discrim or "No") == "Yes"),
    }
    preds = predict(profile)

    with right:
        st.html(LEGEND)
        for t in TASKS:
            st.html(outcome_card(t, preds["demographic"][t], preds["immigration"][t]))


# ----------------------------------------------------------------------------- results tab
MODEL_LABEL = {"Logistic regression": "Logistic Regression", "Random forest": "Random Forest",
               "Separate neural nets": "Separate Neural Nets", "Multi-task neural net": "Multi-Task Neural Net",
               "Multi-task ensemble": "Multi-Task Ensemble"}
FS_LABEL = {"demographic": "Demographics Only", "immigration": "+ Immigration Experiences"}
OUTCOME_LABEL = {t: v[0] for t, v in OUTCOME_UI.items()}
GROUPS = ["White", "Latino", "Black", "Asian"]
IMM_INPUTS = ("Deportation worry", "Immigration-policy stress", "Discrimination by immigration officers")

with tab_results:
    res = load_results()
    overall = pd.DataFrame(res["overall"])
    overall["Feature Set"] = overall.features.map(FS_LABEL)
    overall["Model"] = overall.model.map(MODEL_LABEL)
    mt = overall[overall.model == "Multi-task ensemble"].set_index("features")
    boot = res["bootstrap_gain_deployed"]["mean"]

    c1, c2, c3 = st.columns(3)
    stats = [
        (f'{mt.loc["demographic", "macro_f1_mean"]:.3f}', "Test-Set Average Macro F1",
         "Multi-Task Ensemble · Demographics Only", TEAL),
        (f'{mt.loc["immigration", "macro_f1_mean"]:.3f}', "Test-Set Average Macro F1",
         "Multi-Task Ensemble · + Immigration Experiences", ROSA),
        (f'{boot["gain"]:+.3f}', "Test-Set Gain In Average Macro F1",
         f'Multi-Task Ensemble · 95% CI {boot["ci_low"]:+.3f} To {boot["ci_high"]:+.3f}', INK),
    ]
    for col, (v, l, sub, c) in zip((c1, c2, c3), stats):
        col.html(f'<div class="stat"><div class="v" style="color:{c}">{v}</div><div class="l">{l}</div>'
                 f'<div class="sub">{sub}</div></div>')

    # ---- model comparison, ranked by the immigration-aware score
    with st.container(border=True, key="chart_models"):
        chart_title("Test-Set Average Macro F1 By Model",
                    "Ranked by the + Immigration Experiences score. Each bar averages the three outcomes; "
                    "error bars show ± 1 SD over 5 training seeds.")
        model_sort = (overall[overall.features == "immigration"]
                      .sort_values("macro_f1_mean", ascending=False).Model.tolist())
        chart_df = overall.assign(lo=overall.macro_f1_mean - overall.macro_f1_std,
                                  hi=overall.macro_f1_mean + overall.macro_f1_std)
        base = alt.Chart(chart_df).encode(
            x=alt.X("Model:N", sort=model_sort, title="Model", axis=alt.Axis(labelAngle=0, labelLimit=240)),
            xOffset=alt.XOffset("Feature Set:N", sort=list(FS_LABEL.values())),
        )
        bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width={"band": 0.9}).encode(
            y=alt.Y("macro_f1_mean:Q", title="Average Macro F1", scale=alt.Scale(domain=[0, 0.45])),
            color=alt.Color("Feature Set:N", scale=alt.Scale(domain=list(FS_LABEL.values()), range=[TEAL, ROSA]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Feature Set:N"),
                     alt.Tooltip("macro_f1_mean:Q", title="Average Macro F1", format=".3f"),
                     alt.Tooltip("macro_f1_std:Q", title="SD Over Seeds", format=".3f")],
        )
        labels = base.mark_text(dy=-12, fontSize=11, color=INK).encode(
            y="hi:Q", text=alt.Text("macro_f1_mean:Q", format=".3f"))
        err = base.mark_errorbar(color=INK, thickness=1.5).encode(y=alt.Y("lo:Q", title="Average Macro F1"), y2="hi:Q")
        st.altair_chart(style_chart((bars + err + labels).properties(height=340)), width="stretch")

    g1, g2 = st.columns(2, gap="medium")
    with g1.container(border=True, key="chart_groups"):
        chart_title("Test-Set Gain From Immigration Experiences By Race / Ethnicity",
                    "Increase in average Macro F1 when immigration experiences are added (model used in app), "
                    "ranked from largest to smallest.")
        by_group = pd.DataFrame(res["by_group"])
        mt_g = by_group[by_group.model == "Multi-task ensemble"].set_index("features")
        gain = (mt_g.loc["immigration", GROUPS] - mt_g.loc["demographic", GROUPS]).rename("Gain").reset_index()
        gain.columns = ["Race / Ethnicity", "Gain"]
        gbase = alt.Chart(gain).encode(
            x=alt.X("Race / Ethnicity:N", sort="-y", title="Race / Ethnicity", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Gain:Q", title="Gain In Average Macro F1"),
        )
        gbars = gbase.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=ROSA).encode(
            tooltip=[alt.Tooltip("Race / Ethnicity:N"), alt.Tooltip("Gain:Q", format="+.3f")])
        gtext = gbase.mark_text(dy=-8, fontSize=11, color=INK).encode(text=alt.Text("Gain:Q", format="+.3f"))
        st.altair_chart(style_chart((gbars + gtext).properties(height=300)), width="stretch")

    with g2.container(border=True, key="chart_inputs"):
        chart_title("Which Inputs Matter Most (Test Set)",
                    "How much average Macro F1 drops when one input is randomly shuffled "
                    "(+ Immigration Experiences model). Bigger drop = more important.")
        imp = res["deployed"]["immigration"]["permutation_importance"]
        imp_df = pd.DataFrame([{"Input": k.title(), "Importance": v["mean"],
                                "Input Type": "Immigration" if k in IMM_INPUTS else "Demographic"}
                               for k, v in imp.items()])
        ichart = alt.Chart(imp_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("Input:N", sort="-x", title="Input",
                    axis=alt.Axis(labelLimit=260, labelOverlap=False, titleAngle=0, titleAlign="right",
                                  titleBaseline="bottom", titleX=-6, titleY=-6)),
            x=alt.X("Importance:Q", title=["Drop In Average Macro F1", "When Shuffled"]),
            color=alt.Color("Input Type:N", scale=alt.Scale(domain=["Demographic", "Immigration"],
                                                            range=[TEAL, ROSA]),
                            legend=alt.Legend(orient="top", title="Input Type")),
            tooltip=[alt.Tooltip("Input:N"), alt.Tooltip("Importance:Q", format=".4f")],
        ).properties(height=300)
        st.altair_chart(style_chart(ichart), width="stretch")

    with st.expander("All Test-Set Metrics By Model, Feature Set, And Outcome"):
        by_out = pd.DataFrame(res["by_outcome"])
        cols = ["model", "features", "outcome", "macro_f1_mean", "macro_f1_std", "weighted_f1_mean",
                "balanced_acc_mean", "precision_mean", "recall_mean", "roc_auc_mean", "mae_mean"]
        tbl = by_out[cols].copy()
        tbl["model"] = tbl.model.map(MODEL_LABEL)
        tbl["features"] = tbl.features.map(FS_LABEL)
        tbl["outcome"] = tbl.outcome.map(OUTCOME_LABEL)
        tbl.columns = ["Model", "Feature Set", "Outcome", "Macro F1", "SD", "Weighted F1", "Balanced Accuracy",
                       "Precision", "Recall", "ROC-AUC", "MAE"]
        st.dataframe(tbl.round(3), hide_index=True, width="stretch")
