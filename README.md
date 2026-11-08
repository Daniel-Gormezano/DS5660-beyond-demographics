# Beyond Demographics

**Multi-task deep learning for psychological distress: do immigration-related stressors add predictive signal beyond demographics?**

DS 5660 Deep Learning · Final project · Daniel Gormezano

An interactive Streamlit app runs two trained PyTorch multi-task networks side by side:

* **Demographics only:** age, gender, education, household income, political party, race/ethnicity
* **+ Immigration experiences:** the same inputs plus worry about deportation, stress from U.S. immigration policy, and discrimination by immigration officers

Each network predicts three outcomes from the 2020 CMPS (the "how often over the last 2 weeks" items):

| Outcome | CMPS item |
|---|---|
| Depressive symptoms | Q411_Q416r2 · Feeling down, depressed, or hopeless |
| Anxiety | Q411_Q416r3 · Feeling nervous, anxious, or on edge |
| Uncertainty stress | Q411_Q416r6 · Uncertainty makes me uneasy, anxious or stressed |

Each answer is recoded to the standard 0–3 PHQ/GAD scale: *Not at all*, *Several days*, *More than half the days*, *Nearly every day* (the survey's "Almost every day" and "Every day" are merged).

---

## Repository layout

```
streamlit_app.py        # the app: input -> preprocessing -> PyTorch inference -> output
app_theme.py            # papel picado / talavera theme (pure CSS + SVG)
src/data.py             # variable selection, recoding, stratified 70/15/15 split
src/features.py         # preprocessing (fit on train only, reused by the app)
src/models.py           # MultiTaskMLP and SingleTaskMLP (PyTorch)
src/train.py            # all experiments: baselines, separate nets, multi-task net
src/figures.py          # report figures
models/                 # deployed weights + preprocessing + calibration (no raw data)
results/                # metrics (CSV/JSON) and figures
requirements.txt        # app dependencies (CPU-only PyTorch)
requirements-train.txt  # training dependencies
```

## Reproduce the results

1. Download the CMPS 2020 public-use data from ICPSR (study 39096) and save the tab-delimited file as `data/2020_data.tsv`.
2. Run:

```bash
pip install -r requirements-train.txt
python -m src.data          # -> data/processed/{train,val,test}.csv
python -m src.train         # -> models/, results/   (about 20 minutes on a laptop CPU)
python -m src.figures       # -> results/fig_*.png
```

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app only needs `models/` and `results/`; it never loads the raw survey data.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **GitHub** repository. `data/` is git-ignored: the raw CMPS file must not be redistributed under ICPSR's terms of use.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app**.
3. Pick the repository and branch, and set the main file to `streamlit_app.py`.
4. Click **Deploy**. The first build installs CPU-only PyTorch and takes a few minutes. You get a public `*.streamlit.app` URL.

## Method summary

* **Sample:** the CMPS core (baseline) sample of White, Latino, Black and Asian respondents (`BASELINE == 1`, `MAIN_ETH_QUOTA` 1–4), N = 14,826.
* **Split:** fixed 70/15/15, stratified by race/ethnicity × depressive-symptom level.
* **Preprocessing:** numeric inputs standardized with training-set statistics; categories one-hot encoded; "prefer not to say" income gets the training median plus a missing-value flag.
* **Models:** multinomial logistic regression, random forest, separate single-task MLPs, and the multi-task MLP (shared fully connected layers → ReLU → dropout, then one head per outcome). Each is trained on both feature sets. The deployed model is an ensemble of the five seed-trained multi-task networks; averaging their probabilities makes predictions more stable.
* **Training:** AdamW, mini-batch 256, class-weighted cross-entropy (summed over the three heads for the multi-task model), and early stopping on validation Macro F1.
* **Tuning:** hyperparameters are chosen on the validation set only. The final configuration is re-trained with 5 seeds, and results are reported as mean ± SD on the test set.
* **Metrics:** Macro F1 (primary), weighted F1, balanced accuracy, macro precision and recall, one-vs-rest ROC-AUC, mean absolute error on the ordered levels, confusion matrices, and per-group Macro F1. A paired bootstrap gives a confidence interval for the gain from immigration features, and permutation importance shows which inputs matter.
* **App probabilities:** class weighting helps Macro F1 but inflates rare-level probabilities. Before display, the app removes the class weights from the logits (prior correction) and applies one temperature per outcome fitted on the validation set. Calibration is checked on the test set (see `results/results.json → deployed → calibration_check_test`).

## Results (test set, N = 2,224)

Macro F1 averaged over the three outcomes (mean ± SD over 5 seeds; logistic regression and the ensemble are deterministic once trained).

| Model | Demographics only | + Immigration experiences | Gain |
|---|---|---|---|
| Logistic regression | 0.345 | 0.354 | +0.009 |
| Random forest | 0.346 ± 0.002 | 0.375 ± 0.003 | +0.029 |
| Separate neural nets | 0.344 ± 0.004 | 0.359 ± 0.002 | +0.016 |
| Multi-task neural net | 0.337 ± 0.005 | 0.371 ± 0.007 | +0.035 |
| **Multi-task ensemble (deployed)** | 0.339 | 0.376 | +0.037 |

* **Immigration experiences add signal for every model family.** For the deployed ensemble the gain is +0.037 Macro F1 (paired bootstrap 95% CI +0.023 to +0.052). It is significant for each outcome: depressive +0.031, anxiety +0.039, uncertainty +0.041.
* **Every group benefits:** White +0.052, Latino +0.036, Black +0.055, Asian +0.040.
* **Multi-task vs. separate networks:** with immigration features, one shared network (0.371) beats three separate networks (0.359). Ensembling five seeds (0.376) edges past the random forest (0.375), the strongest baseline.
* **Most important inputs** (permutation importance): Age (0.067), Immigration-policy stress (0.022), Income (0.012).
* **Calibration of the app's percentages** (test set, P(most days or more)): depressive predicted 22% vs observed 23% (ECE 0.022); anxiety predicted 25% vs observed 26% (ECE 0.026); uncertainty predicted 27% vs observed 28% (ECE 0.036).

Outcome distribution in the analytic sample (Not at all / Several days / More than half / Nearly every day): depressive 53.7%/23.7%/12.4%/10.3%; anxiety 47.5%/27.1%/13.1%/12.3%; uncertainty 43.9%/28.3%/14.3%/13.4%.

### Limitations

* Predicting individual distress from survey answers is hard. Macro F1 near 0.38 is well above chance (0.25) but far from diagnostic, and the app is a research tool, not a screening instrument.
* The results are associations in cross-sectional survey data, not causal effects.
* Survey weights are not used in training, so performance reflects the CMPS sample rather than the U.S. population.
* Most errors are confusions between neighboring levels (see `results/fig_confusion_matrices.png`). The middle levels are the hardest to predict.

### Figures

`results/fig_model_comparison.png`, `fig_multitask_by_outcome.png`, `fig_multitask_by_group.png`, `fig_confusion_matrices.png`, `fig_permutation_importance.png`, and `fig_training_curves.png`.
