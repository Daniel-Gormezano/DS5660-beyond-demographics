::: center
**Beyond Demographics:\
Multi-task Learning for Psychological Distress**

Daniel Gormezano^1^

^1^Vanderbilt University, Dr. Huo, DS5660: Deep Learning
:::

**Application:** <https://beyond-demographics.streamlit.app/>\
**Code and Saved Results:**
[github.com/Daniel-Gormezano/DS5660-beyond-demographics](https://github.com/Daniel-Gormezano/DS5660-beyond-demographics)

# Abstract {#abstract .unnumbered}

This project explores whether immigration-related stressors improve the
prediction of psychological distress beyond demographic information. I
compared two feature sets using 14,826 respondents from the 2020
Collaborative Multi-racial Post-election Survey (CMPS). Multiple models,
including logistic regression, random forest, separate neural networks,
a multi-task neural network, and a five-network multi-task ensemble,
were used to predict four response levels (0, 1, 2, and 3, based on
symptom frequency) for each of three survey items: depressive symptoms,
anxiety, and uncertainty stress. On a held-out test set of 2,224
respondents, adding immigration-related variables increased the
multi-task ensemble's test-set average Macro F1 from 0.3393 to 0.3762,
an observed gain of 0.0369 (95% paired-bootstrap interval,
0.0227--0.0517). Every model comparison improved, although the ensemble
and random forest performed almost identically. Improvements were
observed across all racial and ethnic groups, with gains of 0.0551 for
Black, 0.0522 for White, 0.0398 for Asian, and 0.0362 for Latino
respondents. Permutation importance identified age as the strongest
predictor (Macro F1 decrease of 0.0673), followed by immigration-policy
stress (0.0218), income (0.0120), and deportation worry (0.0113). These
findings suggest immigration-related stressors provide meaningful
predictive information beyond traditional demographic characteristics,
supporting their consideration in future research on psychological
distress.

# Problem and Purpose

My work with the Organization for Latino Health Advocacy motivated this
question [@advocacy]. Demographic information describes a person's
background, but it does not directly capture worry about deportation or
stress related to immigration policy. Prior research connects
deportation threat to psychological distress among Latino citizens and
noncitizens [@johnson]. Research on state policy climates also finds an
association with poor mental-health days, although not with every
distress measure [@hatzenbuehler].

I tested whether three reported immigration experiences add predictive
information to six demographic variables, and whether the observed
improvement differs across racial and ethnic groups. The application is
intended for public health researchers, social scientists, and advocacy
organizations exploring these associations. Its outputs estimate survey
response patterns and not future illness or clinical diagnoses.

# Data and Outcomes

## Analytic Sample

I used the 2020 CMPS public-use dataset, ICPSR 39096, version 1 [@cmps].
Although associated with the 2020 election, data were collected from
April through August 2021. Of the 17,545 original records, 14,826
respondents met the study's inclusion criteria for the four
racial/ethnic groups: 3,956 Black, 3,942 Latino, 3,928 Asian, and 3,000
White respondents. The analysis included respondents regardless of
whether they voted in the 2020 election.

::: tabularx
\@L1.14inL1.45inY@ Outcome & CMPS Variable & Item Wording, Referring to
the Previous Two Weeks\
Depressive symptoms & `Q411_Q416R2` & Feeling down, depressed, or
hopeless\
Anxiety & `Q411_Q416R3` & Feeling nervous, anxious, or on edge\
Uncertainty stress & `Q411_Q416R6` & Uncertainty makes me uneasy,
anxious or stressed\
:::

I recoded original responses 1, 2, and 3 to levels 0, 1, and 2.
Responses 4 and 5 merge into level 3. The four labels are *Not at all*,
*Several days*, *More than half the days*, and *Nearly every day*. These
are individual survey items. The sample is imbalanced: 53.7%, 47.5%, and
43.9% report *Not at all* for depressive symptoms, anxiety, and
uncertainty, respectively.

## Inputs and Preprocessing

The demographics-only model uses six original variables. The expanded
model adds deportation worry, immigration-policy stress, and reported
unfair treatment by immigration officers. Nine variables become 18
numerical input columns because one-hot encoding expands some variables
and income includes a missingness indicator
(Table [\[tab:features\]](#tab:features){reference-type="ref"
reference="tab:features"}).

::: tabularx
\@L1.22inL1.11inY r@ Variable & CMPS Source & Encoding & Columns\
Age & `S5` & 2021 minus birth year, clipped to 18--90 & 1\
Gender & `S3B` & Three one-hot categories & 3\
Education & `S13` & Six ordered categories, coded 0--5 & 1\
Household income & `Q813` & Twelve ordered brackets, coded 0--11, plus
missingness indicator & 2\
Political party & `Q21` & Four one-hot categories & 4\
Race/ethnicity & `MAIN_ETH_QUOTA` & Four one-hot categories & 4\
& **15**\
Deportation worry & `Q492` & Ordered 0--3, from no worry to a lot & 1\
Policy stress & `Q504R8` & Reported stress from immigration policy,
1--10 & 1\
Officer discrimination & `Q281R3` & Binary indicator, 0 or 1 & 1\
& **18**\
:::

All fitted preprocessing uses the training set only. Age, education,
income, deportation worry, and policy stress are standardized with
training means and standard deviations. Gender, party, and race use
fixed one-hot categories. Binary indicators remain 0/1. Income refusal
occurs in 8.8% of the analytic sample. I imputed missing income with the
training median and retain a separate indicator so an imputed value can
be distinguished from a reported value.

Education and income each remain a single input feature because ordinal
encoding preserves their category order by assigning numerical values,
assuming equal spacing between categories. The neural network can learn
a nonlinear response, but the spacing is still a modeling assumption.
Saved preprocessing parameters ensure that the application applies the
same transformations.

# Model architecture

I used a fully connected multi-task network because the inputs are
tabular and the three targets concern related aspects of distress. The
shared layers can learn patterns useful across outcomes, while separate
output heads allow the targets to differ.

Both selected models have shared layers of 64 and 32 units, with ReLU
activation to introduce nonlinearity and dropout of 0.2 after each layer
to reduce overfitting. Each of three heads has a 32-unit hidden layer,
ReLU, dropout, and four output logits:
$$\underbrace{d\rightarrow64\rightarrow32}_{\text{shared layers}}
\quad\longrightarrow\quad
\underbrace{32\rightarrow32\rightarrow4}_{\text{one head per outcome}},
\qquad d\in\{15,18\}.$$ For a batch of $N$ respondents, the input is
$N\times d$ and each head returns $N\times4$ logits. Softmax converts
the logits to probabilities. The demographics-only network has 6,668
trainable parameters and the expanded network has 6,860. Each deployed
ensemble averages five networks with the same selected architecture.

# Training and Experimental Methodology

## Split, Optimization, and Comparisons

I used one fixed 70/15/15 split: 10,378 training respondents, 2,224
validation respondents, and 2,224 test respondents. The split uses seed
2026 and stratifies jointly by race/ethnicity and depressive-response
level. All models use the same split. Hyperparameters, early stopping,
and probability-calibration parameters use validation data.

To address imbalance, the loss adds three class-weighted cross-entropy
terms. For task $t$, class $k$ has weight
$w_{tk}=n_{\mathrm{train}}/(4n_{tk})$. The batch loss matches the
weighted-mean reduction in the implementation. These are
inverse-frequency *class weights*: $$\mathcal{L}(B)=\sum_{t=1}^{3}
\frac{-\sum_{i\in B}w_{t,y_{it}}\log p_t(y_{it}\mid x_i)}
     {\sum_{i\in B}w_{t,y_{it}}}.$$

::: tabularx
\@L1.70inY@ Setting & Implementation\
Optimizer / batch & AdamW; batch size 256; weight decay $10^{-4}$\
Stopping & At most 200 epochs; patience 15; restore the checkpoint with
the best mean validation Macro F1\
Neural search & 12 combinations: widths (64,32), (128,64), or (256,128);
dropout 0.2 or 0.4; learning rate 0.001 or 0.003\
Selected learning rates & Demographics only: 0.001; expanded features:
0.003\
Repeated fits & Seed labels 0, 1, 2, 3, and 4 for stochastic models\
Recorded runtime & 11.7 minutes for the experimental pipeline on CPU;
two PyTorch threads\
:::

The baselines test whether the neural approach adds value over simpler
alternatives. Multinomial logistic regression uses balanced class
weights and tunes $C\in\{0.01,0.1,1,10\}$. Random forest uses 300 trees,
balanced-subsample weights, maximum depths of 8, 16, or unlimited, and
minimum leaf sizes of 5 or 20. A separate-network baseline fits one MLP
per outcome using the same neural search space. Each feature set is
tuned independently. The multi-task ensemble averages the five fitted
multi-task networks' probabilities.

## Evaluation and Probability Calibration

The primary metric is Macro F1: F1 is averaged equally across the four
classes within each outcome, then equally across the three outcomes.
This gives rare response levels weight that overall accuracy would not.
Additional metrics include macro precision, macro recall, and the
percentage of predictions that differ from the observed response by one
level or at least two levels.

A paired bootstrap resamples the 2,224 test respondents 2,000 times,
applying the same sampled indices to both feature sets. The 2.5th and
97.5th percentiles give the confidence interval for the gain. Models
stay fixed, so these intervals do not capture variation from retraining
or changing the split. Permutation importance shuffles each original
variable 30 times, keeping its encoded columns together.

Classification and displayed probabilities use different postprocessing.
The F1 comparisons use the original class-weighted softmax outputs. For
display, the application subtracts log class weights from each network's
logits and applies a temperature fitted by validation negative
log-likelihood: $$\widetilde p_t(x)=\frac{1}{5}\sum_{s=1}^{5}
\operatorname{softmax}\!\left(\frac{z_{s,t}(x)-\log w_t}{T_{s,t}}\right).$$
Predicted probabilities were calibrated using temperatures selected on
the validation set and adjusted for class weighting during training.
Because these adjustments can change the predicted response category,
the reported Macro F1 scores reflect model performance before
calibration.

# Results and Interpretation

## Model Comparison

Adding immigration variables improved average test Macro F1 for every
model family (Table [\[tab:models\]](#tab:models){reference-type="ref"
reference="tab:models"}). The multi-task ensemble increased from 0.3393
to 0.3762. Its observed gain is 0.0369, with a paired-bootstrap 95%
interval of 0.0227--0.0517. This supports additional predictive
information in the three immigration variables under this experimental
setup.

::: tabularx
\@Y r r r@ Model & Demographics & + Immigration & Gain\
Logistic regression & 0.3452 & 0.3545 & +0.0094\
Random forest & $0.3465\pm0.0019$ & $0.3753\pm0.0025$ & +0.0288\
Separate neural nets & $0.3435\pm0.0042$ & $0.3594\pm0.0019$ & +0.0159\
Multi-task neural net & $0.3366\pm0.0047$ & $0.3715\pm0.0071$ & +0.0349\
**Multi-task ensemble** & **0.3393** & **0.3762** & **+0.0369**\
:::

With immigration variables, the shared network scored 0.3715 versus
0.3594 for separate networks, consistent with useful sharing across
outcomes. This does not establish that multi-task learning always
performs better. The ensemble exceeded random forest by only 0.0009.

## Outcomes, Groups, and Useful Inputs

The observed ensemble gain was positive for each outcome
(Table [\[tab:taskresults\]](#tab:taskresults){reference-type="ref"
reference="tab:taskresults"}). The confidence intervals exclude zero for
all three. The central finding is consistent across the targets.

::: tabularx
\@Y r r r r@ Outcome & Demographics & + Immigration & Gain & 95%
Interval\
Depressive symptoms & 0.3453 & 0.3757 & +0.0304 & \[0.0107, 0.0510\]\
Anxiety & 0.3346 & 0.3741 & +0.0395 & \[0.0200, 0.0585\]\
Uncertainty stress & 0.3382 & 0.3789 & +0.0407 & \[0.0190, 0.0621\]\
Three-outcome average & 0.3393 & 0.3762 & +0.0369 & \[0.0227, 0.0517\]\
:::

<figure id="fig:groups" data-latex-placement="!htbp">
<img src="./app_groups_importance.png" />
<figcaption>Subgroup gains and permutation importance from the
application’s results view. Left: change in average Macro F1 within each
racial/ethnic group. Right: mean decrease in Macro F1 after shuffling an
original input variable.</figcaption>
</figure>

Observed gains were +0.0551 in average Macro F1 performance for Black
respondents, +0.0522 for White respondents, +0.0398 for Asian
respondents, and +0.0362 for Latino respondents (Figure 1). Improvement
in all four groups does not establish equal performance or fairness. For
example, the expanded ensemble's average Macro F1 was 0.4058 among Black
respondents and 0.3193 among White respondents.

Permutation importance was largest for age (0.0673), followed by policy
stress (0.0218), income (0.0120), and deportation worry (0.0113) (Figure
2). Officer discrimination contributed a smaller decrease of 0.0020.
Correlated inputs can share information, reducing their individual
importance.

# Failure Analysis and Limitations

## Errors in Response-Level Classification

The main failure is distinguishing intermediate response levels. In the
expanded ensemble, recall for *Several days* was only 18.6% for
depressive symptoms, 18.6% for anxiety, and 23.0% for uncertainty
(Figure 2, Table 6). Recall for *Not at all* was 64.2%, 62.2%, and
57.8%, respectively. Class weighting helps keep rare categories in the
objective, but it does not solve the overlap between response levels.

<figure id="fig:confusion" data-latex-placement="!htbp">
<img src="./confusion_matrices.png" />
<figcaption>Test-set confusion matrices for the expanded ensemble before
display calibration. Rows are actual responses and columns are
predictions. “Most Days” and “Daily” refer to <em>More than half the
days</em> and the merged <em>Nearly every day</em> category. Each panel
contains 2,224 respondents.</figcaption>
</figure>

::: samepage
Across the three targets, 59.0% of classification errors were between
adjacent levels, but 41.0% were farther apart (Table 6). Errors of at
least two response levels affected 22.6% of all test respondents for
depressive symptoms, 23.2% for anxiety, and 21.4% for uncertainty. The
improved F1 therefore still leaves substantial individual-level error.
:::

::: tabularx
\@Y r r r r r r@ Outcome & & & & & &\

Depressive symptoms & 37.4% & 40.5% & 64.2% & 18.6% & 29.6% & 22.6%\
Anxiety & 37.7% & 40.1% & 62.2% & 18.6% & 31.8% & 23.2%\
Uncertainty stress & 37.7% & 39.7% & 57.8% & 23.0% & 35.4% & 21.4%\
:::

## Calibration of the Probabilities Shown in the App

The saved calibration check evaluates the combined event $Y\geq2$,
meaning symptoms on more than half the days or more often. Mean
predicted probabilities are close to observed rates
(Table [\[tab:calibration\]](#tab:calibration){reference-type="ref"
reference="tab:calibration"}). The expected calibration error (ECE) uses
ten probability bins and weights each bin's absolute
prediction--observation gap by its sample size.

::: tabularx
\@Y r r r@ Outcome & Mean Predicted & Observed Rate & ECE\
Depressive symptoms & 22.4% & 22.6% & 0.0222\
Anxiety & 25.1% & 25.9% & 0.0258\
Uncertainty stress & 27.1% & 27.7% & 0.0360\
:::

This checks one aggregated event, not all four classes or subgroup
calibration. The demographics-only model had lower ECE for every
outcome, so adding immigration variables did not improve this
calibration measure.

## Scope and Reproducibility Limits

- **Association and Timing.** Cross-sectional self-reports do not
  establish causation. Policy stress and distress may overlap
  conceptually or share reporting tendencies. Changing an app input is
  not an intervention.

- **Generalization.** Results apply to this sample and fixed split. No
  survey weighting, external validation, or later-wave test was
  performed, and population representativeness and performance elsewhere
  are unverified.

- **Exact Retraining.** Saved checkpoints support repeatable inference.
  Exact retraining is not guaranteed: PyTorch is seeded after model
  construction, and dependency versions are not fully pinned.

# Deployment and Inference

The Streamlit Community Cloud app loads five PyTorch networks per
feature set. It also loads `models/preprocess.json` and
`models/model_card.json` for preprocessing, architecture, and
calibration settings. Resource caching avoids reloading these files on
each interaction.

Input widgets enforce supported ranges and categories. The saved
preprocessor converts each profile into 15- and 18-column arrays. Both
ensembles run on CPU in evaluation mode with gradients disabled. Each
outcome card shows four calibrated probabilities and the summed
probability of the two highest-frequency categories. Predictions come
from trained weights, and the Model Results tab displays saved test
metrics from `results/results.json`.

The public app loaded and displayed predictions and results during
report preparation. Preprocessing checks confirmed correct dimensions,
finite values with missing income, and unchanged demographic inputs when
only immigration responses change. Caching and small networks reduce
repeated work.

::: samepage
**Reproduction.** Obtain the CMPS data from ICPSR and save the file as
`data/2020_data.tsv`. Run these commands from the repository root:

> pip install -r requirements-train.txt\
> python -m src.data\
> python -m src.train\
> python -m src.figures
:::

::: samepage
To run the saved models without retraining, use
`pip install -r requirements.txt` followed by
`streamlit run streamlit_app.py`. Raw respondent data are omitted from
the deployment.
:::

# Individual Technical Contribution

I worked by myself on this project. I defined the research question and
the demographic-versus-expanded comparison, selected the CMPS variables,
and directed the implementation and evaluation. My technical
contribution included the preprocessing pipeline, the multi-task model
and baseline comparisons, validation-based tuning, ensemble
construction, test-set evaluation, and deployment. These components are
identifiable in the repository [@repo]:

::: tabularx
\@L1.28inY@ Component & Implementation and Responsibility\
Data and Inputs & `src/data.py`, `src/features.py`: Sample definition,
item recoding, split, and shared preprocessing\
Models and Experiments & `src/models.py`, `src/train.py`: Shared layers
and task heads, baselines, optimization, calibration, and evaluation\
Results and Deployment & `results/`, `streamlit_app.py`: saved evidence,
real inference, and side-by-side interpretation\
:::

I used Claude to draft code and presentation material, assist with
variable mapping, and help debug. I selected the design and metrics, ran
and checked the work, and directed revisions. ChatGPT also assisted with
organizing this report and checking the saved code and results.

# Conclusion

Immigration-related variables improved prediction across the tested
model families and all four observed subgroup comparisons. The main
contribution is a reproducible comparison and a working application that
makes those differences inspectable. Moderate predictive performance,
weak intermediate-class recall, and the absence of external validation
limit individual use. The application is a research demonstration and
not a clinical screening instrument; however, it does highlight the need
for future research to examine immigration-related stressors alongside
traditional demographic factors when studying psychological distress.

::: thebibliography
9 Organization for Latino Health Advocacy (OLHA). "Organization for
Latino Health Advocacy." *Organization for Latino Health Advocacy*,
<https://latinohealthadvocacy.org/>. Accessed 14 Nov. 2026. Johnson AL,
Levesque C, Lewis NA Jr, Asad AL. Deportation threat predicts Latino US
citizens and noncitizens' psychological distress, 2011 to 2018. *Proc
Natl Acad Sci U S A*. 2024;121(9):e2306554121.
[doi:10.1073/pnas.2306554121](https://doi.org/10.1073/pnas.2306554121).
Hatzenbuehler ML, Prins SJ, Flake M, et al. Immigration policies and
mental health morbidity among Latinos: A state-level analysis. *Soc Sci
Med*. 2017;174:169--178.
[doi:10.1016/j.socscimed.2016.11.040](https://doi.org/10.1016/j.socscimed.2016.11.040).
Frasure L, Wong J, Vargas E, Barreto M. *Collaborative Multi-racial
Post-election Survey (CMPS), United States, 2020*. Version 1. ICPSR
39096. Inter-university Consortium for Political and Social Research;
2024.
[doi:10.3886/ICPSR39096.v1](https://doi.org/10.3886/ICPSR39096.v1).
Gormezano D. *Beyond Demographics*. DS 5660 project repository. Commit
5daf372. Accessed November 12, 2026.
<https://github.com/Daniel-Gormezano/DS5660-beyond-demographics>.
:::
