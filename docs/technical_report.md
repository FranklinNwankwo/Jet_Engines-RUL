# Technical report

The full methods, results and decisions behind [`README.md`](../README.md). The README has the summary; this document has the detail. For the phase-by-phase build history see [`architecture_walkthrough.md`](architecture_walkthrough.md); for the original spec and how it changed, see [`project_brief.md`](project_brief.md).

## Table of contents

1. [The problem, and why it isn't regression](#1-the-problem-and-why-it-isnt-regression)
2. [Feature engineering](#2-feature-engineering)
3. [Modeling: a ladder, not a single model](#3-modeling-a-ladder-not-a-single-model)
4. [Evaluation: not RMSE](#4-evaluation-not-rmse)
5. [Cross-validation: does the ranking hold?](#5-cross-validation-does-the-ranking-hold)
6. [Does survival analysis actually beat regression? Tested, with a mixed answer](#6-does-survival-analysis-actually-beat-regression-tested-with-a-mixed-answer)
7. [Generalization: FD002/FD003/FD004](#7-generalization-fd002fd003fd004)
8. [Hyperparameter tuning: a modest win, with weaker CV support](#8-hyperparameter-tuning-a-modest-win-with-weaker-cv-support)
9. [Interval calibration: two subgroup checks, at opposite ends](#9-interval-calibration-two-subgroup-checks-at-opposite-ends)
10. [Deployment: a FastAPI service + dashboard, live on Streamlit Community Cloud](#10-deployment-a-fastapi-service--dashboard-live-on-streamlit-community-cloud)
11. [Bugs caught along the way](#11-bugs-caught-along-the-way)
12. [Scope notes](#12-scope-notes)

---

## 1. The problem, and why it isn't regression

Predict how many operating cycles a turbofan engine has left, from multivariate sensor telemetry (21 sensors + 3 operating settings per cycle), using NASA's C-MAPSS degradation simulation dataset (FD001 subset: single operating regime, single fault mode).

The naive approach; regress sensor features directly onto observed RUL, is a well-known mistake in this space, and it's not cosmetic. In any real predictive-maintenance deployment, most monitored units are still running when the model is built: their failure time is unknown and lies in the future. This is **right censoring**. A regressor trained only on completed failure trajectories (or one that treats a censored unit's last-known age as its true RUL) is trained on a biased sample, it implicitly excludes "units that would have survived longer than we watched," which systematically biases predictions toward shorter lifetimes.

Survival analysis models the conditional distribution of time-to-failure given "known to have survived to at least time *t*," which is the correct estimand. C-MAPSS's structure maps directly onto this: full run-to-failure trajectories for training (`event=1` for every row, `duration` = the row's RUL capped at 125 cycles — see the label note below), truncated trajectories for test (`event=0`, right-censored at the last observed cycle). That `(duration, event)` construction, not a feature-engineering detail, a first-class data artifact, is what every model in this project is actually fit on. See `src/turbofan_rul/labeling.py`.

One catch: C-MAPSS's training engines all run to failure, so the training data itself contains none of the censoring this argument is about. The argument was therefore tested rather than assumed, by simulating a fleet where some engines haven't failed yet ([notebook 13](#6-does-survival-analysis-actually-beat-regression-tested-with-a-mixed-answer)). Result: survival models clearly beat the naive regression that treats an unfinished engine as failed, but a regression that simply drops unfinished engines turned out to be competitive on this dataset.

---

## 2. Feature engineering

Raw sensor readings are noisy; the degradation signal lives in the *trend*, not the instantaneous value. The pipeline (`src/turbofan_rul/features.py`, built in `notebooks/02_feature_engineering.ipynb`):

- **Sensor selection** (Phase 1 EDA): coefficient-of-variation screening flagged 9 of 21 sensors as near-constant, leaving 12 kept sensors. One caveat, recorded in `notebooks/01_eda.ipynb`: CV divides by the mean, so `sensor_8` and `sensor_13` (both near 2388) get tiny CVs partly because of their large means. Much of the C-MAPSS literature keeps them; dropping them here is a deliberate judgment call, not proof they carry no signal.
- **Rolling statistics**: mean/std/min/max/trend-slope per sensor across 5/15/30-cycle windows, `min_periods=1` so early-life rows aren't discarded.
- **EWMA**: exponentially-weighted moving averages (spans 10/30) as a second smoothing representation alongside the rolling windows. Several EWM features rank among RSF's most important (`notebooks/05`).
- **Piecewise-linear RUL labels**: capped at 125 cycles, the standard C-MAPSS convention — early-life cycles carry no meaningful degradation signal, so an uncapped linear countdown from cycle 1 would force models to fit a trend that isn't there.

**Label note — a censoring alternative was tested and rejected.** With the cap, every training row is labeled `event=1`, including the 8,131 of 20,631 FD001 rows sitting at the cap, so those rows are, strictly, labeled as "fails in exactly 125 cycles" when their true remaining life is longer. The statistically cleaner option is to mark those rows as censored at 125 (`event=0`). `labeling.build_train_survival_frame` supports this via `censor_above_cap=True`, and it was tested: Cox improved slightly, RSF was unchanged, but Weibull AFT broke badly, its FD001 NASA score went from about 1.3×10³ to about 2×10¹⁰ and its 90% interval width from about 100 to 184 cycles, because with the censored rows its fitted distribution puts far more mass on very long lifetimes. Since Weibull supplies the deployed interval, the default stays `censor_above_cap=False` (all rows `event=1`), matching standard C-MAPSS practice. The trade-off is documented in `labeling.py`.

Every engineered column is grouped strictly per-engine; cross-engine leakage in rolling windows is checked directly in `tests/test_features.py`.

---

## 3. Modeling: a ladder, not a single model

Four approaches, each with a stated reason for inclusion:

| Model | Why it's here | Key limitation |
|---|---|---|
| **Cox PH** | Reference standard; interpretable hazard ratios; the Schoenfeld-residual test also gives a diagnostic | Assumes proportional hazards (tested directly) |
| **Weibull AFT** | Wear-out is physically a Weibull process; fully parametric, so RUL quantiles/confidence intervals come directly from the fitted distribution | Fixed parametric form; can't capture arbitrary nonlinear interactions |
| **Random Survival Forest** | Drops both the PH and parametric assumptions; captures nonlinear multi-sensor interactions | Confidence intervals under-cover; predictions structurally capped at the training time ceiling |
| **Deep hazard networks** (stretch) | Tests whether temporal structure can be learned directly from raw sensor windows, with no hand-engineered features at all | Didn't beat feature-engineered RSF with 80 training engines; intervals badly under-cover |

**Cox PH's own diagnostic motivates the next models.** Fitting Cox PH (with engine-clustered robust standard errors, since each engine contributes many correlated rows) and running the Schoenfeld-residual test on its output showed **14 of 25 covariates violate the proportional-hazards assumption** at p<0.05 (`notebooks/03_cox_baseline.ipynb`). That count is if anything overstated — the test treats each row as independent, so its p-values are too small — but even discounted, it's the reason a model that doesn't assume proportional hazards (Weibull AFT, then RSF) was tried next.

**Weibull AFT's derivation was validated.** The survival function, hazard function, and AFT log-likelihood are derived by hand in `notebooks/04_weibull_aft.ipynb`, then checked with a from-scratch MLE fit against `lifelines.WeibullFitter` — agreement to **≤5.4×10⁻⁷ relative error** on both parameters, and to 4.6×10⁻⁷ when the censoring term of the likelihood is exercised too (8,031 censored rows). The wear-out evidence comes from a Weibull fit to the 100 engines' actual lifetimes: shape **ρ = 4.41 (95% CI [3.80, 5.01])**, clearly above 1, so failure risk rises with age — the wear-out signature the physical framing predicts. (The covariate AFT model's own shape estimate, 2.67, is not used as evidence: it depends on the row-level label construction, and its CI treats correlated rows from the same engine as independent, so it is far too narrow.)

**Deep models (Phase 7, stretch).** The discrete-time hazard loss; masked binary cross-entropy, derived from the same survival likelihood as the other models, was implemented from scratch in NumPy with hand-derived backprop, gradient-checked against finite differences, *and* cross-validated against the Keras implementation actually used for training (both give the identical loss on identical inputs). Training is made reproducible (`keras.utils.set_random_seed` plus `tf.config.experimental.enable_op_determinism()` — two runs give identical output), but a single seed is still one draw from the initialization lottery, so a plain LSTM and a Conv1D→LSTM (a known strong C-MAPSS baseline architecture) were compared across 3 seeds each. Result: the plain LSTM had the higher validation concordance on all 3 seeds (mean 0.880 vs. 0.871, a gap 2.5× the larger seed-to-seed std) — a consistent but small edge, from only 3 seeds. On the official test set, with all three seeds of each scored, the two architectures tie (LSTM mean NASA 9.21 vs 9.69 per engine, Conv1D→LSTM mean C-index 0.844 vs 0.839, both within one standard deviation), and **none of the six runs beats RSF on any headline metric**.

---

## 4. Evaluation: not RMSE

RMSE and MAE are reported as secondary metrics — they're symmetric, so they treat a 20-cycle overprediction (an engine fails in service) the same as a 20-cycle underprediction (a part gets replaced a bit early). Those aren't equally bad. `src/turbofan_rul/evaluation.py` implements, from the C-MAPSS literature rather than imported, the metric that encodes that asymmetry directly:

**NASA scoring function**: for prediction error *d* = predicted − true RUL,

```
penalty = exp(d/10) − 1   if d > 0   (late — dangerous)
penalty = exp(−d/13) − 1  if d ≤ 0   (early — merely costly)
```

Both branches are zero at *d*=0 and grow exponentially, but the late branch divides by the smaller constant, it grows faster. A unit test asserts specifically that late errors score worse than early ones of the same magnitude; if the two constants were ever swapped, the metric would silently reward the exact behavior it exists to punish.

Alongside it: concordance index (does the model rank engines by risk correctly, independent of magnitude), late-prediction rate (how *often* a model is dangerously wrong, separate from *how badly*), interval coverage (are the confidence intervals calibrated), and calibration-by-risk-bin (is a model biased specifically for the highest-risk engines, which matters more than being biased on average).

### Results (FD001 test set, 100 engines)

| Model | NASA score | C-index (test) | Late rate | 90% coverage | RMSE | MAE |
|---|---|---|---|---|---|---|
| Cox PH | 1672.5 | 0.821 | 54% | — | 22.01 | 16.91 |
| Weibull AFT | 1288.2 | 0.818 | 48% | 90.0% | 23.14 | 18.27 |
| LSTM discrete hazard (stretch)* | 921.3 ± 67.3 | 0.839 ± 0.005 | 60% | 75.0% | 17.80 ± 0.32 | 12.83 |
| Conv1D→LSTM discrete hazard (stretch)* | 968.8 ± 76.1 | 0.844 ± 0.003 | 57% | 78.0% | 18.24 ± 0.06 | 12.96 |
| **Random Survival Forest** | **462.5** | **0.864** | 65% | 86.0% | **16.27** | **11.99** |

*Deep-model figures are the mean (± standard deviation) of 3 seeds per architecture (42, 1, 7), every run scored on the test set (notebook 07, section 10); training is deterministic, so each seed reproduces exactly. None of the six runs reaches RSF on NASA score, C-index or RMSE: the best runs get C-index 0.848 (Conv1D→LSTM) and NASA 847 (8.47 per engine, LSTM), and even those gaps are about 3× (C-index) and 5× (NASA) the between-seed spread. The seed-42 LSTM, previously reported alone, was its best run on NASA (8.47 vs 9.78 and 9.39 per engine). TensorFlow's results shift slightly across hardware even with deterministic ops; these are the repository's reproduced numbers. On the test set the two architectures tie within one standard deviation.

**Which C-index this is.** The C-index column is computed per engine on the 100 official test engines (each engine's predicted RUL vs. its true RUL) — the same metric used for every cross-model and cross-subset comparison in this README. The notebooks also report a *row-level validation* C-index on the 20 held-out training engines (Cox 0.875, Weibull 0.864, RSF 0.901); that's a different metric on different data, useful within a notebook but not comparable to the test C-index. An earlier version of this README mixed the two (see [bug #7](#11-bugs-caught-along-the-way)).

**This table is the Phase 6 model-selection comparison**: every model trained on the same 80 engines, RSF with `min_samples_leaf=30` (the value used throughout Phases 3–6). The RSF actually deployed is retrained on all 100 engines with a retuned `min_samples_leaf=5` — see [Hyperparameter tuning](#8-hyperparameter-tuning-a-modest-win-with-weaker-cv-support) for the deployed model's numbers.

**Two findings from this table:**

- **RMSE and the NASA score rank Cox and Weibull differently on FD001.** Cox wins on RMSE (22.01 vs. 23.14); Weibull wins on NASA score (1288.2 vs. 1672.5) because its errors sit on the safer side. A model comparison run on RMSE alone would have picked the model that's worse for the maintenance decision. (This disagreement is specific to FD001: on FD004, Cox wins on both — see [Generalization](#7-generalization-fd002fd003fd004).)
- **The best model by NASA score is also the most frequently late.** RSF has both the lowest NASA score *and* the highest late-prediction rate (65%). It's late more often, but by smaller margins — the exponential penalty means many small late errors cost less than a few large ones. Reporting only the aggregate score would have hidden this.

**RSF's two real weaknesses**, and why they shape the deployment design below: intervals under-cover (86% actual vs. 90% nominal), and predictions are structurally capped at the ~125-cycle training time ceiling (a non-parametric survival curve is only defined over the time support it was trained on) — 11 of 100 test engines have true RUL above that cap, and RSF cannot say so.

---

## 5. Cross-validation: does the ranking hold?

Every number in the results table above came from one 80/20 engine-level split. `notebooks/08_cross_validation.ipynb` runs 5-fold `GroupKFold` (by engine, so no engine's cycles ever cross a fold boundary) over Cox PH, Weibull AFT, and RSF, and checks two separate questions rather than conflating them.

**Evaluation design:** every training engine runs to failure, so a held-out engine's last row always has RUL=0 — evaluating there would be degenerate. Instead, each held-out engine gets one random truncation point (30–90% of its observed lifetime, mirroring how NASA actually constructs the official test set) and is scored against its true remaining life from there. See `src/turbofan_rul/cross_validation.py`.

*Deep models (Phase 7) aren't included — their own notebook already established more run-to-run variance than useful to compound across 5 additional refits, and they were never deployment candidates.*

### Does the RSF-leads ranking hold? Yes.

| Model | CV concordance (mean ± std) | CV NASA score per fold, 20 engines (mean ± std) |
|---|---|---|
| Cox PH | 0.798 ± 0.052 | 426.6 ± 298.7 |
| Weibull AFT | 0.803 ± 0.048 | 537.8 ± 778.9 |
| **RSF** | **0.850 ± 0.019** | **233.0 ± 189.2** |

**Concordance:** RSF has the strictly best C-index in 4 of 5 folds. In the other (fold 2), Weibull is best (0.883) and RSF ties Cox (0.878). **NASA score:** RSF has the best mean and is best in 3 of 5 folds; Weibull wins folds 1 and 2. More interesting than the win counts: **RSF's concordance standard deviation is roughly a third of the other two models'.** It isn't just better on average, it's the most *stable* model across different engine splits, something a single split can't show. Weibull's NASA score in particular swings wildly (5.3 to 96.2 per-engine mean across folds) — fold 4 had a small number of badly-late predictions that a single-split evaluation could easily have missed entirely.

**The absolute numbers are a little lower than on the official test set** (RSF 0.850 vs. 0.864, both engine-level C-index) — expected, not a discrepancy: this notebook's random mid-life truncation points are a different task than NASA's specific chosen truncation points, and RSF here uses a coarser `row_stride=5` for fold-loop runtime rather than Phase 5's `3`. The CV numbers answer "how stable is this ranking"; the official-test-set numbers answer "what does the model score on the real benchmark".

---

## 6. Does survival analysis actually beat regression? Tested, with a mixed answer

The whole framing above rests on right censoring, engines still running when the model is built, yet every C-MAPSS training engine runs to failure, so nothing above ever tested it. (The one censoring variant tried earlier, marking rows above the 125-cycle cap as censored, was a different thing and was rejected because it broke Weibull; see the label note.) `notebooks/13_censoring_experiment.ipynb` creates the missing situation and measures it.

**Design.** Administrative censoring, as in a fleet observed up to a fixed date (`cross_validation.sample_administrative_censoring`): each engine's observation window is drawn independently of its lifetime (so censoring is non-informative, the assumption survival models need), and engines whose window ends before failure are censored — 0%, 25%, 50% and 75% of the 100 training engines, with 5 random draws per level. Five approaches are trained on the same censored data: RSF, Cox and Weibull using the censored rows properly, and two random-forest regressions with RSF's exact hyperparameters, one that **drops** unfinished engines and one that **treats censoring as failure**. All are scored on the unchanged official test set. The hypothesis and a decision rule were written into the notebook before it was run.

| Approach (trained with 50% of engines censored) | Test C-index | RMSE | NASA score / engine | Mean error (cycles) |
|---|---|---|---|---|
| **RSF** (survival) | **0.852 ± 0.003** | **17.28 ± 0.29** | 5.75 ± 0.55 | +2.8 |
| Regression: drop unfinished engines | 0.851 ± 0.004 | 17.40 ± 0.50 | **4.89 ± 0.56** | −1.3 |
| Regression: treat censoring as failure | 0.839 ± 0.003 | 23.13 ± 1.09 | 9.79 ± 1.44 | −11.7 |
| Cox PH (survival) | 0.812 ± 0.005 | 22.80 ± 0.40 | 18.62 ± 3.69 | +0.5 |
| Weibull AFT (survival) | 0.814 ± 0.004 | 23.87 ± 0.35 | 15.53 ± 1.48 | −1.6 |

*Mean ± std over 5 censoring draws. Mean error = predicted − true RUL; negative means predicting too little life.*

- **Confirmed: survival beats "treat censoring as failure" decisively.** That regression learns to predict too little life (−11.7 cycles at 50% censoring, −21.7 at 75%) and its RMSE climbs from 16.3 to 23.1 to 31.3 cycles as censoring goes 0% → 50% → 75%. Its low late-prediction rate (29% at 50%) looks safe only because it's systematically early.
- **Not confirmed: a regression that drops unfinished engines is essentially tied with RSF** on C-index and RMSE at every level, and **beats** RSF on the NASA score (4.9 vs 5.7 at 50%; 6.3 vs 9.0 at 75%) because RSF drifts slightly late as censoring rises. The pre-registered rule — RSF beating *both* regressions by more than the noise at 50% — is not met.
- **Why dropping works here:** the 125-cycle cap. Every row more than 125 cycles from failure has the same capped target whatever the engine's total life, so discarding long-lived (likelier-censored) engines loses little target information — the selection bias that makes "train only on completed failures" dangerous in general is largely absorbed by the cap.
- **RSF's real advantage under censoring is stability, not accuracy:** at 75% censoring its test C-index varies about a third as much across draws as the drop-censored regression's (std 0.003 vs 0.011), and it uses every engine instead of discarding up to three quarters of the fleet.
- **Weibull did not blow up** under engine-level censoring, unlike the earlier capped-row variant: these censored rows are short windows, not a mass of rows all censored at exactly 125.

**What this changes.** "Regression is wrong under censoring" is true, and now measured, for the naive version. It is not true on this dataset that survival analysis beats every regression strategy. The case for the survival models here is that they use all the data correctly, are more stable under censoring, and give a calibrated interval (Weibull) that regression doesn't. Limitations: FD001 only, one censoring mechanism, untuned regressors (RSF-matched hyperparameters), the 125-cycle cap, and an uncensored test set; an uncapped target is the obvious next test.

---

## 7. Generalization: FD002/FD003/FD004

C-MAPSS has four subsets varying two independent axes: operating regimes (1 vs. 6) and fault modes (1 vs. 2). Everything above uses FD001 (1 regime, 1 fault mode), the easiest subset. This section checks whether the pipeline holds up on the harder ones.

**A new pipeline capability was needed first.** FD002/FD004's 6 operating regimes shift a sensor's absolute level independent of degradation — measured: `sensor_2`'s raw variance on FD002 is **99.99% explained by which regime an engine is in**, not wear (std 37.3 overall vs. 0.44 within-regime). Without correcting for this, rolling-window features would mostly track regime-switching. `features.py` gained `fit_regime_clusters` (KMeans on the 3 operating settings, `n_clusters=6` — confirmed by within-cluster std collapsing to ~0) and within-cluster normalization, fit on training data only. FD001's pipeline is unchanged (it has a single regime).

**FD002 and FD003 were run as evaluation-only checks** (RSF only, no tuning, no deployment, the same weight as the original Phase 1 EDA, not a second full build), each isolating one complexity axis before FD004 combines both:

| Subset | Regimes | Fault modes | C-index | NASA score (mean/engine) | MAE |
|---|---|---|---|---|---|
| FD001 (baseline) | 1 | 1 | 0.864 | 4.6 | 11.99 |
| FD003 (fault modes only) | 1 | 2 | 0.869 | 8.2 | 13.34 |
| FD002 (regimes only) | 6 | 1 | 0.858 | 39.9 | 19.46 |

FD003 generalizes well, on par with the FD001 baseline (concordance even slightly higher, NASA score and MAE somewhat worse) with zero pipeline changes. FD002 is a clear step down even with the clustering fix working correctly, some of which reflects a much coarser `row_stride` used only for notebook runtime, not a final tuned model.

**FD004 (both axes combined) got the full model ladder** — Cox PH, Weibull AFT, and RSF, each evaluated against FD004's official test set exactly as Phase 6 did for FD001:

| Model | FD001 C-index | FD004 C-index | FD001 NASA mean | FD004 NASA mean |
|---|---|---|---|---|
| Cox PH | 0.821 | 0.811 | 16.7 | 54.6 |
| Weibull AFT | 0.818 | 0.812 | 12.9 | 69.8 |
| **RSF** | **0.864** | **0.849** | **4.6** | **27.5** |

**The ranking holds.** RSF leads on both concordance and NASA score on FD004, as on FD001 — consistent across a single split, 5-fold CV and a harder subset. Every model's absolute score is worse on FD004, as expected for a strictly harder problem evaluated with an untuned (`min_samples_leaf=30`), coarser-strided (`row_stride=15`) RSF — this notebook did not fold in the later hyperparameter tuning or RSF coverage fix, to keep "does the existing recipe generalize" a clean, unconfounded question. One FD001 finding does *not* carry over: on FD004, Cox beats Weibull on both RMSE (34.9 vs. 35.6) and NASA score (54.6 vs. 69.8 per engine), so the RMSE-vs-NASA disagreement above is a property of FD001, not a general rule.

**A bug caught along the way:** `data_loading.py`'s published-statistics validation (built in Phase 1 specifically to catch a corrupted mirror) caught that FD004's train/test engine counts were transposed in this project's own reference table (248/249 instead of the correct 249/248) — a transcription error from when FD002-004 support was first added, not a mirror problem.

---

## 8. Hyperparameter tuning: a modest win, with weaker CV support

Tuned against the same 5-fold CV protocol as Phase 11 — searching and reporting on the same split is how overfitting to a holdout happens invisibly. `notebooks/11_hyperparameter_tuning.ipynb`.

**Cox PH / Weibull AFT penalizers** (`[0.01, 0.1, 0.5, 1.0]`, full 5-fold grid): the project's existing default of **0.1 was already at or near the optimum** for both. Confirmed, not changed.

**RSF** (the deployed model): a search over `min_samples_leaf` and `max_features` found `max_features="sqrt"` clearly better than `"log2"`. A narrower single-fold search found leaf sizes 3, 5 and 8 tied (0.8737), and `min_samples_leaf=5` (the middle of the tie) was taken forward and compared with the original default of 30 across all 5 folds. **The CV evidence is mixed:** leaf=5 has the higher mean concordance (0.855 vs. 0.841) but wins only **2 of 5 folds** — its higher mean comes mostly from one large gain in fold 2 (0.910 vs. 0.851) — and its fold-to-fold std is about 2.5× larger (0.032 vs. 0.013), giving up much of the stability Phase 11 flagged as one of RSF's distinct virtues. (Before the EWM bug fix — [bug #6](#11-bugs-caught-along-the-way) — leaf=5 appeared to win 4 of 5 folds; that apparent consistency came partly from corrupted features.)

**Checked on NASA's official FD001 test set**, like-for-like — both models trained on all 100 training engines with the same `row_stride=3`, only the leaf size changed:

| Metric | leaf=30 | Tuned (leaf=5, deployed) |
|---|---|---|
| C-index (test) | 0.859 | **0.864** |
| NASA score (mean per engine) | 4.51 | **4.04** |
| MAE | 12.04 | **11.32** |
| RMSE | 16.27 | **15.55** |
| Late-prediction rate | 0.62 | **0.57** |
| 90% coverage | 0.86 | **0.87** |

**leaf=5 is better on every metric on the official test set**, though most margins are small. The deployed FD001 model keeps `min_samples_leaf=5` on that basis: the test set favours it across the board; the CV evidence is weaker (higher mean, but 2 of 5 fold wins and more variance); and since the official test set is the one every model in this project is reported on, a small advantage there isn't a fully independent confirmation. FD004 inherits the same value without an independent search (a stated scope gap, not an assumed transfer).

*Correction to an earlier version of this section,* which reported a concordance drop from 0.901 to 0.864 when tuning. That was not a real effect: 0.901 was a row-level validation C-index from a model trained on 80 engines, compared against an engine-level test C-index from a model trained on 100 — two different metrics on different data ([bug #7](#11-bugs-caught-along-the-way)). Measured like-for-like, concordance goes slightly *up*.

**A second, previously undisclosed cost of this decision:** smaller leaves mean deeper, larger trees. The deployed FD001 RSF artifact grew from ~44MB (`leaf=30`) to **~216MB** (`leaf=5`) — a real storage and load-time cost.

### A better point estimate, judged by the NASA score first (notebook 18)

RMSE is the number C-MAPSS papers compare, but the decision metric here is the NASA score, so the question was whether the point estimate could improve on NASA without getting worse elsewhere. **Where the error comes from:** the 11 FD001 test engines with more than 125 cycles left are 11% of engines but about a quarter of RSF's squared error, because RSF can't predict past the cap (it under-predicts them by ~23 cycles); without them its RMSE is 14.1, not 15.6.

**Candidates, fixed in advance:** RSF with caps 115 / 125 (deployed) / 135 / 145; an average of RSF and a gradient-boosted regressor on the capped target (300 iterations, learning rate 0.05, untuned); gradient boosting alone as a reference. **Protocol:** the project's 5-fold cross-validation (notebooks 08 and 11), 5 truncation draws per held-out engine. **Rule:** ≥ 5% better mean NASA per engine, ≥ 4 of 5 folds won, not worse on held-out points with ≤ 150 cycles left, RMSE at most 5% worse.

| Cross-validation (FD001) | NASA / engine | ≤150 cycles left | RMSE | Folds won | Passes |
|---|---|---|---|---|---|
| RSF cap 125 (deployed) | 78.8 | 6.76 | 23.6 | — | baseline |
| RSF cap 135 / 145 | 66.5 / 60.4 | 8.24 / 10.14 | 23.5 / 24.0 | 3 / 3 | no: worse at ≤150 |
| RSF cap 115 | 106.8 | 5.45 | 23.9 | 1 | no |
| **Blend RSF + gradient boosting** | **68.4** | **5.50** | **22.3** | **4** | **yes** |

The ≤150-cycle rule did its job: larger caps won overall only by predicting more life for very young engines, which the exponential early-prediction penalty rewards but which isn't where the model is used. (The first run's code had left gradient boosting alone among the candidates, against the written plan, and picked it; the code was corrected to match the plan, and gradient boosting stays a reference.)

**Official test set (checked once), then the retraining gate:** NASA per engine 3.12 vs 4.05 (−23%), RMSE 14.2 vs 15.6, MAE 10.4 vs 11.3, C-index 0.877 vs 0.864; the late rate rises from 56% to 62% but late errors are smaller (notebook 18, section 3b: engines late by more than 30 cycles 5 → 2, mean late error 11.3 → 9.2 cycles, worst late miss within 25 cycles of failure 21.9 → 16.4). Exported as a challenger, it passed all four gate rules, with interval coverage unchanged (the interval never uses the point estimate), so it is now FD001's deployed recipe (`POINT_MODEL_BY_SUBSET`). It adds a 1.5 MB model. (These are the repository's reproduced numbers. Gradient boosting's fit shifts slightly with numpy/pandas versions: a second environment gave NASA 3.07 and RMSE 14.16 for the same code; the RSF reproduces exactly.)

### The same experiment on FD004: failed the rule, recorded, not deployed (notebook 19)

Same candidates (RSF vs the blend, gradient boosting alone as reference), same protocol, same rule, FD004's deployed settings (leaf 5, `row_stride=15`, regime normalization).

| Cross-validation (FD004) | NASA / point | ≤150 cycles left | RMSE | Folds won (overall) | Folds won (≤150) |
|---|---|---|---|---|---|
| RSF (deployed) | 140,773 | 14.86 | 37.2 | — | — |
| Blend RSF + gradient boosting | 144,695 | **9.60** | **36.0** | 3 (4 required) | **5** |
| Gradient boosting (reference) | 149,831 | 8.41 | 35.5 | — | — |

**The blend failed rules (a) and (b), so FD004 keeps RSF.** It passed (c) and (d). **Why it failed:** FD004 engines live up to 543 cycles, so cross-validation scores points up to 366 cycles from failure, where every model predicts just under the 125-cycle cap and the early penalty grows exponentially: 99.99% of the overall score comes from points with more than 150 cycles left, and 95% from one held-out point (engine #118, 366 cycles left; RSF 119.9, blend 119.5). FD001 shows the same effect more mildly (92% from points beyond 150, longest 241), which is why notebook 18 already had rule (c).

**Official test set, scored once for the record:** NASA per engine 19.4 vs 22.4, RMSE 27.0 vs 28.2, C-index 0.875 vs 0.862, near-failure late errors above 10 cycles 8 vs 12. Better on every metric, but the rule set before the run decides, and rewriting a rule after it failed and after the test set was seen would turn the protocol into a formality. **For future experiments** rules (a) and (b) should score only points with at most 150 cycles left (or use a capped penalty); re-running this one under the corrected rule on the same, already-seen data would not be independent evidence.

---

## 9. Interval calibration: two subgroup checks, at opposite ends

### The long-lived end: calibrated RSF fails above the cap (notebook 12)

Phase 6 found RSF's 90% intervals under-cover (~86-87% actual). `notebooks/12_rsf_interval_calibration.ipynb` fixes this directly with **cross-conformal prediction** — a model-agnostic technique that widens an interval by a single correction, calibrated on out-of-fold predictions, until it hits target coverage, rather than hoping the model's quantiles happen to be well-calibrated probabilities. The resulting coverage is an empirical result to be checked, not a guarantee: the standard finite-sample guarantee of split conformal doesn't strictly carry over to the cross-conformal variant, or to rows that are correlated within an engine.

**Calibration** used the same 5-fold CV infrastructure as Phases 11/13 — each of the 100 training engines scored by a model fit on the other 4 folds (with the deployed recipe: `min_samples_leaf=5`, `row_stride=3`), pooling calibration data without holding any engine out of the final deployed model. Result: a correction of **q = 5.00 cycles**, applied to both interval bounds — identical to the value `export_deployment_artifacts.py` computes for the deployed artifact.

**On the official test set:** coverage lands at almost exactly 90% (0.870 → 0.900), at a cost: intervals widen ~20% (49.6 → 59.6 cycles mean width).

**On aggregate numbers alone, calibrated RSF now looks strictly better than the deployed Weibull interval** — better-covered (0.900 vs. 0.890) *and* 40% narrower (59.6 vs. 98.6 cycles mean width). But does it hold on the subgroup that matters?

| | Coverage on the 89 engines within RSF's training ceiling | Coverage on the 11 engines *above* it |
|---|---|---|
| Calibrated RSF | 97.8% | **27.3%** |
| Weibull (deployed) | — | 90.9% |

**It doesn't.** RSF's survival curve is only defined over its training time support (~125 cycles), a known structural limit from Phase 5. A uniform additive correction can't manufacture support the model never had; it just shifts a hard ceiling from ~125 to ~130. The aggregate 90% coverage is almost entirely carried by the majority of engines within that ceiling, while coverage on the exact engines Weibull exists to cover collapses to worse than a coin flip. This is precisely the kind of subgroup check Phase 6's calibration-by-risk-bin analysis was built to catch, now applied to a new model.

**Decision at the time: Weibull stays the primary deployed interval**, now for a demonstrated reason. The calibrated RSF interval is still useful (tighter, better-calibrated for the non-ceiling-affected majority), so it's exposed as a clearly-labeled **supplementary field** (`rsf_calibrated_interval_supplementary`) in both the API and dashboard — informational, not authoritative. `export_deployment_artifacts.py` computes the correction reproducibly as part of a normal build for every subset it exports (5 additional RSF fits — a deliberate cost for a clean-checkout guarantee). FD004 gets its own correction this way (**q = 36.00 cycles**, reflecting its much harder prediction problem), but unlike FD001's, its resulting coverage on FD004's official test set was only checked later (below).

### The near-failure end: Weibull fails close to failure (notebooks 14–15)

`notebooks/14_interval_coverage_check.ipynb` scored every official test engine through the deployed `inference.predict` (FD001 first, reproducing notebook 12's numbers exactly, then FD004). FD004's calibrated RSF interval reached 87.9% overall, but by widening every interval to 122.6 cycles (Weibull: 98.8), and covered only 55% of the 67 engines above the cap: the same structural failure. Breaking coverage down by true remaining life also found a new problem in the *deployed* interval:

| Engines within 25 cycles of failure | Weibull 90% interval | Calibrated RSF |
|---|---|---|
| FD001 (19 engines) | 58% | 100% |
| FD004 (50 engines) | 54% | 100% |

Every Weibull miss in that group was on the dangerous side: the true remaining life was *below* the lower bound, so the interval promised more life than the engine had. Notebook 12 had missed this because its calibration scored each training engine once, between 30% and 90% of its life, so almost no calibration point was close to failure.

**The fix was selected on training data only** (`notebooks/15_interval_fix.ipynb`, `src/turbofan_rul/interval_calibration.py`). Each training engine was scored at 10 random points between 30% and 99% of its life by fold models that never saw it (5 folds, the deployed recipes): 956 points for FD001, 2,417 for FD004. Six candidates were compared: Weibull alone; calibrated RSF; Weibull with a separate conformal correction per band of the RSF prediction ("Mondrian"); and a hybrid (lower bound = the more cautious of RSF's and Weibull's, upper bound = Weibull's) with no correction, one correction, or per-band corrections. Corrections were cross-fitted (calibrated on half the engines, scored on the other half). The rule: ≥ 88% coverage overall and ≥ 85% within 25 cycles of failure on both subsets, then the narrowest wins. Training data alone reproduced the problem (Weibull: 62% near failure on both subsets) and selected the **hybrid with no correction**, about one cycle wider on average than Weibull.

**Confirmed on the official test sets:**

| | FD001 Weibull | FD001 hybrid | FD004 Weibull | FD004 hybrid |
|---|---|---|---|---|
| Coverage | 89.0% | **97.0%** | 80.6% | **88.7%** |
| Within 25 cycles of failure | 57.9% | **100%** | 54.0% | **94.0%** |
| Above the 125-cycle cap | 90.9% | 90.9% | 73.1% | 73.1% |
| Dangerous misses | 8 | **0** | 23 | **3** |
| Mean width (cycles) | 98.6 | 99.7 | 98.8 | 100.0 |

**Decision: the hybrid is the deployed interval.** No model was retrained: it is computed from the already-exported RSF and Weibull models. The export metadata records it (`"interval": {"base": "hybrid", "mode": "none", ...}`), and `inference.py` serves it by default for bundles exported earlier. The API still returns Weibull's own interval (`weibull_interval_90`) and the calibrated RSF interval for comparison.

Caveats: the test sets had already been used by notebook 14 to find the problem, so the confirmation isn't fully independent, and the selection rule was finalised after a first look at FD001's out-of-fold table (not the test set). FD004 above the cap stays at 73%: those misses are on the safe side, and every candidate shared Weibull's upper bound, so fixing them needs a model of long remaining life, not calibration. That is tested next.

### Training without the cap: measured, not deployed (notebook 16)

Every model is trained on remaining life clipped at 125 cycles, so no interval's upper bound can reach far past it. `notebooks/16_uncapped_upper_bound.ipynb` repeats notebook 15's out-of-fold protocol with the cap removed (`out_of_fold_intervals(cap=...)`), for the point estimate and for the upper bound, with rules fixed before the test sets were used.

**Point estimate: the cap stays.** Uncapped RSF is worse on both subsets out of fold (FD001 C-index 0.864 → 0.852, RMSE 24.4 → 28.0; FD004 0.846 → 0.840, 37.0 → 40.4). It gains above the cap but loses more between 26 and 125 cycles, where most engines are: early-life sensors carry almost no degradation signal.

**Upper bound: the uncapped RSF fixes the long-lived band.** Keeping notebook 15's lower bound and taking the upper bound from an uncapped RSF is the only option that clears 85% in every band on both subsets; an uncapped Weibull upper bound just misses on FD001 (84% above the cap). The leaf-5 forest would add about 1.2 GB of models, so the leaf-30 version (about 200 MB, chosen for size before its results were known) was the one checked on the test sets:

| Test set | Above the cap | Overall | Mean width (cycles) |
|---|---|---|---|
| FD001, deployed → uncapped upper | 91% → 100% | 97.0% → 99.0% | 100 → 127 |
| FD004, deployed → uncapped upper | 73% → 100% | 88.7% → 98.4% | 100 → 160 |

The width lands on engines predicted healthy (FD004 median 128 → 217 cycles); near failure the interval gets slightly tighter (43 → 36).

**Decision: recorded, not deployed.** The misses it fixes are on the safe side; the upper bound barely drives maintenance decisions (which run off how soon an engine could fail); it over-covers (98–99% against a nominal 90%); and it costs a second forest per subset (about 200 MB), more live-demo memory and a longer Docker build. A use that needs a trustworthy upper bound, such as fleet planning for healthy engines, would change that.

---

## 10. Deployment: a FastAPI service + dashboard, live on Streamlit Community Cloud

**Model pairing, driven by the evaluation above:** RSF supplies the point estimate and risk ranking (best NASA score, best concordance; on FD001 averaged with gradient boosting since notebook 18; FD004 keeps RSF alone, notebook 19); **the confidence interval combines RSF and Weibull AFT** (section 9): its lower bound is the more cautious of the two, since Weibull's alone promises too much life near failure, and its upper bound is Weibull's, since RSF can't extrapolate past its training ceiling. Cox PH isn't deployed: RSF beats it on every point-estimate and ranking metric, and Cox produces no natural interval to compete with Weibull's. (Against Weibull alone, Cox is not uniformly worse, on FD001 it has the better test C-index, RMSE and MAE, and on FD004 the better RMSE and NASA score, but neither is the deployed point model.) Applies per-subset: FD001 and FD004 each get their own RSF + Weibull pair.

**In the Docker deployment: two separate services communicating over HTTP.** `api/main.py` (FastAPI) owns every model artifact and all scoring logic; `app/streamlit_app.py` holds no model artifacts and calls the API's `/predict`, `/subsets`, and `/subsets/{subset}/metadata` endpoints over plain HTTP, using `API_BASE_URL` (an environment variable, not a hardcoded address) to find it — `http://localhost:8000` for local development, or `http://api:8000` under Docker Compose's service-name DNS. A direct, measurable consequence: the dashboard image's dependency list (`requirements-app.txt`) doesn't include scikit-learn, scikit-survival, or lifelines at all, and the model code is only imported lazily in embedded mode (below), so in the Docker deployment the dashboard cannot load a model at all.

**The dashboard itself** is themed via `.streamlit/config.toml` (dark "engine console" palette, custom fonts) and built for reading a prediction at a glance: headline cards for predicted RUL, the 90% interval and a colour-coded risk level; a single "where this engine sits" chart placing the point estimate, both intervals and (for test engines) the true RUL against shaded critical/warning/healthy zones; per-sensor small-multiple charts with a rolling-mean trend line; and a CSV upload path with a downloadable example file. Charts use Altair, which ships with Streamlit, so the theme adds no dependencies.

**Live demo ([jetengines-rul.streamlit.app](https://jetengines-rul.streamlit.app/)): one exception.** The public demo runs on Streamlit Community Cloud, which can only host a Streamlit app, not a separate API service. The free alternatives for hosting the API were checked and ruled out: Hugging Face now requires a paid plan for Docker Spaces, Render's free tier (512MB) is too small for the ~340MB of models, and Google Cloud Run needs a billing card. So the dashboard has a second, *embedded* mode: when `API_BASE_URL` is not set and `MODEL_REPO` is, it downloads the exported `fd001/` and `fd004/` model folders from a Hugging Face model repo and calls `turbofan_rul.inference.predict` in-process — the exact function the API calls, so predictions are identical (verified in both modes: FD001 test engine 1 gets 123 cycles [52, 237], FD004's 33 [13, 59]). The Docker Compose deployment is unchanged and still runs two separate services. Both modes load the same ~0.6GB of models, well within Community Cloud's memory limit. Because the models are pickled scikit-learn/scikit-survival/lifelines objects, `app/requirements.txt` (the Community Cloud dependency file) pins those libraries to the exact versions of the environment that exported the models (Python 3.11, selected on Community Cloud too); a mismatch — found in testing: models pickled under pandas 3 fail to load under pandas 2 — makes the app show a clear "library versions don't match" message instead of crashing, and a scikit-learn version mismatch is flagged in the sidebar. Updating the live models means re-exporting, re-uploading both folders to the Hugging Face repo, and keeping the pins in step.

```
turbofan-rul/
├── docs/project_brief.md              # original spec, and what changed from it
├── notebooks/                         # 01 EDA → 12 RSF interval calibration → 13 censoring experiment → 14–15 interval fix → 16 uncapped upper bound → 17 drift → 18–19 point estimate (FD001, FD004)
├── src/turbofan_rul/
│   ├── data_loading.py                # download + validate C-MAPSS (all 4 subsets)
│   ├── labeling.py                    # piecewise RUL, (duration, event) construction
│   ├── features.py                    # rolling/EWMA pipeline + regime clustering/normalization
│   ├── models.py                      # Cox/Weibull/RSF fit + engine-level split
│   ├── deep_survival.py               # from-scratch discrete-hazard MLP + Keras helpers
│   ├── cross_validation.py            # GroupKFold + truncation-point sampling for CV
│   ├── conformal.py                    # cross-conformal RSF interval calibration
│   ├── interval_calibration.py        # out-of-fold interval candidates, Mondrian calibration, the served interval
│   ├── evaluation.py                  # NASA score, C-index, calibration
│   ├── inference.py                   # single scoring function -- the API and the dashboard's embedded mode
│   ├── manifest.py                    # content-derived model versions + provenance + test metrics
│   ├── drift.py                       # input drift monitor (PSI, calibrated thresholds)
│   ├── gate.py                        # champion/challenger retraining gate
│   ├── publish.py                     # upload + tag versions on Hugging Face
│   └── export_deployment_artifacts.py # reproducible artifact build, per subset (see below)
├── api/
│   ├── main.py                        # FastAPI — POST /predict, GET /subsets, GET /subsets/{s}/metadata
│   └── Dockerfile                     # bakes FD001 + FD004 model artifacts in at build time
├── app/
│   ├── streamlit_app.py               # HTTP client of the API; embedded mode for the public demo
│   ├── requirements.txt               # Community Cloud deps (embedded mode, versions pinned)
│   └── Dockerfile                     # lean image, zero ML dependencies (HTTP mode)
├── .streamlit/config.toml             # dashboard theme
├── docker-compose.yml                 # local two-service orchestration
├── requirements-api.txt               # lean, API-only deps (no jupyter/matplotlib/tensorflow)
├── requirements-app.txt               # leaner still -- streamlit + pandas + requests, nothing ML
├── tests/                             # 232 tests across 17 files
└── models/
    ├── fd001/                         # RSF + Weibull + metadata for FD001
    └── fd004/                         # RSF + Weibull + regime-clustering artifacts for FD004
```

```bash
# Local, without Docker (bash; in PowerShell use  $env:PYTHONPATH = "src"  etc.):
pip install -r requirements.txt
export PYTHONPATH=src
python -m turbofan_rul.export_deployment_artifacts --subset FD001   # refits on ALL training engines
python -m turbofan_rul.export_deployment_artifacts --subset FD004   # optional -- FD004 support
uvicorn api.main:app --reload                          # API on :8000
API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py   # dashboard on :8501 (HTTP mode)
MODELS_DIR=models streamlit run app/streamlit_app.py                    # or: no API, models loaded in-app
pytest tests/ -v                                       # 232 tests

# Containerized, both services:
docker compose up --build
# API on :8000, dashboard on :8501, dashboard reaches the API at http://api:8000
# automatically via Compose's service-name DNS -- see docker-compose.yml.
```

`export_deployment_artifacts.py` exists as a standalone, deterministic script rather than relying on notebook state: model selection is already done by Phase 6, so the deployed models are refit on all training engines (not the 80% split used for evaluation), there's no more reason to withhold data once the comparison is finished. FD001 uses its original, unmodified recipe (no clustering); FD004 additionally fits and saves the regime-clustering KMeans model and normalization stats, since inference needs to reproduce the exact same normalization at request time.

`inference.py` is the single scoring function — called by the API, and by the dashboard only in its embedded (live-demo) mode, so both paths produce identical predictions. It flags when RSF's point estimate falls outside Weibull's own confidence interval: the two are independent models scoring the same input, nothing forces agreement, and a dashboard showing a point estimate outside its own interval looks broken unless that's labeled as an expected consequence of the two-model design rather than a bug. The API degrades gracefully if only FD001's artifacts have been built, FD004 support is optional, requested per-request via a `subset` field, and a request for a subset that hasn't been built returns a clear 422 rather than a crash.

The risk level is the safety call, so it reads the cautious number: **critical** when the served interval's lower bound (or the point estimate, if lower) is under 20 cycles, **warning** when the point estimate is under 50, otherwise healthy. It originally used the point estimate alone, which contradicted deciding maintenance from the lower bound and, on out-of-fold training data, flagged only 91% (FD001) and 71% (FD004) of engines within 20 cycles of failure as critical. Using the lower bound for both bands was rejected: the interval is about 100 cycles wide, so the warning would fire on almost every engine. The split rule, chosen on that out-of-fold data, flags 100% of near-failure engines on both subsets while 90% / 93% of engines with 50+ cycles left still read healthy (6–10% get a false critical). Confirmed on the test sets: critical 13/13 (was 12/13) on FD001 and 40/40 (was 24/40) on FD004, with 4 and 10 false criticals among engines with 50+ cycles left. The thresholds themselves (20, 50) are an illustrative policy; a real deployment would set them from the maintenance lead time.

### Docker: built and smoke-tested in CI

`api/Dockerfile` builds the FD001 and FD004 models **inside the image at build time** (not mounted or copied from the host — `models/` is gitignored and 200+MB, so there's nothing to copy from a clean clone anyway) by running `export_deployment_artifacts.py` once per subset as separate build steps. The trade-off: a slow `docker build` (the same several-minutes-per-subset cost as running the script directly, since it's the same script — FD004's export took about 6.5 minutes in the most recent test) in exchange for an instant `docker run` with no cold-start model-fitting delay, the right way around for a service, not a notebook. `app/Dockerfile` is lean by construction: it runs the dashboard in HTTP mode, which never imports the modeling code, so its image never installs scikit-learn/scikit-survival/lifelines at all.

**Verified in CI.** The images are built on GitHub Actions (`.github/workflows/docker.yml`): `docker compose build`, `docker compose up --wait` (which waits on the API's healthcheck), then `scripts/docker_smoke_test.py` run inside each container. In the API container it checks that `/health` lists FD001 and FD004, that test engine 1 of each subset gets a prediction in the expected range with an ordered 90% interval (currently FD001 123 cycles [52, 237] and FD004 33 [13, 59], the same numbers as embedded mode), and that an empty request is rejected with 422. In the dashboard container it checks that the API is reachable at `http://api:8000` over the Compose network, then runs the real dashboard headlessly (Streamlit `AppTest`) in HTTP mode and makes a prediction through the API — proving the two services work together, not just that each starts. The first run passed in about 9 minutes, most of it the FD001 and FD004 exports inside the API image build. The job runs weekly, on demand, and on any push that changes what goes into the images.

**Framework note:** deep survival modeling in Phase 7 uses `tensorflow-cpu` rather than PyTorch. `tensorflow-cpu` is a genuine CPU-only build. This was an environment constraint, not a framework preference.

### Operations: versions, drift and a retraining gate

**Versioning (`manifest.py`, `publish.py`).** Every export writes `models/<subset>/manifest.json`: a `model_version` derived from the SHA-256 of the files that determine predictions (so identical models get the same version and any change gets a new one), the git commit, the training-data hashes, library versions, training settings, and test-set metrics measured through `inference.predict`. Loading re-hashes the files and reports the version only if they still match, so a rebuilt model can't borrow an old version. `/health`, `/predict` and the dashboard sidebar report it. `python -m turbofan_rul.publish` checks the large files against the SHA-256 Hugging Face stores, uploads only what differs, and tags the commit with each version; the live demo is pinned with `MODEL_REVISION`. The first publish found that refitting Weibull changes its pickle (lifelines stores the fit time) while RSF stays byte-identical, so only ~2 MB had to be uploaded.

**Drift monitoring (`drift.py`, notebook 17).** The API keeps the last 500 requests per model and compares the operating settings and each sensor's 15-cycle rolling mean with a reference saved at export, using the Population Stability Index (worst feature sets the status; `GET /drift`). Three traps from the first run shaped the design: a full-life reference made FD001's own test fleet look drifted (test engines are healthier on average, and sensors track wear), so the reference uses training engines observed partway through life; replaying from an empty fleet made windows single-age, so replays use a steady-state fleet; and the textbook 0.1 / 0.25 thresholds fired on most normal windows (correlated requests, worst of 15–20 features), so thresholds are calibrated on training engines only (half builds a reference, the other half is replayed). Final replays:

| Fleet → model | Drift windows | Model on that fleet |
|---|---|---|
| FD001 → FD001, FD004 → FD004 | 0% | its normal accuracy |
| FD002 → FD001 (six regimes) | 100% | RMSE 55.8, coverage 10% |
| FD003 → FD001 (new fault mode) | 100% | RMSE 22.1, NASA 19.3 vs 3.12 |
| FD002 → FD004 | 100% | RMSE 27.4 — fine: drift is not harm |

The price of no false alarms is sensitivity: a window needs about 30% foreign requests before it fires.

**Per-engine input check (`drift.familiarity`).** PSI needs hundreds of requests, so each single prediction gets its own check: how far its inputs sit outside what the model saw in training (all training rows from cycle 30, failure included). Operating settings, which don't wear, are checked against the full min–max range and drive a red *unfamiliar* verdict; sensors' rolling means are checked against the 0.5th–99.5th percentile band and drive an amber *caution*. Thresholds come from held-out training engines (99.9th percentile for red, 99th for amber). Two design errors were caught by measuring before shipping: a range taken from the fleet-like drift reference flagged 12% of engines in their last 10 cycles (it stops at 99% of life; the model trained on full lives), and percentile bands on the raw operating settings put one normal FD001 engine in red because of a single noisy reading. Result on the official test sets: FD001 engines under the FD001 model are 100% familiar; FD002's are 90% unfamiliar and 10% caution; FD003's are 45% caution (median 58 cycles left, so a different fault rather than wear) and never red; under the FD004 model, trained on all regimes and both fault modes, engines from every subset are 98–100% familiar. Amber is still expected in an engine's last ~10 cycles (12% of held-out training engines there), and the message says so.

**Retraining gate (`gate.py`, `.github/workflows/retrain.yml`).** A run retrains a challenger on GitHub's machine with the live demo's pinned libraries, downloads the champion's manifest from Hugging Face, and applies four rules fixed in advance: NASA score per engine at most 5% worse, C-index at most 0.01 lower, interval coverage at least 85%, near-failure coverage at most 5 points lower. The challenger is published (and tagged) only if all pass and publishing was requested; making it live stays a manual change of `MODEL_REVISION`. Both outcomes were exercised: the deployed recipe retrained passed with identical metrics; a challenger trained on 30 of 100 engines was rejected on the NASA rule (4.59 vs 4.05) while its C-index and coverage passed, which is why the gate checks more than one metric. The rules use the official test set, acceptable for a no-worse-than check but not for choosing among many challengers.

---

## 11. Bugs caught along the way

Seven bugs found during the project, and how each was fixed:

1. **A silent scope mismatch between Phase 1 and Phase 5.** RSF's "full feature set" was computed as "every column except identifiers and targets" — which turned out to include the *raw* value of all 21 sensors and 3 operating settings, not just the 12 Phase 1 flagged as informative, since the feature-engineering step adds columns without dropping the raw ones it started from. Caught only when `export_deployment_artifacts.py` had to enumerate exact column names for the deployment contract. **Resolved by keeping the model as evaluated** (retraining post hoc would invalidate the reported Phase 6 metrics) — and then measuring whether the gap mattered instead of just documenting it: a second RSF with the same hyperparameters (`min_samples_leaf=5`, as deployed), fit on the intended 205 engineered columns only and evaluated on the same official FD001 test set, differed from the deployed 229-column model by less than ordinary random-forest run-to-run variance, in no consistent direction (C-index −0.0004, NASA score 0.065 *better*, MAE +0.09, late rate +0.03, coverage −0.03). The extra raw columns made no measurable difference, so the deployed model was kept as is.

2. **A gradient check that actually failed, and why.** The first run of Phase 7's from-scratch/Keras loss cross-validation showed 100% relative error and a sigmoid overflow — traced to running the check on unstandardized raw sensor values before the scaling step existed in the notebook cell order. Fixed by reordering (scale first, then check), verified down to 1.7×10⁻⁶ relative error.

3. **A `lifelines` API misuse, not a math error.** `check_assumptions` initially raised a shape-mismatch error from being passed the full 234-column training frame instead of the 25 columns Cox was actually fit on. Fixed by switching to `lifelines.statistics.proportional_hazard_test` against the exact fitted-column subset — which also produced a cleaner, quantified result (14/25 covariates fail, with engine-clustered standard errors) instead of a wall of advisory text.

4. **A transcription error in this project's own reference data, caught by validation built for a different purpose.** `data_loading.py` validates downloaded row/unit counts against published dataset statistics specifically to catch a stale or corrupted mirror. When FD004 support was added (Phase 12), it caught something else: this project's own hardcoded train/test engine counts for FD004 were transposed (248/249 instead of the correct 249/248).

5. **A fresh-clone smoke test caught two crash-on-missing-model bugs that months of work inside one long-lived environment never surfaced.** Both `api/main.py`'s startup check and `test_inference.py`'s skip condition decided whether a subset's model was "available" by checking for `deployment_metadata.json` — a small file that's tracked in git. The actual model (`rsf.joblib`, 25-220MB) is gitignored, so on a genuinely fresh clone the metadata check passed while the model file didn't exist, and both the API and the test suite crashed with a raw `FileNotFoundError` instead of a clear "run the export script first" message. Found by exporting a clean, tracked-files-only copy of the repo into a fresh virtual environment and running the install-and-test sequence a new user would. Fixed by checking for `rsf.joblib` itself in both places; re-verified with the same fresh-clone process afterward, including a full model export and a live prediction, both producing results identical to every prior run.

6. **EWM features silently misaligned on every cross-validation fold.** `features.add_ewm_features` computed EWMs with `groupby().apply(...)` and then `reset_index(level=0)`, which assumes the frame's row index runs 0..n-1. The full training frame satisfies that, so the main pipeline and the deployed model were unaffected — but every CV fold's training frame is a filtered subset with gaps in its index, so on folds the EWM values were misaligned with their rows or missing. It affected only RSF (Cox and Weibull use no EWM features), and it mattered: before the fix, the RSF tuning search made `min_samples_leaf=5` look like a 4-of-5-fold winner; afterwards it wins 2 of 5. Fixed with `groupby().transform(...)`, which aligns by index; `build_feature_matrix` now also rejects frames with duplicate indexes or rows out of cycle order, and tests cover filtered frames specifically.

7. **Two different C-indices compared as if they were one.** The model notebooks report a *row-level validation* C-index (every row of the 20 held-out training engines), while the official test set needs an *engine-level* C-index (one prediction per test engine). Earlier versions of this README and several notebooks put the two side by side — RSF's 0.901 validation figure next to 0.864 on the test set — which produced an apparent "concordance regression" from tuning that didn't exist, and inflated every cross-subset comparison against FD001. Fixed by computing the test C-index for every model and using it for all cross-model, cross-notebook and cross-subset comparisons; the validation C-index is kept only within the notebook that produces it.

---

## 12. Scope notes

- **FD001 gets the full, tuned treatment; FD004 is a transfer test**: it gets the full ladder but deliberately inherits FD001's tuned hyperparameters, to check whether the recipe holds without re-tuning. Its served interval covers 88.7% of test engines, but only 73% of those with more than 125 cycles left (safe-side misses; an uncapped upper bound fixing them was measured and not deployed, see section 9). FD002/FD003 are evaluation-only (RSF, no deployment) — lighter checks isolating one complexity axis each, not full builds.

- **Phase 7 (deep survival) is a stretch phase.** It would only have been deployed if it beat RSF; it didn't.

- **Training labels use `event=1` for every row**, including rows at the 125-cycle cap; the censored alternative was tested and rejected because it breaks the deployed Weibull interval (see [Feature engineering](#2-feature-engineering)).

- **The survival-vs-regression comparison uses simulated censoring** (FD001, one mechanism, untuned regressors); on this data a regression that drops unfinished engines matches RSF's accuracy. See [notebook 13's section](#6-does-survival-analysis-actually-beat-regression-tested-with-a-mixed-answer).

- **Operations are demonstrated on replays.** No live traffic exists, the drift window resets when the API restarts, and the retraining gate uses the official test set (see the Operations subsection).

- **The public demo runs in one process** (the dashboard's embedded mode on Streamlit Community Cloud), not as the two-container deployment — a hosting constraint, documented in the Deployment section. Its models live in a Hugging Face model repo and must be re-uploaded after every re-export, with the pinned library versions kept in step.