# turbofan-rul

Remaining Useful Life (RUL) prediction for NASA's C-MAPSS turbofan engines — framed and solved as a **survival analysis / time-to-event problem**, not plain regression. Deployed as an API + dashboard that returns a RUL point estimate with a confidence interval, not a notebook with a final R² score.

Full technical spec: [`docs/project_brief.md`](docs/project_brief.md). This README is the narrative version — what was built, why, and what it actually measures.

**Update:** the Phase 6 comparison used a single 80/20 split on FD001 alone. `notebooks/08_cross_validation.ipynb` checks split-stability; `notebooks/09_fd002_fd003_generalization_check.ipynb` and `notebooks/10_fd004_full_pipeline.ipynb` check generalization to harder subsets with genuine distribution shift. See [Cross-validation](#cross-validation-does-the-ranking-hold) and [Generalization](#generalization-fd002fd003fd004) below.

---

## The problem, and why it isn't regression

Predict how many operating cycles a turbofan engine has left, from multivariate sensor telemetry (21 sensors + 3 operating settings per cycle), using NASA's C-MAPSS degradation simulation dataset (FD001 subset: single operating regime, single fault mode).

The naive approach — regress sensor features directly onto observed RUL — is a well-known mistake in this space, and it's not cosmetic. In any real predictive-maintenance deployment, most monitored units are still running when the model is built: their failure time is unknown and lies in the future. This is **right censoring**. A regressor trained only on completed failure trajectories (or one that treats a censored unit's last-known age as its true RUL) is trained on a biased sample — it implicitly excludes "units that would have survived longer than we watched," which systematically biases predictions toward shorter lifetimes.

Survival analysis models the conditional distribution of time-to-failure given "known to have survived to at least time *t*," which is the correct estimand. C-MAPSS's structure maps directly onto this: full run-to-failure trajectories for training (`event=1` for every row, `duration` = the row's RUL capped at 125 cycles — see the label note below), truncated trajectories for test (`event=0`, right-censored at the last observed cycle). That `(duration, event)` construction — not a feature-engineering detail, a first-class data artifact — is what every model in this project is actually fit on. See `src/turbofan_rul/labeling.py`.

---

## Feature engineering

Raw sensor readings are noisy; the degradation signal lives in the *trend*, not the instantaneous value. The pipeline (`src/turbofan_rul/features.py`, built in `notebooks/02_feature_engineering.ipynb`):

- **Sensor selection** (Phase 1 EDA): coefficient-of-variation screening flagged 9 of 21 sensors as near-constant, leaving 12 kept sensors — with the actual measured CVs, not an assumed threshold. One caveat, recorded in `notebooks/01_eda.ipynb`: CV divides by the mean, so `sensor_8` and `sensor_13` (both near 2388) get tiny CVs partly because of their large means. Much of the C-MAPSS literature keeps them; dropping them here is a deliberate judgment call, not proof they carry no signal.
- **Rolling statistics**: mean/std/min/max/trend-slope per sensor across 5/15/30-cycle windows, `min_periods=1` so early-life rows aren't discarded (an engine newly under monitoring won't have a full window of history either).
- **EWMA**: exponentially-weighted moving averages (spans 10/30) as a second smoothing representation alongside the rolling windows. Several EWM features rank among RSF's most important (`notebooks/05`).
- **Piecewise-linear RUL labels**: capped at 125 cycles, the standard C-MAPSS convention — early-life cycles carry no meaningful degradation signal, so an uncapped linear countdown from cycle 1 would force models to fit a trend that isn't there.

**Label note — a censoring alternative was tested and rejected.** With the cap, every training row is labeled `event=1`, including the 8,131 of 20,631 FD001 rows sitting at the cap — so those rows are, strictly, labeled as "fails in exactly 125 cycles" when their true remaining life is longer. The statistically cleaner option is to mark those rows as censored at 125 (`event=0`). `labeling.build_train_survival_frame` supports this via `censor_above_cap=True`, and it was tested: Cox improved slightly, RSF was unchanged, but Weibull AFT broke badly — its FD001 NASA score went from about 1.3×10³ to about 2×10¹⁰ and its 90% interval width from about 100 to 184 cycles, because with the censored rows its fitted distribution puts far more mass on very long lifetimes. Since Weibull supplies the deployed interval, the default stays `censor_above_cap=False` (all rows `event=1`), matching standard C-MAPSS practice. The trade-off is documented in `labeling.py`.

Every engineered column is grouped strictly per-engine; cross-engine leakage in rolling windows is checked directly in `tests/test_features.py`, not just assumed away by the groupby.

---

## Modeling: a ladder, not a single model

Four approaches, each included for a stated reason rather than a leaderboard entry:

| Model | Why it's here | Key limitation |
|---|---|---|
| **Cox PH** | Reference standard; interpretable hazard ratios; the Schoenfeld-residual test gives a *diagnostic*, not just a baseline | Assumes proportional hazards — tested directly, not assumed |
| **Weibull AFT** | Wear-out is physically a Weibull process; fully parametric, so RUL quantiles/confidence intervals come directly from the fitted distribution | Fixed parametric form; can't capture arbitrary nonlinear interactions |
| **Random Survival Forest** | Drops both the PH and parametric assumptions; captures nonlinear multi-sensor interactions | Confidence intervals under-cover; predictions structurally capped at the training time ceiling |
| **Deep hazard networks** (stretch) | Tests whether temporal structure can be learned directly from raw sensor windows, with no hand-engineered features at all | Didn't beat feature-engineered RSF with 80 training engines; intervals badly under-cover |

**The ladder order isn't arbitrary — Cox PH's own diagnostic justifies what comes next.** Fitting Cox PH (with engine-clustered robust standard errors, since each engine contributes many correlated rows) and running the Schoenfeld-residual test on its output showed **14 of 25 covariates violate the proportional-hazards assumption** at p<0.05 (`notebooks/03_cox_baseline.ipynb`). That count is if anything overstated — the test treats each row as independent, so its p-values are too small — but even discounted, it's quantified evidence, not a hand-wave, for why a model that doesn't assume proportional hazards (Weibull AFT, then RSF) was tried next.

**Weibull AFT's derivation was validated, not just cited.** The survival function, hazard function, and AFT log-likelihood are derived by hand in `notebooks/04_weibull_aft.ipynb`, then checked with a from-scratch MLE fit against `lifelines.WeibullFitter` — agreement to **≤5.4×10⁻⁷ relative error** on both parameters, and to 4.6×10⁻⁷ when the censoring term of the likelihood is exercised too (8,031 censored rows). The wear-out evidence comes from a Weibull fit to the 100 engines' actual lifetimes: shape **ρ = 4.41 (95% CI [3.80, 5.01])**, clearly above 1, so failure risk rises with age — the signature the physical framing predicted, checked rather than assumed. (The covariate AFT model's own shape estimate, 2.67, is not used as evidence: it depends on the row-level label construction, and its CI treats correlated rows from the same engine as independent, so it is far too narrow.)

**The deep models (Phase 7, stretch) get the same rigor, adapted to a different validation problem.** No survival-analysis deep learning library was available in the build environment (see Deployment section), so the discrete-time hazard loss — masked binary cross-entropy, derived from the same survival likelihood as the other models — was implemented from scratch in NumPy with hand-derived backprop, gradient-checked against finite differences, *and* cross-validated against the Keras implementation actually used for training (both compute the identical loss value on identical inputs — checked, not assumed). Training is made reproducible (`keras.utils.set_random_seed` plus `tf.config.experimental.enable_op_determinism()` — two runs give identical output), but a single seed is still one draw from the initialization lottery, so a plain LSTM and a Conv1D→LSTM (a known strong C-MAPSS baseline architecture) were compared across 3 seeds each. Result: the plain LSTM had the higher validation concordance on all 3 seeds (mean 0.880 vs. 0.871, a gap 2.5× the larger seed-to-seed std) — a consistent but small edge, from only 3 seeds. On the official test set the picture is mixed (the LSTM has the better NASA score and MAE, the Conv1D→LSTM a slightly higher C-index), and **neither beats RSF on any of them**. Reported as directly as a win would have been.

---

## Evaluation: not RMSE

RMSE and MAE are reported, but as explicitly secondary — they're symmetric, so they treat a 20-cycle overprediction (an engine fails in service) the same as a 20-cycle underprediction (a part gets replaced a bit early). Those aren't equally bad. `src/turbofan_rul/evaluation.py` implements, from the C-MAPSS literature rather than imported, the metric that encodes that asymmetry directly:

**NASA scoring function**: for prediction error *d* = predicted − true RUL,

```
penalty = exp(d/10) − 1   if d > 0   (late — dangerous)
penalty = exp(−d/13) − 1  if d ≤ 0   (early — merely costly)
```

Both branches are zero at *d*=0 and grow exponentially, but the late branch divides by the smaller constant — it grows faster. A unit test asserts specifically that late errors score worse than early ones of the same magnitude; if the two constants were ever swapped, the metric would silently reward the exact behavior it exists to punish.

Alongside it: concordance index (does the model rank engines by risk correctly, independent of magnitude), late-prediction rate (how *often* a model is dangerously wrong, separate from *how badly*), interval coverage (are the confidence intervals calibrated), and calibration-by-risk-bin (is a model biased specifically for the highest-risk engines, which matters more than being biased on average).

### Results (FD001 test set, 100 engines)

| Model | NASA score | C-index (test) | Late rate | 90% coverage | RMSE | MAE |
|---|---|---|---|---|---|---|
| Cox PH | 1672.5 | 0.821 | 54% | — | 22.01 | 16.91 |
| Weibull AFT | 1288.2 | 0.818 | 48% | 90.0% | 23.14 | 18.27 |
| LSTM discrete hazard (stretch)* | 846.4 | 0.838 | 58% | 73.0% | 17.43 | 12.41 |
| Conv1D→LSTM discrete hazard (stretch)* | 967.3 | 0.843 | 57% | 74.0% | 18.34 | 13.34 |
| **Random Survival Forest** | **462.5** | **0.864** | 65% | 86.0% | **16.27** | **11.99** |

*Deep-model figures are from one seed each (seed 42); training is deterministic, so these numbers reproduce exactly. Only the LSTM-vs-Conv1D→LSTM validation-concordance comparison was repeated across 3 seeds — the single-seed test metrics above shouldn't be read as the architectures' settled ranking.

**Which C-index this is.** The C-index column is computed per engine on the 100 official test engines (each engine's predicted RUL vs. its true RUL) — the same metric used for every cross-model and cross-subset comparison in this README. The notebooks also report a *row-level validation* C-index on the 20 held-out training engines (Cox 0.875, Weibull 0.864, RSF 0.901); that's a different metric on different data, useful within a notebook but not comparable to the test C-index. An earlier version of this README mixed the two (see [bug #7](#bugs-caught-along-the-way)).

**This table is the Phase 6 model-selection comparison**: every model trained on the same 80 engines, RSF with `min_samples_leaf=30` (the value used throughout Phases 3–6). The RSF actually deployed is retrained on all 100 engines with a retuned `min_samples_leaf=5` — see [Hyperparameter tuning](#hyperparameter-tuning-a-modest-win-with-weaker-cv-support) for the deployed model's numbers.

**Two findings worth the interview time they'd take to explain:**

- **RMSE and the NASA score rank Cox and Weibull differently on FD001.** Cox wins on RMSE (22.01 vs. 23.14); Weibull wins on NASA score (1288.2 vs. 1672.5) because its errors sit on the safer side. A model comparison run on RMSE alone would have picked the model that's worse for the maintenance decision. (This disagreement is specific to FD001: on FD004, Cox wins on both — see [Generalization](#generalization-fd002fd003fd004).)
- **The best model by NASA score is also the most frequently late.** RSF has both the lowest NASA score *and* the highest late-prediction rate (65%). It's late more often, but by smaller margins — the exponential penalty means many small late errors cost less than a few large ones. Reporting only the aggregate score would have hidden this.

**RSF's two real weaknesses**, and why they shape the deployment design below: intervals under-cover (86% actual vs. 90% nominal), and predictions are structurally capped at the ~125-cycle training time ceiling (a non-parametric survival curve is only defined over the time support it was trained on) — 11 of 100 test engines have true RUL above that cap, and RSF cannot say so.

---

## Cross-validation: does the ranking hold?

Every number in the results table above came from one 80/20 engine-level split. `notebooks/08_cross_validation.ipynb` runs 5-fold `GroupKFold` (by engine, so no engine's cycles ever cross a fold boundary) over Cox PH, Weibull AFT, and RSF, and checks two separate questions rather than conflating them.

**Evaluation design:** every training engine runs to failure, so a held-out engine's last row always has RUL=0 — evaluating there would be degenerate. Instead, each held-out engine gets one random truncation point (30–90% of its observed lifetime, mirroring how NASA actually constructs the official test set) and is scored against its true remaining life from there. See `src/turbofan_rul/cross_validation.py`.

*Deep models (Phase 7) aren't included — their own notebook already established more run-to-run variance than useful to compound across 5 additional refits, and they were never deployment candidates.*

### Does the RSF-leads ranking hold? Yes — and it's more than that.

| Model | CV concordance (mean ± std) | CV NASA score per fold, 20 engines (mean ± std) |
|---|---|---|
| Cox PH | 0.798 ± 0.052 | 426.6 ± 298.7 |
| Weibull AFT | 0.803 ± 0.048 | 537.8 ± 778.9 |
| **RSF** | **0.850 ± 0.019** | **233.0 ± 189.2** |

**Concordance:** RSF has the strictly best C-index in 4 of 5 folds. In the other (fold 2), Weibull is best (0.883) and RSF ties Cox (0.878). **NASA score:** RSF has the best mean and is best in 3 of 5 folds; Weibull wins folds 1 and 2. More interesting than the win counts: **RSF's concordance standard deviation is roughly a third of the other two models'.** It isn't just better on average — it's the most *stable* model across different engine splits, which is a real, separate property Phase 6's single split had no way to show. Weibull's NASA score in particular swings wildly (5.3 to 96.2 per-engine mean across folds) — fold 4 had a small number of badly-late predictions that a single-split evaluation could easily have missed entirely.

**The absolute numbers are a little lower than on the official test set** (RSF 0.850 vs. 0.864, both engine-level C-index) — expected, not a discrepancy: this notebook's random mid-life truncation points are a different task than NASA's specific chosen truncation points, and RSF here uses a coarser `row_stride=5` for fold-loop runtime rather than Phase 5's `3`. The CV numbers answer "how stable is this ranking"; the official-test-set numbers answer "what does the model score on the real benchmark".

---

## Generalization: FD002/FD003/FD004

C-MAPSS has four subsets varying two independent axes: operating regimes (1 vs. 6) and fault modes (1 vs. 2). Everything above uses FD001 (1 regime, 1 fault mode) — the easiest subset. This section checks whether the pipeline holds up on the harder ones.

**A new pipeline capability was needed first.** FD002/FD004's 6 operating regimes shift a sensor's absolute level independent of degradation — confirmed by a quantified check, not assumed: `sensor_2`'s raw variance on FD002 is **99.99% explained by which regime an engine is in**, not wear (std 37.3 overall vs. 0.44 within-regime). Without correcting for this, rolling-window features would mostly track regime-switching. `features.py` gained `fit_regime_clusters` (KMeans on the 3 operating settings, `n_clusters=6` — confirmed by within-cluster std collapsing to ~0) and within-cluster normalization, fit on training data only. **FD001's own pipeline is untouched** — retrofitting it with clustering machinery it's never needed would silently change a model whose Phase 6 numbers are already reported.

**FD002 and FD003 were run as evaluation-only checks** (RSF only, no tuning, no deployment — the same weight as the original Phase 1 EDA, not a second full build), each isolating one complexity axis before FD004 combines both:

| Subset | Regimes | Fault modes | C-index | NASA score (mean/engine) | MAE |
|---|---|---|---|---|---|
| FD001 (baseline) | 1 | 1 | 0.864 | 4.6 | 11.99 |
| FD003 (fault modes only) | 1 | 2 | 0.869 | 8.2 | 13.34 |
| FD002 (regimes only) | 6 | 1 | 0.858 | 39.9 | 19.46 |

FD003 generalizes well — on par with the FD001 baseline (concordance even slightly higher, NASA score and MAE somewhat worse) with zero pipeline changes. FD002 is a real step down even with the clustering fix working correctly, some of which reflects a much coarser `row_stride` used only for notebook runtime, not a final tuned model.

**FD004 (both axes combined) got the full model ladder** — Cox PH, Weibull AFT, and RSF, each evaluated against FD004's official test set exactly as Phase 6 did for FD001:

| Model | FD001 C-index | FD004 C-index | FD001 NASA mean | FD004 NASA mean |
|---|---|---|---|---|
| Cox PH | 0.821 | 0.811 | 16.7 | 54.6 |
| Weibull AFT | 0.818 | 0.812 | 12.9 | 69.8 |
| **RSF** | **0.864** | **0.849** | **4.6** | **27.5** |

**The ranking holds.** RSF leads on both concordance and NASA score on FD004, exactly as on FD001 — now three independent pieces of evidence for the same conclusion (single split, 5-fold CV, and a structurally different, harder subset), not one result restated three ways. Every model's absolute score is worse on FD004, as expected for a strictly harder problem evaluated with an untuned (`min_samples_leaf=30`), coarser-strided (`row_stride=15`) RSF — this notebook deliberately did not fold in the later hyperparameter tuning or RSF coverage fix, to keep "does the existing recipe generalize" a clean, unconfounded question. One FD001 finding does *not* carry over: on FD004, Cox beats Weibull on both RMSE (34.9 vs. 35.6) and NASA score (54.6 vs. 69.8 per engine), so the RMSE-vs-NASA disagreement above is a property of FD001, not a general rule.

**A real bug caught along the way:** `data_loading.py`'s published-statistics validation (built in Phase 1 specifically to catch a corrupted mirror) caught that FD004's train/test engine counts were transposed in this project's own reference table (248/249 instead of the correct 249/248) — a transcription error from when FD002-004 support was first added, not a mirror problem. The validation step did exactly what it was built for, just against this codebase's own bug instead of an external one.

---

## Hyperparameter tuning: a modest win, with weaker CV support

Tuned against the same 5-fold CV protocol as Phase 11 — searching and reporting on the same split is how overfitting to a holdout happens invisibly. `notebooks/11_hyperparameter_tuning.ipynb`.

**Cox PH / Weibull AFT penalizers** (`[0.01, 0.1, 0.5, 1.0]`, full 5-fold grid): the project's existing default of **0.1 was already at or near the optimum** for both. Confirmed, not changed.

**RSF** (the deployed model): a search over `min_samples_leaf` and `max_features` found `max_features="sqrt"` clearly better than `"log2"`. A narrower single-fold search found leaf sizes 3, 5 and 8 tied (0.8737), and `min_samples_leaf=5` (the middle of the tie) was taken forward and compared with the original default of 30 across all 5 folds. **The CV evidence is mixed:** leaf=5 has the higher mean concordance (0.855 vs. 0.841) but wins only **2 of 5 folds** — its higher mean comes mostly from one large gain in fold 2 (0.910 vs. 0.851) — and its fold-to-fold std is about 2.5× larger (0.032 vs. 0.013), giving up much of the stability Phase 11 flagged as one of RSF's distinct virtues. (Before the EWM bug fix — [bug #6](#bugs-caught-along-the-way) — leaf=5 appeared to win 4 of 5 folds; that apparent consistency came partly from corrupted features.)

**Checked on NASA's official FD001 test set**, like-for-like — both models trained on all 100 training engines with the same `row_stride=3`, only the leaf size changed:

| Metric | leaf=30 | Tuned (leaf=5, deployed) |
|---|---|---|
| C-index (test) | 0.859 | **0.864** |
| NASA score (mean per engine) | 4.51 | **4.04** |
| MAE | 12.04 | **11.32** |
| RMSE | 16.27 | **15.55** |
| Late-prediction rate | 0.62 | **0.57** |
| 90% coverage | 0.86 | **0.87** |

**leaf=5 is better on every metric on the official test set**, though most margins are small. The deployed FD001 model keeps `min_samples_leaf=5` on that basis, with the evidence stated plainly: the test set favours it across the board; the CV evidence is weaker (higher mean, but 2 of 5 fold wins and more variance); and since the official test set is the one every model in this project is reported on, a small advantage there isn't a fully independent confirmation. FD004 inherits the same value without an independent search (a stated scope gap, not an assumed transfer).

*Correction to an earlier version of this section,* which reported a concordance drop from 0.901 to 0.864 when tuning. That was not a real effect: 0.901 was a row-level validation C-index from a model trained on 80 engines, compared against an engine-level test C-index from a model trained on 100 — two different metrics on different data ([bug #7](#bugs-caught-along-the-way)). Measured like-for-like, concordance goes slightly *up*.

**A second, previously undisclosed cost of this decision:** smaller leaves mean deeper, larger trees. The deployed FD001 RSF artifact grew from ~44MB (`leaf=30`) to **~216MB** (`leaf=5`) — a real deployment-relevant cost (storage, load time) that wasn't part of the original CV-vs-official-test comparison and is disclosed here rather than left for someone to discover by inspecting file sizes.

---

## RSF interval calibration: a fix that validated the original design instead of replacing it

Phase 6 found RSF's 90% intervals under-cover (~86-87% actual). `notebooks/12_rsf_interval_calibration.ipynb` fixes this directly with **cross-conformal prediction** — a model-agnostic technique that widens an interval by a single correction, calibrated on out-of-fold predictions, until it hits target coverage, rather than hoping the model's quantiles happen to be well-calibrated probabilities. The resulting coverage is an empirical result to be checked, not a guarantee: the standard finite-sample guarantee of split conformal doesn't strictly carry over to the cross-conformal variant, or to rows that are correlated within an engine.

**Calibration** used the same 5-fold CV infrastructure as Phases 11/13 — each of the 100 training engines scored by a model fit on the other 4 folds (with the deployed recipe: `min_samples_leaf=5`, `row_stride=3`), pooling calibration data without holding any engine out of the final deployed model. Result: a correction of **q = 5.00 cycles**, applied to both interval bounds — identical to the value `export_deployment_artifacts.py` computes for the deployed artifact.

**Checked on the real official test set, not just the calibration data:** coverage lands at almost exactly 90% (0.870 → 0.900), at a real, quantified cost — intervals widen ~20% (49.6 → 59.6 cycles mean width).

**On aggregate numbers alone, calibrated RSF now looks strictly better than the deployed Weibull interval** — better-covered (0.900 vs. 0.890) *and* 40% narrower (59.6 vs. 98.6 cycles mean width). Worth checking before concluding anything: does this hold on the specific subgroup that matters?

| | Coverage on the 89 engines within RSF's training ceiling | Coverage on the 11 engines *above* it |
|---|---|---|
| Calibrated RSF | 97.8% | **27.3%** |
| Weibull (deployed) | — | 90.9% |

**It doesn't.** RSF's survival curve is only defined over its training time support (~125 cycles) — a known structural limit from Phase 5. A uniform additive correction can't manufacture support the model never had; it just shifts a hard ceiling from ~125 to ~130. The aggregate 90% coverage is almost entirely carried by the majority of engines within that ceiling, while coverage on the exact engines Weibull exists to cover collapses to worse than a coin flip. This is precisely the kind of subgroup check Phase 6's calibration-by-risk-bin analysis was built to catch, now applied to a new model.

**Decision: Weibull stays the primary deployed interval — this result confirms the Phase 6 pairing for a reason now demonstrated, not just assumed.** The calibrated RSF interval is still useful (tighter, better-calibrated for the non-ceiling-affected majority), so it's exposed as a clearly-labeled **supplementary field** (`rsf_calibrated_interval_supplementary`) in both the API and dashboard — informational, explicitly not authoritative. `export_deployment_artifacts.py` computes the correction reproducibly as part of a normal build for every subset it exports (5 additional RSF fits — a deliberate cost for a clean-checkout guarantee). FD004 gets its own correction this way (**q = 36.00 cycles**, reflecting its much harder prediction problem), but unlike FD001's, its resulting coverage on FD004's official test set hasn't been checked in a notebook — a stated scope gap.

---

## Deployment: two containerized services, not a notebook

**Model pairing, driven by the evaluation above, not convenience:** RSF supplies the point estimate and risk ranking (best NASA score, best concordance); **Weibull AFT supplies the confidence interval**, since RSF's own intervals under-cover and can't extrapolate past its training ceiling while Weibull's are calibrated and unbounded. Cox PH isn't deployed: RSF beats it on every point-estimate and ranking metric, and Cox produces no natural interval to compete with Weibull's. (Against Weibull alone, Cox is not uniformly worse — on FD001 it has the better test C-index, RMSE and MAE, and on FD004 the better RMSE and NASA score — but neither is the deployed point model.) Applies per-subset: FD001 and FD004 each get their own RSF + Weibull pair.

**Two genuinely separate services, communicating over HTTP** — not a single process wearing two hats. `api/main.py` (FastAPI) owns every model artifact and all scoring logic; `app/streamlit_app.py` holds no model artifacts at all and calls the API's `/predict`, `/subsets`, and `/subsets/{subset}/metadata` endpoints over plain HTTP, using `API_BASE_URL` (an environment variable, not a hardcoded address) to find it — `http://localhost:8000` for local development, `http://api:8000` under Docker Compose's service-name DNS, or a public URL once deployed. An earlier version of this app called `inference.predict()` in-process as a documented single-process demo shortcut; that's gone now, not just described differently, since a real two-container cloud deployment needs the two services to actually be separate. A direct, measurable consequence: the dashboard's own dependency list (`requirements-app.txt`) no longer includes scikit-learn, scikit-survival, or lifelines at all — it literally cannot load a model anymore, by construction, not by convention.

```
turbofan-rul/
├── docs/project_brief.md              # full technical spec
├── notebooks/                         # 01 EDA → 12 RSF interval calibration
├── src/turbofan_rul/
│   ├── data_loading.py                # download + validate C-MAPSS (all 4 subsets)
│   ├── labeling.py                    # piecewise RUL, (duration, event) construction
│   ├── features.py                    # rolling/EWMA pipeline + regime clustering/normalization
│   ├── models.py                      # Cox/Weibull/RSF fit + engine-level split
│   ├── deep_survival.py               # from-scratch discrete-hazard MLP + Keras helpers
│   ├── cross_validation.py            # GroupKFold + truncation-point sampling for CV
│   ├── conformal.py                    # cross-conformal RSF interval calibration
│   ├── evaluation.py                  # NASA score, C-index, calibration
│   ├── inference.py                   # single scoring function -- used by the API ONLY now
│   └── export_deployment_artifacts.py # reproducible artifact build, per subset (see below)
├── api/
│   ├── main.py                        # FastAPI — POST /predict, GET /subsets, GET /subsets/{s}/metadata
│   └── Dockerfile                     # bakes FD001 + FD004 model artifacts in at build time
├── app/
│   ├── streamlit_app.py               # pure HTTP client of the API -- no model artifacts
│   └── Dockerfile                     # lean image, zero ML dependencies
├── docker-compose.yml                 # local two-service orchestration
├── requirements-api.txt               # lean, API-only deps (no jupyter/matplotlib/tensorflow)
├── requirements-app.txt               # leaner still -- streamlit + pandas + requests, nothing ML
├── tests/                             # 134 tests across 10 files
└── models/
    ├── fd001/                         # RSF + Weibull + metadata for FD001
    └── fd004/                         # RSF + Weibull + regime-clustering artifacts for FD004
```

```bash
# Local, without Docker:
pip install -r requirements.txt
python -m turbofan_rul.export_deployment_artifacts --subset FD001   # refits on ALL training engines
python -m turbofan_rul.export_deployment_artifacts --subset FD004   # optional -- FD004 support
uvicorn api.main:app --reload                          # API on :8000
API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py   # dashboard on :8501
PYTHONPATH=src pytest tests/ -v                         # 134 tests

# Containerized, both services:
docker compose up --build
# API on :8000, dashboard on :8501, dashboard reaches the API at http://api:8000
# automatically via Compose's service-name DNS -- see docker-compose.yml.
```

`export_deployment_artifacts.py` exists as a standalone, deterministic script rather than relying on notebook state: model selection is already done by Phase 6, so the deployed models are refit on all training engines (not the 80% split held out for honest evaluation) — there's no more reason to withhold data once the comparison is finished. FD001 uses its original, unmodified recipe (no clustering); FD004 additionally fits and saves the regime-clustering KMeans model and normalization stats, since inference needs to reproduce the exact same normalization at request time.

`inference.py` is the single scoring function — now used only by the API, since the dashboard was refactored (above) to call it over HTTP instead of importing it directly. It flags, explicitly, when RSF's point estimate falls outside Weibull's own confidence interval: the two are independent models scoring the same input, nothing forces agreement, and a dashboard showing a point estimate outside its own interval looks broken unless that's labeled as an expected consequence of the two-model design rather than a bug. The API degrades gracefully if only FD001's artifacts have been built — FD004 support is optional, requested per-request via a `subset` field, and a request for a subset that hasn't been built returns a clear 422 rather than a crash.

### Docker: written, reasoned through, honestly not verified in this build environment

`api/Dockerfile` builds the FD001 and FD004 models **inside the image at build time** (not mounted or copied from the host — `models/` is gitignored and 200+MB, so there's nothing to copy from a clean clone anyway) by running `export_deployment_artifacts.py` once per subset as separate build steps. That's a real, deliberate trade: a slow `docker build` (the same several-minutes-per-subset cost as running the script directly, since it's the same script — FD004's export took about 6.5 minutes in the most recent test) in exchange for an instant `docker run` with no cold-start model-fitting delay — the right way around for a service, not a notebook. `app/Dockerfile` is lean by construction: since the dashboard no longer imports any modeling code, its image never installs scikit-learn/scikit-survival/lifelines at all.

**Honest limitation:** this sandboxed build environment blocks all container registries (Docker Hub, GHCR both returned `403 Forbidden` on `docker pull`) — its network allowlist covers package registries (PyPI, npm, GitHub) but not container registries. That means `docker build` could not actually be run and verified here, even though Docker itself installs and its daemon runs fine. The Dockerfiles and `docker-compose.yml` are written to standard, well-established patterns and reasoned through in their own comments — and `docker compose config` accepts the compose file — but, unlike everything else in this project, the images were not built and run end-to-end before being shipped. In any normal environment (a local machine, GitHub Actions, or a cloud build service), `docker build` reaches Docker Hub without issue; this is a constraint of this particular tool sandbox, not of the Dockerfiles themselves.

**What IS verified, isolating Docker as the one remaining gap:** the underlying pipeline each Dockerfile runs — `pip install`, `export_deployment_artifacts.py`, `uvicorn`, `pytest` — was fresh-clone tested directly (see bug #5 below): a tracked-files-only copy of this repo exported into an isolated directory, a brand-new virtual environment, install from `requirements.txt`, a full FD001 model export from nothing (real downloads, real fits, real calibration), the API starting and serving a live prediction, and the full test suite — all producing results identical to every prior run in the main development environment. The same steps were later re-run for the API image's current contents — a clean Python 3.12 environment, `requirements-api.txt`, both FD001 and FD004 exported from nothing, and the API serving both — and the dashboard's image contents (only `src/` and `app/`, with only streamlit/pandas/requests installed) were checked to start and import cleanly without any ML library. That closes the "does this even work outside one long-lived sandbox" question for everything except the container layer itself. Verify with `docker compose up --build` before trusting that specific layer the way the rest of this README's claims can now be trusted.

**Framework note:** deep survival modeling in Phase 7 uses `tensorflow-cpu` rather than PyTorch — PyPI's default `torch` package bundles several GB of CUDA dependencies that exceeded the sandboxed build environment's disk, and the official CPU-only wheel index is outside its network allowlist. `tensorflow-cpu` is a genuine CPU-only build. This was an environment constraint, not a framework preference; a normal environment would just `pip install torch`. (Not relevant to the deployed containers above — neither image installs tensorflow at all, since Phase 7 was never a deployment candidate.)

---

## Bugs caught along the way

Rigor claims are only as good as what they catch. Seven real ones, kept rather than quietly fixed and forgotten:

1. **A silent scope mismatch between Phase 1 and Phase 5.** RSF's "full feature set" was computed as "every column except identifiers and targets" — which turned out to include the *raw* value of all 21 sensors and 3 operating settings, not just the 12 Phase 1 flagged as informative, since the feature-engineering step adds columns without dropping the raw ones it started from. Caught only when `export_deployment_artifacts.py` had to enumerate exact column names for the deployment contract. **Resolved by keeping the model as evaluated** (retraining post hoc would invalidate the reported Phase 6 metrics) — and then measuring whether the gap mattered instead of just documenting it: a second RSF with the same hyperparameters (`min_samples_leaf=5`, as deployed), fit on the intended 205 engineered columns only and evaluated on the same official FD001 test set, differed from the deployed 229-column model by less than ordinary random-forest run-to-run variance, in no consistent direction (C-index −0.0004, NASA score 0.065 *better*, MAE +0.09, late rate +0.03, coverage −0.03). The extra raw columns make no measurable difference, so the deployed model was kept as is.
2. **A gradient check that actually failed, and why.** The first run of Phase 7's from-scratch/Keras loss cross-validation showed 100% relative error and a sigmoid overflow — traced to running the check on unstandardized raw sensor values before the scaling step existed in the notebook cell order. Fixed by reordering (scale first, then check), verified down to 1.7×10⁻⁶ relative error. Kept in the notebook rather than edited out, since the catch is the point of running a gradient check at all.
3. **A `lifelines` API misuse, not a math error.** `check_assumptions` initially raised a shape-mismatch error from being passed the full 234-column training frame instead of the 25 columns Cox was actually fit on. Fixed by switching to `lifelines.statistics.proportional_hazard_test` against the exact fitted-column subset — which also produced a cleaner, quantified result (14/25 covariates fail, with engine-clustered standard errors) instead of a wall of advisory text.
4. **A transcription error in this project's own reference data, caught by validation built for a different purpose.** `data_loading.py` validates downloaded row/unit counts against published dataset statistics specifically to catch a stale or corrupted mirror. When FD004 support was added (Phase 12), it caught something else: this project's own hardcoded train/test engine counts for FD004 were transposed (248/249 instead of the correct 249/248). The check did exactly its job, just against an internal bug instead of the external one it was written for — a reminder that a validation step's value isn't limited to the failure mode that motivated it.
5. **A fresh-clone smoke test caught two crash-on-missing-model bugs that months of work inside one long-lived environment never surfaced.** Both `api/main.py`'s startup check and `test_inference.py`'s skip condition decided whether a subset's model was "available" by checking for `deployment_metadata.json` — a small file that's tracked in git. The actual model (`rsf.joblib`, 25-220MB) is gitignored, so on a genuinely fresh clone the metadata check passed while the model file didn't exist, and both the API and the test suite crashed with a raw `FileNotFoundError` instead of a clear "run the export script first" message. Invisible in the development environment, where the model file always existed once built and was never deleted — found only by actually exporting a clean, tracked-files-only copy of the repo into an isolated directory with a fresh virtual environment and running the real install-and-test sequence a new user would. Fixed by checking for `rsf.joblib` itself in both places; re-verified with the same fresh-clone process afterward, including a full model export and a live prediction, both producing results identical to every prior run.
6. **EWM features silently misaligned on every cross-validation fold.** `features.add_ewm_features` computed EWMs with `groupby().apply(...)` and then `reset_index(level=0)`, which assumes the frame's row index runs 0..n-1. The full training frame satisfies that, so the main pipeline and the deployed model were unaffected — but every CV fold's training frame is a filtered subset with gaps in its index, so on folds the EWM values were misaligned with their rows or missing. It affected only RSF (Cox and Weibull use no EWM features), and it mattered: before the fix, the RSF tuning search made `min_samples_leaf=5` look like a 4-of-5-fold winner; afterwards it wins 2 of 5. Fixed with `groupby().transform(...)`, which aligns by index; `build_feature_matrix` now also rejects frames with duplicate indexes or rows out of cycle order, and tests cover filtered frames specifically.
7. **Two different C-indices compared as if they were one.** The model notebooks report a *row-level validation* C-index (every row of the 20 held-out training engines), while the official test set needs an *engine-level* C-index (one prediction per test engine). Earlier versions of this README and several notebooks put the two side by side — RSF's 0.901 validation figure next to 0.864 on the test set — which produced an apparent "concordance regression" from tuning that didn't exist, and inflated every cross-subset comparison against FD001. Fixed by computing the test C-index for every model and using it for all cross-model, cross-notebook and cross-subset comparisons; the validation C-index is kept only within the notebook that produces it.

---

## Scope notes

- **FD001 gets the full, tuned, interval-calibrated treatment; FD004 gets the full ladder but inherits FD001's tuned hyperparameters**, and while the export script computes its own RSF interval correction (36.00 cycles), the calibrated interval's coverage on FD004's test set hasn't been checked. FD002/FD003 are evaluation-only (RSF, no deployment) — deliberately lighter checks isolating one complexity axis each, not full builds.
- **Phase 7 (deep survival) is explicitly a stretch.** Justified only if it measurably improved on RSF; it didn't, and that's reported as the result rather than omitted or reframed as a win.
- **Training labels use `event=1` for every row**, including rows at the 125-cycle cap; the censored alternative was tested and rejected because it breaks the deployed Weibull interval (see [Feature engineering](#feature-engineering)).
- **Docker files exist but are unverified in this build environment** — see the Deployment section's honest disclosure. No hosted/public cloud deployment yet; that's the last remaining step once Docker is verified to actually build (should be straightforward anywhere with normal container-registry access).