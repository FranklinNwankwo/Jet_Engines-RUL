# Project Brief: Turbofan Engine Remaining Useful Life (RUL) Prediction via Survival Analysis

**Author:** Franklin Nwankwo
**Status:** Updated to reflect the actual final implementation across all phases (originally drafted before Phase 1; this revision brings every section in line with what was actually built, including work that went beyond the original scope). Where the original plan changed, that's stated explicitly rather than silently edited over.

## Table of contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Modeling Approach](#4-modeling-approach)
5. [Feature Engineering](#5-feature-engineering)
6. [Evaluation Plan](#6-evaluation-plan)
7. [Deployment](#7-deployment)
8. [Repository Structure](#8-repository-structure)
9. [Deliverables](#9-deliverables)
10. [Known Gaps](#10-known-gaps)
11. [Success Criteria](#11-success-criteria)

---

## 1. Problem Statement

Predict the Remaining Useful Life (RUL) of aircraft turbofan engines from multivariate sensor telemetry, using NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. Each engine unit runs from a healthy state to failure under one of several operating regimes and fault modes, with 21 sensor channels and 3 operating-condition settings recorded per operational cycle.

This is framed and solved as a **survival analysis / time-to-event problem**, not as plain regression or classification. The distinction is not cosmetic, it determines whether the model is statistically valid at all under realistic deployment conditions.

### Why this is a survival problem, not a regression problem

In real predictive-maintenance deployments, the majority of monitored units are still operating at the time a model is built or queried, their true failure time is unknown and lies somewhere in the future. This is **right censoring**. A model trained naively on (features → observed RUL) using only completed failure trajectories, or one that treats a censored unit's last-known age as its true RUL, is trained on a biased sample: it implicitly excludes "units that would have survived longer than we watched," which systematically biases predictions toward shorter lifetimes.

Survival analysis methods instead model the conditional distribution of time-to-failure given "known to have survived to at least time *t*," which is the correct estimand for this problem. This project uses C-MAPSS's structure (complete run-to-failure trajectories for training, truncated/censored trajectories for test) to implement and evaluate that distinction explicitly and correctly, rather than falling into the common mistake of treating this as vanilla regression.

Because C-MAPSS's training engines all run to failure, the training data itself contains no censoring, so this argument was also **tested** rather than assumed: `notebooks/13_censoring_experiment.ipynb` simulates a fleet in which 25–75% of training engines haven't failed yet (Section 6, item 8). Survival models clearly beat a regression that treats unfinished engines as failed; a regression that simply drops them proved competitive on this dataset.

---

## 2. Objectives

1. Build a feature engineering pipeline that extracts degradation signal from raw, noisy multivariate sensor time-series.
2. Implement and compare a ladder of survival modeling approaches of increasing flexibility, with an explicit, defensible rationale for each.
3. Evaluate models using domain-appropriate metrics, not RMSE alone, including a concordance index and the asymmetric NASA scoring function from the C-MAPSS literature.
4. Deploy the best model as an API + dashboard that accepts recent sensor readings and returns a **RUL point estimate with a calibrated confidence interval**, not a bare number.
5. Produce a portfolio artifact (code, notebooks, README, live demo) that demonstrates survival-analysis fluency as a clear differentiator from typical "RUL regression on C-MAPSS" portfolio projects.

**All five objectives were met.** Beyond them, the project also ended up covering: 5-fold cross-validation of the model ranking, generalization checks on three additional C-MAPSS subsets, hyperparameter tuning validated against both CV and the official test set, a conformal-prediction fix for RSF's interval calibration, a near-failure fix for the deployed interval, containerization, and a public live demo ([jetengines-rul.streamlit.app](https://jetengines-rul.streamlit.app/)), some of which were in the original five objectives, added as the project's scope grew. See Sections 8–10 below for the honest state of each.

---

## 3. Dataset

**Source:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset.

**Primary subset:** FD001 (single operating condition, single fault mode — HPC degradation). Selected as the starting subset to keep the modeling problem well-posed while the survival pipeline is built and validated. **This remains the only subset with the full treatment** — tuned hyperparameters, conformal-calibrated intervals, and full deployment (API + dashboard).

**FD002/FD003/FD004 status (originally scoped as future work — since built, with real limits):**

- **FD002** (6 operating regimes, 1 fault mode) and **FD003** (1 regime, 2 fault modes) were run as **evaluation-only checks** — Random Survival Forest only, no hyperparameter search, no deployment — each isolating one complexity axis in turn (`notebooks/09_fd002_fd003_generalization_check.ipynb`). FD003 generalized close to the FD001 baseline; FD002 was a real step down in performance even with the regime-normalization fix working correctly.

- **FD004** (6 regimes, 2 fault modes, both axes combined) got the **full model ladder** (Cox PH, Weibull AFT, RSF) and full official-test-set evaluation (`notebooks/10_fd004_full_pipeline.ipynb`), confirming RSF's lead held under genuine distribution shift. It is also **deployed** (its own API/dashboard subset option), deliberately with FD001's tuned hyperparameters: FD004 is a transfer test of the FD001 recipe, not a second tuning exercise. Its deployed interval's coverage was checked on the official test set in `notebooks/14`–`15` (88.7% overall, 94% within 25 cycles of failure, 73% above the 125-cycle cap).

**Structure per subset:**

- `train_FD00X.txt`: full run-to-failure trajectories (unit, cycle, 3 operating settings, 21 sensor readings) — used as the **event** population (event = 1, failure observed).
- `test_FD00X.txt`: truncated trajectories, each ending at some cycle prior to failure — used as the **censored** population (event = 0, survival time only known to exceed the last observed cycle).
- `RUL_FD00X.txt`: ground-truth remaining cycles for each test engine at the point of truncation — used only for final evaluation, never as a training feature.

**Censoring structure, built explicitly:**

A `(duration, event)` pair is constructed for every row the models are fit or scored on. Training: one pair per cycle, `duration` = that cycle's remaining life capped at 125 (see the labeling convention in Section 4), `event` = 1, since every training engine runs to an observed failure. Test: one pair per engine at its last observed cycle, `event` = 0 (right-censored — failure not observed within the data). This is the standard input format for all survival libraries used in this project (`lifelines`, `scikit-survival`) and is treated as a first-class data artifact, not an implementation detail — see `src/turbofan_rul/labeling.py`.

---

## 4. Modeling Approach

Models are benchmarked as a ladder of increasing flexibility, each included with an explicit rationale rather than chosen by default:

| Model | Library | Rationale for inclusion | Key assumption/limitation | Outcome |
|---|---|---|---|---|
| Cox Proportional Hazards | `lifelines` | Reference standard in survival analysis; interpretable hazard ratios per feature; establishes a baseline concordance and a diagnostic surface (Schoenfeld residuals) for testing whether hazards are actually proportional over time | Assumes proportional hazards and (typically) linear covariate effects | **Confirmed violated**: 14/25 covariates fail the proportional-hazards test at p<0.05 (fit with engine-clustered robust standard errors; the count is if anything overstated, since the test treats correlated rows as independent) — the stated justification for trying Weibull and RSF next |
| Weibull AFT (Accelerated Failure Time) | `lifelines` | Mechanical wear-out is physically well-modeled by a Weibull hazard function; fully parametric, so RUL quantiles/confidence intervals come directly from the fitted distribution | Assumes a specific parametric family | A Weibull fit to the 100 engines' lifetimes gives shape ρ = 4.41 (95% CI [3.80, 5.01]), clearly > 1 — increasing hazard (wear-out), as predicted. (The covariate model's own shape, 2.67, isn't used as evidence: it depends on the row-level labels and its CI is far too narrow.) Not the concordance leader, but its intervals are unbounded and covered 90% of FD001 test engines (80% on the harder FD004) — kept as the source of the deployed interval's upper bound for that reason; its lower bound alone covered only ~55% of engines within 25 cycles of failure (notebook 14), so the deployed lower bound is the more cautious of Weibull's and RSF's |
| Random Survival Forest | `scikit-survival` | Drops both the proportional-hazards and parametric-distribution assumptions; captures nonlinear multi-sensor interactions | Its own confidence intervals under-cover; predictions structurally capped at the training time ceiling (~125 cycles) | **Deployment leader** on concordance and NASA score, confirmed across a single split, 5-fold CV, and the FD004 distribution-shift check. Interval under-coverage fixed via conformal calibration for FD001 (see Section 6) but the fix itself doesn't cover engines past the training ceiling — so the deployed interval takes only its lower bound from RSF (when more cautious than Weibull's) |
| Deep survival models (stretch) — discrete-time hazard head on raw sensor windows, LSTM and Conv1D→LSTM encoders compared | `tensorflow-cpu` (not PyTorch — see note below) | Tests whether temporal structure can be learned directly from raw sensor windows, with zero hand-engineered features, as a check on whether Phase 2's feature engineering was actually earning its keep | Didn't beat feature-engineered RSF with ~80 training engines; intervals under-cover (73–74% vs. 90% nominal) | Neither architecture beat RSF on the official test set (mean of 3 seeds each: C-index 0.839 LSTM / 0.844 Conv1D→LSTM vs. RSF's 0.864; NASA score 921 / 969 vs. 463). Across a 3-seed comparison, the plain LSTM had the higher validation concordance on all 3 seeds (0.880 vs. 0.871) — the CNN front-end didn't help at this dataset size. Not deployed, exactly as scoped: "justified only if it measurably improves on RSF." |

**Labeling convention:** Piecewise-linear RUL labeling, RUL is capped at a maximum value (125 cycles) for early-life cycles, since engines do not show meaningful degradation signal from cycle 1 and an uncapped linear RUL label would force models to fit a nonexistent early-life trend. Standard C-MAPSS convention, applied deliberately.

**Censoring the capped rows — tested and rejected.** With the cap, every training row keeps `event=1`, including the 8,131 of 20,631 FD001 rows sitting at the cap, which are therefore labeled as failing in exactly 125 cycles when their true remaining life is longer. The statistically cleaner alternative, marking those rows censored at 125 (`labeling.build_train_survival_frame(..., censor_above_cap=True)`), was tested: Cox improved slightly and RSF was unchanged, but Weibull AFT broke (FD001 NASA score from ~1.3×10³ to ~2×10¹⁰, 90% interval width from ~100 to 184 cycles). Since Weibull supplies the deployed interval, the default stays all-`event=1`.

**Deployment model selection:** Random Survival Forest for point estimate and risk ranking, Weibull AFT for the confidence interval, as originally anticipated in this section, and confirmed by the full evaluation in Phase 6. Notebook 18 later refined the point estimate for FD001: averaging RSF with a gradient-boosted regressor was selected on cross-validation by NASA score (−23% on the test set) and promoted by the retraining gate. The same experiment on FD004 (notebook 19) failed the pre-set rule because of a measurement artifact and was recorded, not deployed, so FD004 keeps RSF alone. A later subgroup check (notebooks 14–15) refined the interval: its upper bound stays Weibull's, but its lower bound is now the more cautious of Weibull's and RSF's, because Weibull's alone promised too much life to engines close to failure.

---

## 5. Feature Engineering

Raw sensor readings are noisy and autocorrelated; the degradation signal lives in the *trend*, not the instantaneous value. Features are engineered in layers:

1. **Sensor selection.** Drop near-constant sensors (near-zero variance) identified during EDA; document which sensors are kept and why. For FD001/FD003 (single regime), done via coefficient-of-variation screening on raw values. Caveat: CV divides by the mean, so FD001's `sensor_8` and `sensor_13` (both near 2388) are dropped partly because of their large means; much of the C-MAPSS literature keeps them, so this is a recorded judgment call, not proof they carry no signal. **For FD002/FD004 (multi-regime), this must happen *after* regime normalization (layer 6 below)** — raw variance on a multi-regime subset is dominated by which regime an engine is in, not by how informative the sensor is (confirmed: one sensor's raw variance on FD002 was 99.99% explained by regime, not degradation).
2. **Rolling statistics.** Rolling mean, standard deviation, min, and max per sensor over multiple window sizes (5, 15, 30 cycles).
3. **Trend features.** Rolling linear-regression slope per sensor over a window — a direct proxy for instantaneous rate of degradation.
4. **Exponentially weighted moving averages.** Spans 10 and 30, as a second smoothing/trend-extraction representation alongside the rolling windows (several rank among RSF's most important features). Computed with a per-engine `groupby().transform`, which stays aligned on any row index — an earlier `groupby().apply` version misaligned them on cross-validation folds (see Section 10).
5. **Cycle-based features.** Current cycle count (`cycle_count`) — deliberately *not* normalized by total lifetime, since that would require knowing an engine's eventual failure cycle (label leakage).
6. **Regime clustering + normalization (multi-regime subsets only — added when FD002/FD004 work began, not in the original plan).** KMeans on the 3 operating settings (`n_clusters=6`, confirmed by within-cluster std collapsing to ~0 on real data) identifies which of the discrete operating regimes an engine is in; sensor values are then z-scored *within* their regime, fit on training data only, before any rolling/EWMA feature is computed on them. This is what makes FD002/FD004 usable at all — see `src/turbofan_rul/features.py`.
7. **Composite health index (implemented, optional flag).** PCA reconstruction-error trained on early-life ("healthy") cycles, scored across the full trajectory as a single scalar "distance from healthy" signal (`features.add_health_index`) — available via `build_feature_matrix(..., include_health_index=True)`, off by default.

Every engineered feature is documented with a stated hypothesis for *why* it should correlate with failure risk. All feature-engineering logic lives in `src/turbofan_rul/features.py`, used identically at training time and at inference time, never duplicated between notebook and service code.

---

## 6. Evaluation Plan

RMSE alone is treated as insufficient and explicitly secondary. The evaluation suite is layered:

1. **Concordance index (C-index).** The primary discrimination metric, ranks units by relative risk, independent of predicted magnitude. Two variants are used and never mixed: a *row-level validation* C-index on held-out training engines (within the model notebooks) and an *engine-level test* C-index on the official test engines, which is the one used for every cross-model and cross-subset comparison.
2. **NASA asymmetric scoring function.** Implemented directly from the C-MAPSS literature (`src/turbofan_rul/evaluation.py`), not imported, penalizes late predictions (over-estimating remaining life) exponentially more than early ones, reflecting that a missed failure is worse than an early replacement. Unit-tested specifically for this asymmetry.
3. **RMSE / MAE.** Reported as secondary, diagnostic metrics only.
4. **Calibration.** Two checks: empirical coverage of each model's 90% interval on the test engines, and calibration-by-risk-bin — test engines grouped into quintiles of predicted RUL, comparing mean predicted vs. mean observed RUL per bin to see whether a model is biased specifically for the highest-risk engines. (The original plan's comparison of predicted survival curves against Kaplan-Meier curves was not carried out; interval coverage and per-bin bias were used instead.)

**Extended beyond the original plan, once Phase 6 raised the question of whether a single split could be trusted:**

5. **5-fold cross-validation** (`notebooks/08_cross_validation.ipynb`, `src/turbofan_rul/cross_validation.py`). Held-out engines scored at a randomly sampled mid-life truncation point (mirroring how NASA constructs the actual test set), since a training engine's final row trivially has RUL=0. Confirmed the RSF-leads ranking wasn't a single-split artifact: RSF's mean C-index was 0.850 vs. 0.798 (Cox) and 0.803 (Weibull), strictly best in 4 of 5 folds (Weibull won the fifth), and its mean NASA score was best, winning 3 of 5 folds. RSF's fold-to-fold standard deviation in concordance is also roughly a third of Cox's/Weibull's — the most stable model, not just the best on average.
6. **Hyperparameter tuning against that CV protocol** (`notebooks/11_hyperparameter_tuning.ipynb`). Cox/Weibull penalizers confirmed already near-optimal. RSF's `min_samples_leaf` retuned from 30 to 5. The CV support is mixed — higher mean concordance (0.855 vs. 0.841) but only 2 of 5 fold wins and about 2.5× the fold-to-fold variance — so the choice was checked on the official test set, like-for-like (both trained on all 100 engines): leaf=5 was better on every metric (C-index 0.859 → 0.864, NASA score per engine 4.51 → 4.04, MAE 12.0 → 11.3, RMSE 16.3 → 15.5, late rate 62% → 57%, coverage 86% → 87%), mostly by small margins. Deployed on that basis. An earlier version of this brief reported a concordance *drop* (0.901 → 0.864); that compared a row-level validation C-index with an engine-level test C-index and was not a real effect.
7. **Conformal interval calibration** (`notebooks/12_rsf_interval_calibration.ipynb`, `src/turbofan_rul/conformal.py`). Cross-conformal prediction fixed RSF's under-coverage (0.870 → 0.900 on the real test set, an empirical result rather than a guaranteed one) — but checking by subgroup found the aggregate win was hiding near-total failure (27% coverage) on engines above RSF's structural ceiling, exactly the case Weibull's interval exists to cover. This confirmed, at the time, the original Cox/Weibull/RSF pairing decision.

8. **Censoring experiment — testing the survival framing itself** (`notebooks/13_censoring_experiment.ipynb`, `cross_validation.sample_administrative_censoring` / `build_censored_train_frame`). Administrative, non-informative censoring is simulated on 0/25/50/75% of training engines (5 draws per level); RSF, Cox and Weibull are compared with two RSF-matched random-forest regressions — one dropping unfinished engines, one treating censoring as failure — on the official test set, against a decision rule written before running. Result, at 50% censoring: RSF test C-index 0.852 / RMSE 17.3 / NASA 5.7 per engine; "censoring = failure" regression 0.839 / 23.1 / 9.8 (biased 12 cycles early); "drop censored" regression 0.851 / 17.4 / 4.9 — i.e. tied with RSF on accuracy and better on NASA score. The pre-registered survival advantage over *both* regressions is therefore **not confirmed**: the 125-cycle cap absorbs most of the selection bias from dropping long-lived engines. RSF's measured advantages are stability across censoring draws (C-index std 0.003 vs 0.011 at 75%) and using all the data.

9. **Deployed-interval check and near-failure fix** (`notebooks/14_interval_coverage_check.ipynb`, `notebooks/15_interval_fix.ipynb`, `src/turbofan_rul/interval_calibration.py`). Scoring every test engine through the deployed `inference.predict` closed the FD004 calibration gap (calibrated RSF 87.9%, 55% above the cap) and found that the deployed Weibull interval covered only 58% (FD001) and 54% (FD004) of engines within 25 cycles of failure, every miss promising more life than the engine had. A replacement was selected on out-of-fold training data (each engine scored at 10 points across its life; six candidates; a fixed coverage-then-width rule): lower bound = the more cautious of RSF's and Weibull's, upper bound = Weibull's. Confirmed on the test sets: near-failure coverage 100% (FD001) and 94% (FD004), overall 97.0% and 88.7%, about one cycle wider. Now the deployed interval; no model was retrained.

10. **Training without the 125-cycle cap** (`notebooks/16_uncapped_upper_bound.ipynb`). Same out-of-fold protocol, with the cap removed. The point estimate gets worse (FD001 RMSE 24.4 → 28.0, C-index 0.864 → 0.852), so the cap stays. An upper bound from an uncapped RSF (leaf 30, about 200 MB) lifts above-cap test coverage from 91% to 100% (FD001) and from 73% to 100% (FD004), with intervals 27 and 60 cycles wider on average. Recorded, not deployed: those misses are on the safe side, the interval would over-cover (98–99%), and it would add a second forest per subset.

All models are compared in comparison tables across every axis above; the deployment model selection is justified against those tables, not a single number.

---

## 7. Deployment

**The architecture became two genuinely separate containerized services rather than one process; and the public demo, constrained by free hosting, runs the dashboard in a single-process embedded mode (see Frontend and Public demo below).**

- **Feature pipeline reuse.** The exact feature-engineering code path used in training is called at inference time (`src/turbofan_rul/inference.py`) — no re-implementation.

- **Model serialization.** Fitted models and preprocessing state (scalers, regime-clustering KMeans + normalization stats, the conformal interval correction) are serialized (`joblib`/JSON) via `src/turbofan_rul/export_deployment_artifacts.py` — a standalone, deterministic script, not dependent on notebook execution state. Refits on **all** training engines (evaluation's train/val split has already served its purpose by the time this runs).

- **API.** FastAPI (`api/main.py`) — `POST /predict` (subset-selectable, returns RUL point estimate, the combined RSF/Weibull 90% interval, a risk flag (critical when the interval's lower bound is under 20 cycles, warning when the point estimate is under 50), and, for comparison, Weibull's own interval and a supplementary conformally-calibrated RSF interval), plus `GET /subsets` and `GET /subsets/{subset}/metadata` so a frontend can discover available models and required input columns without touching model artifacts directly.

- **Frontend.** **Streamlit** (`app/streamlit_app.py`) Two modes: in the Docker deployment it is a **pure HTTP client** of the API (no model code imported; API address via `API_BASE_URL`); for the public demo it runs in **embedded mode**, downloading the exported models from a Hugging Face model repo (`MODEL_REPO`) and calling the same `inference.predict` the API uses, so predictions are identical in both modes. Engine selection or CSV upload (with a downloadable example file), predicted RUL / 90% interval / colour-coded risk level, a chart placing the prediction and intervals against risk zones, and per-sensor trend charts; themed via `.streamlit/config.toml`.

- **Public demo.** Live at [jetengines-rul.streamlit.app](https://jetengines-rul.streamlit.app/) on Streamlit Community Cloud, which can only host a Streamlit app, hence embedded mode rather than the two-service layout. Free hosts for a separate API were checked and ruled out (Hugging Face Docker Spaces now need a paid plan; Render's free tier is too small for the ~340MB of models; Cloud Run needs a billing card). Because the models are pickled, `app/requirements.txt` — the file Community Cloud installs — pins scikit-learn, scikit-survival, lifelines, numpy, pandas, scipy and joblib to the export environment's exact versions, on Python 3.11.

- **Containerization.** Separate Dockerfiles for the API (`api/Dockerfile`, bakes the FD001 and FD004 models in at build time) and the dashboard (`app/Dockerfile`, deliberately dependency-light since it holds no model code), plus `docker-compose.yml` for local orchestration. **Built and smoke-tested in CI.** The images are built on GitHub Actions (`.github/workflows/docker.yml`): both images are built and started with Docker Compose, and `scripts/docker_smoke_test.py` runs inside each container, real FD001 and FD004 predictions from the API, and a dashboard prediction made through the API over the Compose network. Runs weekly, on demand, and on pushes that change the images; the first run passed.

- **Operations.** Versioned models (`manifest.py`: content-derived version, provenance, test metrics; reported by the API and dashboard; published and tagged on Hugging Face by `publish.py`, with the live demo pinned to a tag), input drift monitoring in the API (`drift.py`, `GET /drift`, PSI with thresholds calibrated on training engines, demonstrated on replayed fleets in `notebooks/17_drift_monitoring.ipynb`), and a champion/challenger retraining gate (`gate.py`, `.github/workflows/retrain.yml`) that publishes a retrained model only if it is no worse on four fixed rules; a challenger trained on 30% of the engines was rejected. Every prediction also carries a per-engine input check (familiar / caution / unfamiliar), shown as a badge on the dashboard: 100% of FD001 test engines read familiar and 90% of FD002's unfamiliar under the FD001 model.

- **Testing.** 232 tests across 17 files (`tests/`), covering feature engineering, labeling, models, RSF prediction, deep-survival helpers, evaluation, cross-validation, regime clustering, conformal calibration, interval selection, inference, versioning, publishing, drift monitoring and the retraining gate. Beyond the unit tests, the API was checked end to end over real HTTP (including rejection of NaN/infinite inputs with a clean 422), and the dashboard was run headlessly (Streamlit `AppTest` plus browser screenshots at desktop and phone widths) in both modes for FD001, FD004, and the API-down case, giving identical predictions in both. Embedded mode is covered by `tests/test_dashboard_embedded.py`, which GitHub Actions runs against the live demo's Hugging Face models with the demo's pinned dependencies.

---

## 8. Repository Structure

```
turbofan-rul/
├── data/
│   ├── raw/                            # gitignored, regenerated via data_loading.download_cmapss()
│   └── processed/                      # small JSON/CSV artifacts kept; .parquet gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_cox_baseline.ipynb
│   ├── 04_weibull_aft.ipynb
│   ├── 05_random_survival_forest.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_deep_survival_stretch.ipynb          # optional, as planned -- LSTM + Conv1D->LSTM
│   ├── 08_cross_validation.ipynb               # added: 5-fold CV of the Phase 6 ranking
│   ├── 09_fd002_fd003_generalization_check.ipynb  # added: eval-only generalization
│   ├── 10_fd004_full_pipeline.ipynb            # added: full ladder on FD004
│   ├── 11_hyperparameter_tuning.ipynb          # added: tuning vs. CV and official test set
│   ├── 12_rsf_interval_calibration.ipynb       # added: conformal calibration fix
│   ├── 13_censoring_experiment.ipynb           # added: survival vs. regression under simulated censoring
│   ├── 14_interval_coverage_check.ipynb        # added: deployed intervals by subgroup (FD001 + FD004)
│   ├── 15_interval_fix.ipynb                   # added: near-failure fix, selected on training data
│   ├── 16_uncapped_upper_bound.ipynb           # added: training without the cap — measured, not deployed
│   ├── 17_drift_monitoring.ipynb               # added: drift monitor replayed on FD001-FD004 fleets
│   ├── 18_point_estimate_candidates.ipynb      # added: cap / blend candidates by NASA score; blend deployed
│   └── 19_fd004_point_estimate.ipynb           # added: the blend on FD004 — failed the rule, recorded, not deployed
├── src/turbofan_rul/
│   ├── data_loading.py                 # all 4 subsets, not just FD001
│   ├── features.py                     # rolling stats, trends, health index, regime clustering
│   ├── labeling.py                     # piecewise RUL labels, (duration, event) construction
│   ├── models.py                       # Cox/Weibull/RSF wrappers, save/load, engine-level split
│   ├── evaluation.py                   # C-index, NASA score, calibration
│   ├── plotting.py                     # shared dark-theme style() and color constants
│   ├── deep_survival.py                # added: from-scratch hazard MLP + Keras helpers
│   ├── cross_validation.py             # added: GroupKFold, truncation points, simulated fleet censoring
│   ├── conformal.py                    # added: cross-conformal interval calibration
│   ├── interval_calibration.py         # added: out-of-fold interval candidates + the served interval
│   ├── inference.py                    # added: single scoring function (API + dashboard embedded mode)
│   ├── manifest.py, publish.py         # added: model versions + Hugging Face publishing
│   ├── drift.py, gate.py               # added: drift monitoring + retraining gate
│   └── export_deployment_artifacts.py  # added: reproducible artifact build script
├── api/
│   ├── main.py                         # FastAPI: /predict, /subsets, /subsets/{s}/metadata
│   └── Dockerfile                      # added: bakes FD001 + FD004 models; built and smoke-tested in CI
├── app/
│   ├── streamlit_app.py                # NOT gradio_app.py -- switched to Streamlit on request
│   ├── requirements.txt                # added: Community Cloud deps for the live demo (pinned)
│   └── Dockerfile                      # added: built and smoke-tested in CI
├── .streamlit/config.toml              # added: dashboard theme
├── docker-compose.yml                  # added
├── requirements.txt                    # full dev/notebook dependencies
├── requirements-api.txt                # added: lean, API-only
├── requirements-app.txt                # added: leaner still, zero ML dependencies
├── .dockerignore                       # added
├── tests/                              # 232 tests across 17 files
├── models/
│   ├── fd001/                          # RSF + Weibull + metadata, tuned + calibrated
│   └── fd004/                          # RSF + Weibull + regime-clustering artifacts
├── docs/
│   ├── project_brief.md                # this document
│   ├── technical_report.md             # added: full methods, results, decisions and bug log
│   ├── architecture_walkthrough.md     # added: chronological build log, every phase
│   └── images/dashboard.png            # added: dashboard screenshot used in the README
└── README.md                           # short summary: problem, results at a glance, findings, how to run
```

Notebooks are used for exploration, derivation, and reporting. Any logic the API depends on lives in `src/`, imported identically by notebooks and services, never inline-only.

---

## 9. Deliverables

1. Cleaned, documented EDA identifying kept/dropped sensors with justification, done for FD001, FD002, FD003, FD004.
2. Reusable, unit-tested feature-engineering and labeling modules, done, plus regime clustering for multi-regime subsets.
3. Four trained and compared survival models, each with a stated rationale for inclusion — done; deep survival extended to two architectures (LSTM, Conv1D→LSTM).
4. A from-scratch mathematical derivation of the Weibull survival function, hazard function, and AFT log-likelihood, done, validated to ≤5.4×10⁻⁷ relative error against `lifelines`, including the likelihood's censoring term.
5. A model-comparison table across C-index, NASA score, RMSE/MAE, and calibration quality, with a stated deployment justification, done, and re-run under 5-fold CV and on FD004 to confirm it wasn't a single-split artifact.
6. A deployed FastAPI + **Streamlit** application returning RUL, confidence interval, and risk flag, done, as two separate HTTP services, containerized and smoke-tested in CI, with a public live demo at [jetengines-rul.streamlit.app](https://jetengines-rul.streamlit.app/).
7. A README presenting the project, done: a short summary with results at a glance, key findings and the live link, with the full methods and results moved to `docs/technical_report.md`.

**Beyond the original scope:** 5-fold cross-validation, generalization checks on FD002/FD003/FD004, hyperparameter tuning validated on the official test set, conformal interval calibration, a near-failure fix for the deployed interval, Docker containerization, a themed public demo on Streamlit Community Cloud, and an operations layer (model versioning, drift monitoring, a retraining gate).

---

## 10. Known Gaps

- **FD004 reuses FD001's tuned hyperparameters by design** — it tests whether the FD001 recipe transfers without re-tuning.

- **The operations layer is demonstrated, not battle-tested.** Drift is shown on replayed fleets, its 500-request window lives in the API's memory and resets on restart, and it only exists in the API deployment (the live demo has no API). The retraining gate judges on the official test set, fine for a no-worse-than check but not for choosing among many challengers.

- **FD004's interval under-covers long-lived engines.** 73% of FD004 test engines with more than 125 cycles left are covered (target 90%). The misses are on the safe side (the engine outlives the interval), and no calibration candidate in notebook 15 fixed them, because they share Weibull's upper bound. An upper bound from an RSF trained without the cap does fix them (100% on the test set), but was deliberately not deployed — see Section 6, item 10.

- **Training labels keep `event=1` at the 125-cycle cap.** The censored alternative was tested and rejected because it breaks the deployed Weibull interval (Section 4); the labels are therefore a convention, not an exact description of those rows.

- **A cross-validation bug was found and fixed late.** `features.add_ewm_features` used `groupby().apply` plus `reset_index`, which misaligned EWM features on any frame whose row index wasn't 0..n-1 — every CV fold, though not the full training frame or the deployed model. It affected only RSF's CV and tuning numbers (Cox/Weibull use no EWM features); all affected notebooks were re-run after the fix, and the main change was that the leaf=5 tuning result went from looking like a 4-of-5-fold winner to 2 of 5.

- **A fresh-clone smoke test was performed and caught two real bugs.** A tracked-files-only export of this repo into an isolated directory with a fresh virtual environment surfaced that both `api/main.py`'s startup check and `test_inference.py`'s skip condition were checking for a *tracked* file (`deployment_metadata.json`) as a proxy for whether a model was built, when the actual model file (`rsf.joblib`) is gitignored, invisible in a long-lived development environment where the model always existed once built, but a crash-on-missing-model bug on a genuinely fresh clone. Fixed in both places; re-verified with a full fresh install → model export → live API prediction → test run, all producing results identical to the main development environment. This closes the "does it work outside one long-lived sandbox" question for the Python pipeline specifically; the Docker layer is now verified separately, by the Docker CI job (Section 7).

- **The public demo is single-process.** It runs the dashboard's embedded mode on Streamlit Community Cloud rather than the two-service layout, because of free-hosting limits (Section 7). Its models are hosted in a Hugging Face model repo and must be re-uploaded after every re-export, with `app/requirements.txt`'s pinned versions kept matching the export environment, otherwise the live app serves stale models or fails to load them.

- **The survival-vs-regression evidence comes from simulated censoring.** FD001 only, one censoring mechanism, regressors with RSF-matched rather than tuned hyperparameters, and the 125-cycle cap, which, the experiment showed, largely neutralises the bias of dropping unfinished engines. An uncapped target would be the natural next test.

- **FD002/FD003 are evaluation-only by design**, not partial implementations, RSF only, no tuning, no deployment, deliberately the same weight as an EDA notebook rather than a second full build.

- **Phase 7's deep-learning numbers are 3 seeds per architecture, not more.** Training is deterministic (`keras.utils.set_random_seed` plus `tf.config.experimental.enable_op_determinism()`, so a given seed reproduces exactly), and every seed is scored on the official test set (notebook 07, section 10): none of the six runs reaches RSF, by margins several times the seed spread. Three seeds are enough for that conclusion but too few to rank the two deep architectures, which tie on the test set.

---

## 11. Success Criteria

- The censoring structure is correctly built and explicitly documented. **Met** — and the survival framing was tested under simulated censoring (Section 6, item 8), with a mixed result reported as measured.

- At least three survival models are implemented, benchmarked, and compared on identical held-out data. **Met and exceeded** — four models (including two deep-learning architectures), compared on a single split, 5-fold CV, and a fourth dataset subset.

- The NASA asymmetric scoring function is implemented from the literature and reported alongside C-index and RMSE/MAE. **Met.**

- Calibration is checked and reported for the deployed model's confidence intervals. **Met and extended** — conformal calibration added as a dedicated fix, then two subgroup checks at opposite ends of the remaining-life range, the second of which changed the deployed interval (notebooks 14–15).

- A live API + dashboard accepts sensor input and returns RUL with a confidence interval and risk flag. **Met** — publicly live at [jetengines-rul.streamlit.app](https://jetengines-rul.streamlit.app/) (embedded mode), and as two properly separated HTTP services in the Docker deployment (built and smoke-tested in CI).

- The README and project brief together allow a technical reviewer to understand *why* every major decision was made. **This document's revision is part of meeting that criterion** — every section above states what changed from the original plan and why, not just the final state.