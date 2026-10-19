# Turbofan engine remaining useful life — survival analysis

**Live demo: [jetengines-rul.streamlit.app](https://jetengines-rul.streamlit.app/)** — pick a test engine and click *Predict remaining life* (the first load after a quiet period takes a minute or two).

- **Problem:** predict how many cycles a jet engine has left before failure, from 21 sensor readings per cycle (NASA C-MAPSS), with an uncertainty interval — not just a number.
- **Approach:** framed as survival analysis (time-to-failure), comparing Cox PH, Weibull AFT, a Random Survival Forest and two deep hazard networks.
- **Headline result:** the Random Survival Forest wins on NASA's official FD001 test set — **test C-index 0.864, mean error 11.3 cycles, NASA score 4.0 per engine**, a NASA score 2.8× better than Weibull's and 3.6× better than Cox's. The lead holds across 5-fold cross-validation and on the harder FD004 subset. Averaged with a gradient-boosting regressor, the deployed FD001 point estimate improves to **NASA 3.12 per engine, RMSE 14.2, C-index 0.877** (selected on cross-validation, promoted by the retraining gate).
- **Deployed:** RSF gives the prediction (on FD001, averaged with gradient boosting); the 90% interval combines RSF and Weibull AFT (lower bound = the more cautious of RSF's and Weibull's, upper bound = Weibull's), because each model's interval fails at one end of the remaining-life range. Served by a FastAPI service and a Streamlit dashboard.
- **Tested the premise:** with simulated censoring, survival models clearly beat a regression that treats unfinished engines as failed — but a regression that drops them was competitive, a result reported as found.
- **Engineering:** shared `src/` package, reproducible export script, 232 tests run by GitHub Actions on every push (including a check of the live demo's models), Docker images built and smoke-tested in CI, versioned models, drift monitoring and a champion/challenger retraining gate, live demo.

![Dashboard](docs/images/dashboard.png)

## Table of contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Key findings](#2-key-findings)
3. [How it works](#3-how-it-works)
4. [Run it](#4-run-it)
5. [Repository](#5-repository)
6. [Limitations](#6-limitations)
7. [Further reading](#7-further-reading)

---

## 1. Results at a glance

**Model comparison** — NASA's official FD001 test set (100 engines), all models trained on the same 80 engines:

| Model | NASA score ↓ | Test C-index ↑ | RMSE ↓ | MAE ↓ | Late predictions | 90% interval coverage |
|---|---|---|---|---|---|---|
| Cox PH | 1672.5 | 0.821 | 22.01 | 16.91 | 54% | — |
| Weibull AFT | 1288.2 | 0.818 | 23.14 | 18.27 | 48% | 90% |
| LSTM (deep, stretch)* | 921.3 ± 67 | 0.839 ± 0.005 | 17.80 | 12.83 | 60% | 75% |
| Conv1D→LSTM (deep, stretch)* | 968.8 ± 76 | 0.844 ± 0.003 | 18.24 | 12.96 | 57% | 78% |
| **Random Survival Forest** | **462.5** | **0.864** | **16.27** | **11.99** | 65% | 86% |

*Mean (± standard deviation) of 3 seeds each, all scored on the test set ([notebook 07](notebooks/07_deep_survival_stretch.ipynb), section 10); none of the 6 runs reaches RSF on any column.

**Deployed models** (retrained on all training engines; FD001's point estimate averages RSF with gradient boosting, FD004's is RSF alone):

| | FD001 (1 operating regime, 1 fault mode) | FD004 (6 regimes, 2 fault modes) |
|---|---|---|
| Test C-index | 0.877 (RSF alone 0.864) | 0.862 |
| NASA score per engine | 3.12 (RSF alone 4.05) | 22.4 |
| MAE / RMSE (cycles) | 10.4 / 14.2 (RSF alone 11.3 / 15.6) | 20.7 / 28.2 |
| 90% interval coverage | 97% | 89% |
| …within 25 cycles of failure | 100% (Weibull alone: 58%) | 94% (Weibull alone: 54%) |

All numbers are for the exported models, measured through the same `inference.predict` the API serves (`notebooks/14`, `notebooks/15`, `notebooks/18`).

**Robustness:**
- **5-fold CV:** RSF C-index 0.850 ± 0.019 vs. Cox 0.798 ± 0.052 and Weibull 0.803 ± 0.048. RSF is best in 4 of 5 folds and the most stable.
- **Harder subsets:** RSF C-index 0.869 on FD003 (2 fault modes) and 0.858 on FD002 (6 regimes).

---

## 2. Key findings

1. **RMSE would have picked the wrong model.** On FD001, Cox beats Weibull on RMSE (22.0 vs 23.1), but Weibull beats Cox on the NASA score (1288 vs 1673), because its errors fall on the safe (early) side. The asymmetric NASA score is the primary metric here: a late prediction means an engine fails in service.
2. **The best model is also the most often late.** RSF has the best NASA score *and* the highest late-prediction rate (65%). Its late errors are small, and the NASA penalty grows exponentially, so many small errors cost less than a few large ones.
3. **Both intervals failed a subgroup check, at opposite ends.** Conformal calibration brought RSF's 90% interval to exactly 90% overall coverage, 40% narrower than Weibull's, but it covered only 27% of engines above the 125-cycle training cap (Weibull: 91%). A later check of the deployed Weibull interval found the reverse: it covered only 58% of engines within 25 cycles of failure, and every miss promised more life than the engine had. The served interval now takes the more cautious lower bound of the two models and Weibull's upper bound, selected on training data only: near-failure coverage went to 100% (FD001) and 94% (FD004) for about one cycle of extra width. [Details](docs/technical_report.md#9-interval-calibration-two-subgroup-checks-at-opposite-ends).
4. **Survival vs. regression, measured.** With 50% of training engines censored, RSF gets RMSE 17.3. The regression that treats unfinished engines as failed gets 23.1 and predicts about 12 cycles too little life. The regression that drops unfinished engines ties RSF (17.4) and has a better NASA score, because the 125-cycle cap absorbs most of the bias from dropping them. RSF's measured edges are stability and using all the data. [Details](docs/technical_report.md#6-does-survival-analysis-actually-beat-regression-tested-with-a-mixed-answer).
5. **A better point estimate, judged by the NASA score first.** The RSF's error is concentrated: the 11 test engines with more than 125 cycles left are 11% of engines but a quarter of its squared error (it can't predict past the cap). Of the candidates fixed in advance, larger caps "won" on cross-validation only by predicting more life for very young engines and were rejected by a rule aimed at exactly that; averaging RSF with gradient boosting won on every metric, passed the test-set check and the retraining gate, and is now deployed for FD001. It is late on more engines (62% vs 56%) but by less: engines late by more than 30 cycles fell from 5 to 2. [Details](notebooks/18_point_estimate_candidates.ipynb). The same experiment on FD004 failed the pre-set rule and was **not deployed**, although the blend was better wherever the model is used (all 5 folds at ≤150 cycles left; test NASA 19.4 vs 22.4): the overall score it failed on came 99.99% from engines 150+ cycles from failure, where every model predicts just under the cap, and 95% from a single held-out point. The rule is fixed for future experiments rather than rewritten after the fact. [Details](notebooks/19_fd004_point_estimate.ipynb).
6. **Two evaluation bugs changed conclusions.** Comparing a row-level validation C-index (0.901) with an engine-level test C-index (0.864) had produced a fake "regression" after tuning. A feature misalignment on cross-validation folds had inflated a tuning result from 2 to 4 of 5 fold wins. Both were fixed, and every affected notebook was re-run. [All 7 bugs](docs/technical_report.md#11-bugs-caught-along-the-way).

---

## 3. How it works

**Data and labels.** NASA C-MAPSS: training engines run to failure; test engines are cut off before failure, with the true remaining life given separately. Each training row gets a `(duration, event)` label, with remaining life capped at 125 cycles, the standard C-MAPSS convention (`src/turbofan_rul/labeling.py`).

**Features.** 12 informative sensors (9 near-constant ones dropped), then rolling mean/std/min/max/slope over 5/15/30-cycle windows, exponentially weighted averages and cycle count — all computed per engine and backward-looking only. For the multi-regime subsets, sensors are first normalised within each of the 6 operating regimes, which are found by KMeans (`features.py`).

**Models.**

| Model | Role |
|---|---|
| Cox PH | Interpretable baseline. Its proportional-hazards test fails for 14 of 25 features, which motivates the next models. |
| Weibull AFT | Parametric wear-out model (shape ρ = 4.41 from engine lifetimes, i.e. rising failure risk). Provides the interval's upper bound, which can reach past the training cap. |
| Random Survival Forest | No proportional-hazards or distribution assumptions. Its point estimate is the area under the survival curve. Deployed for prediction, and its lower bound keeps the interval cautious near failure. |
| Gradient boosting | A regressor on the same capped target and features, averaged with RSF for FD001's point estimate (notebook 18); tested and not deployed for FD004 (notebook 19). Not used for the interval. |
| LSTM / Conv1D→LSTM | Discrete-time hazard networks on raw sensor windows (stretch). They didn't beat RSF. |

**Evaluation.** The NASA score is the primary metric:

```
penalty = exp(d/10) − 1   if d > 0  (late — dangerous)
penalty = exp(−d/13) − 1  if d ≤ 0  (early — merely costly)
```

where d = predicted − true RUL. It's reported alongside the test C-index, RMSE/MAE, late-prediction rate, interval coverage and per-risk-bin bias.

**Serving.** `export_deployment_artifacts.py` refits the chosen models on all training engines. `inference.py` is the single scoring function. The Docker setup runs a FastAPI service (`/predict`, `/subsets`, `/subsets/{subset}/metadata`) and a Streamlit dashboard that calls it over HTTP.

**The live demo** runs on Streamlit Community Cloud, which only hosts Streamlit apps. There, the dashboard runs in *embedded mode*: it downloads the exported models from a Hugging Face model repo and calls the same `inference.predict`, so predictions are identical in both modes. Library versions in `app/requirements.txt` are pinned to the export environment, because the models are pickled.

---

## 4. Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src                       # PowerShell: $env:PYTHONPATH = "src"
python -m turbofan_rul.export_deployment_artifacts --subset FD001
python -m turbofan_rul.export_deployment_artifacts --subset FD004   # optional
pytest tests/ -v                            # 232 tests

# Dashboard against the API (two terminals):
uvicorn api.main:app --reload
API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py

# Or the dashboard alone, loading the models itself:
MODELS_DIR=models streamlit run app/streamlit_app.py

# Or both services in Docker:
docker compose up --build
```

The notebooks in `notebooks/` run in order, `01_eda` through `19_fd004_point_estimate`.

**CI.** `.github/workflows/tests.yml` runs the test suite on every push. A second job installs exactly what the live demo installs (`app/requirements.txt`, Python 3.11), downloads the models from Hugging Face and runs the dashboard in embedded mode with real predictions (`tests/test_dashboard_embedded.py`), so a library-version mismatch with the uploaded models fails CI instead of the live site. It needs one repository variable: **Settings → Secrets and variables → Actions → Variables → `MODEL_REPO`** = your Hugging Face model repo. To run the embedded-mode test locally, set `MODELS_DIR=models` (or `MODEL_REPO`) before `pytest`.

**Docker CI.** `.github/workflows/docker.yml` builds both images with `docker compose`, starts them, and runs `scripts/docker_smoke_test.py` inside each container: the API must serve FD001 and FD004 predictions in the expected range (and reject a bad request with 422), and the dashboard must reach the API over the Compose network and make a prediction through it. It runs weekly, on demand, and on pushes that change anything inside the images (about 9 minutes, most of it the in-image model export).

**Operations.** Each exported model carries a `manifest.json` with a version derived from its files (e.g. `fd001-3328802d81b0`), the git commit, data hashes, library versions and test-set metrics; the API and dashboard report which version answered, and `python -m turbofan_rul.publish` uploads and tags versions on Hugging Face (the live demo is pinned to one with `MODEL_REVISION`). The API's `GET /drift` compares the last 500 requests with a training reference (PSI, thresholds calibrated on training engines): replays show no false alarms on FD001's and FD004's own fleets and immediate alarms for FD002 and FD003 traffic ([notebook 17](notebooks/17_drift_monitoring.ipynb)). `.github/workflows/retrain.yml` retrains a challenger, compares it with the published champion on fixed rules (`gate.py`) and publishes it only if it passes; a challenger trained on 30% of the engines was rejected (NASA score 13% worse). Every prediction also carries an **input check** (`input_check` in the API, a badge on the dashboard): *familiar*, *caution* (sensors outside the training range) or *unfamiliar* (operating conditions never seen in training). Calibrated on held-out training engines; on the test sets it marks 100% of FD001's engines familiar, 90% of FD002's unfamiliar and 45% of FD003's as caution under the FD001 model. The live demo lets visitors pick engines from other fleets to see it.

---

## 5. Repository

```
├── notebooks/            # 01 EDA → 12 interval calibration → 13 censoring experiment → 14–15 interval fix → 16 uncapped upper bound → 17 drift → 18 point-estimate candidates
├── src/turbofan_rul/     # data loading, labeling, features, models, evaluation, CV, conformal, inference, export,
│                         #   operations: manifest (versions), drift, gate (retraining), publish (Hugging Face)
├── tests/                # 232 tests (GitHub Actions: .github/workflows/tests.yml; retraining: retrain.yml)
├── scripts/              # Docker smoke test (GitHub Actions: .github/workflows/docker.yml)
├── api/                  # FastAPI service + Dockerfile
├── app/                  # Streamlit dashboard, Community Cloud requirements, Dockerfile
├── .streamlit/           # dashboard theme
├── docs/                 # technical report, project brief, architecture walkthrough
└── docker-compose.yml
```

---

## 6. Limitations

- **FD004 is a transfer test.** It deliberately reuses FD001's tuned hyperparameters, to check whether the FD001 recipe holds on a harder subset without re-tuning. FD002 and FD003 are evaluation-only.
- **Long-lived engines on FD004.** The interval covers 73% of FD004 test engines with more than 125 cycles left. These misses are on the safe side (the engine outlives the interval). An upper bound from an RSF trained without the 125-cycle cap reaches 100%, but makes intervals for healthy engines much wider and adds about 200 MB of models; it was measured and deliberately not deployed ([notebook 16](notebooks/16_uncapped_upper_bound.ipynb)). Without the cap, the *point estimate* gets worse (RMSE 24.4 → 28.0 out of fold on FD001), so the cap stays.
- **FD004's point estimate stays RSF-only.** The FD001 blend was tested on FD004 and failed the pre-set rule because of a measurement artifact (overall cross-validated NASA is dominated by a few very young engines), not because it was worse where it is used. It stays undeployed: the rule was not changed after it failed ([notebook 19](notebooks/19_fd004_point_estimate.ipynb)).
- **Censoring convention.** Training labels treat rows beyond the 125-cycle cap as failures at 125, the standard convention. Censoring those rows instead broke the Weibull model, so it was rejected.
- **The survival-vs-regression evidence comes from simulated censoring** on FD001, with untuned regressors.
- **The live demo runs in one process** (embedded mode), not as two services. This is a free-hosting constraint, and it means `/drift` exists only in the API deployment.
- **Operations are demonstrated, not battle-tested.** Drift is shown on replayed fleets (there is no live traffic), its window lives in memory and resets on restart, and the retraining gate judges challengers on the official test set, which is fine for a no-worse-than check but must not be used to pick between many challengers.

---

## 7. Further reading

- [`docs/technical_report.md`](docs/technical_report.md) — the full methods, every result table, the decisions and the bug log.
- [`docs/project_brief.md`](docs/project_brief.md) — the original specification and how each part changed.
- [`docs/architecture_walkthrough.md`](docs/architecture_walkthrough.md) — the build history, phase by phase.
