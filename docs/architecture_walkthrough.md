# Architecture Walkthrough: How turbofan-rul Was Built, Step by Step

This is the "how did we get here" document; every phase, what triggered it, what was actually built, why, and what came out of it. Read top to bottom for the full architecture, or jump to any phase. `docs/project_brief.md` is the spec (what the project is); this is the build log (how it happened and why each piece exists).

## Table of contents

1. [The framing (before any code)](#1-the-framing-before-any-code)
2. [Phase 0: Project brief](#2-phase-0-project-brief)
3. [Phase 1: Data acquisition + EDA](#3-phase-1-data-acquisition--eda)
4. [Phase 2: Labeling + feature engineering](#4-phase-2-labeling--feature-engineering)
5. [Phase 3: Cox Proportional Hazards baseline](#5-phase-3-cox-proportional-hazards-baseline)
6. [Phase 4: Weibull AFT](#6-phase-4-weibull-aft)
7. [Phase 5: Random Survival Forest](#7-phase-5-random-survival-forest)
8. [Phase 6: Evaluation](#8-phase-6-evaluation)
9. [Phase 7: Deep survival model (stretch) — the phase with the most course-correction](#9-phase-7-deep-survival-model-stretch--the-phase-with-the-most-course-correction)
10. [Phase 8/9: Deployment (API + dashboard)](#10-phase-89-deployment-api--dashboard)
11. [Phase 10: README consolidation](#11-phase-10-readme-consolidation)
12. [Phase 11: 5-fold cross-validation](#12-phase-11-5-fold-cross-validation)
13. [Phase 12: FD004 generalization (a-through-d)](#13-phase-12-fd004-generalization-a-through-d)
14. [Phase 13: hyperparameter tuning](#14-phase-13-hyperparameter-tuning)
15. [Phase 14: RSF interval calibration](#15-phase-14-rsf-interval-calibration)
16. [Phase 15: proper two-service architecture + containerization](#16-phase-15-proper-two-service-architecture--containerization)
17. [Phase 16: fresh-clone verification](#17-phase-16-fresh-clone-verification)
18. [Phase 17: measuring the Phase 8/9 feature-scope bug instead of just documenting it](#18-phase-17-measuring-the-phase-89-feature-scope-bug-instead-of-just-documenting-it)
19. [Phase 18: correctness review and corrections](#19-phase-18-correctness-review-and-corrections)
20. [Phase 19: dashboard redesign + public deployment](#20-phase-19-dashboard-redesign--public-deployment)
21. [Phase 20: testing the survival framing itself](#21-phase-20-testing-the-survival-framing-itself)
22. [Phase 21: README rewritten for readers, detail moved to a technical report](#22-phase-21-readme-rewritten-for-readers-detail-moved-to-a-technical-report)
23. [Phase 22: continuous integration](#23-phase-22-continuous-integration)
24. [Phase 23: Docker, finally built](#24-phase-23-docker-finally-built)
25. [Phase 24: the deployed interval, checked at the other end](#25-phase-24-the-deployed-interval-checked-at-the-other-end)
26. [Phase 25: training without the cap, measured and not deployed](#26-phase-25-training-without-the-cap-measured-and-not-deployed)
27. [Phase 26: the operations layer — versions, drift, retraining gate](#27-phase-26-the-operations-layer--versions-drift-retraining-gate)
28. [Phase 27: a better point estimate, decided by the gate](#28-phase-27-a-better-point-estimate-decided-by-the-gate)
29. [Phase 28: the same experiment on FD004, recorded not deployed](#29-phase-28-the-same-experiment-on-fd004-recorded-not-deployed)
30. [Phase 29: every deep-model seed on the test set](#30-phase-29-every-deep-model-seed-on-the-test-set)
31. [Where things stand](#31-where-things-stand)

---

## 1. The framing (before any code)

**What was decided:** C-MAPSS turbofan dataset, framed as **survival analysis**, not regression. The reasoning that shaped everything downstream: most real-world equipment hasn't failed yet when you build a model, that's *right censoring*. A regressor trained on observed failures only, or one that treats a censored unit's current age as its true RUL, is trained on a biased sample. Survival analysis models "time-to-event given known to have survived this long," which is the correct estimand.

**Why it mattered structurally:** this decision is why the project has a `labeling.py` module building `(duration, event)` pairs instead of a single `RUL` regression target, why every model is a survival model, and why evaluation includes concordance/NASA-score instead of just RMSE.

---

## 2. Phase 0: Project brief

**What was built:** `docs/project_brief.md` — problem statement, objectives, dataset description, the model ladder (Cox → Weibull → RSF, with a stretch deep-learning phase), evaluation plan, deployment plan, repo structure, explicit non-goals (FD002-004 deferred), success criteria.

**Why:** locks in scope before writing code, so later phases point back to a decision record instead of re-litigating them. This is also the document referenced by name throughout every later phase ("see project_brief.md").

---

## 3. Phase 1: Data acquisition + EDA

**What was built:** `data_loading.py` (downloads from a verified GitHub mirror of the raw NASA files, validates row/unit counts against published dataset statistics after every download so a stale mirror fails loudly) and `notebooks/01_eda.ipynb` (raw sensor trajectory plots, per-sensor coefficient-of-variation screening, correlation heatmap, operating-condition variability check).

**Why URL download:** reproducibility; a fresh clone can regenerate the data without a manual step, and the mirror is checked so silently-wrong data can't slip in unnoticed.

**Why FD002-004 deferred:** they add six operating regimes instead of one, which needs an operating-condition clustering/normalization step the feature pipeline doesn't have. Attempting that alongside the core survival-analysis build would have split effort across two hard problems.

**Real result:** 9 of 21 sensors flagged as near-constant (dropped), 12 kept sensors, one flagged (not auto-dropped) redundant pair (`sensor_9`/`sensor_14`, r = 0.96). One caveat recorded in the notebook: CV divides by the mean, so `sensor_8` and `sensor_13` (both near 2388) are dropped partly because of their large means; much of the C-MAPSS literature keeps them, so this is a judgment call, not proof they carry no signal.

---

## 4. Phase 2: Labeling + feature engineering

**What was built:** `labeling.py` (piecewise-linear RUL capped at 125 cycles; the standard C-MAPSS convention, since early-life cycles carry no degradation signal, plus the `(duration, event)` construction: every training row gets `event=1`, since every training engine runs to an observed failure, and `duration` = the row's remaining life capped at 125 — so the 8,131 of 20,631 FD001 rows at the cap are labeled as failing in exactly 125 cycles, a convention rather than their exact time-to-failure; see Phase 18 for the censored alternative, tested and rejected) and `features.py` (rolling mean/std/min/max/trend-slope across 5/15/30-cycle windows, EWMA, all strictly grouped per-engine so no window ever crosses an engine boundary, checked directly in tests, not just assumed).

**Why this labeling scheme:** it's the one detail that makes the survival framing real rather than decorative, the exact mechanism by which "we know this engine fails eventually" gets encoded without ever using the future.

---

## 5. Phase 3: Cox Proportional Hazards baseline

**What was built:** `models.py`'s `train_val_split_by_engine` (splits on engine ID, never on rows, a row-level split would leak, since consecutive cycles from one engine are correlated) and `notebooks/03_cox_baseline.ipynb`, fitting Cox on a **curated 25-feature subset** (one rolling window's mean+slope per sensor, not all 205 engineered columns).

**Why curated features for Cox specifically:** Cox is being used *for* its interpretability, hazard ratios only mean something if the feature set is small enough to read. Feeding it all 205 correlated columns would produce technically-computable but untrustworthy coefficients.

**Real result:** held-out (row-level validation) concordance 0.875. More importantly: the Schoenfeld-residual proportional-hazards test showed **14 of 25 covariates violate the assumption** at p<0.05 (with engine-clustered robust standard errors, added in Phase 18; the count is if anything overstated, since the test treats correlated rows as independent). This number is the stated reason the next two models get tried, not just "more models to compare."

**Bug caught:** `check_assumptions` initially threw a shape-mismatch error from being passed the full 234-column frame instead of the 25 fitted columns. Fixed by switching to `lifelines.statistics.proportional_hazard_test` against the exact subset.

---

## 6. Phase 4: Weibull AFT

**What was built:** `notebooks/04_weibull_aft.ipynb` — the Weibull survival/hazard/AFT log-likelihood derived by hand in markdown, then validated by fitting an intercept-only version from scratch (NumPy + `scipy.optimize`) and checking it against `lifelines.WeibullFitter` on the same data.

**Why derive AND validate:** this is the project's recurring pattern; don't just cite a formula, prove the implementation matches it. Agreement here was **≤5.4×10⁻⁷ relative error** on both parameters, and 4.6×10⁻⁷ once the likelihood's censoring term was exercised too (8,031 censored rows — a check added in Phase 18).

**Real result:** a Weibull fit to the 100 engines' lifetimes gives shape **ρ = 4.41 (95% CI [3.80, 5.01])**, clearly > 1 — increasing hazard (wear-out), the physical prediction the Weibull choice was based on. (The covariate AFT model's own shape, 2.665 with CI [2.63, 2.70], was originally cited as this evidence; Phase 18 replaced it, since it depends on the row-level labels and its CI treats correlated rows as independent.) Validation concordance 0.864, slightly below Cox's 0.875. 90% confidence interval hit **90.0% empirical coverage** on the real test set, Weibull's actual value-add isn't beating Cox on ranking, it's producing calibrated intervals, which Cox can't.

**Caveat surfaced and documented:** predictions aren't capped at 125 even though training labels were, the fitted distribution is continuous and unbounded, so it can (correctly) predict beyond the cap for healthy-looking engines.

---

## 7. Phase 5: Random Survival Forest

**What was built:** RSF fit on the **full** feature set (229 columns, vs. Cox's 25) via `models.fit_random_survival_forest`, with row-subsampling (`row_stride=3`) justified by an actual measured comparison (strides 1/2/3/5 → validation concordance 0.9033/0.9013/0.9008/0.8972, with stride 1 costing about 7× stride 3's fit time) rather than picked by convenience.

**Why RSF gets the full feature set:** it doesn't share Cox's interpretability goal and handles collinearity natively.

**Real result:** validation concordance **0.901** (row-level, same split as Cox and Weibull), the best of the three so far, consistent with Phase 3's finding that proportional hazards is badly violated (RSF doesn't assume it). Permutation importance (measured on held-out engines, not training rows) showed engineered rolling/EWMA features outranking the raw sensors they came from, the empirical payoff of Phase 2's feature engineering.

**Real limitation found and fixed:** RSF's survival curve is only defined over the training time support (~125 cycles), so the **median** RUL estimate silently saturates at the ceiling for 28 of 100 test engines. Switched the point estimate to **restricted mean survival time** (area under the curve) instead, MAE improved from 13.35 to 11.99, and only 3 engines saturate instead of 28.

---

## 8. Phase 6: Evaluation

**What was built:** `evaluation.py` — the **NASA asymmetric scoring function**, implemented from the C-MAPSS literature (not imported), penalizing late predictions (predicted more life than exists which is outrightly dangerous) harder than early ones (predicted less, merely costly), via two different exponential-decay constants. Unit-tested specifically for the asymmetry: if the two constants were ever swapped, the metric would silently reward the wrong behavior.

**Why not RMSE as the headline:** RMSE is symmetric, it can't distinguish a dangerous late error from a merely-costly early one.

**Real findings, the two that matter most:**

1. **RMSE and the NASA score disagree** about which of Cox/Weibull is better. Cox wins RMSE (22.01 vs 23.14); Weibull wins NASA score (1288.2 vs 1672.5) because its errors sit on the safer side. A comparison run on RMSE alone would have picked the wrong model for the actual maintenance decision.

2. **RSF has the best NASA score AND the highest late-prediction rate (65%).** It's late more often but by smaller margins — the exponential penalty means many small late errors cost less than a few large ones.

**Engine-level test concordance** (added in Phase 18, so every cross-model comparison uses the same metric): Cox 0.821, Weibull 0.818, RSF 0.864.

**Deployment decision, made here:** RSF for point estimate/ranking, **Weibull AFT for the confidence interval** (RSF's own intervals under-cover at 86%, and RSF can't extrapolate past its training ceiling). Cox isn't deployed: RSF beats it on every point-estimate and ranking metric, and it has no natural interval. (Against Weibull, Cox isn't uniformly worse — it has the better test C-index, RMSE and MAE — but neither is the deployed point model.)

---

## 9. Phase 7: Deep survival model (stretch) — the phase with the most course-correction

This phase went through three distinct versions, which is worth tracing since it's the clearest example of the project's iterate-until-complete pattern.

**v1 — from-scratch NumPy MLP.** Built a discrete-time hazard neural net entirely in NumPy — forward pass, hand-derived backprop, gradient-checked against finite differences.

**v2 — Keras LSTM.** Used `tensorflow-cpu`, a genuine CPU-only PyPI package. Rebuilt around an actual `keras.layers.LSTM`, with the NumPy work repositioned as validation of the loss function (cross-checked to produce the identical value as the Keras loss on identical inputs) rather than the deliverable itself. Caught a real overfitting problem in the process, 150 fixed epochs scored *worse* than 80, fixed with early stopping on validation loss.

**v3 — follow-up checks.** The coverage weakness and (at the time) run-to-run non-determinism were documented in the notebook, quantifying both rather than leaving them for a reader to notice in a column of numbers.

**Effective Baseline:** Built the combined **Conv1D→LSTM** architecture and, given the just-documented non-determinism, compared it against the plain LSTM across **3 seeds each** rather than one run.

**Final honest result (re-run in Phase 18 with deterministic training — `keras.utils.set_random_seed` plus `tf.config.experimental.enable_op_determinism()`):** the plain LSTM had the higher validation concordance on all 3 seeds (0.880 vs. 0.871, a gap 2.5× the larger seed-to-seed std) — the CNN front-end didn't help on ~80 training engines. On the official test set, neither beats RSF: C-index 0.838 (LSTM) / 0.844 (Conv1D→LSTM) vs. 0.864, NASA score 847 / 926 vs. 463, interval coverage 74% / 78% (seed 42; all three seeds were later scored on the test set, Phase 29). Reported as directly as a win would have been. (An earlier version called the two architectures "statistically tied" at 0.879 vs. 0.871 and compared them with RSF's 0.901 — a validation figure; see Phase 18.)

---

## 10. Phase 8/9: Deployment (API + dashboard)

**What was built:** `export_deployment_artifacts.py` (a standalone, deterministic script, not notebook state, that refits RSF + Weibull on **all** training engines, since the 80/20 split's job was evaluation, which is already done); `inference.py` (the single scoring function, at this point imported directly by both the API and the dashboard rather than duplicated — the dashboard moved to HTTP-only in Phase 15, then gained an optional embedded mode for the public demo in Phase 19); `api/main.py` (FastAPI, tested with real predictions and all error paths via `TestClient`); `app/streamlit_app.py` (verified end-to-end through headless-browser automation, not just "it started without crashing").

**Real bug caught:** building the export script forced explicit enumeration of every column RSF needs, which surfaced that Phase 5's "full feature set" silently included raw passthrough values for all 21 sensors and 3 operating settings, not just the 12 Phase 1 flagged as informative. **Decision at the time:** keep the model exactly as evaluated (retraining then would have invalidated the reported Phase 6 numbers), document the gap in three places, and adjust the API contract to require the raw telemetry the model actually needs. **Later measured, not left as an assumption — see Phase 17.**

**Also caught:** RSF's point estimate can fall outside Weibull's own confidence interval (they're independent models scoring the same input so nothing forces agreement). Surfaced explicitly in the API response (`point_estimate_outside_interval`) rather than left to look like a bug.

---

## 11. Phase 10: README consolidation

**What changed:** the README had grown into a phase-by-phase status log through the build. Rewritten as a top-to-bottom narrative; problem framing → censoring argument → feature engineering → model ladder (each model justified by what the previous one's diagnostics showed) → evaluation methodology → results → deployment → a dedicated "bugs caught along the way" section → scope notes. `project_brief.md` stayed untouched as the technical spec; the README became the story.

---

## 12. Phase 11: 5-fold cross-validation

**What was built:** `cross_validation.py` — `GroupKFold` by engine (5 folds), with held-out engines scored at a **randomly sampled mid-life truncation point** (30-90% of observed lifetime) rather than their final cycle, which would trivially be RUL=0 for a full run-to-failure trajectory. This mirrors how NASA actually built the official test set.

**Real result (after the Phase 18 EWM fix):** RSF has the strictly best concordance in 4 of 5 folds — in the fifth (fold 2), Weibull is best (0.883) and RSF ties Cox (0.878) — and the best mean NASA score, winning 3 of 5 folds. That confirms the Phase 6 ranking isn't an artifact of one lucky split. The more interesting finding: **RSF's standard deviation across folds (0.019) is roughly a third of Cox's and Weibull's (~0.05)** — it's not just the best model on average, it's the most *stable* one, a property the original single split had no way to show. Absolute numbers are a little lower than on the official test set (RSF 0.850 vs. 0.864, both engine-level) — expected, since random mid-life truncation is a different task than NASA's specific chosen truncation points, and the CV RSF uses a coarser `row_stride=5`.

---

## 13. Phase 12: FD004 generalization (a-through-d)

**12a — the missing pipeline piece.** `features.py` gained regime clustering (`fit_regime_clusters`, KMeans on the 3 operating settings) and within-cluster normalization, validated on real FD002 data: one sensor's raw variance was **99.99% explained by which regime an engine was in**, not degradation, confirming the step does real work, not assumed to.

**12b — FD002/FD003, evaluation-only.** RSF only, no tuning, no deployment, deliberately the same weight as the original Phase 1 EDA. FD003 (fault modes only) generalized close to the FD001 baseline; FD002 (regimes only) was a real step down even with the clustering fix working correctly.

**12c — FD004, full ladder.** Cox PH, Weibull AFT, and RSF, evaluated against FD004's official test set exactly as Phase 6 did for FD001. RSF won again (concordance 0.849 vs. Cox's 0.811 and Weibull's 0.812; NASA score per engine 27.5 vs. 54.6 and 69.8), a third independent piece of evidence for the same ranking, this time under genuine distribution shift rather than just a different split of the same data. One FD001 finding didn't carry over: on FD004, Cox beats Weibull on both RMSE and NASA score, so FD001's RMSE-vs-NASA disagreement isn't a general rule.

**A real bug caught mid-build:** `data_loading.py`'s published-statistics validation — built in Phase 1 to catch a corrupted mirror, caught that FD004's train/test engine counts were transposed in this project's own reference table (248/249 instead of 249/248), a transcription error from when FD002-004 support was first added. The check did exactly its job, just against an internal bug instead of the external one it was written for.

**12d — deployment extended, not tripled.** `export_deployment_artifacts.py`, `inference.py`, the API, and the dashboard all became subset-parameterized. Verified two ways: FD001's predictions came out byte-identical to before the refactor (nothing broke), and FD004 worked end-to-end through both `TestClient` and a real headless-browser click-through of the subset selector.

---

## 14. Phase 13: hyperparameter tuning

**What was built:** Cox/Weibull penalizer search across a full 5-fold grid — the existing default (0.1) turned out already near-optimal, confirmed rather than assumed. For RSF, `max_features="sqrt"` clearly beat `"log2"`; a narrower single-fold search found leaf sizes 3, 5 and 8 tied, and `min_samples_leaf=5` was compared with the original default of 30 across all 5 folds.

**The cross-validation evidence (as re-run after the Phase 18 EWM fix) is mixed:** leaf=5 has the higher mean concordance (0.855 vs. 0.841) but wins only 2 of 5 folds — mostly from one big gain in fold 2 — and has about 2.5× the fold-to-fold variance (std 0.032 vs. 0.013), giving up the stability Phase 11 had flagged as one of RSF's virtues.

**So it was checked on the official FD001 test set, like-for-like** (both trained on all 100 engines, same stride): leaf=5 was better on every metric — C-index 0.859 → 0.864, NASA score per engine 4.51 → 4.04, MAE 12.0 → 11.3, RMSE 16.3 → 15.5, late rate 62% → 57%, coverage 86% → 87% — mostly by small margins. Deployed on that basis, with the weak CV support stated. A cost not shown in that table: the tuned model's file size grew from ~44MB to ~216MB (smaller leaves mean deeper, larger trees).

**Corrected in Phase 18:** the original version of this phase reported CV wins in 4 of 5 folds (inflated by the EWM bug) and a concordance "regression" on the test set (0.901 → 0.864). The regression wasn't real: 0.901 was a row-level validation C-index from an 80-engine model, compared against an engine-level test C-index from a 100-engine model.

---

## 15. Phase 14: RSF interval calibration

**What was built:** cross-conformal prediction (`conformal.py`) — each of the 100 training engines scored by a model fit on the other 4 folds, pooling calibration data without holding any engine out of the final deployed model. Calibrated correction: 5.00 cycles, added to both interval bounds (re-run in Phase 18 with the fold models using the deployed `row_stride=3`, so the notebook's correction matches the deployed artifact's exactly).

**Validated on the real official test set, and here's where it got interesting:** coverage moved from 0.870 to almost exactly 0.900 — it held on unseen engines, not just on the calibration set, though as an empirical result rather than a guarantee. In aggregate, the calibrated RSF interval now looked *strictly better* than the deployed Weibull interval: better coverage (0.900 vs. 0.890) and 40% narrower (59.6 vs. 98.6 cycles mean width). That would have justified simplifying the whole deployment to RSF alone, except checking by subgroup (the same instinct behind Phase 6's calibration-by-risk-bin analysis) found the aggregate number was hiding a near-total failure: coverage on the 11 engines above RSF's structural training-time ceiling collapsed to 27.3%, versus Weibull's 90.9% on those same engines. A uniform correction can't manufacture support a model never had.

**Decision:** Weibull stays the primary deployed interval, this result *confirmed* the original Phase 6 pairing for a reason now demonstrated, not assumed. The calibrated RSF interval ships as a clearly-labeled, explicitly non-authoritative supplementary field. The export script computes the correction for every subset it builds, FD004 included (36.00 cycles), though FD004's calibrated coverage wasn't checked on its test set until Phase 24, which also changed the deployed interval.

---

## 16. Phase 15: proper two-service architecture + containerization

**The HTTP refactor.** `app/streamlit_app.py` had been calling `inference.predict()` in-process as a documented demo simplification. Now the dashboard imports no modeling code at all, and calls the API's `/predict`, `/subsets`, and new `/subsets/{subset}/metadata` endpoints (added so the frontend can discover required columns without touching model artifacts) over HTTP, with the API's address read from an `API_BASE_URL` environment variable rather than hardcoded. Verified for real, a live `uvicorn` server and a live Streamlit process, driven through an actual browser click via Playwright, with the API's own access log confirming the full round-trip (`/health` → `/subsets` → `/subsets/{subset}/metadata` → `/predict`) actually happened.

**A direct, measurable consequence of that refactor:** splitting `requirements.txt` into `requirements-api.txt` and `requirements-app.txt` showed the dashboard's dependency list no longer needed scikit-learn, scikit-survival, or lifelines at all, it literally cannot load a model anymore, by construction. (Still true of the Docker dashboard image; Phase 19 added a separate, opt-in embedded mode for the public demo.)

**Docker: written, honestly not verified.** Separate Dockerfiles for the API (bakes the FD001 model in at build time, accepting a slow build for an instant start — FD004 added in Phase 18) and the dashboard (lean, zero ML dependencies), plus a `docker-compose.yml`. Installing Docker itself and starting its daemon worked fine in this sandbox — but every `docker build` failed with a 403 on the base-image pull, from both Docker Hub and GHCR. The sandbox's network allowlist covers package registries (PyPI, npm, GitHub) but no container registry at all. Disclosed directly rather than claimed as working: this is the one piece of the project that was written to standard patterns and reasoned through, but not executed until Phase 23 built it in CI.

---

## 17. Phase 16: fresh-clone verification

**What was done:** the repo was initialized as a git repository specifically so `.gitignore` could be trusted to define exactly what a real clone would contain, then `git archive` exported precisely that tracked set into a completely separate directory. A fresh Python virtual environment, `pip install -r requirements.txt` from nothing, and the full test suite immediately surfaced 19 errors, not because anything was broken, but because the fresh directory correctly had no pre-built models (as designed — they're gitignored, too large to version).

**Two real bugs, invisible until this exact test.** Both `api/main.py`'s startup check and `test_inference.py`'s skip condition inferred "is a model built?" from the presence of `deployment_metadata.json`, a small file that IS tracked in git. The actual model (`rsf.joblib`) is not. Fixed in both places to check for `rsf.joblib` directly.

**Re-verified for real, not just patched and assumed fixed:** a full FD001 export run from the fresh clone (genuine downloads, confirmed by the log — nothing cached), the API starting and serving a live prediction over HTTP, and the complete test suite; every result identical to the main development environment, including the exact same conformal correction (5.00 cycles) and the exact same point prediction (116.9 cycles) seen throughout this entire project. `project_brief.md` and this document were both updated to reflect that this gap is closed, and the README gained a fifth documented bug.

---

## 18. Phase 17: measuring the Phase 8/9 feature-scope bug instead of just documenting it

**What prompted it:** a direct challenge to the Phase 8/9 decision — "keep the model as evaluated, document the gap" was pragmatic, not evidenced. No one had ever measured how much the leak actually mattered, and every phase since (CV, tuning, calibration) had silently inherited the same confounded feature set. A proposal to restrict RSF to Cox/Weibull's exact 25-column curated set was considered and set aside: that would strip not just the 24 columns that were never supposed to be there, but 180 of RSF's 205 legitimately-scoped engineered columns too — a materially different, narrower experiment than fixing the actual bug.

**What was measured:** a second RSF, identical hyperparameters (`min_samples_leaf=5`, matching the Phase 13-tuned deployed model), fit on the corrected 205-column engineered-only feature set, evaluated against the same official FD001 test set as the deployed 229-column model. Every metric moved by less than ordinary random-forest run-to-run variance, in no consistent direction: C-index −0.0004, NASA score actually 0.065 *better*, MAE +0.09, late rate +0.03, coverage −0.03. The "plausible, given tree ensembles' random per-split feature subsampling" hypothesis from Phase 8/9, now confirmed with a number instead of asserted.

**Decision:** nothing retrained, nothing rerun. Per the plan agreed before running the ablation — a negligible result means tightening the documentation, not redoing Phase 6, the CV baseline, tuning, or calibration on a corrected model. `README.md`'s bug #1 now reports this comparison in place of the earlier unmeasured "documenting the gap" language (added in Phase 18 — an earlier version of this document said it had been added before it actually was). The ablation was run on the full training frame, which the Phase 18 EWM bug didn't affect, so its numbers stand.

---

## 19. Phase 18: correctness review and corrections

**What prompted it:** a file-by-file review of every source file, test, notebook, deployment file and document, checking each claim against what the code actually does and re-running everything affected.

**Real bugs found and fixed:**
- **EWM features misaligned on CV folds.** `features.add_ewm_features` used `groupby().apply` + `reset_index(level=0)`, which only lines up on a frame whose index runs 0..n-1. The full training frame (and so the deployed model) was fine, but every CV fold's filtered frame got misaligned or missing EWM values. Fixed with `groupby().transform`; `build_feature_matrix` now rejects frames with duplicate indexes or rows out of cycle order. It affected only RSF's CV and tuning numbers — and it mattered: Phase 13's leaf=5 result went from "wins 4 of 5 folds" to 2 of 5.
- **Two C-indices compared as one.** Row-level validation C (0.901 for RSF) had been put side by side with engine-level test C (0.864). Every model now reports an engine-level test C-index, used for all cross-model, cross-notebook and cross-subset comparisons; this removed Phase 13's phantom "regression" and corrected Phase 7's and Phase 12's comparisons.
- **API returned 500 instead of 422 on infinite inputs.** `SensorReading` now rejects NaN/infinity (`allow_inf_nan=False`), and a custom validation-error handler stops the error response from trying to echo the invalid number back. The dashboard also gained readable API-error messages and a clean message if the API goes down mid-session.
- **Dependency floors.** `requirements-api.txt` gained `joblib` (imported directly by `models.py`); `requirements-app.txt` raised Streamlit to `>=1.50`, since older versions reject the `width="stretch"` charts the dashboard uses (raised again to `>=1.51` in Phase 19, for the new Altair charts).

**Method changes:**
- Cox fit with engine-clustered robust standard errors (14/25 PH violations, not 18/25); Weibull's wear-out evidence switched to the per-engine lifetime fit (ρ = 4.41); a censoring-term check added to the Weibull derivation.
- Keras training made deterministic, so deep-model results reproduce exactly per seed.
- RSF calibration notebook aligned with the deployed recipe (`row_stride=3`).
- Input validation tightened in `data_loading.py`, `evaluation.py` and `cross_validation.py` (which now raises instead of silently dropping engines).

**Tested and rejected — censoring the capped rows.** `labeling.build_train_survival_frame(..., censor_above_cap=True)` marks rows above the 125-cycle cap as censored instead of failing at 125. Cox improved slightly and RSF was unchanged, but Weibull AFT broke (FD001 NASA score from ~1.3×10³ to ~2×10¹⁰, interval width ~100 → 184 cycles). Since Weibull supplies the deployed interval, the default stays all-`event=1`.

**Deployment:** `api/Dockerfile` now exports FD004 as well as FD001; both exports and the API serving both subsets were reproduced outside Docker in a clean Python 3.12 environment, and `docker compose config` accepts the compose file.

**Re-run:** all 12 notebooks re-executed on the corrected code; the test suite grew to 134 tests across 10 files (146 after Phase 20); README and project brief updated to match.

---

## 20. Phase 19: dashboard redesign + public deployment

**What prompted it:** the dashboard worked but looked like a default Streamlit page, and the project had no public URL.

**The redesign.** A theme in `.streamlit/config.toml` (dark navy/amber "engine console" palette, Inter / Space Grotesk / JetBrains Mono fonts) and a rebuilt `app/streamlit_app.py`: headline cards for predicted RUL, the Weibull interval and a colour-coded risk level; one "where this engine sits" chart showing the point estimate, both intervals and the true RUL against shaded risk zones; per-sensor small multiples with a rolling-mean trend; an example-CSV download on the upload tab; and results that persist across reruns. Charts use Altair (ships with Streamlit). Checked in a headless browser on desktop and phone widths; two chart bugs found that way were fixed (interval bars not drawing because layers shared a colour scale, and labels colliding). Testing the new charts on older Streamlit versions showed `st.altair_chart(width=...)` needs 1.51, so the floor moved from 1.50 to 1.51.

**The hosting decision.** The obvious free host for the API, a Hugging Face Docker Space, now requires a paid plan; Render's free tier (512MB) is too small for the ~340MB of models; Google Cloud Run needs a billing card. Streamlit Community Cloud is free but only runs a Streamlit app. So the dashboard gained a second mode rather than the project gaining a paid service: when `API_BASE_URL` is unset and `MODEL_REPO` is set, it downloads the exported `fd001/` and `fd004/` folders from a public Hugging Face model repo and calls `inference.predict` in-process. The model code is imported lazily, so the Docker dashboard image (HTTP mode) still never installs it. Measured first: both bundles load in ~0.6GB and about 1.5s, comfortably inside Community Cloud's limits.

**A real problem caught before it shipped:** the first embedded-mode test failed to load the Weibull model — test models pickled under pandas 3 could not be unpickled under pandas 2. Pickled scikit-learn/scikit-survival/lifelines models are tied to the library versions that saved them, so `app/requirements.txt` (the file Community Cloud reads, which sits next to the entrypoint and so takes precedence over the root `requirements.txt`) pins those libraries to the export environment's exact versions; the app now reports a version mismatch clearly instead of crashing. A second version issue surfaced during deployment: a fresh local venv had silently been created with Python 3.14, which TensorFlow doesn't support yet and which forced a newer numpy; the venv was rebuilt on Python 3.11, the models re-exported in it, and the pins regenerated from its `pip freeze`, with Community Cloud set to 3.11 to match.

**Verified:** identical predictions in both modes (FD001 test engine 1: 121 cycles [52, 237]; FD004: 33 [13, 59]); HTTP mode re-tested in an environment with no ML libraries installed; the live app at [https://jetengines-rul.streamlit.app/](https://jetengines-rul.streamlit.app/) runs end to end. Housekeeping found along the way: research models that notebooks 03/04/05/07 save directly into `models/` weren't covered by `.gitignore` (now ignored), and `.gitignore`/`.dockerignore` had excluded the whole `.streamlit/` folder, which would have kept the theme off GitHub and out of Docker (now only `secrets.toml` is excluded).

---

## 21. Phase 20: testing the survival framing itself

**What prompted it:** a review of the project pointed out that its central claim — survival analysis is necessary because of right censoring — had never been tested. Every C-MAPSS training engine runs to failure, and the only censoring ever tried (capped rows, Phase 18) was rejected, so the training data contained no censoring at all.

**What was built:** two helpers in `cross_validation.py` with 12 new tests. `sample_administrative_censoring` simulates a fleet observed up to a fixed date: each engine's observation window is drawn independently of its lifetime (non-informative censoring), with a fleet-wide scale set so an exact fraction of engines is censored. `build_censored_train_frame` truncates those engines and labels their rows (`event=0` inside the window, the usual capped `event=1` when the window reaches 125 cycles), hiding their true RUL. A test confirms that truncating after featurizing equals featurizing the truncated trajectories. `notebooks/13_censoring_experiment.ipynb` runs RSF, Cox, Weibull and two RSF-matched random-forest regressions ("drop unfinished engines" and "treat censoring as failure") at 0/25/50/75% censoring with 5 draws each, scored on the official test set, against a hypothesis and decision rule written into the notebook before it ran.

**Real result — partly confirmed:** survival models decisively beat the "censoring = failure" regression (50% censoring: RMSE 17.3 vs 23.1, C-index 0.852 vs 0.839; that regression is biased about 12 cycles early, 22 at 75%). But the "drop unfinished engines" regression tied RSF on C-index and RMSE (0.851 / 17.4) and beat it on the NASA score (4.9 vs 5.7 per engine), because RSF drifts slightly late under censoring. The pre-registered rule — RSF beating *both* regressions by more than the noise — was not met. The explanation is the 125-cycle cap: rows far from failure share the same capped target, so discarding long-lived engines loses little. RSF's measured advantages are stability (C-index std across draws 0.003 vs 0.011 at 75%) and using all the data. Weibull, notably, did not blow up under engine-level censoring.

**Decision:** nothing in the deployment changes — RSF + Weibull remain the best-supported pairing, and the Weibull interval is still something regression doesn't provide — but the README and brief now state the survival case as measured ("beats the naive regression; more stable; ties a careful one on accuracy here") rather than as an unqualified premise.

---

## 22. Phase 21: README rewritten for readers, detail moved to a technical report

**What prompted it:** the README had grown to 320 lines of method detail, with the headline results around line 70 and a defensive tone ("checked, not assumed", "stated plainly") repeated throughout. A recruiter or interviewer reads the top of a README and stops.

**What changed:** the README is now about 140 lines: a ten-line summary (problem, approach, headline result, deployment, the censoring finding, live link), a dashboard screenshot, a "Results at a glance" section with both result tables, five key findings, a short how-it-works section, run commands, limitations and links. The previous README's full content moved to `docs/technical_report.md`, with the defensive phrasing cut from 39 instances to 6 and the Docker disclosure condensed. No result or claim was changed.

---

## 23. Phase 22: continuous integration

**What was built:** `.github/workflows/tests.yml` with two jobs. `unit-tests` installs `requirements.txt` on Python 3.11 and runs the whole suite on every push and pull request (model-dependent tests skip, as on a fresh clone). `live-demo-models` installs `app/requirements.txt` — exactly what Streamlit Community Cloud installs — downloads the models from the Hugging Face repo named in the `MODEL_REPO` repository variable, and runs the new `tests/test_dashboard_embedded.py`: the dashboard is started headlessly in embedded mode and makes real FD001 and FD004 predictions.

**Why the second job matters:** the live demo's main failure mode is the pinned library versions drifting away from the environment that exported the uploaded models. Checked both ways before shipping: with version-matched models all 4 embedded-mode tests pass; with models exported under pandas 3, 3 of 4 fail. The suite was then 150 tests across 11 files.

---

## 24. Phase 23: Docker, finally built

**The gap:** since Phase 15 the Dockerfiles had been written and reasoned through but never built, because the sandbox blocked every container registry. That was the last claim in the project resting on reasoning rather than execution.

**What was built:** a separate workflow, `.github/workflows/docker.yml`, kept out of `tests.yml` because the API image exports both models during its build. It runs `docker compose build`, starts both services with `docker compose up --wait` (gated on the API's healthcheck), waits for the dashboard's health endpoint, then pipes `scripts/docker_smoke_test.py` into each container:

- **API container:** `/health` lists FD001 and FD004; test engine 1 of each gets a prediction in the expected range with an ordered interval; an empty request is rejected with 422.
- **Dashboard container:** the API is reachable at `http://api:8000` over the Compose network, and the real dashboard, run headlessly in HTTP mode, makes a prediction through it.

Running the checks inside the containers, rather than from the runner, is what tests the service-name networking the deployment depends on.

**Verified:** the smoke script was first run against the real API and a dashboard environment with no ML libraries, outside Docker (all checks pass, same 121 [52, 237] and 33 [13, 59] predictions). The first GitHub Actions run then built both images and passed in about 9 minutes. The job runs weekly, on demand, and on pushes that touch anything inside the images.

---

## 25. Phase 24: the deployed interval, checked at the other end

**What prompted it:** a review flagged that FD004's calibrated RSF interval had never been checked on its test set. Closing that took one evaluation, not retraining: `notebooks/14_interval_coverage_check.ipynb` scored every official test engine through the deployed `inference.predict`, with FD001 as a reproduction check (it matched notebook 12 exactly).

**The gap itself:** FD004's calibrated RSF interval reached 87.9% overall but only by widening everything (122.6 cycles vs Weibull's 98.8), and it covered 55% of engines above the cap. It stays supplementary.

**The real finding:** breaking coverage down by true remaining life showed the *deployed* Weibull interval covered only 58% (FD001) and 54% (FD004) of engines within 25 cycles of failure, and every miss promised more life than the engine had. Phase 14 had checked the long-lived end and kept Weibull for it; nobody had checked the short end, where a late call matters most. Phase 14's calibration couldn't have seen it either: it scored engines between 30% and 90% of their life.

**The fix, selected without the test set:** `interval_calibration.py` scores each training engine at 10 points across 30–99% of its life with fold models that never saw it, and `notebooks/15_interval_fix.ipynb` compares six candidate intervals (Weibull, calibrated RSF, per-band "Mondrian" conformal corrections, and hybrids) with corrections cross-fitted across engine halves. The rule (≥ 88% coverage overall and ≥ 85% near failure on both subsets, then narrowest) picked the hybrid with no correction: lower bound = the more cautious of RSF's and Weibull's, upper bound = Weibull's.

**Confirmed on the test sets:** near-failure coverage 58% → 100% (FD001) and 54% → 94% (FD004); overall 89.0% → 97.0% and 80.6% → 88.7%; dangerous misses 8 → 0 and 23 → 3; about one cycle wider on average. Above-cap coverage is unchanged (FD004 73%), since the upper bound is the same.

**Deployment:** no retraining and no re-upload. The interval is computed from the already-exported models; the export now records it in the metadata, and `inference.py` serves it by default for older bundles, so the live demo picked it up with the code change alone. The API also returns Weibull's own interval (`weibull_interval_90`). 13 new tests; the suite was then 163 tests across 12 files.

**Caveat, stated in the notebook:** the test sets had already been used to find the problem, so the confirmation isn't fully independent; the out-of-fold results are the main evidence.

---

## 26. Phase 25: training without the cap, measured and not deployed

**What prompted it:** after Phase 24, FD004's interval still covered only 73% of engines with more than 125 cycles left. Every model learns remaining life clipped at 125, so no upper bound can reach far past it.

**What was built:** `notebooks/16_uncapped_upper_bound.ipynb` reruns Phase 24's out-of-fold protocol with the cap removed, as two pre-stated questions. Point estimate: does uncapped RSF beat capped RSF on C-index and RMSE on both subsets? No (FD001 RMSE 24.4 → 28.0), so the cap stays. Upper bound: keeping Phase 24's lower bound, does an uncapped upper bound clear 85% in every band? Only the uncapped RSF does; an uncapped Weibull just misses on FD001 (84%).

**A size constraint surfaced:** uncapped leaf-5 forests are about 3× the deployed ones (1.2 GB for both subsets), so a leaf-30 version (about 200 MB, the project's original RSF setting) was added before its results were known. It passed on training data and on the test sets: above-cap coverage 91% → 100% (FD001) and 73% → 100% (FD004), intervals 27 and 60 cycles wider, almost all of it on engines predicted healthy.

**Decision: recorded, not deployed.** The misses it fixes are on the safe side, the upper bound barely drives maintenance decisions, the interval would over-cover (98–99% vs 90%), and it costs a second forest per subset. Stated in the notebook with what would change the decision. `run_all.ps1` checks all three results.

---

## 27. Phase 26: the operations layer — versions, drift, retraining gate

**Why:** nothing tied a running model to how it was made, nothing noticed unfamiliar inputs, and nothing stopped a worse retrain from shipping. These are the questions an ML system design interview asks first.

**Versioning.** `manifest.py` writes a manifest per exported model: a version derived from the SHA-256 of the prediction files, the git commit, data hashes, library versions and test metrics measured through the serving path. Loading re-hashes and reports the version only if it still matches. `publish.py` compares local files with the hashes Hugging Face stores, uploads only what differs and tags the commit; the live demo is pinned with `MODEL_REVISION`, and its sidebar shows the version. The first real publish found that Weibull's pickle changes on every refit (lifelines records the fit time) while RSF's doesn't, so 1.75 MB were uploaded instead of ~350 MB. A bug was caught before that publish: `--full` would have uploaded every model file rather than only the changed ones.

**Drift monitoring.** `drift.py` and `GET /drift` compare the last 500 requests with a training reference using PSI. The first replay broke the "no false alarm" expectation three ways, and each became a design decision: a fleet-like reference (not full lives), steady-state replays (not an empty fleet), and thresholds calibrated on training engines (not the textbook 0.1 / 0.25, which fired on 83% of normal windows). `notebooks/17_drift_monitoring.ipynb`: FD001's and FD004's own fleets never alarm; FD002 and FD003 sent to the FD001 model alarm in every window; FD002 sent to the FD004 model also alarms although that model predicts it well, so the expectation that it wouldn't was wrong, and an alert means "check", not "the model is failing". Building a data-loss challenger later exposed a crash in threshold calibration for small fleets; fixed with a fallback to the textbook thresholds.

**Retraining gate.** `gate.py` applies four rules fixed in advance (NASA score per engine ≤ champion +5%, C-index ≥ champion −0.01, coverage ≥ 85%, near-failure coverage ≥ champion −5 points) to the champion's published manifest and the challenger's. `.github/workflows/retrain.yml` retrains a challenger with the live demo's pinned libraries, runs the gate, writes the table to the run summary, and publishes only if it passed and publishing was requested. Exercised both ways, on GitHub as well as locally: the deployed recipe retrained passed with identical metrics on GitHub's Linux machine and a Windows laptop; a challenger trained on 30 of 100 engines was rejected on the NASA rule alone (4.59 vs 4.05).

**The "familiar inputs?" badge.** `/drift` needs traffic, which the live demo doesn't have, so each prediction now carries its own input check (`drift.familiarity`), shown as a badge, and the demo lets visitors score engines from other fleets. Measured before shipping, it caught two of its own design errors: a range taken from the fleet-like reference flagged 12% of engines near failure (fixed by using all training rows, since the model trained on full lives), and percentile bands on raw operating settings sent one normal FD001 engine to red over a single noisy reading (fixed with the full min–max range for settings). Final: 100% of FD001 test engines familiar, 90% of FD002's unfamiliar, 45% of FD003's caution under the FD001 model; FD004's model, trained on every regime and both faults, finds all subsets familiar.

---

## 28. Phase 27: a better point estimate, decided by the gate

**The question, reframed:** "make RMSE better" became "improve the point estimate, judged by the NASA score first", the same priority the gate enforces, after a fair challenge that RMSE isn't this project's primary metric (it's the number papers compare).

**What was built:** `notebooks/18_point_estimate_candidates.ipynb`. It breaks down the error first (11% of test engines, those above the cap, carry a quarter of RSF's squared error), then compares candidates fixed in advance on the project's cross-validation: four caps, and an RSF + gradient-boosting average, with gradient boosting alone as a reference. The rule required a ≥ 5% NASA gain, 4 of 5 folds, no loss on points ≤ 150 cycles from failure, and RMSE within 5%. That third condition was written for a predicted trap and caught it: caps 135 and 145 won overall only by predicting more life for very young engines.

**A mistake caught in my own selection code:** it left gradient boosting alone among the candidates, against the written plan, and picked it. Corrected to match the plan; disclosed in the notebook.

**Result:** the blend won (cross-validated NASA 68.4 vs 78.8, 4 of 5 folds), passed the pre-set test check (NASA 3.12 vs 4.05 per engine, RMSE 14.2 vs 15.6, C-index 0.877 vs 0.864), then passed the retraining gate as a real challenger, the first time the gate decided something that mattered. `export_deployment_artifacts.py` gained `--point-model`, inference averages the two members and reports both (`point_estimate_components`), and FD001's deployed recipe is now the blend; FD004 stays RSF-only.

---

## 29. Phase 28: the same experiment on FD004, recorded not deployed

**Why:** after Phase 27 the two deployed models differed (FD001 blend, FD004 RSF) for no tested reason. FD004 has six operating conditions and two fault modes, so FD001's result couldn't simply be copied.

**What was built:** `notebooks/19_fd004_point_estimate.ipynb`, notebook 18's candidates, protocol and rule unchanged, with FD004's deployed settings.

**Result: the blend failed rules (a) and (b), so FD004 keeps RSF.** Cross-validated NASA 144,695 vs 140,773, winning 3 of 5 folds (4 required). A diagnostic showed what those numbers measure: 99.99% of the score comes from points more than 150 cycles from failure, where every model predicts just under the cap, and 95% from one held-out point (engine #118 with 366 cycles left; 119.9 vs 119.5 predicted). Where the model is used, the blend won all 5 folds (NASA 9.6 vs 14.9 at ≤150 cycles left), and the test set, scored once for the record, agreed (NASA 19.4 vs 22.4, RMSE 27.0 vs 28.2, C-index 0.875 vs 0.862).

**The decision, and why:** not deployed. The rule was fixed before the run, and changing it after it failed, with the test result already seen, would make the protocol meaningless. The finding is kept instead: overall cross-validated NASA is unreliable when engines live far beyond the cap, so future rules should score only points with at most 150 cycles left. The two subsets now differ for a tested, documented reason.

---

## 30. Phase 29: every deep-model seed on the test set

**Why:** notebook 07 trained 3 seeds of each deep architecture but scored only seed 42 on the official test set, so the deep models' test numbers came from one run each, a stated gap since Phase 7.

**What was built:** section 10 of `notebooks/07_deep_survival_stretch.ipynb`. It keeps all six trained models (no retraining, no selection) and scores each on the 100 test engines.

**Result:** none of the six runs reaches RSF on test C-index, NASA score or RMSE. Means over 3 seeds: LSTM C-index 0.839 ± 0.005, NASA 9.21 ± 0.67 per engine; Conv1D→LSTM 0.844 ± 0.003, 9.69 ± 0.76; RSF 0.864 and 4.63. Even the best runs' gaps are about 3× (C-index) and 5× (NASA) the seed spread, so the Phase 6 conclusion doesn't depend on the seed. Two smaller findings: the seed-42 LSTM reported alone was its best run on NASA (8.47 vs 9.78 and 9.39), and the validation ranking of the two architectures (plain LSTM ahead on all 3 seeds) doesn't carry over to the test set, where they tie.

---

## 31. Where things stand

**Done, verified with real execution:** the full survival-analysis framing and censoring argument; the complete model ladder (Cox → Weibull → RSF → two deep-learning architectures), each addition justified by the previous model's diagnostics; domain-appropriate evaluation (NASA score primary, RMSE secondary); 5-fold cross-validation confirming the ranking isn't a single-split artifact; generalization checks across three additional dataset subsets; hyperparameter tuning validated against both CV and the real official test set; a conformal-prediction fix for RSF's interval coverage, validated by subgroup rather than trusted in aggregate; the Phase 8/9 feature-scope leak, measured rather than left as a documented assumption; a properly separated two-service HTTP architecture, fresh-clone verified end to end (install → export → serve → test, from nothing); a full correctness review (Phase 18) with every notebook re-run; the survival framing tested under simulated censoring, with a mixed result reported as measured (Phase 20); a short README with results first and a full technical report (Phase 21); a redesigned dashboard live at [https://jetengines-rul.streamlit.app/](https://jetengines-rul.streamlit.app/) (Phase 19); 224 tests across 16 files run by GitHub Actions (Phase 22 onwards); both Docker images built and smoke-tested in CI (Phase 23); the deployed interval checked by subgroup at both ends of the remaining-life range, with a near-failure fix selected on training data (Phase 24); training without the cap measured for both the point estimate and the upper bound, and declined on cost-benefit grounds (Phase 25); versioned models, drift monitoring and a champion/challenger retraining gate (Phase 26); a blended point estimate selected by NASA score and promoted by that gate (Phase 27), and the same experiment on FD004 failing the pre-set rule and kept undeployed (Phase 28); every deep-model seed scored on the test set (Phase 29); and a README, project brief and architecture walkthrough that match the actual current state rather than the original plan.

**Genuinely outstanding — everything left, and nothing more:**

- The operations layer is demonstrated on replays: no live traffic, the drift window resets when the API restarts, and the gate uses the official test set (fine for no-worse-than, not for choosing among challengers).
- The public demo runs in one process (embedded mode), not as the two-service deployment — a free-hosting constraint (Phase 19). Its models must be re-uploaded to the Hugging Face repo after every re-export, with `app/requirements.txt`'s pins kept in step.
- FD004's point estimate is RSF alone: the blend failed the pre-set rule on a measurement artifact (Phase 28). A fairer rule is recorded for future experiments; testing the blend under it needs data not already seen.
- FD004 reuses FD001's hyperparameters by design (a transfer test). Its interval covers 73% of test engines above the 125-cycle cap — safe-side misses. An uncapped upper bound fixes them (Phase 25) and is ready to deploy if a use ever needs it.