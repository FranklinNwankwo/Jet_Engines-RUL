import warnings

import numpy as np
import pandas as pd
import pytest

from turbofan_rul.models import (
    fit_cox_ph,
    fit_random_survival_forest,
    predict_rul_from_survival,
    standardize_features,
    train_val_split_by_engine,
)


def _synthetic_df(n_engines=20, cycles_per_engine=10):
    rows = []
    for unit in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            rows.append({"unit_number": unit, "time_cycles": cycle, "x": unit * 10 + cycle})
    return pd.DataFrame(rows)


def test_engine_split_has_no_overlapping_units():
    df = _synthetic_df()
    train_df, val_df = train_val_split_by_engine(df, val_fraction=0.25, random_state=0)
    train_units = set(train_df.unit_number.unique())
    val_units = set(val_df.unit_number.unique())
    assert train_units.isdisjoint(val_units)
    assert train_units | val_units == set(df.unit_number.unique())


def test_engine_split_keeps_all_rows_of_each_engine_together():
    df = _synthetic_df()
    train_df, val_df = train_val_split_by_engine(df, val_fraction=0.25, random_state=0)
    for unit in val_df.unit_number.unique():
        assert unit not in train_df.unit_number.unique()
        # every row for this engine should be in val, none missing
        assert (val_df.unit_number == unit).sum() == (df.unit_number == unit).sum()


def test_engine_split_respects_approx_fraction():
    df = _synthetic_df(n_engines=100)
    train_df, val_df = train_val_split_by_engine(df, val_fraction=0.2, random_state=0)
    n_val_units = val_df.unit_number.nunique()
    assert 15 <= n_val_units <= 25  # ~20% of 100, some tolerance for rounding


def test_standardize_features_scaler_fit_on_train_only():
    df = _synthetic_df(n_engines=10)
    train_df, val_df = train_val_split_by_engine(df, val_fraction=0.3, random_state=1)

    # Deliberately shift val's feature far outside train's range -- if the
    # scaler were fit on train+val instead of train alone, this shift would
    # change the scaler's mean/std and this test's assertions would break.
    val_df = val_df.copy()
    val_df["x"] = val_df["x"] + 10_000

    train_scaled, val_scaled, scaler = standardize_features(train_df, val_df, ["x"])

    assert np.isclose(scaler.mean_[0], train_df["x"].mean())
    # train's own values should be ~standardized (mean ~0)
    assert abs(train_scaled["x"].mean()) < 1e-6
    # val's shifted values should now be wildly out of range under train's scaler
    assert val_scaled["x"].mean() > 100


# ---------------------------------------------------------------------------
# Model wrappers -- small synthetic survival data so these stay fast.
# ---------------------------------------------------------------------------

CAP = 30


def _synthetic_survival_df(n_engines=30, random_state=0):
    """Engines of varying length with one degrading sensor, labeled the same
    way as labeling.build_train_survival_frame: rows with more than CAP
    cycles left are censored at CAP."""
    rng = np.random.default_rng(random_state)
    rows = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(20, 60))
        for cycle in range(1, life + 1):
            true_rul = life - cycle
            rows.append({
                "unit_number": unit,
                "time_cycles": cycle,
                "x": -true_rul / 10 + rng.normal(0, 0.3),  # rises as failure approaches
                "z": rng.normal(),                          # pure noise
                "duration": min(true_rul, CAP),
                "event": int(true_rul <= CAP),
            })
    return pd.DataFrame(rows)


def test_fit_cox_ph_cluster_col_changes_only_standard_errors():
    """Clustered (robust) standard errors must leave the coefficients
    unchanged and only change the uncertainty estimates."""
    df = _synthetic_survival_df()
    plain = fit_cox_ph(df, ["x", "z"])
    clustered = fit_cox_ph(df, ["x", "z"], cluster_col="unit_number")

    assert np.allclose(plain.params_.values, clustered.params_.values)
    assert not np.allclose(plain.standard_errors_.values, clustered.standard_errors_.values)
    assert plain.params_["x"] > 0  # higher x (closer to failure) -> higher hazard


def test_fit_cox_ph_lets_convergence_warnings_through(monkeypatch):
    """Routine lifelines warnings are silenced, but a ConvergenceWarning must
    never be swallowed. CoxPHFitter.fit is replaced with a stub that emits one
    of each, so the test doesn't depend on reproducing a real non-converging fit."""
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning

    def fake_fit(self, *args, **kwargs):
        warnings.warn("routine advisory", UserWarning)
        warnings.warn("did not converge", ConvergenceWarning)
        return self

    monkeypatch.setattr(CoxPHFitter, "fit", fake_fit)
    df = _synthetic_survival_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_cox_ph(df, ["x", "z"])

    categories = [w.category for w in caught]
    assert ConvergenceWarning in categories
    assert UserWarning not in categories


def test_rsf_fit_and_predict_respect_censoring_and_ceiling():
    df = _synthetic_survival_df()
    assert (df["event"] == 0).any()  # premise: the data actually contains censored rows

    rsf = fit_random_survival_forest(df, ["x", "z"], n_estimators=20, min_samples_leaf=5, row_stride=1)

    out = predict_rul_from_survival(rsf, df[["x", "z"]].iloc[:50])
    assert out.index.equals(df.index[:50])
    assert (out["pred_ci_low"] <= out["pred_ci_high"]).all()
    assert (out["pred_rul"] >= 0).all()
    assert (out["pred_rul"] <= rsf.unique_times_.max() + 1e-9).all()  # RMST can't exceed the training ceiling


def test_rsf_censored_rows_are_treated_differently_from_events():
    """Relabeling censored rows as event=1 must change the fitted model --
    if it didn't, the event column would be getting ignored. (Uses synthetic
    data with censoring; the project's default labeling has none.)"""
    df = _synthetic_survival_df()
    as_failures = df.assign(event=1)
    kwargs = dict(n_estimators=20, min_samples_leaf=5, row_stride=1, random_state=0)
    censored_model = fit_random_survival_forest(df, ["x", "z"], **kwargs)
    failure_model = fit_random_survival_forest(as_failures, ["x", "z"], **kwargs)

    X = df[["x", "z"]].iloc[:30]
    rmst_censored = predict_rul_from_survival(censored_model, X)["pred_rul"].to_numpy()
    rmst_failures = predict_rul_from_survival(failure_model, X)["pred_rul"].to_numpy()
    assert not np.allclose(rmst_censored, rmst_failures)


def test_predict_rul_inverted_percentiles_raise():
    df = _synthetic_survival_df()
    rsf = fit_random_survival_forest(df, ["x", "z"], n_estimators=10, min_samples_leaf=5, row_stride=2)
    with pytest.raises(AssertionError):
        predict_rul_from_survival(rsf, df[["x", "z"]].iloc[:20], lower_p=0.05, upper_p=0.95)
