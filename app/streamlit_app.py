"""
Streamlit dashboard: pick a subset, pick or upload an engine's recent
sensor history, see predicted RUL, confidence interval, and risk flag.

Two backends, chosen by environment variables (or Streamlit secrets, which
Community Cloud also exposes as environment variables):

* HTTP mode (default) -- a pure client of the FastAPI service
  (api/main.py). Every prediction is a POST to `/predict`; the sensor list
  and required columns come from `/subsets` and `/subsets/{subset}/metadata`.
  The API's address is `API_BASE_URL` (default `http://localhost:8000`), so
  the same code works against localhost, a Docker Compose service name
  (`http://api:8000`) or a public URL. This is the two-service deployment
  that docker-compose.yml runs; in this mode the app never imports the
  model code or loads a model artifact.

* Embedded mode -- used when `API_BASE_URL` is NOT set and either
  `MODEL_REPO` (a Hugging Face model repo holding the exported `fd001/` and
  `fd004/` folders) or `MODELS_DIR` (a local folder with the same layout)
  is. The app then loads the deployment bundles itself and calls
  `turbofan_rul.inference.predict` -- the exact function the API calls --
  in-process. This exists for the public demo on Streamlit Community Cloud,
  which can only host a Streamlit app, not a separate API service.

Both modes return the same response dict, so everything below the backend
functions is shared.

Look and feel: colours and fonts come from `.streamlit/config.toml` (the
theme); the small amount of CSS below only styles the custom HTML cards.
Charts use Altair, which ships with Streamlit -- no extra dependency.

Run with:
  API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py   # HTTP mode
  MODELS_DIR=models streamlit run app/streamlit_app.py                    # embedded mode
"""

from __future__ import annotations

import html
import os
import sys
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# data_loading is used in both modes, only for the "sample engine" picker,
# which reads the public C-MAPSS dataset directly. The model-facing modules
# (turbofan_rul.inference) are imported lazily, in embedded mode only, so the
# HTTP-mode Docker image never needs scikit-learn, scikit-survival or lifelines.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from turbofan_rul import data_loading  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO")            # e.g. "your-hf-username/turbofan-rul-models"
MODEL_REVISION = os.environ.get("MODEL_REVISION")    # optional: pin a commit/tag of that repo
MODELS_DIR = os.environ.get("MODELS_DIR")            # local alternative to MODEL_REPO
EMBEDDED = "API_BASE_URL" not in os.environ and bool(MODEL_REPO or MODELS_DIR)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
REQUEST_TIMEOUT = 30
SUBSETS = ["FD001", "FD004"]
# Same value the API's /metadata endpoint reports.
RECOMMENDED_MIN_HISTORY_CYCLES = 30

# Display-only copy of the API's risk thresholds (inference.RISK_THRESHOLDS),
# used to shade the risk zones on the RUL chart. The risk level itself always
# comes from the API response, never from these numbers.
CRITICAL_BELOW, WARNING_BELOW = 20, 50

RISK_STYLE = {
    "critical": {"label": "Critical", "color": "#EF4444", "note": "Under 20 cycles left — schedule maintenance now."},
    "warning": {"label": "Warning", "color": "#F5A524", "note": "Under 50 cycles left — plan maintenance soon."},
    "healthy": {"label": "Healthy", "color": "#22C55E", "note": "50+ cycles left — no action needed yet."},
}

SUBSET_INFO = {
    "FD001": {"short": "1 regime · 1 fault", "desc": "1 operating regime · 1 fault mode", "tag": "baseline"},
    "FD004": {"short": "6 regimes · 2 faults", "desc": "6 operating regimes · 2 fault modes", "tag": "hardest subset"},
}
# Fleets a visitor can draw sample engines from. Engines from a fleet the
# selected model wasn't trained on show what the input check does.
FLEETS = {
    "FD001": "1 operating regime · 1 fault mode",
    "FD002": "6 operating regimes · 1 fault mode",
    "FD003": "1 operating regime · 2 fault modes",
    "FD004": "6 operating regimes · 2 fault modes",
}
INPUT_CHECK_STYLE = {
    "familiar": {"label": "Familiar inputs", "color": "#22C55E"},
    "caution": {"label": "Caution: sensors outside training range", "color": "#F5A524"},
    "unfamiliar": {"label": "Unfamiliar operating conditions", "color": "#EF4444"},
}

st.set_page_config(
    page_title="Turbofan RUL · Survival Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Only styles the custom HTML blocks below; everything else is themed by
# .streamlit/config.toml.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      .hero {
        padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
        border: 1px solid #22314A;
        background:
          radial-gradient(1200px 300px at 0% 0%, rgba(245,165,36,0.16), transparent 60%),
          radial-gradient(900px 300px at 100% 100%, rgba(56,189,248,0.12), transparent 60%),
          #101A2C;
      }
      .hero .eyebrow { color: #F5A524; font-weight: 600; letter-spacing: .14em;
                       font-size: .78rem; text-transform: uppercase; }
      .hero h1 { margin: .25rem 0 .4rem 0; font-size: 2.3rem; line-height: 1.15; }
      .hero p { color: #A9B6CC; margin: 0; max-width: 760px; }
      .pills { margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .45rem; }
      .pill { font-size: .78rem; padding: .22rem .65rem; border-radius: 999px;
              border: 1px solid #2A3B57; color: #CBD5E1; background: #0E1626; }
      .card { border: 1px solid #22314A; border-radius: 14px; padding: 1.1rem 1.25rem;
              background: #101A2C; height: 100%; }
      .card .label { color: #8EA0BC; font-size: .8rem; text-transform: uppercase;
                     letter-spacing: .08em; font-weight: 600; }
      .card .value { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem;
                     font-weight: 700; line-height: 1.1; margin-top: .3rem; }
      .card .unit { font-size: 1rem; color: #8EA0BC; font-weight: 500; margin-left: .3rem; }
      .card .sub { color: #A9B6CC; font-size: .86rem; margin-top: .35rem; }
      .risk-badge { display: inline-flex; align-items: center; gap: .5rem;
                    padding: .3rem .8rem; border-radius: 999px; font-weight: 700;
                    font-size: 1.05rem; margin-top: .45rem; }
      .risk-dot { width: .6rem; height: .6rem; border-radius: 50%; display: inline-block; }
      .status { display: inline-flex; align-items: center; gap: .45rem; font-size: .85rem;
                color: #A9B6CC; }
      .status .dot { width: .55rem; height: .55rem; border-radius: 50%; background: #22C55E;
                     box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
      .section-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
                       font-size: 1.15rem; margin: 1.2rem 0 .4rem 0; }
      .muted { color: #8EA0BC; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class BackendError(Exception):
    """A readable, user-facing failure from either backend."""


@st.cache_data(ttl=60)
def _http_subsets() -> list[str]:
    """Ask the API which subsets it has loaded -- never assumed or
    hardcoded, and re-checked periodically (ttl) in case the API is
    redeployed with different artifacts while this app keeps running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/subsets", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["subsets"]
    except requests.RequestException:
        return []


@st.cache_data(ttl=300)
def _http_metadata(subset: str) -> dict:
    resp = requests.get(f"{API_BASE_URL}/subsets/{subset}/metadata", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data
def get_demo_engines(subset: str):
    """Test engines for the "pick a sample engine" demo path -- lets a
    reviewer try the dashboard without preparing their own CSV. Reads the
    public dataset directly; see module docstring for why this doesn't
    count as scoring logic living in the frontend."""
    data_loading.download_cmapss(subset, data_dir=DATA_DIR)
    cmapss = data_loading.load_subset(subset, data_dir=DATA_DIR)
    return cmapss.test, cmapss.rul.set_index("unit_number")["RUL"]


def call_predict(subset: str, readings: pd.DataFrame) -> dict:
    """POST to the API's /predict endpoint. Raises requests.HTTPError on a
    4xx/5xx response -- callers catch that, not any turbofan_rul exception
    type, since this app has no import of that package's model-facing code."""
    payload = {
        "subset": subset,
        "readings": readings.to_dict(orient="records"),
    }
    resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def error_detail(e: requests.HTTPError) -> str:
    """Turn an API error response into a readable message.

    The API returns `{"detail": "..."}` for its own errors and
    `{"detail": [{"loc": [...], "msg": "..."}, ...]}` for request-validation
    errors (e.g. a NaN or missing column in an uploaded CSV). A non-JSON body
    (e.g. a proxy's HTML error page) must not crash the error handler itself.
    """
    if e.response is None:
        return str(e)
    try:
        detail = e.response.json().get("detail", str(e))
    except ValueError:
        return f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    if isinstance(detail, list):
        parts = []
        for err in detail[:5]:
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            parts.append(f"{loc}: {err.get('msg', err)}" if loc else str(err.get("msg", err)))
        more = f" (+{len(detail) - 5} more)" if len(detail) > 5 else ""
        return "; ".join(parts) + more
    return str(detail)


@st.cache_resource(show_spinner="Loading models (first visit after a restart can take a minute)...")
def _embedded_models() -> tuple[dict, list[str]]:
    """Embedded mode: download (or locate) the exported model folders and
    load them once per server process. Returns (bundles by subset, any
    scikit-learn version-mismatch warnings raised while unpickling). Raises
    BackendError with a readable message if the models can't be found."""
    from turbofan_rul import inference  # lazy: heavy ML imports, embedded mode only

    if MODEL_REPO:
        from huggingface_hub import snapshot_download
        try:
            root = Path(snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION,
                                          repo_type="model"))
        except Exception as e:  # network error, missing repo, bad revision...
            raise BackendError(f"Could not download models from Hugging Face repo "
                               f"`{MODEL_REPO}`: {e}") from e
    else:
        root = Path(MODELS_DIR)

    bundles, version_warnings = {}, []
    for subset in SUBSETS:
        subset_dir = root / subset.lower()
        if not (subset_dir / "rsf.joblib").exists():
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                bundles[subset] = inference.load_deployment_bundle(subset_dir)
            except Exception as e:
                # Almost always a library-version mismatch between the
                # environment that exported the models and this one (e.g.
                # models pickled under pandas 3 loaded under pandas 2).
                raise BackendError(
                    f"Could not load the {subset} models ({type(e).__name__}: {str(e)[:200]}). "
                    f"This usually means the library versions in app/requirements.txt don't "
                    f"match the environment that ran export_deployment_artifacts.py."
                ) from e
        version_warnings += [str(w.message) for w in caught
                             if w.category.__name__ == "InconsistentVersionWarning"]
    if "FD001" not in bundles:
        raise BackendError(f"No exported models found under `{root}` -- expected "
                           f"`fd001/rsf.joblib` (and optionally `fd004/`).")
    # Pickled scikit-learn models loaded under a different scikit-learn
    # version can silently give different predictions, so the warnings are
    # returned for the sidebar to show rather than swallowed.
    return bundles, version_warnings


def _embedded_bundles() -> dict:
    return _embedded_models()[0]


def list_subsets() -> list[str]:
    if EMBEDDED:
        return list(_embedded_bundles())
    return _http_subsets()


def subset_metadata(subset: str) -> dict:
    if EMBEDDED:
        bundle = _embedded_bundles()[subset]
        return {
            "subset": bundle.subset,
            "kept_sensors": bundle.kept_sensors,
            "raw_passthrough_cols": bundle.raw_passthrough_cols,
            "uses_regime_clustering": bundle.uses_regime_clustering,
            "recommended_min_history_cycles": RECOMMENDED_MIN_HISTORY_CYCLES,
            "model_version": bundle.model_version,
            "test_metrics": (bundle.manifest or {}).get("test_metrics"),
        }
    try:
        return _http_metadata(subset)
    except requests.RequestException as e:
        raise BackendError(f"Could not load metadata for {subset} from the API at "
                           f"`{API_BASE_URL}`: {e}") from e


def predict(subset: str, readings: pd.DataFrame) -> dict:
    """Score one engine. Same response dict in both modes."""
    if EMBEDDED:
        from turbofan_rul import inference
        bundle = _embedded_bundles()[subset]
        # The API's request schema keeps only its known columns, requires
        # numbers and rejects inf/NaN; do the same here so both modes accept
        # and reject the same inputs.
        cols = ["time_cycles"] + bundle.raw_passthrough_cols
        missing = [c for c in cols if c not in readings.columns]
        if missing:
            raise BackendError(f"missing required columns: {missing}")
        try:
            df = readings[cols].apply(pd.to_numeric)
        except (ValueError, TypeError) as e:
            raise BackendError(f"non-numeric value in readings: {e}") from e
        df = df.replace([np.inf, -np.inf], np.nan)
        try:
            return inference.predict(bundle, df)
        except inference.InvalidReadingsError as e:
            raise BackendError(str(e)) from e
    try:
        return call_predict(subset, readings)
    except requests.HTTPError as e:
        raise BackendError(error_detail(e)) from e
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the API at `{API_BASE_URL}`: {e}") from e


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def sensor_chart(engine_df: pd.DataFrame, sensors: list[str]) -> alt.FacetChart:
    """Small multiples, one panel per sensor with its own y-scale (the
    sensors live on very different scales), raw reading plus a 10-cycle
    rolling mean so the degradation trend is visible through the noise."""
    long = engine_df.melt(id_vars="time_cycles", value_vars=sensors,
                          var_name="sensor", value_name="reading")
    long["trend"] = (long.groupby("sensor")["reading"]
                     .transform(lambda s: s.rolling(10, min_periods=1).mean()))
    base = alt.Chart().encode(x=alt.X("time_cycles:Q", title="Cycle"))
    raw = base.mark_line(strokeWidth=1, opacity=0.35, color="#38BDF8").encode(
        y=alt.Y("reading:Q", title=None, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("time_cycles:Q", title="Cycle"),
                 alt.Tooltip("reading:Q", title="Reading", format=".3f")],
    )
    trend = base.mark_line(strokeWidth=2.4, color="#F5A524").encode(y="trend:Q")
    return (
        alt.layer(raw, trend, data=long)
        .properties(height=150)
        .facet(facet=alt.Facet("sensor:N", title=None,
                               header=alt.Header(labelColor="#CBD5E1", labelFontSize=13,
                                                 labelFontWeight="bold")),
               columns=2)
        .resolve_scale(y="independent")
        .configure_view(stroke=None)
        .configure_axis(gridColor="#1C2940", domainColor="#2A3B57",
                        labelColor="#8EA0BC", titleColor="#8EA0BC")
    )


def rul_chart(result: dict, true_rul: float | None) -> alt.LayerChart:
    """One horizontal "ruler" in cycles: shaded risk zones, the served 90%
    interval, RSF's supplementary interval, the point estimate,
    and (for demo engines) the true RUL. Every layer uses fixed colours
    rather than a colour encoding, so the layers can't fight over a shared
    colour scale."""
    pred = result["predicted_rul_cycles"]
    w_low, w_high = result["confidence_interval_90"]

    # Each band is a rectangle at a fixed vertical position (in arbitrary
    # 0-3 units) rather than a categorical row, so the chart keeps a fixed
    # height and both bars stay readable at phone width.
    bands = [("90% interval (RSF + Weibull)", w_low, w_high, "#38BDF8", 1.45, 2.45)]
    rsf_cal = result.get("rsf_calibrated_interval_supplementary")
    if rsf_cal:
        r_low, r_high = rsf_cal["interval_90"]
        # The conformal correction can push the lower bound below zero;
        # remaining life can't be negative, so draw it from 0.
        bands.append(("RSF calibrated (supplementary)", max(r_low, 0.0), r_high, "#A78BFA", 0.6, 1.05))

    x_max = max(w_high, pred, true_rul or 0, 130) * 1.08
    x_enc = dict(scale=alt.Scale(domain=[0, x_max], nice=False))
    y_scale = alt.Scale(domain=[0, 3])

    layers = []
    for start, end, color in [(0, CRITICAL_BELOW, "#EF4444"),
                              (CRITICAL_BELOW, WARNING_BELOW, "#F5A524"),
                              (WARNING_BELOW, x_max, "#22C55E")]:
        layers.append(alt.Chart(pd.DataFrame([{"start": start, "end": end}]))
                      .mark_rect(color=color, opacity=0.12)
                      .encode(x=alt.X("start:Q", title="Remaining useful life (cycles)", **x_enc),
                              x2="end:Q"))
    for name, low, high, color, y0, y1 in bands:
        layers.append(alt.Chart(pd.DataFrame([{"band": name, "low": low, "high": high, "y0": y0, "y1": y1}]))
                      .mark_rect(color=color, cornerRadius=5, opacity=0.9)
                      .encode(x=alt.X("low:Q", **x_enc), x2="high:Q",
                              y=alt.Y("y0:Q", scale=y_scale, axis=None), y2="y1:Q",
                              tooltip=[alt.Tooltip("band:N", title="Interval"),
                                       alt.Tooltip("low:Q", title="From", format=".0f"),
                                       alt.Tooltip("high:Q", title="To", format=".0f")]))
    rules = [("Predicted (RSF)", pred, "#F5A524", [1, 0])]
    if true_rul is not None:
        rules.append(("Actual RUL", float(true_rul), "#E6EDF7", [5, 4]))
    for name, x, color, dash in rules:
        layers.append(alt.Chart(pd.DataFrame([{"what": name, "x": x}]))
                      .mark_rule(color=color, strokeWidth=3, strokeDash=dash)
                      .encode(x=alt.X("x:Q", **x_enc),
                              tooltip=[alt.Tooltip("what:N", title="Marker"),
                                       alt.Tooltip("x:Q", title="Cycles", format=".0f")]))
    return (
        alt.layer(*layers)
        .properties(height=110)
        .configure_view(stroke=None)
        .configure_axisX(grid=False, domainColor="#2A3B57", tickColor="#2A3B57",
                         labelColor="#A9B6CC", titleColor="#8EA0BC")
    )


def chart_legend(has_true_rul: bool) -> str:
    """Legend for rul_chart, as HTML: Vega-Lite legends need colour
    encodings, which rul_chart deliberately avoids."""
    items = [
        '<span style="color:#F5A524;font-weight:700">━</span> Predicted (RSF)',
        '<span style="color:#38BDF8;font-weight:700">■</span> 90% interval',
        '<span style="color:#A78BFA;font-weight:700">■</span> RSF calibrated',
    ]
    if has_true_rul:
        items.insert(1, '<span style="color:#E6EDF7;font-weight:700">┅</span> Actual RUL')
    items.append('<span style="opacity:.8">Background: '
                 '<span style="color:#EF4444">critical</span> &lt; 20 · '
                 '<span style="color:#F5A524">warning</span> &lt; 50 · '
                 '<span style="color:#22C55E">healthy</span></span>')
    return '<div class="muted" style="display:flex;flex-wrap:wrap;gap:1.2rem">' + \
        "".join(f"<span>{i}</span>" for i in items) + "</div>"


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def card(label: str, value: str, unit: str = "", sub: str = "", value_color: str | None = None) -> str:
    style = f' style="color:{value_color}"' if value_color else ""
    unit_html = f'<span class="unit">{html.escape(unit)}</span>' if unit else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value"{style}>{html.escape(value)}{unit_html}</div>{sub_html}</div>')


def render_input_check(check: dict | None) -> None:
    """The "familiar inputs?" badge (drift.familiarity). Hidden when the
    backend predates it."""
    if not check:
        return
    style = INPUT_CHECK_STYLE[check["status"]]
    st.markdown(
        f'<div style="margin:.6rem 0 .2rem 0"><span class="risk-badge" style="background:{style["color"]}22;'
        f'color:{style["color"]};border:1px solid {style["color"]}66"><span class="risk-dot" '
        f'style="background:{style["color"]}"></span>{style["label"]}</span>'
        f'<span class="muted" style="margin-left:.7rem">{html.escape(check["message"])}</span></div>',
        unsafe_allow_html=True,
    )
    if check["outside_training_range"]:
        with st.expander("Which inputs are outside the training range?"):
            st.dataframe(pd.DataFrame([
                {"input": o["feature"], "value": o["value"],
                 "training range": f'{o["training_range"][0]:g} to {o["training_range"][1]:g}'}
                for o in check["outside_training_range"]]), hide_index=True, width="stretch")
            st.caption("Operating settings are checked against the full range seen in training; sensors "
                       "(15-cycle rolling means) against the 0.5th-99.5th percentile band.")


def render_prediction(result: dict, true_rul: float | None = None) -> None:
    risk = RISK_STYLE[result["risk_level"]]
    pred = result["predicted_rul_cycles"]
    low, high = result["confidence_interval_90"]

    c1, c2, c3 = st.columns(3)
    parts = result.get("point_estimate_components") or {}
    source = ("Average of the Random Survival Forest and gradient boosting"
              if parts.get("gradient_boosting") is not None else "Random Survival Forest point estimate")
    c1.markdown(card("Predicted remaining life", f"{pred:.0f}", "cycles", source), unsafe_allow_html=True)
    c2.markdown(card("90% confidence interval", f"{low:.0f} – {high:.0f}", "cycles",
                     "Lower bound: the more cautious of RSF and Weibull AFT · upper: Weibull AFT"),
                unsafe_allow_html=True)
    badge = (f'<div class="card"><div class="label">Risk level</div>'
             f'<div class="risk-badge" style="background:{risk["color"]}22;color:{risk["color"]};'
             f'border:1px solid {risk["color"]}66"><span class="risk-dot" '
             f'style="background:{risk["color"]}"></span>{risk["label"]}</div>'
             f'<div class="sub">{html.escape(risk["note"])}</div></div>')
    c3.markdown(badge, unsafe_allow_html=True)

    render_input_check(result.get("input_check"))

    st.markdown('<div class="section-title">Where this engine sits</div>', unsafe_allow_html=True)
    st.altair_chart(rul_chart(result, true_rul), width="stretch")
    st.markdown(chart_legend(true_rul is not None), unsafe_allow_html=True)

    if true_rul is not None:
        err = pred - float(true_rul)
        direction = "late (over-estimate)" if err > 0 else "early (under-estimate)" if err < 0 else "exact"
        st.markdown(
            f'<div class="muted">Known test engine — actual remaining life: <b>{true_rul} cycles</b>. '
            f'Prediction error: <b>{err:+.0f} cycles</b>, {direction}. '
            f'A real deployment wouldn\'t have the actual value.</div>',
            unsafe_allow_html=True,
        )

    if result["point_estimate_outside_interval"]:
        st.info(
            "The point estimate falls outside its own interval. This is an expected "
            "consequence of using two independent models (the interval's upper bound comes "
            "from Weibull AFT, the point estimate from the Random Survival Forest, blended with "
            "gradient boosting on FD001) rather "
            "than a bug.",
            icon="ℹ️",
        )

    rsf_cal = result.get("rsf_calibrated_interval_supplementary")
    if rsf_cal:
        cal_low, cal_high = rsf_cal["interval_90"]
        with st.expander("About the purple RSF interval (supplementary, not authoritative)"):
            st.write(f"RSF's own conformally calibrated 90% interval: **{cal_low:.0f} – {cal_high:.0f} cycles**.")
            st.caption(rsf_cal["caveat"])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Predictive maintenance · NASA C-MAPSS</div>
      <h1>How long until this engine fails?</h1>
      <p>Remaining-useful-life prediction for turbofan engines, framed as a
         survival-analysis problem: a Random Survival Forest (blended with gradient boosting
         on FD001) gives the estimate and risk ranking, and a 90% interval combines it with a
         Weibull AFT model.</p>
      <div class="pills">
        <span class="pill">Random Survival Forest</span>
        <span class="pill">Weibull AFT</span>
        <span class="pill">Conformal calibration</span>
        <span class="pill">Scikit-survival · lifelines</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    available_subsets = list_subsets()
except BackendError as e:
    st.error(str(e))
    st.stop()
if not available_subsets:
    st.error(
        f"Could not reach the API at `{API_BASE_URL}` (or it has no subsets loaded). "
        f"Start it with `uvicorn api.main:app --reload`, or set the `API_BASE_URL` "
        f"environment variable if it's running elsewhere. If the API is hosted on a "
        f"free tier, it may be waking up — wait a minute and refresh."
    )
    st.stop()

with st.sidebar:
    st.markdown("### ✈️ Turbofan RUL")
    status = "Models loaded in this app" if EMBEDDED else "Model API online"
    st.markdown(f'<div class="status"><span class="dot"></span>{status}</div>',
                unsafe_allow_html=True)
    if EMBEDDED and _embedded_models()[1]:
        st.warning("These models were saved with a different scikit-learn version than "
                   "the one installed here, so predictions may differ from the notebooks. "
                   "Pin the versions in app/requirements.txt to match the export.")
    st.divider()
    selected_subset = st.radio(
        "Dataset / trained model",
        options=available_subsets,
        format_func=lambda s: f"{s} · {SUBSET_INFO[s]['short']}" if s in SUBSET_INFO else s,
    )

try:
    metadata = subset_metadata(selected_subset)
except BackendError as e:
    st.error(str(e))
    st.stop()
kept_sensors = metadata["kept_sensors"]
raw_passthrough_cols = metadata["raw_passthrough_cols"]

with st.sidebar:
    info = SUBSET_INFO.get(selected_subset, {})
    st.markdown(f"**{selected_subset}** · {info.get('tag', '')}")
    st.caption(
        f"{len(kept_sensors)} informative sensors · "
        f"{'regime-normalised (6 operating regimes)' if metadata['uses_regime_clustering'] else 'single operating regime'} · "
        f"{metadata['recommended_min_history_cycles']}+ cycles of history recommended"
    )
    # Older APIs don't report a version; show it only when there is one.
    version = metadata.get("model_version")
    if version:
        metrics = metadata.get("test_metrics") or {}
        detail = (f" · test C-index {metrics['c_index']:.3f}, interval coverage "
                  f"{metrics['interval_coverage']:.0%}" if metrics.get("c_index") is not None else "")
        st.caption(f"Model version `{version}`{detail}")
    st.divider()
    with st.expander("How it works"):
        st.markdown(
            "1. You send an engine's recent sensor readings.\n"
            "2. The API rebuilds the same rolling-window features used in training.\n"
            "3. **Random Survival Forest** estimates the survival curve; the area under it "
            "is its predicted remaining life. On FD001 that is averaged with a **gradient "
            "boosting** regressor, which cut the NASA score by about a quarter in testing.\n"
            "4. The **90% interval** combines two models: its lower bound is the more "
            "cautious of the forest's and a **Weibull AFT** model's, so it doesn't promise "
            "too much life near failure; its upper bound is Weibull's, which can extrapolate "
            "beyond the 125-cycle training cap where the forest cannot.\n"
            "5. Risk: critical < 20 cycles, warning < 50, otherwise healthy."
        )
    if EMBEDDED:
        st.caption(f"Models: `{MODEL_REPO or MODELS_DIR}` (in-app)")
    else:
        st.caption(f"API: `{API_BASE_URL}`")

if "results" not in st.session_state:
    st.session_state.results = {}

tab_demo, tab_upload = st.tabs(["🛩️  Sample engine", "📤  Upload your own readings"])

with tab_demo:
    test_df, _ = get_demo_engines(selected_subset)          # the model's own fleet (upload tab template)
    fleet_options = [selected_subset] + [f for f in FLEETS if f != selected_subset]
    fleet = st.selectbox(
        "Engines from", fleet_options, index=0,
        format_func=lambda f: (f"{f} — the fleet this model was trained on" if f == selected_subset
                               else f"{f} — a different fleet ({FLEETS[f]}), to see the input check"),
    )
    fleet_df, rul_lookup = get_demo_engines(fleet)
    unit_ids = sorted(fleet_df["unit_number"].unique())

    left, right = st.columns([2, 1], vertical_alignment="bottom")
    selected_unit = left.selectbox(
        "Pick one of the official test engines", unit_ids, index=0,
        format_func=lambda u: f"Engine {u}",
    )
    engine_df = fleet_df[fleet_df.unit_number == selected_unit].sort_values("time_cycles")
    readings = engine_df[["time_cycles"] + raw_passthrough_cols].reset_index(drop=True)
    predict_clicked = right.button("Predict remaining life", type="primary",
                                   key="demo_predict", width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.markdown(card("Cycles observed", f"{len(readings)}", "",
                     "History sent to the model"), unsafe_allow_html=True)
    m2.markdown(card("Sensors used", f"{len(kept_sensors)}", "of 21",
                     "Near-constant sensors are dropped"), unsafe_allow_html=True)
    m3.markdown(card("Model", selected_subset, "",
                     info.get("desc", "") if fleet == selected_subset
                     else f"scoring an {fleet} engine ({FLEETS[fleet]})"), unsafe_allow_html=True)

    key = ("demo", selected_subset, fleet, int(selected_unit))
    if predict_clicked:
        try:
            with st.spinner("Scoring engine..."):
                st.session_state.results[key] = predict(selected_subset, readings)
        except BackendError as e:
            st.error(f"Could not score this engine: {e}")

    if key in st.session_state.results:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        render_prediction(st.session_state.results[key], true_rul=rul_lookup.loc[selected_unit])

    st.markdown('<div class="section-title">Sensor history</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">First four informative sensors: raw reading (blue) and '
                '10-cycle rolling mean (amber). Degradation shows up as a drift in the trend.</div>',
                unsafe_allow_html=True)
    st.altair_chart(sensor_chart(engine_df, kept_sensors[:4]), width="stretch")

with tab_upload:
    st.markdown(
        f"Upload a CSV with columns `time_cycles`, `op_setting_1`–`op_setting_3` and "
        f"`sensor_1`–`sensor_21`: one row per cycle, oldest first. It will be scored against "
        f"the **{selected_subset}** model selected in the sidebar."
    )
    template = test_df[test_df.unit_number == test_df.unit_number.min()][["time_cycles"] + raw_passthrough_cols]
    st.download_button(
        "Download an example CSV", template.to_csv(index=False).encode(),
        file_name=f"example_engine_{selected_subset.lower()}.csv", mime="text/csv",
    )
    uploaded = st.file_uploader("Sensor readings CSV", type="csv")
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
        else:
            st.dataframe(uploaded_df.tail(5), width="stretch", hide_index=True)
            key = ("upload", selected_subset, uploaded.file_id)
            if st.button("Predict remaining life", type="primary", key="upload_predict"):
                try:
                    with st.spinner("Scoring engine..."):
                        st.session_state.results[key] = predict(selected_subset, uploaded_df)
                except BackendError as e:
                    st.error(f"Could not score this file: {e}")
            if key in st.session_state.results:
                render_prediction(st.session_state.results[key])

st.markdown(
    '<div class="muted" style="margin-top:2.5rem;text-align:center">'
    "NASA C-MAPSS turbofan degradation data · survival analysis with scikit-survival and lifelines · "
    + ("models run inside this app (the FastAPI service is used in the Docker deployment)"
       if EMBEDDED else "predictions served by a separate FastAPI service")
    + "</div>",
    unsafe_allow_html=True,
)
