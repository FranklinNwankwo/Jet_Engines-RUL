"""
Streamlit dashboard: pick a subset, pick or upload an engine's recent
sensor history, see predicted RUL, confidence interval, and risk flag.

This app is a PURE HTTP CLIENT of the FastAPI service (api/main.py) -- it
never imports `turbofan_rul.inference`, never loads a model artifact, and
has no access to `models/` at all. Every prediction is a POST to the API's
`/predict` endpoint; the sensor list and required-columns schema needed to
build that request and render the sensor chart come from the API's
`/subsets` and `/subsets/{subset}/metadata` endpoints, not from reading
deployment metadata directly. This is the intended production shape for
this project's two-service deployment (see README's Deployment section) --
an earlier version of this file called `inference.predict` in-process as a
documented single-process demo simplification; that simplification is
gone, not just described differently, now that the project is being
containerized as two separate services.

The API's base URL is read from the `API_BASE_URL` environment variable
(defaulting to `http://localhost:8000` for local development), so the same
image works whether the API is reached at localhost (running both services
by hand), a Docker Compose service name (e.g. `http://api:8000`), or a
public URL after cloud deployment -- nothing about that address is
hardcoded.

Run with: API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Only used for the "sample engine" demo picker, which reads the public
# C-MAPSS dataset directly -- this is demo convenience data loading, not
# model/scoring logic, so it staying local (rather than round-tripping
# through the API) doesn't reintroduce the in-process-scoring problem this
# rewrite exists to remove. Nothing here ever touches a fitted model.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from turbofan_rul import data_loading  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="Turbofan RUL Prediction", layout="wide")


@st.cache_data(ttl=60)
def get_available_subsets() -> list[str]:
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
def get_subset_metadata(subset: str) -> dict:
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


def render_prediction(result: dict, true_rul: float | None = None) -> None:
    risk_colors = {"critical": "🔴", "warning": "🟡", "healthy": "🟢"}
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL", f"{result['predicted_rul_cycles']:.0f} cycles")
    low, high = result["confidence_interval_90"]
    col2.metric("90% confidence interval", f"[{low:.0f}, {high:.0f}]")
    col3.metric("Risk level", f"{risk_colors[result['risk_level']]} {result['risk_level']}")

    if result["point_estimate_outside_interval"]:
        st.info(
            "The point estimate falls outside its own interval. This is an expected "
            "consequence of using two independent models (Random Survival Forest for the "
            "point estimate, Weibull AFT for the interval) rather than a bug -- see "
            "`inference.py` for why.",
            icon="ℹ️",
        )

    rsf_cal = result.get("rsf_calibrated_interval_supplementary")
    if rsf_cal:
        cal_low, cal_high = rsf_cal["interval_90"]
        with st.expander("RSF's own calibrated interval (supplementary, not authoritative)"):
            st.write(f"[{cal_low:.0f}, {cal_high:.0f}] cycles")
            st.caption(rsf_cal["caveat"])

    if true_rul is not None:
        st.caption(f"Ground truth RUL for this demo engine: {true_rul} cycles "
                   f"(shown because this is a known test engine -- a real deployment wouldn't have this)")


st.title("Turbofan Engine — Remaining Useful Life")
st.caption(
    "Random Survival Forest (point estimate + risk ranking) and Weibull AFT (confidence interval), "
    "per the Phase 6 model comparison in this project. Served over HTTP from a separate FastAPI "
    "service -- this dashboard holds no model artifacts itself. Not RMSE-optimized regression -- "
    "see docs/project_brief.md for why this is framed as a survival-analysis problem."
)

available_subsets = get_available_subsets()
if not available_subsets:
    st.error(
        f"Could not reach the API at `{API_BASE_URL}` (or it has no subsets loaded). "
        f"Start it with `uvicorn api.main:app --reload`, or set the `API_BASE_URL` "
        f"environment variable if it's running elsewhere."
    )
    st.stop()

subset_labels = {
    "FD001": "FD001 — single regime, single fault mode",
    "FD004": "FD004 — 6 regimes, 2 fault modes (harder; see notebooks/10)",
}
selected_subset = st.sidebar.radio(
    "Dataset / trained model",
    options=available_subsets,
    format_func=lambda s: subset_labels.get(s, s),
)
st.sidebar.caption(f"API: `{API_BASE_URL}`")

try:
    metadata = get_subset_metadata(selected_subset)
except requests.RequestException as e:
    st.error(f"Could not load metadata for {selected_subset} from the API at `{API_BASE_URL}`: {e}")
    st.stop()
kept_sensors = metadata["kept_sensors"]
raw_passthrough_cols = metadata["raw_passthrough_cols"]

tab_demo, tab_upload = st.tabs(["Sample engine", "Upload your own readings"])

with tab_demo:
    test_df, rul_lookup = get_demo_engines(selected_subset)
    unit_ids = sorted(test_df["unit_number"].unique())
    selected_unit = st.selectbox("Pick a test engine", unit_ids, index=0)

    engine_df = test_df[test_df.unit_number == selected_unit].sort_values("time_cycles")
    readings = engine_df[["time_cycles"] + raw_passthrough_cols].reset_index(drop=True)

    st.line_chart(engine_df.set_index("time_cycles")[kept_sensors[:4]],
                  height=250, width="stretch")
    st.caption(f"{len(readings)} cycles observed for engine {selected_unit} "
               "(first 4 kept sensors shown)")

    if st.button("Predict RUL", type="primary", key="demo_predict"):
        try:
            result = call_predict(selected_subset, readings)
            render_prediction(result, true_rul=rul_lookup.loc[selected_unit])
        except requests.HTTPError as e:
            st.error(f"Could not score this engine: {error_detail(e)}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")

with tab_upload:
    st.write(
        f"Upload a CSV with columns: `time_cycles`, `op_setting_1`-`op_setting_3`, "
        f"`sensor_1`-`sensor_21`, one row per cycle, oldest first. Scored against "
        f"the **{selected_subset}** model selected in the sidebar."
    )
    uploaded = st.file_uploader("Sensor readings CSV", type="csv")
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.dataframe(uploaded_df.tail(5), width="stretch")
            if st.button("Predict RUL", type="primary", key="upload_predict"):
                result = call_predict(selected_subset, uploaded_df)
                render_prediction(result)
        except requests.HTTPError as e:
            st.error(f"Could not score this file: {error_detail(e)}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
        except Exception as e:
            st.error(f"Could not read this file: {e}")