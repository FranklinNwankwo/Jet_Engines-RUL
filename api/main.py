"""
FastAPI service: recent sensor readings in, RUL + confidence interval +
risk flag out.

All scoring logic lives in `turbofan_rul.inference` -- this module is a thin
HTTP wrapper around it. The Streamlit dashboard (`app/streamlit_app.py`) is a
pure HTTP client of this API: it calls `/subsets`, `/subsets/{subset}/metadata`
and `/predict` over HTTP and imports no modeling code itself, so this service
is the only place models are loaded.

Multi-subset support (Phase 12d): both FD001 and FD004 bundles are loaded
at startup (whichever have been built via `export_deployment_artifacts.py`
-- FD001 is required, FD004 is optional and the API degrades gracefully
without it, since not every environment will have run the FD004 export).
The caller picks a subset per request; nothing is auto-detected.

Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from turbofan_rul import inference

MODELS_DIR = Path(__file__).parent.parent / "models"
AVAILABLE_SUBSETS = ["FD001", "FD004"]

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup, not per-request -- the deployed FD001 RSF
    # artifact alone is ~200+ MB (min_samples_leaf=5 grows deep trees), and
    # re-loading it on every call would dominate latency.
    #
    # Checked via rsf.joblib specifically, not deployment_metadata.json:
    # the metadata file is small and tracked in git, so it can exist on a
    # completely fresh clone with no model actually built yet. rsf.joblib
    # is gitignored (too large to version), so its presence is the real
    # signal that `export_deployment_artifacts.py` has actually been run.
    # Checking the tracked file alone (an earlier version of this code did)
    # let startup get as far as attempting to load a model that was never
    # there, crashing with a raw FileNotFoundError instead of this clear
    # message -- caught by an actual fresh-clone test, not assumed fixed.
    bundles = {}
    for subset in AVAILABLE_SUBSETS:
        subset_dir = MODELS_DIR / subset.lower()
        if (subset_dir / "rsf.joblib").exists():
            bundles[subset] = inference.load_deployment_bundle(subset_dir)
    if "FD001" not in bundles:
        raise RuntimeError(
            "FD001 deployment artifacts not found -- run "
            "`python -m turbofan_rul.export_deployment_artifacts --subset FD001` first."
        )
    _state["bundles"] = bundles
    yield
    _state.clear()


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predicts remaining useful life for turbofan engines from recent sensor telemetry, "
                "using a Random Survival Forest (point estimate/ranking) and a 90% interval combining "
                "RSF and a Weibull AFT model (lower bound = the more cautious of the two, upper bound = "
                "Weibull) -- see docs/technical_report.md for the model selection rationale. "
                "Supports FD001 (single regime/fault mode) and, if built, FD004 (6 regimes, 2 fault "
                "modes) via the `subset` field.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return request-validation errors as a 422 WITHOUT echoing the rejected
    input values back. FastAPI's default handler includes each bad value, and
    a rejected NaN or infinity can't be encoded as JSON -- so the default
    handler would turn a clean 422 into a 500 crash on exactly the inputs
    `SensorReading` exists to reject."""
    errors = [{k: v for k, v in err.items() if k not in ("input", "ctx")} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


class SensorReading(BaseModel):
    """One cycle of raw C-MAPSS telemetry. Field names match the raw dataset
    schema directly (op_setting_1-3, sensor_1-21) -- see
    `turbofan_rul.data_loading.COLUMN_NAMES`.

    NaN and infinite values are rejected at validation time (422), so a
    malformed reading can never reach the models."""
    model_config = ConfigDict(allow_inf_nan=False)

    time_cycles: int = Field(..., description="Cycle number for this reading, strictly increasing across a request")
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    subset: str = Field("FD001", description=f"Which trained subset to score against. One of {AVAILABLE_SUBSETS}.")
    readings: list[SensorReading] = Field(
        ..., min_length=1,
        description="Recent consecutive cycles for ONE engine, oldest first. "
                    "30+ cycles recommended for stable rolling-window features; fewer is accepted "
                    "but the earliest predictions for a newly-monitored engine will be noisier.",
    )


class PredictResponse(BaseModel):
    subset: str
    predicted_rul_cycles: float
    confidence_interval_90: list[float]
    point_estimate_outside_interval: bool
    risk_level: str
    cycles_observed: int
    model_source: dict
    weibull_interval_90: list[float]
    rsf_calibrated_interval_supplementary: dict


class SubsetMetadata(BaseModel):
    subset: str
    kept_sensors: list[str]
    raw_passthrough_cols: list[str]
    uses_regime_clustering: bool
    recommended_min_history_cycles: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "subsets_loaded": list(_state.get("bundles", {}).keys())}


@app.get("/subsets")
def list_subsets() -> dict:
    """Which subsets this API instance has loaded models for -- the
    dashboard calls this first to populate its subset picker, rather than
    hardcoding the list or reaching into model artifacts directly."""
    return {"subsets": list(_state.get("bundles", {}).keys())}


@app.get("/subsets/{subset}/metadata", response_model=SubsetMetadata)
def subset_metadata(subset: str) -> dict:
    """Everything a caller needs to BUILD a valid /predict request or render
    a sensor chart, without needing to load model artifacts itself: which
    sensors are informative (`kept_sensors`, for charting) and which raw
    columns a request must supply (`raw_passthrough_cols`). This is what
    lets the dashboard be a pure HTTP client with no direct access to
    `models/` -- see app/streamlit_app.py."""
    bundles = _state.get("bundles", {})
    if subset not in bundles:
        raise HTTPException(
            status_code=404,
            detail=f"subset '{subset}' not available -- loaded subsets: {list(bundles.keys())}.",
        )
    bundle = bundles[subset]
    return {
        "subset": bundle.subset,
        "kept_sensors": bundle.kept_sensors,
        "raw_passthrough_cols": bundle.raw_passthrough_cols,
        "uses_regime_clustering": bundle.uses_regime_clustering,
        "recommended_min_history_cycles": 30,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    bundles = _state["bundles"]
    if request.subset not in bundles:
        raise HTTPException(
            status_code=422,
            detail=f"subset '{request.subset}' not available -- loaded subsets: {list(bundles.keys())}. "
                   f"Build it with `python -m turbofan_rul.export_deployment_artifacts --subset {request.subset}`.",
        )
    readings_df = pd.DataFrame([r.model_dump() for r in request.readings])
    try:
        return inference.predict(bundles[request.subset], readings_df)
    except inference.InvalidReadingsError as e:
        # Maps a caller mistake (bad input shape) to 422, distinct from an
        # unexpected 500 -- the InvalidReadingsError/ValueError split in
        # inference.py exists specifically so this distinction is possible.
        raise HTTPException(status_code=422, detail=str(e))
