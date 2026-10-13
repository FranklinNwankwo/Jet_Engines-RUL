"""Embedded-mode smoke test for the dashboard (the mode the live demo uses).

Starts app/streamlit_app.py headlessly with Streamlit's AppTest, with the
models loaded in-app -- from the Hugging Face repo named in MODEL_REPO (as on
Streamlit Community Cloud) or from a local MODELS_DIR -- and makes real
predictions for a sample FD001 and FD004 test engine.

Its main job is catching the failure that would break the live demo: the
library versions pinned in app/requirements.txt no longer matching the
environment that exported the models (pickled models can fail to load, or
load with a scikit-learn version warning, under different versions).

Skipped unless MODEL_REPO or MODELS_DIR is set, so a plain `pytest` run
without models still passes. In CI it runs in the `live-demo-models` job of
.github/workflows/tests.yml.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("MODEL_REPO") or os.environ.get("MODELS_DIR")),
    reason="set MODEL_REPO (Hugging Face repo) or MODELS_DIR (local folder) to run the embedded-mode test",
)

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    saved = os.environ.pop("API_BASE_URL", None)   # API_BASE_URL would force HTTP mode
    try:
        # Generous timeout: the first run downloads ~340MB of models.
        yield testing.AppTest.from_file(str(APP), default_timeout=900).run()
    finally:
        if saved is not None:
            os.environ["API_BASE_URL"] = saved


def _latest_result(at, subset):
    results = [r for key, r in at.session_state["results"].items() if key[1] == subset]
    assert results, f"no prediction stored for {subset}"
    return results[-1]


def test_app_starts_in_embedded_mode(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    status = " ".join(m.value for m in app.sidebar.markdown)
    assert "Models loaded in this app" in status


def test_models_load_without_library_version_mismatch(app):
    # The sidebar shows a warning when unpickling raised a scikit-learn
    # InconsistentVersionWarning, i.e. the pins don't match the export.
    assert not app.sidebar.warning, [w.value for w in app.sidebar.warning]


def test_fd001_sample_engine_prediction(app):
    app.sidebar.radio[0].set_value("FD001").run()
    app.button(key="demo_predict").click().run()
    assert not app.exception and not app.error

    r = _latest_result(app, "FD001")
    low, high = r["confidence_interval_90"]
    # Engine 1's true RUL is 112 cycles; the deployed models predict ~121.
    assert 80 <= r["predicted_rul_cycles"] <= 160
    assert 0 <= low < high
    assert r["risk_level"] == "healthy"


def test_fd004_sample_engine_prediction(app):
    # .options holds the display labels ("FD004 · 6 regimes · 2 faults").
    assert any(o.startswith("FD004") for o in app.sidebar.radio[0].options), "FD004 models not found"
    app.sidebar.radio[0].set_value("FD004").run()
    app.button(key="demo_predict").click().run()
    assert not app.exception and not app.error

    r = _latest_result(app, "FD004")
    low, high = r["confidence_interval_90"]
    # Engine 1's true RUL is 22 cycles; the deployed models predict ~33.
    assert 0 <= r["predicted_rul_cycles"] <= 80
    assert 0 <= low < high
    assert r["risk_level"] in {"critical", "warning"}
