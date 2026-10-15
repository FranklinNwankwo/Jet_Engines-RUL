"""Smoke test for the Docker Compose deployment, run INSIDE the containers.

Used by .github/workflows/docker.yml after `docker compose up`:

  docker compose exec -T api python - api < scripts/docker_smoke_test.py
  docker compose exec -T app python - app < scripts/docker_smoke_test.py

`api` mode (inside the API container): /health reports both subsets, /predict
returns plausible results for FD001 and FD004 test engine 1, and a bad
request gets a clean 422. Uses the C-MAPSS files the image downloaded while
exporting the models (data/raw/).

`app` mode (inside the dashboard container): the dashboard reaches the API
over the Compose network at API_BASE_URL (http://api:8000), and a headless
run of the real dashboard (Streamlit AppTest) makes a prediction through it.
"""

import os
import sys

import requests

MODE = sys.argv[1] if len(sys.argv) > 1 else "api"
CHECKS = {"FD001": (80, 160), "FD004": (0, 80)}   # engine 1: true RUL 112 / 22


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)
    print(f"ok   {message}")


def api_mode():
    sys.path.insert(0, "src")
    from turbofan_rul import data_loading

    base = "http://localhost:8000"
    health = requests.get(f"{base}/health", timeout=30).json()
    check(set(CHECKS) <= set(health["subsets_loaded"]), f"/health lists {sorted(CHECKS)}: {health}")
    versions = health.get("model_versions", {})
    check(all(versions.get(s, "unversioned") != "unversioned" for s in CHECKS),
          f"the image's export wrote verified manifests: {versions}")

    for subset, (lo, hi) in CHECKS.items():
        meta = requests.get(f"{base}/subsets/{subset}/metadata", timeout=30).json()
        cmapss = data_loading.load_subset(subset, data_dir="data/raw")
        engine = cmapss.test[cmapss.test.unit_number == 1].sort_values("time_cycles")
        readings = engine[["time_cycles"] + meta["raw_passthrough_cols"]].to_dict(orient="records")
        resp = requests.post(f"{base}/predict", json={"subset": subset, "readings": readings}, timeout=60)
        check(resp.status_code == 200, f"{subset} /predict returns 200 (got {resp.status_code})")
        r = resp.json()
        check(r["model_version"] == versions[subset], f"{subset} prediction reports model {r['model_version']}")
        check((r.get("input_check") or {}).get("status") == "familiar",
              f"{subset} engine 1 passes the input check: {(r.get('input_check') or {}).get('status')}")
        low, high = r["confidence_interval_90"]
        check(lo <= r["predicted_rul_cycles"] <= hi,
              f"{subset} prediction {r['predicted_rul_cycles']:.1f} cycles within [{lo}, {hi}]")
        check(0 <= low < high, f"{subset} interval [{low:.1f}, {high:.1f}] is ordered")

    report = requests.get(f"{base}/drift", timeout=30).json()
    check(all(report.get(s, {}).get("requests_seen", 0) >= 1 for s in CHECKS),
          f"/drift monitors both subsets and saw the requests: "
          f"{ {s: report.get(s, {}).get('status') for s in CHECKS} }")

    bad = requests.post(f"{base}/predict", json={"subset": "FD001", "readings": []}, timeout=30)
    check(bad.status_code == 422, f"empty readings rejected with 422 (got {bad.status_code})")


def app_mode():
    base = os.environ.get("API_BASE_URL", "http://api:8000")
    subsets = requests.get(f"{base}/subsets", timeout=30).json()["subsets"]
    check("FD001" in subsets, f"dashboard container reaches the API at {base}: {subsets}")

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=300).run()
    check(not at.exception and not at.error, "dashboard starts without errors")
    status = " ".join(m.value for m in at.sidebar.markdown)
    check("Model API online" in status, "dashboard is in HTTP mode, talking to the API")
    at.button(key="demo_predict").click().run()
    check(not at.exception and not at.error, "dashboard prediction runs without errors")
    results = list(at.session_state["results"].values())
    check(bool(results), "dashboard stored a prediction")
    lo, hi = CHECKS["FD001"]
    pred = results[-1]["predicted_rul_cycles"]
    check(lo <= pred <= hi, f"dashboard FD001 prediction {pred:.1f} cycles within [{lo}, {hi}]")


{"api": api_mode, "app": app_mode}[MODE]()
print(f"\n{MODE} smoke test passed")
