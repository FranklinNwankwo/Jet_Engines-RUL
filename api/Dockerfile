# API image: builds the FD001 AND FD004 deployment artifacts (models +
# conformal calibration) at DOCKER BUILD time, not at container startup.
#
# Why bake the models into the image rather than mount them at runtime:
# each subset's build (RSF + Weibull fit, plus the 5-fold cross-conformal
# calibration from Phase 14) takes minutes -- fine for a one-time image
# build, but a container that had to do that on every cold start would
# fail most platforms' startup/health-check timeouts.
# The tradeoff is a slow `docker build` in exchange for an instant
# `docker run` -- the right way around for a service, not a notebook.
#
# models/ is gitignored (the RSF artifact alone is 200+MB) and is NOT
# copied in from the host -- it's produced fresh inside the image, so this
# Dockerfile builds correctly from a clean clone with no local model files
# required, the same "reproducible from a clean checkout" guarantee
# export_deployment_artifacts.py already provides outside Docker.

FROM python:3.12-slim

WORKDIR /app

# Only what's needed to install/build wheels for scikit-survival etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/
# The one small (1.7KB) Phase-1 EDA artifact FD001's non-clustered recipe
# needs -- see export_deployment_artifacts.py's _prepare_fd001_style.
COPY data/processed/fd001_sensor_selection.json ./data/processed/fd001_sensor_selection.json

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Build the deployment artifacts now, inside the image -- each line
# downloads its C-MAPSS subset (cached to data/raw/ within the image
# layer), fits RSF + Weibull AFT on all training engines, and runs the
# 5-fold cross-conformal interval calibration from Phase 14. Real network
# + compute cost at build time, by design (see the header above).
#
# FD001 uses the sensor-selection JSON copied above. FD004 needs no extra
# files: its sensor selection and regime clustering are computed inside
# the script. They are separate RUN lines so that if the FD004 step fails,
# a rebuild reuses the cached FD001 layer instead of redoing it.
# The API refuses to start without FD001; FD004 is served when present.
RUN python -m turbofan_rul.export_deployment_artifacts --subset FD001
RUN python -m turbofan_rul.export_deployment_artifacts --subset FD004

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]