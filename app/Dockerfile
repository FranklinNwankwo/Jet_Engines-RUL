# Dashboard image: a pure HTTP client of the API service (see
# app/streamlit_app.py's module docstring). No model artifacts, no
# scikit-learn/scikit-survival/lifelines, requirements-app.txt is short
# specifically because this image doesn't need any of that anymore.
#
# Why src/ is still copied: the "Sample engine" tab imports
# turbofan_rul.data_loading to read the public C-MAPSS test files. That
# module needs only pandas and requests, and no model code is imported.
# The test files are downloaded the first time a subset is opened, so the
# container needs outbound internet access (or a data/raw/ folder mounted
# at /app/data/raw). The "Upload your own readings" tab needs neither.
#
# API_BASE_URL is a runtime environment variable, not baked in at build
# time, the same image works pointed at localhost (docker run by hand),
# a Compose service name (docker-compose.yml sets this to
# http://api:8000), or a public API URL after cloud deployment.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ ./src/
COPY app/ ./app/
# The dashboard theme. Streamlit reads .streamlit/config.toml from the
# working directory (/app), so it must sit at /app/.streamlit/.
COPY .streamlit/config.toml ./.streamlit/config.toml

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=http://localhost:8000

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]