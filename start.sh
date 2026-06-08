#!/usr/bin/env bash
# Render start command. Mirrors render.yaml's startCommand so the service boots
# whether Render uses the blueprint or a dashboard "bash start.sh" override.
# $PORT is provided by Render at runtime.
set -e
python -m streamlit run app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true
