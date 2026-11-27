#!/bin/sh
# permission-rag demo — one command, no install, no API key.
# Seeds the corpus from both connectors, runs 4 queries as different users
# (allowed / denied / cross-connector / public), prints the ACL proof and
# the RAGAS-style eval summary. Needs Python 3.12+ only; Ollama is used
# for answering when a local server is running, otherwise the extractive
# fallback runs instead (the mode is printed per answer).
set -e
cd "$(dirname "$0")"
python3 demo.py
