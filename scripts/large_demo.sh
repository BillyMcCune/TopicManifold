#!/usr/bin/env bash
# Generate a 120-prompt corpus and verify native HDBSCAN clustering.
# Usage: PYTHON_BIN=/path/to/python ./scripts/large_demo.sh [output-directory]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="${1:-$ROOT_DIR/large_demo_run}"
LOG_DIR="$RUN_DIR/logs"
OUTPUT_DIR="$RUN_DIR/topic_output"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing path: $RUN_DIR" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sentence_transformers, hdbscan, sklearn, pandas, numpy' >/dev/null 2>&1; then
  echo "Python dependencies are unavailable. Activate the topicmanifold environment first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" - "$LOG_DIR/large-sample.prompts.jsonl" <<'PY'
import json
import sys

path = sys.argv[1]
topics = {
    "ruby-on-rails": [
        "Ruby on Rails routing and nested resources", "Rails Active Record associations",
        "Rails controller actions and strong parameters", "Rails views with ERB partials",
        "Rails database migrations and schema changes", "Rails model validations and callbacks",
    ],
    "go-concurrency": [
        "Go concurrency worker pools", "Go channels and buffered queues",
        "Go context cancellation", "Go goroutine leak diagnosis",
        "Go mutexes and race conditions", "Go concurrent HTTP service design",
    ],
    "postgres-performance": [
        "Postgres query planner and indexes", "Postgres EXPLAIN ANALYZE for joins",
        "Postgres lock contention", "Postgres composite index design",
        "Postgres slow query optimization", "Postgres connection pool performance",
    ],
    "web-accessibility": [
        "Web accessibility keyboard navigation", "ARIA labels for interactive controls",
        "Accessible React dialogs and focus", "Color contrast for web dashboards",
        "Screen reader testing for forms", "Accessible semantic HTML structure",
    ],
    "kubernetes-operations": [
        "Kubernetes deployment rollout failures", "Kubernetes pod resource limits",
        "Kubernetes service networking", "Kubernetes Helm chart configuration",
        "Kubernetes cluster autoscaling", "Kubernetes log debugging",
    ],
}
questions = [
    "Explain the important concepts and common mistakes for {subject}.",
    "Give me a practical implementation plan for {subject}.",
    "How do I debug a production problem involving {subject}?",
    "What tests should I write when working with {subject}?",
]

with open(path, "w", encoding="utf-8") as f:
    number = 0
    for session_id, subjects in topics.items():
        for subject in subjects:
            for question in questions:
                number += 1
                f.write(json.dumps({
                    "schema_version": 1,
                    "timestamp": f"2026-08-01T13:{number // 60:02d}:{number % 60:02d}Z",
                    "session_id": session_id,
                    "role": "user",
                    "content": question.format(subject=subject),
                    "command": ["codex"],
                    "working_dir": "/example/large-demo",
                }) + "\n")
PY

echo "Generated 120 prompts across five known topics."
"$PYTHON_BIN" "$ROOT_DIR/cluster_logs.py" \
  --input-dir "$LOG_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --min-cluster-size 8 \
  --min-samples 3 \
  --fallback-similarity 0

echo
echo "HDBSCAN-only results: $OUTPUT_DIR/topics.csv"
echo "Wiki metadata: $OUTPUT_DIR/topic_wiki.json"
