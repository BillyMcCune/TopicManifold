#!/usr/bin/env bash
# Create sample interaction logs and turn them into a topic wiki.
# Usage: ./scripts/demo.sh [output-directory]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${1:-$ROOT_DIR/demo_run}"
LOG_DIR="$DEMO_DIR/logs"
OUTPUT_DIR="$DEMO_DIR/topic_output"

if [[ -e "$DEMO_DIR" ]]; then
  echo "Refusing to overwrite existing path: $DEMO_DIR" >&2
  echo "Choose another directory, for example: ./scripts/demo.sh /tmp/topicmanifold-demo" >&2
  exit 1
fi

# Check imports before creating output. This produces an actionable failure
# instead of leaving behind a partial demo directory that blocks a retry.
if ! python3 -c 'import sentence_transformers, hdbscan, sklearn, pandas, numpy' >/dev/null 2>&1; then
  echo "TopicManifold Python dependencies could not be imported." >&2
  echo "Create a clean Python 3.12 environment, then install requirements:" >&2
  echo "  conda create -n topicmanifold python=3.12" >&2
  echo "  conda activate topicmanifold" >&2
  echo "  python -m pip install --upgrade pip" >&2
  echo "  python -m pip install -r $ROOT_DIR/requirements.txt" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

python3 - "$LOG_DIR/sample.prompts.jsonl" <<'PY'
import json
import sys

path = sys.argv[1]
topics = {
    "go-concurrency": [
        "Go concurrency: how should I structure a worker pool?",
        "Go concurrency: explain cancellation propagation with context.Context.",
        "Go concurrency: when should a channel be buffered?",
        "Go concurrency: help me find a goroutine leak.",
    ],
    "postgres-performance": [
        "Postgres performance: why is this query not using my index?",
        "Postgres performance: how do I read EXPLAIN ANALYZE for a slow join?",
        "Postgres performance: suggest an index for account and date filtering.",
        "Postgres performance: what causes lock contention during migrations?",
    ],
    "frontend-accessibility": [
        "Web accessibility: make this modal keyboard accessible.",
        "Web accessibility: which ARIA attributes belong on a tab list?",
        "Web accessibility: test focus management in a React dialog.",
        "Web accessibility: improve dashboard chart color contrast.",
    ],
}

with open(path, "w", encoding="utf-8") as f:
    sequence = 0
    for session_id, prompts in topics.items():
        for content in prompts:
            sequence += 1
            record = {
                "schema_version": 1,
                "timestamp": f"2026-08-01T12:{sequence:02d}:00Z",
                "session_id": session_id,
                "role": "user",
                "content": content,
                "command": ["claude", "chat"],
                "working_dir": "/example/project",
            }
            f.write(json.dumps(record) + "\n")
PY

echo "Created sample interaction ledger: $LOG_DIR/sample.prompts.jsonl"
echo "Embedding and clustering the sample prompts..."

python3 "$ROOT_DIR/cluster_logs.py" \
  --input-dir "$LOG_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --min-cluster-size 3 \
  --min-samples 1

echo
echo "Demo complete. Explore:"
echo "  $OUTPUT_DIR/topic_wiki.json"
echo "  $OUTPUT_DIR/topics.csv"
echo "  $OUTPUT_DIR/assignments.csv"
