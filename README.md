# TopicManifold

TopicManifold lets you log your CLI AI chats and automatically cluster your prompts into topics with BERT-based embeddings.

Track what you've been asking, see recurring themes, and organize your AI usage history.

## Features

**CLI Logger** → Run claude chat (or any other command) in a PTY and record your prompts.

**Prompt Logs** → Stored as .txt and .jsonl in .ai-cli-log/.

**Topic Clustering** → Python script with BERT + HDBSCAN.

**Auto-labeling** → Topic summaries with c-TF-IDF keywords.

**Exports** → topics.csv and assignments.csv for analysis.

## Installation

### Go (Logger)
```bash
git clone https://github.com/BillyMcCune/TopicManifold.git
cd TopicManifold
go build -o bin/ailog ./cmd/ailog
```

### Python (Clustering)
```bash
# pick your environment (venv or conda recommended)
pip install -r requirements.txt
```

## Usage

### Step 1: Log prompts
```bash
./bin/ailog
```

Default runs: `claude chat`

Logs written to `.ai-cli-log/`

### Step 2: Cluster logs
```bash
python cluster_logs.py \
  --input-dir .ai-cli-log \
  --use-chat-jsonl false \
  --include-assistant false \
  --model all-MiniLM-L6-v2 \
  --output-dir topic_output
```

## Output

### topic_output/topics.csv
```
topic_id, size, label, top_terms, exemplar
0, 42, "cache, memory, threads", "cache, memory, threads, concurrency...", "how do I implement a reader-writer lock?"
```

### topic_output/assignments.csv
```
doc_id, topic_id, label, role, text
20250918-claude.prompts.txt:line12, 0, "cache, memory, threads", user, "how do I implement a reader-writer lock?"
```

## Configuration / Flags

`--min-cluster-size` → Controls topic granularity.

`--model` → Swap Sentence-Transformers model.

`--use-chat-jsonl` / `--include-assistant` → Include assistant replies or not.

## Roadmap

- [ ] HTML report with collapsible topic views.
- [ ] Interactive dashboard (Streamlit/Gradio).
- [ ] Multi-model embedding comparison.

## License

MIT 