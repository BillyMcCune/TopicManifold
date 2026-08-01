# TopicManifold

TopicManifold records AI-assisted terminal prompts and turns them into a searchable topic map. It has two intentionally separate stages:

1. `topicmanifold` wraps an interactive CLI in a PTY and records each submitted user prompt.
2. `cluster_logs.py` embeds the prompts with Sentence-Transformers (BERT), clusters them with HDBSCAN, and creates CSV and wiki-style JSON outputs.

The raw log is an append-only JSONL ledger; the topic wiki is a reproducible derived view. This preserves the evidence behind every category rather than making the LLM's labels the source of truth.

## Install

```bash
git clone https://github.com/BillyMcCune/TopicManifold.git
cd TopicManifold
go build -o bin/topicmanifold ./cmd/topicmanifold
```

For analysis, use a clean Python 3.12 environment. This avoids native PyTorch/Conda library mismatches:

```bash
conda create -n topicmanifold python=3.12
conda activate topicmanifold
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Record a terminal session

```bash
# Default command: claude chat
./bin/topicmanifold

# Any interactive command
./bin/topicmanifold -- codex
./bin/topicmanifold -- claude code --resume last
```

Each session creates two files in `.ai-cli-log/`:

- `*.prompts.txt` is convenient for quick inspection.
- `*.prompts.jsonl` is the canonical structured record. Every event includes `schema_version`, UTC `timestamp`, `session_id`, `role`, `content`, command arguments, and working directory.

The PTY wrapper records user input reliably. It deliberately does not pretend to separate assistant text from terminal rendering and control sequences. To include assistant responses, add a provider-specific export adapter that emits the same JSONL fields.

## Analyze and create a topic wiki

```bash
python cluster_logs.py --input-dir .ai-cli-log --output-dir topic_output
```

To see the complete workflow without recording a real terminal session, run:

```bash
./scripts/demo.sh
```

It creates a `demo_run/` folder containing sample structured logs and the resulting wiki. The first run may download the selected embedding model.
If a prior attempt created `demo_run/` before failing, remove that generated folder before retrying.

To test HDBSCAN itself on a larger 120-prompt corpus, run:

```bash
./scripts/large_demo.sh
```

It disables the small-corpus similarity fallback and reports `hdbscan` as the clustering method when the density-based model succeeds.

By default the analyzer reads `*.prompts.jsonl` (and supports legacy `*.chat.jsonl` logs), keeps repeated prompts as useful history, and uses local c-TF-IDF keywords for category names.

For LLM-generated category titles, set an API key and opt in explicitly:

```bash
OPENAI_API_KEY=... python cluster_logs.py \
  --input-dir .ai-cli-log \
  --label-mode llm \
  --llm-model gpt-4o-mini
```

`--llm-api-url` can point at any Chat Completions-compatible endpoint. Only representative prompt text is sent to that endpoint; use `--label-mode keywords` when all analysis must remain local.

## Outputs

- `topics.csv` — flat category summary, terms, and exemplars.
- `assignments.csv` — every prompt with its category and source file.
- `topic_wiki.json` — structured category document with schema version, generation configuration, stable topic IDs, hierarchy-ready `parent_id`, keywords, member document IDs, exemplars, and unclustered prompts.

The wiki is designed for a future dashboard or editor: human or LLM-curated descriptions and parent categories can be layered on top without modifying the original interaction ledger.

## Useful flags

- `--min-cluster-size 5` controls category granularity.
- `--min-samples 1` relaxes HDBSCAN's density threshold; useful for small or diverse datasets that would otherwise be entirely noise.
- `--context-window 2` embeds short follow-up prompts with the preceding prompts from the same session; set `0` for strictly independent prompts.
- `--fallback-similarity 0.65` forms high-similarity groups only when HDBSCAN finds no topics; set `0` to require HDBSCAN exclusively.
- `--model all-MiniLM-L6-v2` selects the Sentence-Transformers model.
- `--deduplicate` removes exact repeated role/text pairs (off by default).
- `--include-assistant true` accepts assistant entries from compatible JSONL exports.
- `--label-mode keywords|llm` selects local keyword or LLM-generated titles.

## Roadmap

- Provider adapters for structured assistant-response capture.
- Human/LLM category descriptions, aliases, and parent-category curation.
- HTML report and interactive dashboard.
- Multi-model embedding comparison.

## License

MIT
