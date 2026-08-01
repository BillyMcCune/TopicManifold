#!/usr/bin/env python3
import argparse
import json
import os
import glob
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import hdbscan

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


# Some terminal frontends inject printable fragments while negotiating input
# modes. Keep them in the raw ledger, but do not let them name a topic.
TERMINAL_PROTOCOL_FRAGMENT = re.compile(
    r"[a-z]{2}:[a-z0-9]{4}(?:/[a-z0-9]+)+(?:[a-z]{2}:[a-z0-9]+(?:/[a-z0-9]+)+)?",
    re.IGNORECASE,
)


def clean_for_topic_terms(text: str) -> str:
    return TERMINAL_PROTOCOL_FRAGMENT.sub(" ", text)


@dataclass
class Doc:
    id: str
    text: str
    source_file: str
    role: str  # "user" or "assistant" or "unknown"


def read_prompts_txt(path: str) -> List[Doc]:
    docs: List[Doc] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text:
                docs.append(Doc(
                    id=f"{os.path.basename(path)}:line{i+1}",
                    text=text,
                    source_file=path,
                    role="user"
                ))
    return docs


def read_chat_jsonl(path: str, include_assistant: bool,
                    include_commands: bool = False) -> List[Doc]:
    docs: List[Doc] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            role = rec.get("role", "unknown")
            content = (rec.get("content") or "").strip()
            if not content:
                continue
            if not include_commands and (rec.get("kind") == "command" or content.startswith("/")):
                continue
            if role == "assistant" and not include_assistant:
                continue
            docs.append(Doc(
                id=f"{os.path.basename(path)}:line{i+1}",
                text=content,
                source_file=path,
                role=role
            ))
    return docs


def collect_docs(input_dir: str, use_chat_jsonl: bool, include_assistant: bool,
                 deduplicate: bool = False, include_commands: bool = False) -> List[Doc]:
    docs: List[Doc] = []
    if use_chat_jsonl:
        # *.prompts.jsonl is emitted by the Go logger.  Keep the older
        # *.chat.jsonl convention so logs from previous versions still work.
        files = sorted(set(
            glob.glob(os.path.join(input_dir, "*.prompts.jsonl")) +
            glob.glob(os.path.join(input_dir, "*.chat.jsonl"))
        ))
        for fp in files:
            docs.extend(read_chat_jsonl(fp, include_assistant, include_commands))
        # Fallback to prompts.txt if no chat.jsonl
        if not files:
            files = sorted(glob.glob(os.path.join(input_dir, "*.prompts.txt")))
            for fp in files:
                docs.extend(read_prompts_txt(fp))
    else:
        files = sorted(glob.glob(os.path.join(input_dir, "*.prompts.txt")))
        for fp in files:
            docs.extend(read_prompts_txt(fp))
        # Fallback to chat.jsonl if no prompts
        if not files:
            files = sorted(set(
                glob.glob(os.path.join(input_dir, "*.prompts.jsonl")) +
                glob.glob(os.path.join(input_dir, "*.chat.jsonl"))
            ))
            for fp in files:
                docs.extend(read_chat_jsonl(fp, include_assistant, include_commands))

    if not deduplicate:
        return docs

    # Optional exact deduplication (keep first). Repeated prompts are normally
    # meaningful interaction history, so this is opt-in.
    seen = set()
    unique_docs: List[Doc] = []
    for d in docs:
        key = (d.text, d.role)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(d)
    return unique_docs


def embed_texts(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings)


def build_embedding_texts(docs: List[Doc], context_window: int) -> List[str]:
    """Add prior prompts from the same log file only for semantic embedding.

    Short follow-ups such as "what are views?" otherwise lose the subject of
    the conversation. Raw text in assignments and the JSONL ledger is never
    changed.
    """
    if context_window <= 0:
        return [doc.text for doc in docs]

    texts: List[str] = []
    history_by_source: Dict[str, List[str]] = {}
    for doc in docs:
        history = history_by_source.setdefault(doc.source_file, [])
        context = history[-context_window:]
        if context:
            context_text = "\n".join(f"Previous prompt: {item[:800]}" for item in context)
            texts.append(f"{context_text}\nCurrent prompt: {doc.text}")
        else:
            texts.append(doc.text)
        history.append(doc.text)
    return texts


def similarity_fallback_clusters(X: np.ndarray, min_cluster_size: int,
                                 similarity_threshold: float) -> np.ndarray:
    """Cluster connected, highly similar documents when HDBSCAN finds none."""
    labels = np.full(len(X), -1, dtype=int)
    similarity = X @ X.T  # embeddings are normalized
    visited = np.zeros(len(X), dtype=bool)
    next_topic = 0
    for start in range(len(X)):
        if visited[start]:
            continue
        component = []
        stack = [start]
        visited[start] = True
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in np.where(similarity[current] >= similarity_threshold)[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        if len(component) >= min_cluster_size:
            labels[component] = next_topic
            next_topic += 1
    return labels


def cluster_embeddings(X: np.ndarray, min_cluster_size: int,
                       min_samples: Optional[int] = None,
                       fallback_similarity: float = 0.65) -> Tuple[np.ndarray, str]:
    if len(X) < min_cluster_size:
        return np.full(len(X), -1, dtype=int), "insufficient_documents"
    # HDBSCAN will label noise as -1
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=False
    )
    labels = clusterer.fit_predict(X)
    if np.any(labels != -1) or fallback_similarity <= 0:
        return labels, "hdbscan"
    return (similarity_fallback_clusters(X, min_cluster_size, fallback_similarity),
            "similarity_graph_fallback")


def llm_labels(texts_by_topic: Dict[int, List[str]], api_key: str,
               model: str, api_url: str) -> Dict[int, str]:
    """Request concise labels from any Chat Completions-compatible endpoint."""
    labels: Dict[int, str] = {}
    for topic_id, texts in texts_by_topic.items():
        examples = "\n".join(f"- {text[:700]}" for text in texts[:12])
        prompt = (
            "Give this cluster of terminal-assistant prompts a precise topic label. "
            "Return only the label, with at most 8 words and no quotation marks.\n\n"
            f"Prompts:\n{examples}"
        )
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            api_url, data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.load(response)
            label = data["choices"][0]["message"]["content"].strip().replace("\n", " ")
            if label:
                labels[topic_id] = label[:160]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            print(f"Warning: could not label topic {topic_id} with the LLM: {exc}", file=sys.stderr)
    return labels


def ctfidf_labels(texts_by_topic: Dict[int, List[str]],
                  ngram_range=(1, 2),
                  topn_terms: int = 6) -> Dict[int, Tuple[str, List[str]]]:
    """
    Compute class-based TF-IDF (c-TF-IDF) labels per topic:
    - Concatenate texts per topic into a single "class doc"
    - Build CountVectorizer over class docs
    - Compute c-TF-IDF = (term_count / total_terms_in_class) * log(1 + n_classes / df_term)
    """
    topic_ids = sorted(texts_by_topic.keys())
    class_docs = [" ".join(clean_for_topic_terms(text) for text in texts_by_topic[t])
                  for t in topic_ids]
    if not class_docs:
        return {}

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=1
    )
    X = vectorizer.fit_transform(class_docs)  # shape: [n_topics, n_terms]
    terms = np.array(vectorizer.get_feature_names_out())

    # term frequency normalized per class
    tf = normalize(X, norm='l1', axis=1)  # each row sums to 1
    # document frequency across classes
    df = (X > 0).sum(axis=0).A1  # number of classes where term appears
    n_classes = X.shape[0]
    idf = np.log(1 + (n_classes / (df + 1e-12)))  # smooth
    ctfidf = tf.multiply(idf)

    labels = {}
    for idx, topic_id in enumerate(topic_ids):
        row = ctfidf.getrow(idx).toarray().ravel()
        if row.sum() == 0:
            labels[topic_id] = ("(misc)", [])
            continue
        top_idx = row.argsort()[::-1][:topn_terms]
        top_terms = [terms[i] for i in top_idx if row[i] > 0]
        # Keywords remain available in full, while the visible fallback title
        # is intentionally compact. LLM mode can replace this with a richer
        # human-readable title.
        label = top_terms[0].title() if top_terms else "(misc)"
        labels[topic_id] = (label if label else "(misc)", top_terms)
    return labels


def choose_exemplars(X: np.ndarray, labels: np.ndarray, docs: List[Doc]) -> Dict[int, int]:
    """
    Pick one exemplar doc per topic: doc with highest cosine sim to cluster centroid
    (since embeddings are normalized, centroid can be normalized mean).
    Returns mapping topic_id -> doc_index
    """
    exemplars = {}
    unique_topics = sorted(t for t in set(labels) if t != -1)
    if not unique_topics:
        return exemplars

    # Pre-normalized X, but centroid mean may not be norm 1 — normalize it.
    for t in unique_topics:
        idxs = np.where(labels == t)[0]
        if len(idxs) == 0:
            continue
        centroid = X[idxs].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        # Cosine sim is dot product (since X is normalized)
        sims = (X[idxs] @ centroid)
        best_local = idxs[int(np.argmax(sims))]
        exemplars[t] = int(best_local)
    return exemplars


def build_outputs(docs: List[Doc],
                  labels: np.ndarray,
                  topic_labels: Dict[int, Tuple[str, List[str]]],
                  exemplar_idx: Dict[int, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, d in enumerate(docs):
        t = int(labels[i])
        lbl = topic_labels.get(t, ("(noise)", []))[0] if t != -1 else "(noise)"
        rows.append({
            "doc_id": d.id,
            "topic_id": t,
            "label": lbl,
            "role": d.role,
            "text": d.text,
            "source_file": d.source_file
        })
    assignments = pd.DataFrame(rows)

    topic_rows = []
    for t, grp in assignments.groupby("topic_id"):
        if t == -1:
            label = "(noise)"
            top_terms = []
            ex_text = ""
        else:
            label, top_terms = topic_labels.get(t, ("(misc)", []))
            ex_i = exemplar_idx.get(t, None)
            ex_text = docs[ex_i].text if ex_i is not None else ""
        topic_rows.append({
            "topic_id": int(t),
            "size": int(len(grp)),
            "label": label,
            "top_terms": ", ".join(top_terms) if top_terms else "",
            "exemplar": ex_text
        })
    topics = pd.DataFrame(topic_rows).sort_values(by=["topic_id"]).reset_index(drop=True)
    return topics, assignments


def build_topic_wiki(docs: List[Doc], labels: np.ndarray,
                     topic_labels: Dict[int, Tuple[str, List[str]]],
                     exemplar_idx: Dict[int, int], args: Any) -> Dict[str, Any]:
    """Create a portable, provenance-preserving category artifact."""
    categories = []
    for topic_id in sorted(t for t in set(labels) if t != -1):
        member_indexes = np.where(labels == topic_id)[0]
        label, terms = topic_labels.get(topic_id, ("(misc)", []))
        exemplar = exemplar_idx.get(topic_id)
        categories.append({
            "id": f"topic-{topic_id}",
            "topic_id": int(topic_id),
            "title": label,
            "parent_id": None,
            "keywords": terms,
            "document_count": int(len(member_indexes)),
            "exemplar": ({"doc_id": docs[exemplar].id, "text": docs[exemplar].text}
                         if exemplar is not None else None),
            "members": [docs[i].id for i in member_indexes],
        })
    noise_indexes = np.where(labels == -1)[0]
    clustered_count = len(docs) - len(noise_indexes)
    topic_count = len(categories)
    if topic_count == 0:
        status = "no_topics_found"
        recommendation = "Try --min-samples 1 for a small or diverse corpus."
    elif len(noise_indexes) > clustered_count:
        status = "partial_coverage"
        recommendation = "Many prompts are unclustered; consider lowering --min-samples."
    else:
        status = "ok"
        recommendation = None
    return {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "generation": {
            "embedding_model": args.model,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "context_window": args.context_window,
            "clustering_method": getattr(args, "clustering_method", "hdbscan"),
            "fallback_similarity": args.fallback_similarity,
            "label_mode": args.label_mode,
            "input_dir": os.path.abspath(args.input_dir),
            "document_count": len(docs),
        },
        "categories": categories,
        "unclustered_document_ids": [docs[i].id for i in noise_indexes],
        "summary": {
            "status": status,
            "topic_count": topic_count,
            "clustered_document_count": clustered_count,
            "unclustered_document_count": int(len(noise_indexes)),
            "cluster_coverage": round(clustered_count / len(docs), 4) if docs else 0,
            "recommendation": recommendation,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Unsupervised topic clustering of CLI logs with BERT + HDBSCAN + c-TF-IDF labels")
    ap.add_argument("--input-dir", type=str, required=True, help="Directory containing *.prompts.jsonl, *.chat.jsonl, and/or *.prompts.txt")
    ap.add_argument("--use-chat-jsonl", type=lambda x: str(x).lower() == "true", default=True, help="Prefer structured JSONL over prompts.txt")
    ap.add_argument("--include-assistant", type=lambda x: str(x).lower() == "true", default=False, help="Include assistant messages")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    ap.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size")
    ap.add_argument("--min-samples", type=int, default=None,
                    help="HDBSCAN density threshold; lower values form more small topics (default: min-cluster-size)")
    ap.add_argument("--context-window", type=int, default=2,
                    help="Prior prompts from the same session included when embedding follow-ups (0 disables)")
    ap.add_argument("--fallback-similarity", type=float, default=0.65,
                    help="Cosine-similarity threshold used only if HDBSCAN finds no topics (0 disables fallback)")
    ap.add_argument("--topn-terms", type=int, default=6, help="Top terms per topic label")
    ap.add_argument("--output-dir", type=str, default="topic_output", help="Where to write topics.csv and assignments.csv")
    ap.add_argument("--deduplicate", action="store_true", help="Drop exact duplicate role/text pairs before clustering")
    ap.add_argument("--include-commands", action="store_true", help="Include slash commands such as /exit as documents")
    ap.add_argument("--label-mode", choices=["keywords", "llm"], default="keywords",
                    help="Use local c-TF-IDF keywords (default) or LLM-generated labels")
    ap.add_argument("--llm-model", default="gpt-4o-mini", help="Model used when --label-mode llm")
    ap.add_argument("--llm-api-url", default="https://api.openai.com/v1/chat/completions",
                    help="Chat Completions-compatible endpoint used for LLM labels")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading logs…")
    docs = collect_docs(args.input_dir, args.use_chat_jsonl, args.include_assistant,
                        args.deduplicate, args.include_commands)
    if not docs:
        print("No documents found. Check --input-dir and file patterns.")
        return

    texts = [d.text for d in docs]
    embedding_texts = build_embedding_texts(docs, args.context_window)
    print(f"Collected {len(texts)} items (context window: {args.context_window}).")

    print("Embedding with Sentence-Transformers…")
    X = embed_texts(embedding_texts, args.model, batch_size=64)

    print("Clustering with HDBSCAN…")
    labels, args.clustering_method = cluster_embeddings(
        X, args.min_cluster_size, args.min_samples, args.fallback_similarity)
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"Found {n_topics} topics (+ {n_noise} noise) via {args.clustering_method}.")
    if n_topics == 0 and len(texts) >= args.min_cluster_size:
        print("No dense topics found. Try --min-samples 1 for a small or diverse corpus.")

    # Build texts per topic for labeling
    texts_by_topic: Dict[int, List[str]] = {}
    for i, t in enumerate(labels):
        if t not in texts_by_topic:
            texts_by_topic[t] = []
        texts_by_topic[t].append(texts[i])

    print("Generating c-TF-IDF topic labels…")
    topic_labels = ctfidf_labels({k: v for k, v in texts_by_topic.items() if k != -1},
                                 ngram_range=(1, 2),
                                 topn_terms=args.topn_terms)

    if args.label_mode == "llm" and topic_labels:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            ap.error("--label-mode llm requires OPENAI_API_KEY")
        generated = llm_labels(
            {topic_id: texts_by_topic[topic_id] for topic_id in topic_labels},
            api_key, args.llm_model, args.llm_api_url,
        )
        for topic_id, label in generated.items():
            _, terms = topic_labels[topic_id]
            topic_labels[topic_id] = (label, terms)

    print("Choosing exemplars…")
    exemplars = choose_exemplars(X, labels, docs)

    print("Building outputs…")
    topics, assignments = build_outputs(docs, labels, topic_labels, exemplars)

    topics_csv = os.path.join(args.output_dir, "topics.csv")
    assignments_csv = os.path.join(args.output_dir, "assignments.csv")
    wiki_json = os.path.join(args.output_dir, "topic_wiki.json")
    topics.to_csv(topics_csv, index=False)
    assignments.to_csv(assignments_csv, index=False)
    with open(wiki_json, "w", encoding="utf-8") as f:
        json.dump(build_topic_wiki(docs, labels, topic_labels, exemplars, args), f,
                  ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  - {topics_csv}")
    print(f"  - {assignments_csv}")
    print(f"  - {wiki_json}")

    # Console summary
    print("\nTop topics:")
    show = topics[topics["topic_id"] != -1].sort_values("size", ascending=False).head(10)
    for _, r in show.iterrows():
        print(f"[{int(r['topic_id']):>3}] n={int(r['size']):<4} | {r['label']}")
        if r['exemplar']:
            print(f"      e.g., {r['exemplar'][:120].replace('\\n',' ')}")


if __name__ == "__main__":
    main()
