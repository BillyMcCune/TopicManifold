#!/usr/bin/env python3
import argparse
import json
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import hdbscan

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


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


def read_chat_jsonl(path: str, include_assistant: bool) -> List[Doc]:
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
            if role == "assistant" and not include_assistant:
                continue
            docs.append(Doc(
                id=f"{os.path.basename(path)}:line{i+1}",
                text=content,
                source_file=path,
                role=role
            ))
    return docs


def collect_docs(input_dir: str, use_chat_jsonl: bool, include_assistant: bool) -> List[Doc]:
    docs: List[Doc] = []
    if use_chat_jsonl:
        files = sorted(glob.glob(os.path.join(input_dir, "*.chat.jsonl")))
        for fp in files:
            docs.extend(read_chat_jsonl(fp, include_assistant))
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
            files = sorted(glob.glob(os.path.join(input_dir, "*.chat.jsonl")))
            for fp in files:
                docs.extend(read_chat_jsonl(fp, include_assistant))

    # Deduplicate exact duplicates (keep first)
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


def cluster_embeddings(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    # HDBSCAN will label noise as -1
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=False
    )
    labels = clusterer.fit_predict(X)
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
    class_docs = [" ".join(texts_by_topic[t]) for t in topic_ids]
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
        label = ", ".join(top_terms[:max(1, topn_terms)])
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


def main():
    ap = argparse.ArgumentParser(description="Unsupervised topic clustering of CLI logs with BERT + HDBSCAN + c-TF-IDF labels")
    ap.add_argument("--input-dir", type=str, required=True, help="Directory containing *.prompts.txt and/or *.chat.jsonl")
    ap.add_argument("--use-chat-jsonl", type=lambda x: str(x).lower() == "true", default=True, help="Prefer *.chat.jsonl over prompts.txt")
    ap.add_argument("--include-assistant", type=lambda x: str(x).lower() == "true", default=False, help="Include assistant messages")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    ap.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size")
    ap.add_argument("--topn-terms", type=int, default=6, help="Top terms per topic label")
    ap.add_argument("--output-dir", type=str, default="topic_output", help="Where to write topics.csv and assignments.csv")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading logs…")
    docs = collect_docs(args.input_dir, args.use_chat_jsonl, args.include_assistant)
    if not docs:
        print("No documents found. Check --input-dir and file patterns.")
        return

    texts = [d.text for d in docs]
    print(f"Collected {len(texts)} items.")

    print("Embedding with Sentence-Transformers…")
    X = embed_texts(texts, args.model, batch_size=64)

    print("Clustering with HDBSCAN…")
    labels = cluster_embeddings(X, args.min_cluster_size)
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"Found {n_topics} topics (+ {n_noise} noise).")

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

    print("Choosing exemplars…")
    exemplars = choose_exemplars(X, labels, docs)

    print("Building outputs…")
    topics, assignments = build_outputs(docs, labels, topic_labels, exemplars)

    topics_csv = os.path.join(args.output_dir, "topics.csv")
    assignments_csv = os.path.join(args.output_dir, "assignments.csv")
    topics.to_csv(topics_csv, index=False)
    assignments.to_csv(assignments_csv, index=False)

    print(f"\nSaved:")
    print(f"  - {topics_csv}")
    print(f"  - {assignments_csv}")

    # Console summary
    print("\nTop topics:")
    show = topics[topics["topic_id"] != -1].sort_values("size", ascending=False).head(10)
    for _, r in show.iterrows():
        print(f"[{int(r['topic_id']):>3}] n={int(r['size']):<4} | {r['label']}")
        if r['exemplar']:
            print(f"      e.g., {r['exemplar'][:120].replace('\\n',' ')}")


if __name__ == "__main__":
    main()
