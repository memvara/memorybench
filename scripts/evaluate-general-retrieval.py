#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict

import numpy as np

DATASETS = ["nfcorpus", "scifact", "arguana"]
CONFIGS = {
    "gemini-retrieval-1536": ("gemini", "gemini-embedding-001", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", 1536),
    "openai-small-1536": ("openai", "text-embedding-3-small", "default", "default", 1536),
    "openai-large-1536": ("openai", "text-embedding-3-large", "default", "default", 1536),
    "openai-large-3072": ("openai", "text-embedding-3-large", "default", "default", 3072),
}


def jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def load_data(root):
    documents, document_texts, queries, query_texts, dataset_meta = [], [], [], [], {}
    document_vector_indices, query_vector_indices = {}, {}
    for dataset in DATASETS:
        directory = os.path.join(root, dataset)
        corpus = jsonl(os.path.join(directory, "corpus.jsonl"))
        query_rows = jsonl(os.path.join(directory, "queries.jsonl"))
        qrels = defaultdict(dict)
        for row in jsonl(os.path.join(directory, "test.jsonl")):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
        doc_start = len(documents)
        for row in corpus:
            text = "\n".join(value for value in [row.get("title"), row.get("text")] if value)
            if text not in document_vector_indices:
                document_vector_indices[text] = len(document_texts)
                document_texts.append(text)
            documents.append((row["_id"], text, dataset, document_vector_indices[text]))
        query_start = len(queries)
        for row in query_rows:
            if row["_id"] not in qrels:
                continue
            if row["text"] not in query_vector_indices:
                query_vector_indices[row["text"]] = len(query_texts)
                query_texts.append(row["text"])
            queries.append((row["_id"], row["text"], dataset, qrels[row["_id"]], query_vector_indices[row["text"]]))
        dataset_meta[dataset] = {
            "document_slice": (doc_start, len(documents)),
            "query_slice": (query_start, len(queries)),
        }
    return documents, document_texts, queries, query_texts, dataset_meta


def cache_for(cache_dir, provider, model, dimensions, task_type, texts):
    digest = hashlib.sha256()
    digest.update(f"{provider}\0{model}\0{dimensions}\0{task_type}".encode())
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode())
    cache_id = f"{provider}-{model}-{dimensions}-{task_type}"
    path = os.path.join(cache_dir, f"{cache_id}-{digest.hexdigest()[:16]}.f32")
    expected = len(texts) * dimensions * 4
    if not os.path.exists(path) or os.path.getsize(path) != expected:
        raise RuntimeError(f"Missing exact {expected}-byte cache: {path}")
    return path


def metrics_for(ranked_ids, qrels):
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    ranks = [index + 1 for index, doc_id in enumerate(ranked_ids) if doc_id in relevant]
    recall10 = len([rank for rank in ranks if rank <= 10]) / len(relevant)
    recall100 = len([rank for rank in ranks if rank <= 100]) / len(relevant)
    mrr10 = 1 / ranks[0] if ranks and ranks[0] <= 10 else 0
    dcg = sum(qrels[doc_id] / math.log2(index + 2) for index, doc_id in enumerate(ranked_ids[:10]) if doc_id in qrels)
    ideal = sum(score / math.log2(index + 2) for index, score in enumerate(sorted(qrels.values(), reverse=True)[:10]))
    return {"recallAt10": recall10, "recallAt100": recall100, "mrrAt10": mrr10, "ndcgAt10": dcg / ideal if ideal else 0}


def aggregate(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/embedding-model-benchmark")
    parser.add_argument("--output", default="general-vectorized")
    args = parser.parse_args()
    documents, document_texts, queries, query_texts, dataset_meta = load_data(args.data_dir)
    cache_dir = os.path.join(args.data_dir, "cache")
    results = []
    for config_id, (provider, model, document_task, query_task, dimensions) in CONFIGS.items():
        doc_path = cache_for(cache_dir, provider, model, dimensions, document_task, document_texts)
        query_path = cache_for(cache_dir, provider, model, dimensions, query_task, query_texts)
        doc_vectors = np.memmap(doc_path, dtype=np.float32, mode="r", shape=(len(document_texts), dimensions))
        query_vectors = np.memmap(query_path, dtype=np.float32, mode="r", shape=(len(query_texts), dimensions))
        started = time.perf_counter()
        groups = {}
        failures = []
        for dataset, meta in dataset_meta.items():
            ds, de = meta["document_slice"]
            qs, qe = meta["query_slice"]
            corpus_ids = [row[0] for row in documents[ds:de]]
            corpus_id_order = np.argsort(np.argsort(np.asarray(corpus_ids)))
            corpus_vectors = doc_vectors[[row[3] for row in documents[ds:de]]]
            rows = []
            block_size = 64
            for start in range(qs, qe, block_size):
                end = min(start + block_size, qe)
                query_block = query_vectors[[row[4] for row in queries[start:end]]]
                scores = np.matmul(query_block, corpus_vectors.T)
                top_count = min(100, scores.shape[1])
                for local_index in range(scores.shape[0]):
                    ordered = np.lexsort((corpus_id_order, -scores[local_index]))[:top_count]
                    ranked_ids = [corpus_ids[index] for index in ordered]
                    query_id, query_text, _, qrels, _ = queries[start + local_index]
                    metrics = metrics_for(ranked_ids, qrels)
                    rows.append(metrics)
                    if metrics["recallAt10"] == 0:
                        failures.append({"dataset": dataset, "queryId": query_id, "query": query_text, "topIds": ranked_ids[:10], "relevantIds": list(qrels)})
            groups[dataset] = {"queries": len(rows), **aggregate(rows)}
        elapsed = time.perf_counter() - started
        overall_rows = []
        for dataset, group in groups.items():
            overall_rows.extend([{key: group[key] for key in ["recallAt10", "recallAt100", "mrrAt10", "ndcgAt10"]}] * group["queries"])
        results.append({"config": config_id, "rankingSeconds": elapsed, "summary": {"overall": {"queries": len(overall_rows), **aggregate(overall_rows)}, **groups}, "failures": failures[:25]})
        print(config_id, results[-1]["summary"]["overall"], f"ranking={elapsed:.2f}s")
    report = {"method": "blocked NumPy float32 exact-cosine ranking", "datasets": DATASETS, "documents": len(documents), "queries": len(queries), "results": results}
    output_dir = os.path.join(args.data_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{args.output}.json")
    with open(json_path, "w") as file:
        json.dump(report, file, indent=2)
    with open(os.path.join(output_dir, f"{args.output}.md"), "w") as file:
        file.write("# General retrieval benchmark\n\n")
        file.write("| Configuration | nDCG@10 | R@10 | R@100 | MRR@10 | Ranking seconds |\n|---|---:|---:|---:|---:|---:|\n")
        for result in results:
            metric = result["summary"]["overall"]
            file.write(f"| {result['config']} | {metric['ndcgAt10']:.3f} | {metric['recallAt10']:.3f} | {metric['recallAt100']:.3f} | {metric['mrrAt10']:.3f} | {result['rankingSeconds']:.2f} |\n")
    print(json_path)


if __name__ == "__main__":
    main()
