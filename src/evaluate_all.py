
import csv
import json
from pathlib import Path

import numpy as np

from tfidf_search import TFIDFRetriever
from bm25_search import BM25Retriever, tokenize
from sbert_search import SBERTRetriever
from hybrid_search import minmax_normalize


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"
RESULTS_DIR = BASE_DIR / "results"

K = 10


def load_queries():
    queries = {}

    with open(
        DATA_DIR / "queries.jsonl",
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]

    return queries


def load_qrels(path):
    qrels = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])

            qrels.setdefault(qid, {})[did] = score

    return qrels


def calculate_metrics(ranked_ids, relevant, k=10):

    ranked_ids = ranked_ids[:k]

    labels = [
        max(0, relevant.get(did, 0))
        for did in ranked_ids
    ]

    # Recall@10
    relevant_set = {
        did for did, rel in relevant.items()
        if rel > 0
    }

    hits = sum(
        did in relevant_set for did in ranked_ids
    )

    recall = (
        hits / len(relevant_set)
        if relevant_set else 0.0
    )

    # MRR@10
    mrr = 0.0

    for rank, did in enumerate(ranked_ids, 1):
        if did in relevant_set:
            mrr = 1.0 / rank
            break

    # nDCG@10
    discounts = np.log2(
        np.arange(2, len(labels) + 2)
    )

    gains = np.power(2.0, labels) - 1
    dcg = np.sum(gains / discounts)

    ideal = sorted(
        [max(0, x) for x in relevant.values()],
        reverse=True
    )[:k]

    ideal_discounts = np.log2(
        np.arange(2, len(ideal) + 2)
    )

    idcg = np.sum(
        (np.power(2.0, ideal) - 1)
        / ideal_discounts
    )

    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    return {
        "ndcg": ndcg,
        "recall": recall,
        "mrr": mrr
    }


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 读取已在训练集选择的最佳 alpha
    with open(
        RESULTS_DIR / "hybrid_config.json",
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    alpha = float(config["alpha"])

    print("Fixed Hybrid alpha:", alpha)

    queries = load_queries()

    # 只读取测试集标注，不做任何调参
    qrels = load_qrels(
        DATA_DIR / "qrels" / "test.tsv"
    )

    test_ids = [
        qid for qid in qrels
        if qid in queries
    ]

    print("Test queries:", len(test_ids))

    tfidf = TFIDFRetriever()
    bm25 = BM25Retriever()
    sbert = SBERTRetriever()

    # 使用 SBERT 文档顺序统一所有评分
    doc_ids = sbert.doc_ids

    tfidf_index = {
        did: i for i, did in enumerate(tfidf.doc_ids)
    }

    bm25_index = {
        did: i for i, did in enumerate(bm25.doc_ids)
    }

    tfidf_order = np.array([
        tfidf_index[did] for did in doc_ids
    ])

    bm25_order = np.array([
        bm25_index[did] for did in doc_ids
    ])

    if (
        len(set(doc_ids)) != len(doc_ids)
        or set(tfidf.doc_ids) != set(doc_ids)
        or set(bm25.doc_ids) != set(doc_ids)
    ):
        raise ValueError("Corpus document IDs do not match.")

    # 一次性编码全部测试查询
    test_texts = [queries[qid] for qid in test_ids]

    query_embeddings = sbert.model.encode(
        test_texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    metric_rows = []
    ranking_rows = []

    for i, qid in enumerate(test_ids):

        query = queries[qid]

        # TF-IDF 分数
        qvector = tfidf.vectorizer.transform([query])
        tfidf_scores = (
            qvector @ tfidf.doc_vectors.T
        ).toarray().ravel()[tfidf_order]

        # BM25 分数
        bm25_scores = bm25.bm25.get_scores(
            tokenize(query)
        )[bm25_order]

        # Sentence-BERT 分数
        sbert_scores = (
            sbert.embeddings @ query_embeddings[i]
        )

        # Hybrid 分数
        hybrid_scores = (
            alpha * minmax_normalize(bm25_scores)
            + (1 - alpha) * minmax_normalize(sbert_scores)
        )

        models = {
            "TF-IDF": tfidf_scores,
            "BM25": bm25_scores,
            "SBERT": sbert_scores,
            "Hybrid": hybrid_scores
        }

        for name, scores in models.items():

            indices = np.argsort(
                -scores, kind="stable"
            )[:K]

            ranked_ids = [
                doc_ids[j] for j in indices
            ]

            metrics = calculate_metrics(
                ranked_ids, qrels[qid], K
            )

            metric_rows.append({
                "query_id": qid,
                "model": name,
                "ndcg_at_10": metrics["ndcg"],
                "recall_at_10": metrics["recall"],
                "mrr_at_10": metrics["mrr"]
            })

            for rank, did in enumerate(ranked_ids, 1):
                ranking_rows.append({
                    "query_id": qid,
                    "model": name,
                    "rank": rank,
                    "doc_id": did,
                    "score": float(scores[indices[rank - 1]]),
                    "relevance": qrels[qid].get(did, 0)
                })

    # 保存逐查询评价结果
    with open(
        RESULTS_DIR / "test_metrics.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=metric_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    # 保存 Top 10 排名，供错误分析使用
    with open(
        RESULTS_DIR / "test_rankings.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=ranking_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    print("\nFINAL TEST RESULTS")
    print("-" * 55)

    for name in ["TF-IDF", "BM25", "SBERT", "Hybrid"]:

        rows = [
            r for r in metric_rows
            if r["model"] == name
        ]

        ndcg = np.mean([
            r["ndcg_at_10"] for r in rows
        ])
        recall = np.mean([
            r["recall_at_10"] for r in rows
        ])
        mrr = np.mean([
            r["mrr_at_10"] for r in rows
        ])

        print(
            f"{name:8s} | "
            f"nDCG@10: {ndcg:.4f} | "
            f"Recall@10: {recall:.4f} | "
            f"MRR@10: {mrr:.4f}"
        )

    print("\nSaved test_metrics.csv and test_rankings.csv")


if __name__ == "__main__":
    main()
