
import csv
import json
from pathlib import Path

import numpy as np

from hybrid_search import HybridRetriever, minmax_normalize
from bm25_search import tokenize


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"
RESULTS_DIR = BASE_DIR / "results"

ALPHAS = np.linspace(0, 1, 11)
K = 10


def load_queries():
    """读取所有查询，建立 query_id -> text 映射"""
    queries = {}

    with open(
        DATA_DIR / "queries.jsonl",
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            item = json.loads(line)
            queries[str(item["_id"])] = item["text"]

    return queries


def load_train_qrels():
    """读取训练集人工相关性标注"""
    qrels = {}

    with open(
        DATA_DIR / "qrels" / "train.tsv",
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            relevance = int(row["score"])

            if query_id not in qrels:
                qrels[query_id] = {}

            qrels[query_id][doc_id] = relevance

    return qrels


def ndcg_at_k(ranked_ids, relevant_docs, k=10):
    """计算单条 Query 的 nDCG@K"""

    relevance = np.array([
        relevant_docs.get(doc_id, 0)
        for doc_id in ranked_ids[:k]
    ], dtype=float)

    discounts = np.log2(np.arange(2, k + 2))

    gains = np.power(2.0, relevance) - 1.0
    dcg = np.sum(gains / discounts[:len(gains)])

    # 理想情况下将相关度最高的论文排在最前
    ideal_relevance = sorted(
        relevant_docs.values(),
        reverse=True
    )[:k]

    ideal_gains = np.power(
        2.0, np.array(ideal_relevance, dtype=float)
    ) - 1.0

    idcg = np.sum(
        ideal_gains / discounts[:len(ideal_gains)]
    )

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(alpha=0.5)

    queries = load_queries()
    train_qrels = load_train_qrels()

    # 只使用有训练集标注的查询
    train_ids = [
        qid for qid in train_qrels
        if qid in queries
        and any(v > 0 for v in train_qrels[qid].values())
    ]

    if not train_ids:
        raise ValueError("No usable training queries found.")

    print("Training queries:", len(train_ids))

    # 批量编码训练集 Query，避免反复调用模型
    query_texts = [queries[qid] for qid in train_ids]

    query_embeddings = retriever.sbert_retriever.model.encode(
        query_texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 保存每个 alpha 的逐查询 nDCG
    scores_by_alpha = {
        round(float(a), 1): []
        for a in ALPHAS
    }

    print("Evaluating alpha values...")

    for i, query_id in enumerate(train_ids):
        query = queries[query_id]

        # 1. BM25 全库打分
        bm25_scores = retriever.bm25_retriever.bm25.get_scores(
            tokenize(query)
        )
        bm25_scores = bm25_scores[retriever.bm25_order]

        # 2. SBERT 全库打分
        sbert_scores = (
            retriever.sbert_retriever.embeddings
            @ query_embeddings[i]
        )

        # 3. 与 Hybrid 检索使用相同的归一化
        bm25_norm = minmax_normalize(bm25_scores)
        sbert_norm = minmax_normalize(sbert_scores)

        # 4. 尝试不同 alpha
        for alpha in ALPHAS:
            alpha = round(float(alpha), 1)

            hybrid_scores = (
                alpha * bm25_norm
                + (1 - alpha) * sbert_norm
            )

            ranked_indices = np.argsort(
                -hybrid_scores,
                kind="stable"
            )[:K]

            ranked_ids = [
                retriever.doc_ids[j]
                for j in ranked_indices
            ]

            score = ndcg_at_k(
                ranked_ids,
                train_qrels[query_id],
                k=K
            )

            scores_by_alpha[alpha].append(score)

    # 5. 计算每个 alpha 的平均 nDCG@10
    results = []

    for alpha, scores in scores_by_alpha.items():
        mean_ndcg = float(np.mean(scores))
        results.append((alpha, mean_ndcg))

        print(
            f"Alpha = {alpha:.1f} | "
            f"Mean nDCG@10 = {mean_ndcg:.4f}"
        )

    # 分数相同时优先选择较小 alpha
    best_alpha, best_score = sorted(
        results,
        key=lambda x: (-x[1], x[0])
    )[0]

    print("\nBest alpha:", best_alpha)
    print("Best training nDCG@10:", round(best_score, 4))

    # 6. 保存调参结果
    with open(
        RESULTS_DIR / "alpha_tuning.csv",
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "mean_ndcg_at_10"])
        writer.writerows(results)

    # 7. 保存选出的参数
    config = {
        "alpha": best_alpha,
        "selection_metric": "train_ndcg_at_10",
        "training_queries": len(train_ids),
        "model": "all-MiniLM-L6-v2"
    }

    with open(
        RESULTS_DIR / "hybrid_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
