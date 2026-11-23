
import json
from pathlib import Path

import pandas as pd

from bm25_search import tokenize


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"
RESULTS_DIR = BASE_DIR / "results"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def jaccard(text_a, text_b):
    """计算两个文本的词集合 Jaccard 相似度"""
    a = set(tokenize(text_a))
    b = set(tokenize(text_b))

    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)


def main():
    # 1. 加载 SciFact
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    corpus_map = {
        str(doc["_id"]): (
            (doc.get("title") or "") + " " +
            (doc.get("text") or "")
        )
        for doc in corpus
    }

    query_map = {
        str(q["_id"]): q["text"]
        for q in queries
    }

    # 2. 加载测试集相关性标注
    qrels = pd.read_csv(
        DATA_DIR / "qrels" / "test.tsv",
        sep="\t"
    )

    # 3. 只使用已标注的相关论文
    qrels = qrels[qrels["score"] > 0].copy()

    overlap_rows = []

    for query_id, group in qrels.groupby("query-id"):
        query_id = str(query_id)

        if query_id not in query_map:
            continue

        query = query_map[query_id]
        overlaps = []

        for doc_id in group["corpus-id"]:
            doc_id = str(doc_id)

            if doc_id in corpus_map:
                overlaps.append(
                    jaccard(query, corpus_map[doc_id])
                )

        if overlaps:
            overlap_rows.append({
                "query_id": query_id,
                "jaccard": max(overlaps)
            })

    overlap_df = pd.DataFrame(overlap_rows)

    # 4. 按三分位数划分低、中、高重叠
    q1 = overlap_df["jaccard"].quantile(1 / 3)
    q2 = overlap_df["jaccard"].quantile(2 / 3)

    print("Jaccard lower boundary:", round(q1, 4))
    print("Jaccard upper boundary:", round(q2, 4))

    if q1 == q2:
        raise ValueError(
            "Quantile boundaries are equal. "
            "Review the Jaccard distribution."
        )

    def assign_group(value):
        if value <= q1:
            return "Low"
        elif value <= q2:
            return "Medium"
        return "High"

    overlap_df["overlap_group"] = (
        overlap_df["jaccard"].apply(assign_group)
    )

    # 5. 加载之前保存的测试集指标
    metrics = pd.read_csv(
        RESULTS_DIR / "test_metrics.csv",
        dtype={"query_id": str}
    )

    merged = metrics.merge(
        overlap_df,
        on="query_id",
        how="inner",
        validate="many_to_one"
    )

    # 6. 分组计算平均 nDCG@10
    summary = (
        merged.groupby(
            ["overlap_group", "model"]
        )[["ndcg_at_10", "recall_at_10", "mrr_at_10"]]
        .mean()
        .reset_index()
    )

    print("\nGROUPED RESULTS")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nQueries in each group:")
    print(
        overlap_df["overlap_group"].value_counts()
    )

    # 7. 保存结果
    overlap_df.to_csv(
        RESULTS_DIR / "query_overlap.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "overlap_results.csv",
        index=False
    )

    print("\nOverlap analysis completed.")


if __name__ == "__main__":
    main()
