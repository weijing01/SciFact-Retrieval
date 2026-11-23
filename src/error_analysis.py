
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"
RESULTS_DIR = BASE_DIR / "results"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

    query_map = {
        str(q["_id"]): q["text"]
        for q in queries
    }

    doc_map = {
        str(d["_id"]): d
        for d in corpus
    }

    rankings = pd.read_csv(
        RESULTS_DIR / "test_rankings.csv",
        dtype={"query_id": str, "doc_id": str}
    )

    metrics = pd.read_csv(
        RESULTS_DIR / "test_metrics.csv",
        dtype={"query_id": str}
    )

    # 整理 BM25 与 SBERT 的逐查询指标
    comparison = metrics[
        metrics["model"].isin(["BM25", "SBERT"])
    ].pivot(
        index="query_id",
        columns="model",
        values="ndcg_at_10"
    ).dropna()

    comparison["difference"] = (
        comparison["SBERT"] - comparison["BM25"]
    )

    print("Queries compared:", len(comparison))

    # 保存所有查询的效果差异
    comparison.sort_values(
        "difference", ascending=False
    ).to_csv(
        RESULTS_DIR / "model_differences.csv"
    )

    # 优先选择一方 nDCG 为 0、
    # 另一方大于 0 的清晰案例
    sbert_cases = comparison[
        (comparison["SBERT"] > 0) &
        (comparison["BM25"] == 0)
    ].sort_values("difference", ascending=False)

    bm25_cases = comparison[
        (comparison["BM25"] > 0) &
        (comparison["SBERT"] == 0)
    ].sort_values("difference")

    # 如果没有完全失败的案例，
    # 则选取差距最大的正向案例
    if sbert_cases.empty:
        sbert_cases = comparison[
            comparison["difference"] > 0
        ].sort_values("difference", ascending=False)

    if bm25_cases.empty:
        bm25_cases = comparison[
            comparison["difference"] < 0
        ].sort_values("difference")

    def show_case(title, candidates):
        print("\n" + "=" * 70)
        print(title)

        if candidates.empty:
            print("No matching case found.")
            return

        qid = str(candidates.index[0])
        row = candidates.iloc[0]

        print("Query ID:", qid)
        print("Query:", query_map[qid])
        print("BM25 nDCG@10:", round(row["BM25"], 4))
        print("SBERT nDCG@10:", round(row["SBERT"], 4))

        for model in ["BM25", "SBERT"]:
            subset = rankings[
                (rankings["query_id"] == qid) &
                (rankings["model"] == model)
            ].sort_values("rank")

            print(f"\n{model} Top 10:")

            for _, item in subset.iterrows():
                did = str(item["doc_id"])
                doc = doc_map[did]

                print(
                    f"Rank {int(item['rank'])}: "
                    f"ID={did} "
                    f"Relevant={int(item['relevance']) > 0}"
                )
                print("Title:", doc.get("title", ""))

    show_case(
        "Case A: Sentence-BERT performs better",
        sbert_cases
    )

    show_case(
        "Case B: BM25 performs better",
        bm25_cases
    )

    print("\nSaved model_differences.csv")


if __name__ == "__main__":
    main()
