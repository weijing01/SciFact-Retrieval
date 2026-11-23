
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


# 项目目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"


def load_jsonl(file_path):
    """读取 JSONL 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    """
    英文分词：
    1. 转换为小写
    2. 提取单词或数字
    """
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self):

        # 1. 加载科学论文
        self.corpus = load_jsonl(
            DATA_DIR / "corpus.jsonl"
        )

        # 2. 保存文档 ID
        self.doc_ids = [
            str(doc["_id"]) for doc in self.corpus
        ]

        # 3. 拼接标题和摘要
        self.documents = [
            (doc.get("title") or "") + " " +
            (doc.get("text") or "")
            for doc in self.corpus
        ]

        print("Documents loaded:", len(self.documents))

        # 4. 对每篇论文分词
        self.tokenized_corpus = [
            tokenize(doc) for doc in self.documents
        ]

        # 5. 建立 BM25 索引
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=1.5,
            b=0.75
        )

        print("BM25 index created.")
        print("Average document length:", round(self.bm25.avgdl, 2))

    def search(self, query, top_k=5):
        """返回 BM25 排名前 K 的论文"""

        # 1. 查询分词
        query_tokens = tokenize(query)

        # 2. 计算每篇论文的 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 3. 从高到低排序
        ranked_indices = np.argsort(
            -scores, kind="stable"
        )[:top_k]

        # 4. 保存检索结果
        results = []

        for idx in ranked_indices:
            doc = self.corpus[idx]

            results.append({
                "doc_id": self.doc_ids[idx],
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": float(scores[idx])
            })

        return results


if __name__ == "__main__":

    retriever = BM25Retriever()

    # 与 TF-IDF 使用相同的测试查询
    query = "Vitamin D reduces the risk of fractures"

    print("\nQuery:", query)

    results = retriever.search(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Document ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Abstract:", result["text"][:250])
