
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"


def load_jsonl(file_path):
    """逐行读取 JSONL 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TFIDFRetriever:
    def __init__(self):
        # 读取完整文档库
        self.corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

        # 保留文档 ID，供后续评价使用
        self.doc_ids = [str(doc["_id"]) for doc in self.corpus]

        # 将标题和摘要组合成检索文本
        self.documents = [
            (doc.get("title") or "") + " " +
            (doc.get("text") or "")
            for doc in self.corpus
        ]

        # 定义 TF-IDF 转换器
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            norm="l2",
            use_idf=True,
            smooth_idf=True
        )

        # 构建所有论文的 TF-IDF 向量
        self.doc_vectors = self.vectorizer.fit_transform(
            self.documents
        )

        print("TF-IDF index created.")
        print("Documents:", len(self.documents))
        print("Matrix shape:", self.doc_vectors.shape)

    def search(self, query, top_k=5):
        """检索与查询最相关的前 K 篇论文"""

        # 1. 将用户查询转换为 TF-IDF 向量
        query_vector = self.vectorizer.transform([query])

        # 2. 计算查询与每篇论文的余弦相似度
        scores = cosine_similarity(
            query_vector,
            self.doc_vectors
        ).ravel()

        # 3. 按相似度从高到低排列
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        # 4. 整理检索结果
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
    retriever = TFIDFRetriever()

    # 使用 SciFact 中的真实查询
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    query = "Vitamin D reduces the risk of fractures"

    print("\nQuery:", query)

    results = retriever.search(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Document ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Abstract:", result["text"][:250])
