
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"
MODEL_DIR = BASE_DIR / "models"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SBERTRetriever:

    def __init__(self):

        # 1. 加载 SciFact 文档
        corpus = load_jsonl(
            DATA_DIR / "corpus.jsonl"
        )

        self.corpus_by_id = {
            str(doc["_id"]): doc
            for doc in corpus
        }

        # 2. 加载预计算的论文 Embedding
        self.embeddings = np.load(
            MODEL_DIR / "scifact_embeddings.npy"
        )

        # 3. 加载 Embedding 对应的文档 ID
        with open(
            MODEL_DIR / "scifact_doc_ids.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.doc_ids = json.load(f)

        # 确保文档数量与向量数量一致
        assert len(self.doc_ids) == len(self.embeddings)

        # 4. 加载 Sentence-BERT 模型
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device
        )

        print("Sentence-BERT index loaded.")
        print("Embedding shape:", self.embeddings.shape)

    def search(self, query, top_k=5):

        # 1. 编码用户输入的 Query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # 2. 计算相似度
        scores = self.embeddings @ query_embedding

        # 3. 根据相似度从大到小排序
        ranked_indices = np.argsort(
            -scores,
            kind="stable"
        )[:top_k]

        # 4. 整理检索结果
        results = []

        for idx in ranked_indices:
            doc_id = self.doc_ids[idx]
            doc = self.corpus_by_id[doc_id]

            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": float(scores[idx])
            })

        return results


if __name__ == "__main__":

    retriever = SBERTRetriever()

    # 与 TF-IDF、BM25 使用相同的查询
    query = "Vitamin D reduces the risk of fractures"

    print("\nQuery:", query)

    results = retriever.search(query, top_k=5)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print("Document ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Abstract:", result["text"][:250])
