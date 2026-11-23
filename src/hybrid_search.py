
import numpy as np

from bm25_search import BM25Retriever, tokenize
from sbert_search import SBERTRetriever


def minmax_normalize(scores):
    """
    Min-Max 归一化
    将分数缩放到 [0, 1]
    """
    scores = np.asarray(scores, dtype=np.float64)

    min_score = scores.min()
    max_score = scores.max()

    # 防止最大值等于最小值时除以零
    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


class HybridRetriever:

    def __init__(self, alpha=0.5):

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha

        # 1. 加载两个已经完成的检索模型
        self.bm25_retriever = BM25Retriever()
        self.sbert_retriever = SBERTRetriever()

        # 2. 以 SBERT 保存的文档顺序为标准
        self.doc_ids = self.sbert_retriever.doc_ids

        # 3. 建立 BM25 文档 ID 到下标的映射
        bm25_id_to_index = {
            doc_id: idx
            for idx, doc_id in enumerate(
                self.bm25_retriever.doc_ids
            )
        }

        # 4. 计算 BM25 对应的文档排列顺序
        self.bm25_order = np.array([
            bm25_id_to_index[doc_id]
            for doc_id in self.doc_ids
        ])

        print("Hybrid retriever initialized.")
        print("Documents:", len(self.doc_ids))
        print("Alpha:", self.alpha)

    def search(self, query, top_k=5):

        # 1. BM25 查询分词
        query_tokens = tokenize(query)

        # 2. 计算全部文档的 BM25 原始分数
        bm25_scores = self.bm25_retriever.bm25.get_scores(
            query_tokens
        )

        # 按 SBERT 的文档 ID 顺序重新排列
        bm25_scores = bm25_scores[self.bm25_order]

        # 3. 对 Query 进行 Sentence-BERT 编码
        query_embedding = self.sbert_retriever.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # 4. 计算全部文档的语义相似度
        sbert_scores = (
            self.sbert_retriever.embeddings @ query_embedding
        )

        # 5. 分别归一化两组分数
        bm25_norm = minmax_normalize(bm25_scores)
        sbert_norm = minmax_normalize(sbert_scores)

        # 6. 计算 Hybrid 融合分数
        hybrid_scores = (
            self.alpha * bm25_norm
            + (1 - self.alpha) * sbert_norm
        )

        # 7. 根据融合分数排序
        ranked_indices = np.argsort(
            -hybrid_scores,
            kind="stable"
        )[:top_k]

        # 8. 整理结果
        results = []

        for idx in ranked_indices:

            doc_id = self.doc_ids[idx]

            doc = self.sbert_retriever.corpus_by_id[
                doc_id
            ]

            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": float(hybrid_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "sbert_score": float(sbert_scores[idx]),
                "bm25_norm": float(bm25_norm[idx]),
                "sbert_norm": float(sbert_norm[idx])
            })

        return results


if __name__ == "__main__":

    # 首次测试使用相同的融合权重
    retriever = HybridRetriever(alpha=0.5)

    query = "Vitamin D reduces the risk of fractures"

    print("\nQuery:", query)

    results = retriever.search(query, top_k=5)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print("Document ID:", result["doc_id"])
        print("Title:", result["title"])

        print(
            "BM25 normalized:",
            round(result["bm25_norm"], 4)
        )

        print(
            "SBERT normalized:",
            round(result["sbert_norm"], 4)
        )

        print(
            "Hybrid score:",
            round(result["score"], 4)
        )

        print("Abstract:", result["text"][:250])
