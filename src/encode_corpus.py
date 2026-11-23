
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
    """读取 SciFact JSONL 数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    # 创建保存向量的目录
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载全部论文
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

    # 2. 保存文档 ID
    doc_ids = [
        str(doc["_id"]) for doc in corpus
    ]

    # 3. 拼接标题和摘要
    documents = [
        (doc.get("title") or "") + " " +
        (doc.get("text") or "")
        for doc in corpus
    ]

    print("Documents:", len(documents))

    # 4. 加载 Sentence-BERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    # 5. 批量编码论文
    embeddings = model.encode(
        documents,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 6. 保存 Embedding
    np.save(
        MODEL_DIR / "scifact_embeddings.npy",
        embeddings
    )

    # 7. 保存论文 ID
    with open(
        MODEL_DIR / "scifact_doc_ids.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(doc_ids, f)

    print("\nEncoding completed.")
    print("Embedding shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
