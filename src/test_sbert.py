
import torch
from sentence_transformers import SentenceTransformer

# 1. 选择模型
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2. 使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 3. 下载并加载预训练模型
model = SentenceTransformer(MODEL_NAME, device=device)

print("Model loaded successfully.")
print("Embedding dimension:", model.get_sentence_embedding_dimension())
print("Max sequence length:", model.max_seq_length)

# 4. 准备三个测试句子
sentences = [
    "A heart attack requires medical treatment.",
    "Myocardial infarction requires clinical care.",
    "Vitamin D is important for bone health."
]

# 5. 将句子转换为语义向量
embeddings = model.encode(
    sentences,
    batch_size=3,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

# 6. 计算句子之间的余弦相似度
similarities = embeddings @ embeddings.T

print("Similarity matrix:")
print(similarities.round(4))
