
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "scifact"


# 读取 JSONL 文件
def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


# 1. 读取科学论文
corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

# 2. 读取科学查询
queries = load_jsonl(DATA_DIR / "queries.jsonl")

# 3. 读取相关性标注
train_qrels = pd.read_csv(
    DATA_DIR / "qrels" / "train.tsv",
    sep="\t"
)

test_qrels = pd.read_csv(
    DATA_DIR / "qrels" / "test.tsv",
    sep="\t"
)

# 4. 查看数据规模
print("Number of documents:", len(corpus))
print("Number of queries:", len(queries))

print("Train relevance records:", len(train_qrels))
print("Test relevance records:", len(test_qrels))

# 5. 查看真实文档与查询
print("\nSample Document:")
print("ID:", corpus[0]["_id"])
print("Title:", corpus[0]["title"])
print("Text:", corpus[0]["text"][:300])

print("\nSample Query:")
print(queries[0])

print("\nQrels columns:")
print(train_qrels.columns.tolist())
