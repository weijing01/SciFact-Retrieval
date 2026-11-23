
import json
from pathlib import Path

from bm25_search import tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "scifact"

cases = [
    {
        "query_id": "1088",
        "relevant_doc_id": "37549932"
    },
    {
        "query_id": "1024",
        "relevant_doc_id": "5373138"
    }
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return {
            str(item["_id"]): item
            for line in f
            if (item := json.loads(line))
        }


queries = load_jsonl(DATA_DIR / "queries.jsonl")
corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

for case in cases:
    qid = case["query_id"]
    did = case["relevant_doc_id"]

    query = queries[qid]["text"]
    doc = corpus[did]

    title = doc.get("title", "")
    abstract = doc.get("text", "")
    document = title + " " + abstract

    query_words = set(tokenize(query))
    document_words = set(tokenize(document))

    common_words = query_words & document_words
    union_words = query_words | document_words

    jaccard = (
        len(common_words) / len(union_words)
        if union_words else 0
    )

    print("\n" + "=" * 70)
    print("QUERY ID:", qid)
    print("QUERY:", query)
    print("DOCUMENT ID:", did)
    print("TITLE:", title)
    print("\nFULL ABSTRACT:\n", abstract)

    print("\nCOMMON WORDS:", sorted(common_words))
    print("JACCARD:", round(jaccard, 4))
