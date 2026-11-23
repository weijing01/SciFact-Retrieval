
import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st


# ==================================================
# 1. 项目路径
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from tfidf_search import TFIDFRetriever
from bm25_search import BM25Retriever, tokenize
from sbert_search import SBERTRetriever
from hybrid_search import minmax_normalize


# ==================================================
# 2. Streamlit 页面设置
# ==================================================

st.set_page_config(
    page_title="Scientific Evidence Retrieval",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. 中英双语词典
# ==================================================

TEXT = {
    "中文": {
        "title": "🔬 科学文献证据检索系统",
        "subtitle": "基于 SciFact 数据集的关键词与语义检索对比",
        "description": (
            "本系统实现 TF-IDF、BM25、Sentence-BERT "
            "和 Hybrid 四种科学文献检索方法。"
        ),
        "settings": "检索设置",
        "dataset": "数据集",
        "documents": "论文数量",
        "alpha": "混合检索权重 α",
        "input": "请输入英文科学观点或问题",
        "placeholder": (
            "在这里输入任意英文科学查询...\n"
            "例如：Does exercise improve cardiovascular health?"
        ),
        "top_k": "每种方法返回的论文数量",
        "search": "🔍 开始检索",
        "loading": "正在检索科学论文...",
        "success": "检索完成！",
        "warning": "请输入科学观点或问题。",
        "results": "检索结果",
        "doc_id": "论文 ID",
        "score": "相关性分数",
        "abstract": "查看完整摘要",
        "no_abstract": "暂无摘要",
        "tfidf": "TF-IDF 关键词检索",
        "bm25": "BM25 关键词检索",
        "sbert": "Sentence-BERT 语义检索",
        "hybrid": "Hybrid 混合检索",
        "help": "建议使用英文查询，以保持与 SciFact 实验设置一致。",
        "footer": (
            "本系统用于检索可能相关的科学论文。"
            "检索分数不代表科学观点的真实性，"
            "也不能代替对论文证据的人工判断。"
        ),
        "method_info": "检索方法说明",
        "tfidf_info": "使用 TF-IDF 词汇权重和余弦相似度进行排序。",
        "bm25_info": "考虑词频饱和与文档长度归一化的传统关键词检索方法。",
        "sbert_info": "使用预训练 Transformer 模型生成 384 维语义向量。",
        "hybrid_info": "对 BM25 与 SBERT 分数归一化后加权融合。",
        "sample": "推荐示例查询",
        "sample_help": "点击下面的示例，即可将其填入搜索框；也可以直接输入自己的问题。",
        "example_1": "维生素 D 与骨折风险",
        "example_2": "BCL-2 与肿瘤维持",
        "example_3": "CTCF 与癌症突变",
        "fixed_alpha": "α 在训练集上选择，测试集评价时保持固定。",
        "error": "模型加载或检索失败",
        "custom_search": "支持任意英文科学查询",
        "clear": "清空输入",
        "no_results": "暂无检索结果"
    },

    "English": {
        "title": "🔬 Scientific Evidence Retrieval System",
        "subtitle": "Keyword vs. Semantic Retrieval on SciFact",
        "description": (
            "This system compares TF-IDF, BM25, Sentence-BERT, "
            "and Hybrid retrieval for scientific evidence search."
        ),
        "settings": "Retrieval Settings",
        "dataset": "Dataset",
        "documents": "Documents",
        "alpha": "Hybrid Weight α",
        "input": "Enter a scientific claim or question in English",
        "placeholder": (
            "Enter any scientific query in English...\n"
            "Example: Does exercise improve cardiovascular health?"
        ),
        "top_k": "Number of papers per method",
        "search": "🔍 Search Papers",
        "loading": "Searching scientific papers...",
        "success": "Retrieval completed!",
        "warning": "Please enter a scientific claim or question.",
        "results": "Retrieval Results",
        "doc_id": "Document ID",
        "score": "Relevance Score",
        "abstract": "View Full Abstract",
        "no_abstract": "No abstract available",
        "tfidf": "TF-IDF Keyword Retrieval",
        "bm25": "BM25 Keyword Retrieval",
        "sbert": "Sentence-BERT Semantic Retrieval",
        "hybrid": "Hybrid Retrieval",
        "help": "English queries are recommended for consistency with the SciFact experiments.",
        "footer": (
            "This system retrieves potentially relevant scientific papers. "
            "Retrieval scores do not indicate whether a scientific claim "
            "is true and do not replace manual evidence assessment."
        ),
        "method_info": "Retrieval Methods",
        "tfidf_info": "Ranks documents using TF-IDF weights and cosine similarity.",
        "bm25_info": "A lexical retrieval method with term-frequency saturation and document-length normalization.",
        "sbert_info": "Uses pretrained Transformer embeddings with 384 dimensions.",
        "hybrid_info": "Combines normalized BM25 and Sentence-BERT scores.",
        "sample": "Recommended Example Queries",
        "sample_help": "Click an example to fill the search box, or enter your own query.",
        "example_1": "Vitamin D and fracture risk",
        "example_2": "BCL-2 and tumor maintenance",
        "example_3": "CTCF and cancer mutations",
        "fixed_alpha": "Alpha was selected using training queries and fixed for test evaluation.",
        "error": "Model loading or retrieval failed",
        "custom_search": "Search any scientific claim in English",
        "clear": "Clear Input",
        "no_results": "No results found"
    }
}


# ==================================================
# 4. 语言选择
# ==================================================

language = st.sidebar.selectbox(
    "🌐 Language / 语言",
    ["中文", "English"],
    index=0,
    key="language_select"
)

t = TEXT[language]


# ==================================================
# 5. 检索模型加载与缓存
# ==================================================

@st.cache_resource(show_spinner="Loading retrieval models...")
def load_retrievers():

    tfidf = TFIDFRetriever()
    bm25 = BM25Retriever()
    sbert = SBERTRetriever()

    doc_ids = sbert.doc_ids

    # 检查文档 ID 是否一致
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Duplicate document IDs found.")

    if set(tfidf.doc_ids) != set(doc_ids):
        raise ValueError("TF-IDF document IDs do not match.")

    if set(bm25.doc_ids) != set(doc_ids):
        raise ValueError("BM25 document IDs do not match.")

    tfidf_map = {
        did: i for i, did in enumerate(tfidf.doc_ids)
    }

    bm25_map = {
        did: i for i, did in enumerate(bm25.doc_ids)
    }

    tfidf_order = np.array([
        tfidf_map[did] for did in doc_ids
    ])

    bm25_order = np.array([
        bm25_map[did] for did in doc_ids
    ])

    # 从实验结果读取最佳 alpha
    config_path = (
        BASE_DIR / "results" / "hybrid_config.json"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    alpha = float(config["alpha"])

    if not 0 <= alpha <= 1:
        raise ValueError("Invalid Hybrid alpha.")

    return {
        "tfidf": tfidf,
        "bm25": bm25,
        "sbert": sbert,
        "doc_ids": doc_ids,
        "tfidf_order": tfidf_order,
        "bm25_order": bm25_order,
        "alpha": alpha
    }


# ==================================================
# 6. 四种检索算法
# ==================================================

def search_all(query, top_k, resources):

    tfidf = resources["tfidf"]
    bm25 = resources["bm25"]
    sbert = resources["sbert"]

    doc_ids = resources["doc_ids"]
    alpha = resources["alpha"]

    # ---------- TF-IDF ----------
    query_vector = tfidf.vectorizer.transform([query])

    tfidf_scores = (
        query_vector @ tfidf.doc_vectors.T
    ).toarray().ravel()

    tfidf_scores = tfidf_scores[
        resources["tfidf_order"]
    ]

    # ---------- BM25 ----------
    query_tokens = tokenize(query)

    bm25_scores = bm25.bm25.get_scores(
        query_tokens
    )

    bm25_scores = bm25_scores[
        resources["bm25_order"]
    ]

    # ---------- Sentence-BERT ----------
    query_embedding = sbert.model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    sbert_scores = (
        sbert.embeddings @ query_embedding
    )

    # ---------- Hybrid ----------
    bm25_norm = minmax_normalize(bm25_scores)
    sbert_norm = minmax_normalize(sbert_scores)

    hybrid_scores = (
        alpha * bm25_norm
        + (1 - alpha) * sbert_norm
    )

    model_scores = {
        "TF-IDF": tfidf_scores,
        "BM25": bm25_scores,
        "Sentence-BERT": sbert_scores,
        "Hybrid": hybrid_scores
    }

    all_results = {}

    for model_name, scores in model_scores.items():

        ranked_indices = np.argsort(
            -scores,
            kind="stable"
        )[:top_k]

        results = []

        for idx in ranked_indices:

            doc_id = doc_ids[idx]
            doc = sbert.corpus_by_id[doc_id]

            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "abstract": doc.get("text", ""),
                "score": float(scores[idx])
            })

        all_results[model_name] = results

    return all_results


# ==================================================
# 7. 检索结果展示
# ==================================================

def display_results(model_name, results, t):

    st.subheader(model_name)

    if not results:
        st.info(t["no_results"])
        return

    for rank, paper in enumerate(results, start=1):

        with st.container(border=True):

            st.markdown(
                f"**{rank}. {paper['title']}**"
            )

            st.caption(
                f"{t['doc_id']}: {paper['doc_id']}"
            )

            st.write(
                f"**{t['score']}:** {paper['score']:.4f}"
            )

            with st.expander(t["abstract"]):

                if paper["abstract"]:
                    st.write(paper["abstract"])
                else:
                    st.write(t["no_abstract"])


# ==================================================
# 8. 页面标题
# ==================================================

st.title(t["title"])
st.caption(t["subtitle"])
st.write(t["description"])

st.divider()


# ==================================================
# 9. 加载模型
# ==================================================

try:
    resources = load_retrievers()

except Exception as e:
    st.error(f"{t['error']}: {e}")
    st.stop()


alpha = resources["alpha"]
document_count = len(resources["doc_ids"])


# ==================================================
# 10. 左侧设置栏
# ==================================================

st.sidebar.divider()
st.sidebar.header(t["settings"])

st.sidebar.write(
    f"**{t['dataset']}:** SciFact (BEIR)"
)

st.sidebar.write(
    f"**{t['documents']}:** {document_count:,}"
)

st.sidebar.write(
    f"**{t['alpha']}:** {alpha:.1f}"
)

st.sidebar.caption(t["fixed_alpha"])

st.sidebar.divider()
st.sidebar.subheader(t["method_info"])

st.sidebar.markdown("**TF-IDF**")
st.sidebar.caption(t["tfidf_info"])

st.sidebar.markdown("**BM25**")
st.sidebar.caption(t["bm25_info"])

st.sidebar.markdown("**Sentence-BERT**")
st.sidebar.caption(t["sbert_info"])

st.sidebar.markdown("**Hybrid**")
st.sidebar.caption(t["hybrid_info"])


# ==================================================
# 11. 三个推荐查询
# ==================================================

EXAMPLES = [
    {
        "label": t["example_1"],
        "query": "Vitamin D reduces the risk of fractures"
    },
    {
        "label": t["example_2"],
        "query": (
            "Silencing of Bcl2 is important for "
            "the maintenance and progression of tumors."
        )
    },
    {
        "label": t["example_3"],
        "query": (
            "Recurrent mutations occur frequently "
            "within CTCF anchor sites adjacent to oncogenes."
        )
    }
]


# ==================================================
# 12. Session State：保存用户输入
# ==================================================

# 只在第一次访问时初始化
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ==================================================
# 13. 推荐示例按钮
# ==================================================

st.subheader(t["sample"])
st.caption(t["sample_help"])

example_columns = st.columns(3)

for i, example in enumerate(EXAMPLES):

    with example_columns[i]:

        if st.button(
            example["label"],
            key=f"example_{i}",
            use_container_width=True
        ):
            # 点击示例时只填入查询框
            # 用户可以继续编辑
            st.session_state.search_query = example["query"]

            # 清除旧搜索结果，避免结果与新输入不一致
            st.session_state.search_results = None


# ==================================================
# 14. 自由搜索区域
# ==================================================

st.info(
    f"SciFact | {document_count:,} documents | "
    f"Hybrid α = {alpha:.1f}"
)

st.caption(t["custom_search"])

with st.form("search_form"):

    query = st.text_area(
        t["input"],
        key="search_query",
        placeholder=t["placeholder"],
        height=120,
        help=t["help"]
    )

    top_k = st.slider(
        t["top_k"],
        min_value=1,
        max_value=10,
        value=5,
        key="top_k"
    )

    submitted = st.form_submit_button(
        t["search"],
        use_container_width=True,
        type="primary"
    )


# ==================================================
# 15. 搜索执行
# ==================================================

if submitted:

    current_query = query.strip()

    if not current_query:

        st.warning(t["warning"])
        st.session_state.search_results = None

    else:

        try:
            with st.spinner(t["loading"]):

                results = search_all(
                    query=current_query,
                    top_k=top_k,
                    resources=resources
                )

            # 保存结果，避免语言切换后消失
            st.session_state.search_results = results
            st.session_state.last_query = current_query

        except Exception as e:
            st.error(f"{t['error']}: {e}")


# ==================================================
# 16. 展示检索结果
# ==================================================

if st.session_state.search_results is not None:

    st.success(t["success"])

    st.header(t["results"])

    st.caption(
        f"Query: {st.session_state.last_query}"
    )

    all_results = st.session_state.search_results

    # 第一行：TF-IDF 与 BM25
    col1, col2 = st.columns(2)

    with col1:
        display_results(
            t["tfidf"],
            all_results["TF-IDF"],
            t
        )

    with col2:
        display_results(
            t["bm25"],
            all_results["BM25"],
            t
        )

    st.divider()

    # 第二行：SBERT 与 Hybrid
    col3, col4 = st.columns(2)

    with col3:
        display_results(
            t["sbert"],
            all_results["Sentence-BERT"],
            t
        )

    with col4:
        display_results(
            f"{t['hybrid']} (α={alpha:.1f})",
            all_results["Hybrid"],
            t
        )


# ==================================================
# 17. 页脚
# ==================================================

st.divider()

st.caption(t["footer"])

st.caption(
    "GITC301 NLP Term Project | SciFact (BEIR) | "
    "TF-IDF · BM25 · Sentence-BERT · Hybrid"
)
