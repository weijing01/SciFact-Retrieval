# 科学文献证据检索系统（Scientific Evidence Retrieval System）

**西江大学 GITC301 — 自然语言处理期末项目**

**项目类型：** 应用型（Applied Project）

**项目英文名称：** Keyword vs. Semantic Retrieval: When Does Sentence-BERT Beat TF-IDF and BM25 for Scientific Evidence Search?

---

## 1. 项目简介

本项目基于 SciFact（BEIR 版本）科学文献数据集，开发了一个能够根据用户输入的科学观点检索相关论文的系统。

系统实现了四种自然语言处理检索方法：

1. **TF-IDF：** 基于词频和逆文档频率的传统关键词检索。
2. **BM25：** 考虑词频饱和与文档长度的关键词检索。
3. **Sentence-BERT：** 基于预训练 Transformer 的语义检索。
4. **Hybrid：** 结合 BM25 和 Sentence-BERT 的混合检索。

项目使用 Streamlit 构建中英双语网页，允许用户输入任意英文科学查询，并同时查看四种检索方法返回的相关论文。

**注意：** 本系统负责科学文献检索，不直接判断科学观点的真实性。

## 2. 项目研究目标

本项目主要研究传统关键词检索与现代语义检索在科学文献搜索中的表现差异。

研究提出三个假设：

**H1：** BM25 的 nDCG@10 高于 TF-IDF。

**H2：** 当查询与相关论文的词汇重叠较低时，Sentence-BERT 优于 BM25；当词汇重叠较高时，BM25 表现更好。

**H3：** 结合 BM25 和 Sentence-BERT 的 Hybrid 检索，其 nDCG@10 高于单独使用任一方法。

通过统一的数据集与评价指标，对这些假设进行实验验证。

## 3. 数据集

本项目使用 SciFact（BEIR 版本）。

| 数据项目 | 内容 |
|---|---|
| 数据集名称 | SciFact |
| 文档数量 | 5,183 |
| 论文内容 | 标题 + 摘要 |
| 用于调参的训练查询 | 809 |
| 测试查询 | 300 |
| 人工相关性标注 | Qrels |

数据集主要由三个部分组成。

**Corpus（文档库）**

包含 5,183 篇科学论文的标题和摘要，是检索系统的搜索对象。

**Queries（查询）**

包含用于检索的英文科学观点。

**Qrels（相关性标注）**

记录哪些论文与哪些查询相关，用于计算检索效果。

项目使用训练集选择 Hybrid 参数，并保留测试集用于最终评价，以避免测试集信息泄漏。

## 4. 检索方法

### 4.1 TF-IDF

TF-IDF（Term Frequency–Inverse Document Frequency）是一种传统文本表示方法。

它根据词语在文档中的出现频率，以及词语在整个文档集合中的稀有程度，为不同词语分配权重。

本项目使用：

- Scikit-learn 的 `TfidfVectorizer`
- 英文小写转换
- L2 向量归一化
- 余弦相似度排序

TF-IDF 的优点是计算简单、效率较高，但其检索主要依赖词汇匹配，难以直接识别不同措辞之间的语义关系。

### 4.2 BM25

BM25（Best Matching 25）是一种经典的关键词检索排序方法。

相比单纯依赖词频的检索方式，BM25 进一步考虑：

- **词频饱和：** 避免同一个词重复出现过多次时，分数无限增长。
- **文档长度归一化：** 减少文档长度差异对检索排名的影响。

本项目使用 `rank-bm25` 库中的 `BM25Okapi`。

参数设置：

| 参数 | 数值 |
|---|---|
| k1 | 1.5 |
| b | 0.75 |

### 4.3 Sentence-BERT 语义检索

Sentence-BERT 是一种基于预训练 Transformer 的文本语义表示方法。

本项目使用：

`sentence-transformers/all-MiniLM-L6-v2`

主要配置：

| 项目 | 设置 |
|---|---|
| 模型 | all-MiniLM-L6-v2 |
| Embedding 维度 | 384 |
| 文档数量 | 5,183 |
| 批次大小 | 8 |
| 最大输入长度 | 256 tokens |
| 是否微调 | 否 |
| 计算设备 | NVIDIA GTX 1650 |

系统首先将全部论文编码为 384 维语义向量，然后保存到本地。

检索时，只需要将用户查询转换为相同维度的向量，并计算查询向量与文档向量之间的余弦相似度。

这种方法可以在一定程度上识别同义表达和语义相近的科学文本。

### 4.4 Hybrid 混合检索

Hybrid 将 BM25 和 Sentence-BERT 的检索分数进行融合。

首先对两种分数分别进行 Min-Max 归一化，再使用加权公式：

\[
S_{\mathrm{Hybrid}}(d)
=\alpha S_{\mathrm{BM25}}^{norm}(d)
+(1-\alpha)S_{\mathrm{SBERT}}^{norm}(d)
\]

其中 α 控制两种方法的权重。

本项目在训练查询上测试 11 个权重：

`0.0, 0.1, 0.2, ..., 1.0`

以平均 nDCG@10 为标准，最终选出：

**最佳 α = 0.3**

对应：

- BM25 权重：30%
- Sentence-BERT 权重：70%

训练集平均 nDCG@10 为 **0.7220**。

## 5. 实验评价

项目使用三个信息检索评价指标。

**nDCG@10**

衡量前 10 篇检索结果的相关性及其排序质量。

**Recall@10**

衡量前 10 篇检索结果覆盖了多少已知相关论文。

**MRR@10**

衡量第一篇相关论文出现在多靠前的位置。

### 5.1 测试集实验结果

在 300 条 SciFact 测试查询上的结果如下：

| 检索方法 | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| TF-IDF | 0.5780 | 0.7135 | 0.5437 |
| BM25 | 0.6519 | 0.7740 | 0.6186 |
| Sentence-BERT | 0.6451 | 0.7833 | 0.6047 |
| **Hybrid** | **0.7149** | **0.8366** | **0.6804** |

Hybrid 在三个评价指标上均取得最高分。

与 BM25 相比，Hybrid 的 nDCG@10 从 0.6519 提升至 0.7149，相对提升约 9.66%。

### 5.2 词汇重叠分析

项目使用 Jaccard 相似度衡量查询与人工标注相关论文之间的词汇重叠程度。

按照 Jaccard 数值，将 300 条测试查询划分为低、中、高三个组，每组 100 条查询。

| 词汇重叠程度 | BM25 nDCG@10 | SBERT nDCG@10 | Hybrid nDCG@10 |
|---|---:|---:|---:|
| Low | 0.4299 | 0.4687 | **0.5223** |
| Medium | 0.6859 | 0.6534 | **0.7390** |
| High | 0.8398 | 0.8131 | **0.8834** |

实验观察到：

- 低词汇重叠时，Sentence-BERT 的平均 nDCG@10 高于 BM25。
- 中、高词汇重叠时，BM25 的平均 nDCG@10 高于 Sentence-BERT。
- Hybrid 在三个组中均取得最高平均 nDCG@10。

## 6. 错误分析

为进一步理解不同检索方法的优势与局限，本项目分析了两个真实案例。

### 案例 A：Sentence-BERT 成功，BM25 失败

**Query ID：** 1088

**查询：**

Silencing of Bcl2 is important for the maintenance and progression of tumors.

**相关论文：**

Antiapoptotic BCL-2 is required for maintenance of a model leukemia.

**文档 ID：** 37549932

| 方法 | nDCG@10 |
|---|---:|
| BM25 | 0.0000 |
| Sentence-BERT | 1.0000 |

查询中的 Bcl2 与论文中的 BCL-2 存在不同书写形式，同时论文使用了与查询不同的生物医学表达。

Sentence-BERT 将相关论文排在第一位，而 BM25 未能在前十名中找到该论文。

这表明语义检索在某些词汇不匹配的情况下具有优势。

### 案例 B：BM25 成功，Sentence-BERT 失败

**Query ID：** 1024

**查询：**

Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes.

**相关论文：**

3D Chromosome Regulatory Landscape of Human Pluripotent Cells.

**文档 ID：** 5373138

| 方法 | nDCG@10 |
|---|---:|
| BM25 | 1.0000 |
| Sentence-BERT | 0.0000 |

查询与相关论文摘要共享 CTCF、anchor 和 mutations 等关键科学词汇。

BM25 将相关论文排在第一位，而 Sentence-BERT 未能在前十名中找到该论文。

该案例表明，精确专业术语匹配仍然具有重要价值。

两个案例的 Jaccard 值分别为 0.0606 和 0.0594，说明相近的总体词汇重叠比例并不一定导致相同的检索表现。

## 7. 开发环境

| 项目 | 环境 |
|---|---|
| 操作系统 | Windows |
| IDE | PyCharm |
| Python | 3.11 |
| GPU | NVIDIA GeForce GTX 1650 4GB |
| PyTorch | 2.14.1+cu126 |
| Sentence Transformers | 6.3.0 |
| Scikit-learn | 1.9.1 |
| rank-bm25 | 0.2.2 |
| Streamlit | 1.67.0 |

本项目在 GPU 环境中完成 Sentence-BERT 文档编码，但也可以使用兼容的 CPU 版 PyTorch 运行。

## 8. 项目文件结构

```text
SciFactRetrieval/
│
├── app.py
├── download_data.py
├── check_data.py
├── requirements.txt
├── requirements-lock.txt
├── README.md
│
├── src/
│   ├── tfidf_search.py
│   ├── bm25_search.py
│   ├── test_sbert.py
│   ├── encode_corpus.py
│   ├── sbert_search.py
│   ├── hybrid_search.py
│   ├── tune_alpha.py
│   ├── evaluate_all.py
│   ├── analyze_overlap.py
│   ├── error_analysis.py
│   └── inspect_cases.py
│
├── data/
│   └── scifact/
│
├── models/
│   ├── scifact_embeddings.npy
│   └── scifact_doc_ids.json
│
└── results/
    ├── alpha_tuning.csv
    ├── hybrid_config.json
    ├── test_metrics.csv
    ├── test_rankings.csv
    ├── query_overlap.csv
    ├── overlap_results.csv
    └── model_differences.csv
```

## 9. 安装与运行

### 9.1 创建虚拟环境

在项目根目录运行：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 9.2 安装依赖

根据电脑硬件选择兼容的 PyTorch 版本。需要使用 GPU 时，可参考 PyTorch 官方安装说明选择对应 CUDA 构建。

然后执行：

```powershell
python -m pip install -r requirements.txt
```

### 9.3 下载 SciFact 数据集

```powershell
python download_data.py
```

检查数据：

```powershell
python check_data.py
```

### 9.4 生成 Sentence-BERT 文档向量

如果没有预先保存的 Embedding 文件，运行：

```powershell
python src/encode_corpus.py
```

完成后将在 `models/` 目录下生成论文向量和对应的文档 ID。

如果已提供且确认与当前文档库一致，可以跳过该步骤。

### 9.5 启动网页

运行：

```powershell
python -m streamlit run app.py
```

浏览器打开：

`http://localhost:8501`

即可使用科学文献检索系统。

## 10. 实验复现

### Hybrid 参数选择

```powershell
python src/tune_alpha.py
```

结果保存到 `results/alpha_tuning.csv`，最佳 α 保存到 `results/hybrid_config.json`。

### 四种模型统一评价

```powershell
python src/evaluate_all.py
```

输出：

- `test_metrics.csv`
- `test_rankings.csv`

### 词汇重叠分析

```powershell
python src/analyze_overlap.py
```

输出：

- `query_overlap.csv`
- `overlap_results.csv`

### 检索错误分析

```powershell
python src/error_analysis.py
```

查看选定案例的论文摘要：

```powershell
python src/inspect_cases.py
```

## 11. Streamlit 网页功能

系统提供中英双语操作界面。

主要功能包括：

- 中文和英文界面切换。
- 自由输入英文科学查询。
- 三个推荐示例查询。
- 同时展示四种模型的检索结果。
- 设置每种方法返回的论文数量（1–10）。
- 展示论文标题、ID 和检索分数。
- 展开查看论文完整摘要。
- 使用训练集确定的 α = 0.3 进行 Hybrid 检索。

目前网页界面支持中文，但检索模型主要面向英文文本，论文标题与摘要保留英文原文。

## 12. 项目局限性

本项目仍存在一些限制。

**第一，文档库规模有限。** 系统只能检索 SciFact 数据集包含的论文，并不是完整的医学数据库。

**第二，Sentence-BERT 存在输入长度限制。** 较长的论文摘要可能被截断。

**第三，科学术语可能造成语义检索错误。** 通用预训练模型不一定能够准确表示所有专业术语。

**第四，检索不等于事实核查。** 系统返回相关论文，但不能直接判断科学观点是否正确。

**第五，统计显著性尚未检验。** 当前实验展示了模型之间的性能差异，但尚未证明这些差异具有统计显著性。

## 13. 项目总结

本项目实现了基于 SciFact 的科学文献证据检索系统，并通过真实测试查询比较了传统关键词检索与现代语义检索的性能。

实验发现，BM25 和 Sentence-BERT 各自存在适合的检索场景。

在低词汇重叠情况下，Sentence-BERT 的平均表现更好；在中、高词汇重叠情况下，BM25 的平均表现更好。

通过融合两者，Hybrid 在本次实验中获得了最高的 nDCG@10、Recall@10 和 MRR@10。

最终，项目实现了一个具有实际交互功能的中英双语科学文献检索应用。