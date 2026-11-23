
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

# 1. 项目路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 2. SciFact 官方 BEIR 下载地址
URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

# 3. 下载及解压路径
ZIP_PATH = DATA_DIR / "scifact.zip"
SCIFACT_DIR = DATA_DIR / "scifact"

# 4. 创建数据目录
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 5. 下载数据集
if not ZIP_PATH.exists():
    print("Downloading SciFact dataset...")
    urlretrieve(URL, ZIP_PATH)
    print("Download completed.")
else:
    print("ZIP file already exists.")

# 6. 解压数据集
if not (SCIFACT_DIR / "corpus.jsonl").exists():
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction completed.")
else:
    print("Dataset already extracted.")

# 7. 检查文件
files = [
    "corpus.jsonl",
    "queries.jsonl",
    "qrels/train.tsv",
    "qrels/test.tsv",
]

for file in files:
    path = SCIFACT_DIR / file
    print(f"{file}: {path.exists()}")

print("Dataset setup finished.")
