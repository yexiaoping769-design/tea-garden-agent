"""
阶段1：知识库构建脚本 —— docs/ 文档 → 切分 → 向量化 → 存入 Chroma
用法：python build_kb.py
运行后生成 kb_db/ 目录（本地向量数据库），之后 rag_cli.py 会读它。
"""
import os
import re
import unicodedata
from pathlib import Path

# 国内镜像下载模型 + 缓存重定向到项目文件夹（不占C盘）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"
DB_DIR = BASE / "kb_db"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# Unicode 私用区（码位无公开定义，PDF 字体映射出错时常表现为这些"乱码"）
PUA_RE = re.compile(r"[\uE000-\uF8FF\U000F0000-\U0010FFFD]")


def normalize_text(text: str) -> str:
    """PDF 提取文本清洗（对 txt 同样无害）：
    1) NFKC 归一化：全角数字/字母/标点转半角（５→5、％→%）
    2) 私用区字符修复：本套标准 PDF 的字体把章节号中的"．"映射到了私用区，
       表现为数字之间的乱码（如 ５?２?４ 实为 5.2.4）→ 按上下文还原为"."
    3) 其余无上下文依据的私用区字符直接剔除，避免污染向量
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\d) ?(?:" + PUA_RE.pattern + r")+ ?(?=\d)", ".", text)
    text = PUA_RE.sub("", text)
    return re.sub(r"[ \t\u3000]+", " ", text)


def read_doc(path: Path) -> str:
    """按扩展名提取纯文本：txt/md 直接读，pdf 用 PyMuPDF 逐页提取。"""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return normalize_text(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        import pymupdf  # 比 PyPDF2 更快，且对中文嵌入字体的兼容性更好
        with pymupdf.open(str(path)) as doc:
            return normalize_text("\n".join(page.get_text() for page in doc))
    return ""


def main():
    # ---- 1. 读取 docs/ 下所有文档 ----
    texts, sources = [], []
    for f in sorted(DOCS_DIR.iterdir()):
        content = read_doc(f)
        if content.strip():
            texts.append(content)
            sources.append(f.name)
    if not texts:
        print(f"docs/ 目录下没有可读文档")
        return

    # ---- 2. 切分：长文档切成 ~300 字的小块 ----
    # 为什么切：嵌入模型对短文本的语义表达最准，且检索时返回"相关段落"
    # 比整篇文档更精准、更省 token。chunk_overlap=50 让相邻块有重叠，
    # 避免一句话恰好被切在边界上而丢失上下文。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks, metadatas = [], []
    for text, src in zip(texts, sources):
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(piece)
                metadatas.append({"source": src})
    print(f"读取 {len(texts)} 份文档 → 切成 {len(chunks)} 个知识块")

    # ---- 3. 向量化：bge-small-zh 是中文语义嵌入模型 ----
    # 检索类任务按官方建议给查询加前缀，文档侧不加
    print(f"加载嵌入模型 {MODEL_NAME}（首次运行会下载约100MB，请稍等）...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunks, normalize_embeddings=True, show_progress_bar=True
    )

    # ---- 4. 存入 Chroma（本地持久化，用余弦相似度） ----
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("tea_knowledge")  # 重建时清掉旧数据
        print("已清除旧知识库")
    except Exception:
        pass
    collection = client.create_collection(
        name="tea_knowledge", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"完成：{collection.count()} 个知识块已存入 {DB_DIR}")


if __name__ == "__main__":
    main()
