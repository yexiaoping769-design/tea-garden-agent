"""
RAG 核心模块：入库（解析/清洗/切分/向量化/写入）+ 检索。
模型与向量库均为进程内单例，供多方复用：
- agent.py     ：作为检索工具喂给 Agent
- rag_cli.py   ：命令行问答
- web_app.py   ：在线上传入库（ingest_file）
- build_kb.py  ：离线全量重建
"""
import os
import re
import uuid
import unicodedata
from pathlib import Path

# 国内镜像下载模型 + 缓存重定向到项目文件夹（不占C盘）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# Unicode 私用区（码位无公开定义，PDF 字体映射出错时常表现为这些"乱码"）
PUA_RE = re.compile(r"[\uE000-\uF8FF\U000F0000-\U0010FFFD]")


def normalize_text(text: str) -> str:
    """PDF 提取文本清洗（对 txt/docx 同样无害）：
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
    """按扩展名提取纯文本：txt/md 直接读，pdf 用 PyMuPDF，docx 用 python-docx。"""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return normalize_text(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        import pymupdf  # 比 PyPDF2 更快，且对中文嵌入字体的兼容性更好
        with pymupdf.open(str(path)) as doc:
            return normalize_text("\n".join(page.get_text() for page in doc))
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return normalize_text("\n".join(p.text for p in doc.paragraphs))
    return ""


def get_splitter() -> RecursiveCharacterTextSplitter:
    """知识块切分器（全量重建与增量入库必须用同一套参数，保证一致性）。
    为什么切 300 字：嵌入模型对短文本的语义表达最准，且检索时返回"相关段落"
    比整篇文档更精准、更省 token。overlap=50 让相邻块有重叠，
    避免一句话恰好被切在边界上而丢失上下文。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )


# ---- 单例：模型约100MB、加载约2秒，进程内只加载一次 ----
_model = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(BASE / "kb_db"))
        _collection = client.get_collection("tea_knowledge")
    return _collection


def ingest_file(path: Path, source_name: str | None = None) -> int:
    """把单个文件增量入库：解析 → 切分 → 向量化 → 写入 Chroma。
    返回入库的知识块数；提取不到内容时抛 ValueError。
    同名来源先删旧块再写入（覆盖语义），避免重复上传导致知识重复。"""
    text = read_doc(path)
    if not text.strip():
        raise ValueError(f"无法从 {path.name} 中提取到文本内容")
    chunks = [p for p in get_splitter().split_text(text) if p.strip()]
    if not chunks:
        raise ValueError(f"{path.name} 切分后没有有效内容")
    src = source_name or path.name
    collection = get_collection()
    collection.delete(where={"source": src})
    embeddings = get_model().encode(chunks, normalize_embeddings=True)
    collection.add(
        ids=[f"upload_{uuid.uuid4().hex}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": src}] * len(chunks),
    )
    return len(chunks)


def search_tea_knowledge(query: str, k: int = 3) -> str:
    """检索本地茶园知识库，返回带来源标注的资料文本。"""
    q_emb = get_model().encode([QUERY_PREFIX + query], normalize_embeddings=True)
    res = get_collection().query(query_embeddings=q_emb.tolist(), n_results=k)
    lines = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        lines.append(f"[来源: {meta['source']} | 相似度 {1 - dist:.2f}]\n{doc}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "茶饼病怎么防治"
    print(search_tea_knowledge(q))
