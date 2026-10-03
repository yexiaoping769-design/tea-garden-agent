"""
知识库构建脚本（全量重建）—— docs/ 文档 → 切分 → 向量化 → 存入 Chroma
用法：python build_kb.py
运行后生成 kb_db/ 目录（本地向量数据库）。
单文件增量入库走 rag_core.ingest_file（web 上传功能使用），不要用它重建。
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rag_core import MODEL_NAME, read_doc, get_splitter  # 解析/切分逻辑与增量入库共用一份

BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"
DB_DIR = BASE / "kb_db"


def main():
    # ---- 1. 读取 docs/ 下所有文档 ----
    texts, sources = [], []
    for f in sorted(DOCS_DIR.iterdir()):
        content = read_doc(f)
        if content.strip():
            texts.append(content)
            sources.append(f.name)
    if not texts:
        print("docs/ 目录下没有可读文档")
        return

    # ---- 2. 切分：长文档切成 ~300 字的小块 ----
    splitter = get_splitter()
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
