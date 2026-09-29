"""
RAG 检索核心模块：模型与向量库的单例加载 + 检索函数。
供 agent.py（作为工具）和 rag_cli.py（命令行）复用，避免重复加载模型。
"""
import os
from pathlib import Path

# 国内镜像下载模型 + 缓存重定向到项目文件夹（不占C盘）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 单例：模型约100MB、加载约2秒，进程内只加载一次
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
