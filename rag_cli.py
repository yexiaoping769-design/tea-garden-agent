"""
阶段1：RAG 检索问答命令行
用法：
    python rag_cli.py "茶饼病怎么防治"
    python rag_cli.py            （进入交互问答）
流程：问题 → 向量化 → Chroma 检索 top3 → 拼进提示词 → DeepSeek 生成带引用的回答
"""
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_deepseek import ChatDeepSeek

load_dotenv(override=True)

BASE = Path(__file__).parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def retrieve(collection, model, question: str, k: int = 3):
    """把问题向量化，取相似度最高的 k 个知识块。"""
    q_emb = model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "source": meta["source"], "score": 1 - dist})
    return hits


def answer(question: str, hits) -> str:
    """把检索片段作为参考资料交给模型，要求带引用回答。"""
    context = "\n\n".join(f"[来源{i+1}] {h['text']}" for i, h in enumerate(hits))
    prompt = f"""你是茶园种植助手。请仅依据下面的参考资料回答问题，在回答中用[来源n]标注引用了哪条资料。
如果资料中没有足够信息，就回答"知识库中暂无相关资料"，不要编造。

参考资料：
{context}

问题：{question}
"""
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
    return llm.invoke(prompt).content


def main():
    question = " ".join(sys.argv[1:]).strip() or input("请输入问题：").strip()

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(BASE / "kb_db"))
    collection = client.get_collection("tea_knowledge")

    hits = retrieve(collection, model, question)

    print("\n=== 检索到的知识片段 ===")
    for i, h in enumerate(hits):
        print(f"[来源{i+1}] {h['source']}（相似度 {h['score']:.2f}）")

    print("\n=== 助手回答 ===")
    print(answer(question, hits))


if __name__ == "__main__":
    main()
