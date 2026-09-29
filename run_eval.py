"""
阶段3：评测脚本 —— 三种用法：

  python run_eval.py retrieve    # 检索评测：官方知识库 top3 是否命中关键词（本地运行，零成本）
  python run_eval.py compare     # 对比评测：从 git 历史恢复"种子版"知识库，与官方版跑同一套题
  python run_eval.py e2e         # 端到端评测：完整 Agent 跑 20 题 + 记录工具调用 + LLM 裁判打分（需 API）

评测设计：
- 检索评测（客观）：对 kb 类问题做向量检索，判定 retrieval_keywords 是否全部出现在 top3 中。
  这是"检索质量"的下限指标——连关键词都检不到，后面的回答一定不靠谱。
- 端到端评测（主观+客观结合）：
  * 工具选择客观核对：expect_tools 是否都被调用（代码直接比对，不靠裁判）
  * 回答质量主观评分：DeepSeek 当裁判，按"质量 0-2 + 诚实性 0-1"打分
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
EVAL_SET_PATH = BASE / "eval_set.json"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def load_eval_set():
    data = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    return data["cases"]


# ================== 检索评测 ==================

def eval_retrieval(collection, cases, label):
    """对 kb 类问题做检索，返回命中率明细。命中 = 所有 retrieval_keywords 都出现在 top3 文本中。"""
    from rag_core import get_model

    model = get_model()
    details, hit_count, total = [], 0, 0
    for c in cases:
        kws = c.get("retrieval_keywords") or []
        if not kws:  # 天气/联网/拒答类没有检索指标，跳过
            continue
        total += 1
        q_emb = model.encode([QUERY_PREFIX + c["question"]], normalize_embeddings=True)
        res = collection.query(query_embeddings=q_emb.tolist(), n_results=3)
        top3_text = "\n".join(res["documents"][0])
        top3_sources = [m["source"] for m in res["metadatas"][0]]
        missing = [k for k in kws if k not in top3_text]
        hit = not missing
        hit_count += hit
        status = "PASS" if hit else f"MISS(缺{missing})"
        details.append({"id": c["id"], "question": c["question"], "hit": hit,
                        "missing": missing, "sources": top3_sources})
        print(f"  [{status:6s}] {c['id']:>2}. {c['question']}")
        print(f"           top3来源: {top3_sources}")

    rate = hit_count / total * 100 if total else 0
    print(f"\n  == {label} 检索命中率: {hit_count}/{total} = {rate:.0f}% ==\n")
    return {"label": label, "hit": hit_count, "total": total,
            "rate": round(rate, 1), "details": details}


def cmd_retrieve():
    from rag_core import get_collection
    cases = load_eval_set()
    result = eval_retrieval(get_collection(), cases, "官方语料版")
    (BASE / "eval_results_retrieve.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_compare():
    """从 git 历史 ce75a27 恢复种子文档 → 建临时旧库 → 与官方库对比命中率。"""
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rag_core import get_model

    old_docs_dir = BASE / "eval_tmp" / "old_docs"
    old_db_dir = BASE / "eval_tmp" / "old_kb_db"
    restore_seed_docs(old_docs_dir)
    build_kb_at(old_docs_dir, old_db_dir)

    cases = load_eval_set()
    official = eval_retrieval(
        chromadb.PersistentClient(path=str(BASE / "kb_db")).get_collection("tea_knowledge"),
        cases, "官方语料版")
    old = eval_retrieval(
        chromadb.PersistentClient(path=str(old_db_dir)).get_collection("tea_knowledge"),
        cases, "种子版(AI生成文档)")

    # 并排对比表
    old_by_id = {d["id"]: d for d in old["details"]}
    print("=" * 72)
    print(f"{'题目':<28}{'种子版':<10}{'官方版':<10}")
    print("-" * 72)
    for d in official["details"]:
        o = old_by_id[d["id"]]
        mark = lambda x: "命中" if x["hit"] else "未命中"
        print(f"{str(d['id'])+'. '+d['question'][:22]:<28}{mark(o):<10}{mark(d):<10}")
    print("-" * 72)
    print(f"{'检索命中率':<28}{old['rate']:>5.0f}%{'':<4}{official['rate']:>5.0f}%")
    print("=" * 72)

    (BASE / "eval_results_compare.json").write_text(
        json.dumps({"official": official, "seed": old},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n明细已存入 eval_results_compare.json")


def restore_seed_docs(dest: Path):
    """从 git 历史 commit ce75a27（阶段1首次提交）中提取当时的 5 份种子文档。"""
    dest.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-tree", "ce75a27", "docs/", "--name-only"],
        capture_output=True).stdout.decode("utf-8")
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    for p in paths:
        name = p.split("/", 1)[1]  # 去掉 "docs/" 前缀
        content = subprocess.run(["git", "show", f"ce75a27:{p}"], capture_output=True).stdout
        (dest / name).write_bytes(content)
    print(f"已从 git 历史恢复 {len(paths)} 份种子文档到 {dest}")


def build_kb_at(docs_dir: Path, db_dir: Path):
    """把指定目录的文档建库到指定位置（与 build_kb.py 同一套切分/向量化参数）。"""
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rag_core import get_model

    chunks, metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""])
    for f in sorted(docs_dir.iterdir()):
        if f.suffix.lower() != ".txt":
            continue
        for piece in splitter.split_text(f.read_text(encoding="utf-8")):
            if piece.strip():
                chunks.append(piece)
                metadatas.append({"source": f.name})

    embeddings = get_model().encode(chunks, normalize_embeddings=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection("tea_knowledge")
    except Exception:
        pass
    col = client.create_collection("tea_knowledge", metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"chunk_{i}" for i in range(len(chunks))], documents=chunks,
            embeddings=embeddings.tolist(), metadatas=metadatas)
    print(f"临时旧库构建完成：{len(chunks)} 个知识块 → {db_dir}\n")


# ================== 端到端评测 ==================

def cmd_e2e():
    from agent import agent  # 导入即完成 Agent 构建（load_dotenv + create_agent）
    from langchain_deepseek import ChatDeepSeek

    judge = ChatDeepSeek(model="deepseek-chat", temperature=0)  # 裁判用低温，评分更稳定
    cases = load_eval_set()

    judge_sys = """你是严格的大模型应用评测裁判。根据【用户问题】【参考要点】【助手回答】评分，只输出JSON：
{"quality": 0, "honesty": 0, "total": 0, "reason": "一句话理由"}
评分标准：
- quality (0-2)：回答是否正确、完整地解决了用户问题。2=正确且覆盖了参考要点中的关键信息；1=部分正确或泛泛而谈；0=错误或答非所问。
- honesty (0-1)：1=没有编造事实，对知识库/工具无法覆盖的内容如实说明；0=编造了具体但不真实的信息。
- total = quality + honesty
注意：被评测系统的知识库包含 2025 年发布的农业农村部标准《NY/T 4803-2025 茶树病虫害防控技术规范》《NY/T 4802-2025 茶园低碳栽培技术规范》，其中列出的药剂（如吡唑醚菌酯、代森锌、矿物油等）和防治指标均为真实内容——不要因为你的训练数据中没有这些新标准，就把回答判为编造。"""

    results, tool_correct = [], 0
    for i, c in enumerate(cases, 1):
        cfg = {"configurable": {"thread_id": f"eval-{c['id']}"}}
        res = agent.invoke({"messages": [("user", c["question"])]}, config=cfg)

        # 客观核对工具调用：expect_tools 是否全部被调用
        called = []
        for m in res["messages"]:
            if getattr(m, "tool_calls", None):
                called += [tc["name"] for tc in m.tool_calls]
        expect = c.get("expect_tools", [])
        tool_ok = all(t in called for t in expect) if expect else True
        tool_correct += tool_ok

        answer = res["messages"][-1].content
        # 主观评分：LLM 裁判
        points = c.get("key_points") or [c.get("expectation", "如实说明无法回答，不编造")]
        judge_user = (f"用户问题：{c['question']}\n参考要点：{'; '.join(points)}\n"
                      f"助手回答：{answer}")
        try:
            text = judge.invoke([("system", judge_sys), ("user", judge_user)]).content
            s, e = text.find("{"), text.rfind("}")
            score = json.loads(text[s:e + 1])
        except Exception as ex:
            score = {"quality": -1, "honesty": -1, "total": -1, "reason": f"裁判解析失败: {ex}"}

        results.append({"id": c["id"], "category": c["category"], "question": c["question"],
                        "tools_called": called, "expect_tools": expect, "tool_ok": tool_ok,
                        "score": score, "answer": answer})
        print(f"[{i}/20] {'工具OK' if tool_ok else '工具异常'} | "
              f"调用:{called or '无'} | 裁判分:{score['total']}/3 | {c['question']}", flush=True)

    n = len(cases)
    passed = sum(1 for r in results if r["score"]["total"] >= 2)
    print("\n" + "=" * 60)
    print(f"工具选择正确率: {tool_correct}/{n}")
    avg = sum(r["score"]["total"] for r in results) / n
    print(f"裁判平均分: {avg:.2f}/3    通过率(total>=2): {passed}/{n}")
    print("=" * 60)

    (BASE / "eval_results_e2e.json").write_text(
        json.dumps({"summary": {"tool_correct": tool_correct, "total": n,
                                "avg_score": round(avg, 2), "passed": passed},
                    "details": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("明细已存入 eval_results_e2e.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "retrieve":
        cmd_retrieve()
    elif cmd == "compare":
        cmd_compare()
    elif cmd == "e2e":
        cmd_e2e()
    else:
        print(__doc__)
