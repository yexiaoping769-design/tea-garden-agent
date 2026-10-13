"""对比评测：带 reranker vs 不带 reranker 的检索效果差异。"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def run_retrieval_eval(cases, use_reranker, label):
    """对 kb 类问题跑检索，统计命中率与来源分布。"""
    from rag_core import hybrid_search

    details, hit_count, total = [], 0, 0
    for c in cases:
        kws = c.get("retrieval_keywords") or []
        if not kws:
            continue
        total += 1
        results = hybrid_search(c["question"], k=3, use_reranker=use_reranker)
        top3_text = "\n".join(doc for doc, _, _ in results)
        top3_sources = [src for _, src, _ in results]
        missing = [k for k in kws if k not in top3_text]
        hit = not missing
        hit_count += hit
        status = "PASS" if hit else f"MISS({missing})"
        details.append({
            "id": c["id"], "question": c["question"],
            "hit": hit, "missing": missing,
            "sources": top3_sources,
        })
        print(f"  [{status:8s}] {c['id']:>2}. {c['question']}")

    rate = hit_count / total * 100 if total else 0
    print(f"\n  == {label}: {hit_count}/{total} = {rate:.1f}% ==\n")
    return {"label": label, "hit": hit_count, "total": total,
            "rate": round(rate, 1), "details": details}


def main():
    data = json.loads((BASE / "eval_set.json").read_text(encoding="utf-8"))
    cases = data["cases"]

    print("=" * 70)
    print("对比评测：bge-reranker 重排模块加入前后")
    print("=" * 70)

    print("\n>>> 版本 A：RRF 融合（无 reranker）<<<")
    rrf_only = run_retrieval_eval(cases, use_reranker=False, label="RRF融合(无reranker)")

    print("\n>>> 版本 B：RRF 融合 + bge-reranker 重排 <<<")
    with_rerank = run_retrieval_eval(cases, use_reranker=True, label="RRF+bge-reranker")

    # 并排对比
    print("=" * 70)
    print("逐项对比")
    print("-" * 70)
    rrf_by_id = {d["id"]: d for d in rrf_only["details"]}
    for d in with_rerank["details"]:
        r = rrf_by_id.get(d["id"])
        changed = ""
        if r and r["hit"] != d["hit"]:
            changed = f"  ← 变化：RRF-only={'PASS' if r['hit'] else 'MISS'} | +rerank={'PASS' if d['hit'] else 'MISS'}"
        elif r and r["sources"] != d["sources"]:
            changed = f"  ← 来源变化：{r['sources']} → {d['sources']}"
        mark = "PASS" if d["hit"] else f"MISS(缺{d['missing']})"
        print(f"  {d['id']:>2}. [{mark}] {d['question'][:30]}{changed}")

    print("-" * 70)
    print(f"  RRF 融合(无重排) : {rrf_only['hit']}/{rrf_only['total']} = {rrf_only['rate']:.1f}%")
    print(f"  RRF + bge-reranker: {with_rerank['hit']}/{with_rerank['total']} = {with_rerank['rate']:.1f}%")
    print("=" * 70)

    # 写结果
    result = {"rrf_only": rrf_only, "with_reranker": with_rerank}
    (BASE / "eval_results_reranker_compare.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n对比结果已存入 eval_results_reranker_compare.json")


if __name__ == "__main__":
    main()
