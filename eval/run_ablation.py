#!/usr/bin/env python3
"""
毕设「RAG 效果验证」可复现实验脚本（消融 + 简单自动指标）。

用法（在项目根目录）：
  .venv/bin/python eval/run_ablation.py
  .venv/bin/python eval/run_ablation.py --gold eval/gold_set.json --out eval/results/latest.json

论文中可写为：
  - 对照：同一批主诉，开启检索（RAG）与关闭检索（无知识库摘录注入）各生成一次回答；
  - 检索侧：期望词在「检索片段」文本中的命中率（验证检索是否对齐语料）；
  - 生成侧：期望词在「模型回答」中的命中率（近似衡量是否更贴近教材表述；非临床准确率）。

依赖：.env 中 AI_API_KEY、已构建 rag_data/；会调用真实大模型 API，产生费用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _term_hit(term: str, text: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    return t in (text or "")


def _rag_blob(sources: list) -> str:
    parts = []
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        parts.append(s.get("title") or "")
        parts.append(s.get("ref") or "")
        parts.append(s.get("excerpt") or "")
    return "\n".join(parts)


def _coverage(terms: list, text: str) -> tuple[float, int, int]:
    if not terms:
        return 0.0, 0, 0
    hits = sum(1 for t in terms if _term_hit(t, text))
    return hits / len(terms), hits, len(terms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold_set.json")
    ap.add_argument("--out", type=Path, default=None, help="汇总 JSON 输出路径")
    ap.add_argument("--sleep", type=float, default=1.0, help="每条请求间隔秒，降低限流概率")
    args = ap.parse_args()

    data = json.loads(args.gold.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if not cases:
        print("gold_set 无 cases")
        return 1

    from ai_service import analyze_with_ai

    rows = []
    for i, c in enumerate(cases):
        cid = c.get("id") or str(i)
        desc = (c.get("description") or "").strip()
        terms = c.get("expected_terms") or []
        if not desc:
            continue
        print(f"[{i + 1}/{len(cases)}] {cid} …", flush=True)
        with_rag = analyze_with_ai(None, desc, skip_rag=False)
        time.sleep(max(0.0, args.sleep))
        no_rag = analyze_with_ai(None, desc, skip_rag=True)
        time.sleep(max(0.0, args.sleep))

        ans_r = (with_rag.get("diagnosis") or with_rag.get("analysis") or "").strip()
        ans_n = (no_rag.get("diagnosis") or no_rag.get("analysis") or "").strip()
        src = with_rag.get("rag_sources") or []
        blob = _rag_blob(src)

        cov_r, hit_r, n_t = _coverage(terms, ans_r)
        cov_n, hit_n, _ = _coverage(terms, ans_n)
        cov_ret, hit_ret, _ = _coverage(terms, blob)

        rows.append(
            {
                "id": cid,
                "description": desc,
                "expected_terms": terms,
                "rag_source_count": len(src),
                "retrieval_term_recall": round(cov_ret, 4),
                "answer_term_recall_with_rag": round(cov_r, 4),
                "answer_term_recall_no_rag": round(cov_n, 4),
                "delta_answer_recall": round(cov_r - cov_n, 4),
            }
        )

    n = len(rows)
    if n == 0:
        return 1

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_file": str(args.gold),
        "n_cases": n,
        "mean_retrieval_term_recall": round(mean("retrieval_term_recall"), 4),
        "mean_answer_term_recall_with_rag": round(mean("answer_term_recall_with_rag"), 4),
        "mean_answer_term_recall_no_rag": round(mean("answer_term_recall_no_rag"), 4),
        "mean_delta_answer_recall": round(mean("delta_answer_recall"), 4),
        "rag_nonempty_rate": round(
            sum(1 for r in rows if r["rag_source_count"] > 0) / n, 4
        ),
        "per_case": rows,
    }

    out_path = args.out
    if out_path is None:
        out_dir = ROOT / "eval" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latest.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_case"}, ensure_ascii=False, indent=2))
    print(f"\n已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
