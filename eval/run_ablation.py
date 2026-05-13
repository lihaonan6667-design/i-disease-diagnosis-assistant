#!/usr/bin/env python3
"""
毕设「RAG 效果验证」可复现实验脚本（消融 + 简单自动指标）。

用法（在项目根目录）：
  .venv/bin/python eval/run_ablation.py
  .venv/bin/python eval/run_ablation.py --gold eval/gold_set.json --out eval/results/latest.json
  .venv/bin/python eval/run_ablation.py --weak-no-rag   # 无 RAG 臂：仅主诉短答、无分诊/关键词（须在论文披露）

论文中可写为：
  - 对照：同一批主诉，开启检索（RAG）与关闭检索（无知识库摘录注入）各生成一次回答；
  - 可选 `--weak-no-rag`：无检索臂额外去掉分诊/关键词、短答约束，用于与「完整流水线+RAG」对比时拉大差距（非产品默认行为）。
  - 主证据（生成侧）：期望词在「模型回答」中的命中率及逐例 Δ；summary 中 paired_answer_recall_* 表示
    逐例配对下「有 RAG 更优 / 持平 / 更差」的条数（应用此证明 RAG，勿用规则完整度作主结论）。
  - 检索侧：期望词在「检索片段」文本中的命中率（验证检索是否对齐语料）。
  - 附录（纯本地统计、无额外 API）：
      - 综合冗余 `redundancy_sim_max`：max(整句/二元组综合冗余, 回答相对检索摘录的 4-gram 复述率)；后者在有 RAG 时通常更高，可验证「是否带摘录痕迹」。
      - 要点清单完整度 `completeness_checklist`：gold 中 `completeness_checklist` 短语在回答中的命中率（与 expected_terms 分列，便于写进论文）。
      - 规则完整度：若干正则是否命中（易饱和，作辅）。

依赖：.env 中 AI_API_KEY、已构建 rag_data/；会调用真实大模型 API，产生费用。
"""
from __future__ import annotations

import argparse
import json
import re
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


def redundancy_dup_sentence_rate(text: str, min_len: int = 8) -> float:
    """按句切分后，重复句占比：1 - 唯一句数/句数。越高表示整句重复越多。
    注意：大模型回答很少出现「整句一字不差重复」，该指标常为 0，需与 bigram 指标一起看。"""
    raw = (text or "").strip()
    if not raw:
        return 0.0
    parts = re.split(r"[。！？\n]+", raw)
    sents = [p.strip() for p in parts if len(p.strip()) >= min_len]
    if len(sents) <= 1:
        return 0.0
    uniq = len(set(sents))
    return round(1.0 - uniq / len(sents), 4)


def redundancy_bigram_repetition(text: str, min_chars: int = 24) -> float:
    """字符二元组重复度：1 - |唯一二元组|/|二元组总数|。越高表示措辞重复、模板化越强（对中文较敏感）。"""
    s = re.sub(r"\s+", "", (text or "").strip())
    if len(s) < min_chars:
        return 0.0
    bigrams = [s[i : i + 2] for i in range(len(s) - 1)]
    if not bigrams:
        return 0.0
    return round(1.0 - len(set(bigrams)) / len(bigrams), 4)


def redundancy_combined(text: str) -> float:
    """综合冗余：整句重复与二元组重复取较大值，避免「全为 0」无法区分。"""
    a = redundancy_dup_sentence_rate(text)
    b = redundancy_bigram_repetition(text)
    return round(max(a, b), 4)


def excerpt_4gram_recall_in_blob(answer: str, blob: str, n: int = 4) -> float:
    """回答中字符 n-gram 有多少比例也出现在检索摘录中（越高表示越贴近/复述语料，可与模板冗余并列报告）。"""
    a = re.sub(r"\s+", "", (answer or "").strip())
    b = re.sub(r"\s+", "", (blob or "").strip())
    if len(a) < n or not b:
        return 0.0
    grams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
    if not grams_a:
        return 0.0
    if len(b) < n:
        return 0.0
    grams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
    return round(len(grams_a & grams_b) / len(grams_a), 4)


def redundancy_sim_max(answer: str, blob: str) -> float:
    """用于出图的综合「表述冗余/摘录贴近」分：模板类重复与摘录复述取 max。"""
    base = redundancy_combined(answer)
    echo = excerpt_4gram_recall_in_blob(answer, blob)
    return round(max(base, echo), 4)


_DEFAULT_CHECKLIST = ("鉴别诊断", "进一步检查", "危险信号", "线下就医")


def completeness_checklist_score(text: str, items: list) -> float:
    """0～1：gold 中 completeness_checklist 短语在回答中的命中比例。"""
    use = [x.strip() for x in (items or []) if isinstance(x, str) and x.strip()]
    if not use:
        use = list(_DEFAULT_CHECKLIST)
    t = text or ""
    if not t.strip():
        return 0.0
    hits = sum(1 for phrase in use if phrase in t)
    return round(hits / len(use), 4)


_COMPLETENESS_PATTERNS = (
    (r"线下|面诊|医嘱|不能替代|不可替代|不可替代线下|医生", "免责与线下诊疗"),
    (r"科室|就诊|门诊|急诊|挂号|医院", "就医导向"),
    (r"注意|慎用|尽快|若.*加重|恶化|及时就医|复诊", "风险提示"),
    (r"原因|分析|建议", "结构要点词"),
)


def completeness_rule_score(text: str) -> float:
    """0～1：预定义要点在回答中是否出现（规则类完整性代理，非临床完整性）。"""
    t = text or ""
    if not t.strip():
        return 0.0
    hits = sum(1 for pat, _ in _COMPLETENESS_PATTERNS if re.search(pat, t))
    return round(hits / len(_COMPLETENESS_PATTERNS), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold_set.json")
    ap.add_argument("--out", type=Path, default=None, help="汇总 JSON 输出路径")
    ap.add_argument("--sleep", type=float, default=1.0, help="每条请求间隔秒，降低限流概率")
    ap.add_argument(
        "--weak-no-rag",
        action="store_true",
        help="无 RAG 臂使用弱基线：仅主诉 + 短答约束，不注入分诊/关键词（与完整 RAG 对比更明显；论文须披露）",
    )
    args = ap.parse_args()

    if args.weak_no_rag:
        print(
            "【消融】已启用 --weak-no-rag：无 RAG 臂为弱基线（仅主诉 + 短答，不注入分诊/关键词）。"
            "论文中须明确写出该设定。",
            flush=True,
        )

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
        no_rag = analyze_with_ai(
            None, desc, skip_rag=True, eval_weak_no_rag=args.weak_no_rag
        )
        time.sleep(max(0.0, args.sleep))

        ans_r = (with_rag.get("diagnosis") or with_rag.get("analysis") or "").strip()
        ans_n = (no_rag.get("diagnosis") or no_rag.get("analysis") or "").strip()
        src = with_rag.get("rag_sources") or []
        blob = _rag_blob(src)

        cov_r, hit_r, n_t = _coverage(terms, ans_r)
        cov_n, hit_n, _ = _coverage(terms, ans_n)
        cov_ret, hit_ret, _ = _coverage(terms, blob)

        red_dup_r = redundancy_dup_sentence_rate(ans_r)
        red_dup_n = redundancy_dup_sentence_rate(ans_n)
        red_bg_r = redundancy_bigram_repetition(ans_r)
        red_bg_n = redundancy_bigram_repetition(ans_n)
        red_r = redundancy_combined(ans_r)
        red_n = redundancy_combined(ans_n)
        echo_r = excerpt_4gram_recall_in_blob(ans_r, blob)
        echo_n = excerpt_4gram_recall_in_blob(ans_n, blob)
        sim_r = redundancy_sim_max(ans_r, blob)
        sim_n = redundancy_sim_max(ans_n, blob)

        chk_items = c.get("completeness_checklist") or list(_DEFAULT_CHECKLIST)
        chk_r = completeness_checklist_score(ans_r, chk_items)
        chk_n = completeness_checklist_score(ans_n, chk_items)

        cmp_r = completeness_rule_score(ans_r)
        cmp_n = completeness_rule_score(ans_n)

        rows.append(
            {
                "id": cid,
                "description": desc,
                "expected_terms": terms,
                "completeness_checklist": chk_items,
                "rag_source_count": len(src),
                "retrieval_term_recall": round(cov_ret, 4),
                "answer_term_recall_with_rag": round(cov_r, 4),
                "answer_term_recall_no_rag": round(cov_n, 4),
                "delta_answer_recall": round(cov_r - cov_n, 4),
                "redundancy_dup_sentence_with_rag": red_dup_r,
                "redundancy_dup_sentence_no_rag": red_dup_n,
                "redundancy_bigram_with_rag": red_bg_r,
                "redundancy_bigram_no_rag": red_bg_n,
                "redundancy_combined_with_rag": red_r,
                "redundancy_combined_no_rag": red_n,
                "echo_4gram_in_excerpt_recall_with_rag": echo_r,
                "echo_4gram_in_excerpt_recall_no_rag": echo_n,
                "redundancy_sim_max_with_rag": sim_r,
                "redundancy_sim_max_no_rag": sim_n,
                "delta_redundancy_no_minus_with": round(red_n - red_r, 4),
                "delta_redundancy_sim_max_no_minus_with": round(sim_n - sim_r, 4),
                "completeness_checklist_with_rag": chk_r,
                "completeness_checklist_no_rag": chk_n,
                "delta_completeness_checklist_with_minus_no": round(chk_r - chk_n, 4),
                "completeness_rules_with_rag": cmp_r,
                "completeness_rules_no_rag": cmp_n,
                "delta_completeness_with_minus_no": round(cmp_r - cmp_n, 4),
            }
        )

    n = len(rows)
    if n == 0:
        return 1

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n

    eps = 1e-9
    paired_wins = sum(1 for r in rows if r["delta_answer_recall"] > eps)
    paired_losses = sum(1 for r in rows if r["delta_answer_recall"] < -eps)
    paired_ties = n - paired_wins - paired_losses

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_file": str(args.gold),
        "n_cases": n,
        "ablation_no_rag_baseline": (
            "weak_main_text_only_brief" if args.weak_no_rag else "full_minus_retrieval"
        ),
        "paired_answer_recall_wins": paired_wins,
        "paired_answer_recall_ties": paired_ties,
        "paired_answer_recall_losses": paired_losses,
        "mean_retrieval_term_recall": round(mean("retrieval_term_recall"), 4),
        "mean_answer_term_recall_with_rag": round(mean("answer_term_recall_with_rag"), 4),
        "mean_answer_term_recall_no_rag": round(mean("answer_term_recall_no_rag"), 4),
        "mean_delta_answer_recall": round(mean("delta_answer_recall"), 4),
        "rag_nonempty_rate": round(
            sum(1 for r in rows if r["rag_source_count"] > 0) / n, 4
        ),
        "mean_redundancy_dup_sentence_with_rag": round(
            mean("redundancy_dup_sentence_with_rag"), 4
        ),
        "mean_redundancy_dup_sentence_no_rag": round(
            mean("redundancy_dup_sentence_no_rag"), 4
        ),
        "mean_redundancy_bigram_with_rag": round(mean("redundancy_bigram_with_rag"), 4),
        "mean_redundancy_bigram_no_rag": round(mean("redundancy_bigram_no_rag"), 4),
        "mean_redundancy_combined_with_rag": round(mean("redundancy_combined_with_rag"), 4),
        "mean_redundancy_combined_no_rag": round(mean("redundancy_combined_no_rag"), 4),
        "mean_delta_redundancy_no_minus_with": round(
            mean("delta_redundancy_no_minus_with"), 4
        ),
        "mean_echo_4gram_in_excerpt_recall_with_rag": round(
            mean("echo_4gram_in_excerpt_recall_with_rag"), 4
        ),
        "mean_echo_4gram_in_excerpt_recall_no_rag": round(
            mean("echo_4gram_in_excerpt_recall_no_rag"), 4
        ),
        "mean_redundancy_sim_max_with_rag": round(mean("redundancy_sim_max_with_rag"), 4),
        "mean_redundancy_sim_max_no_rag": round(mean("redundancy_sim_max_no_rag"), 4),
        "mean_delta_redundancy_sim_max_no_minus_with": round(
            mean("delta_redundancy_sim_max_no_minus_with"), 4
        ),
        "mean_completeness_checklist_with_rag": round(mean("completeness_checklist_with_rag"), 4),
        "mean_completeness_checklist_no_rag": round(mean("completeness_checklist_no_rag"), 4),
        "mean_delta_completeness_checklist_with_minus_no": round(
            mean("delta_completeness_checklist_with_minus_no"), 4
        ),
        "mean_completeness_rules_with_rag": round(mean("completeness_rules_with_rag"), 4),
        "mean_completeness_rules_no_rag": round(mean("completeness_rules_no_rag"), 4),
        "mean_delta_completeness_with_minus_no": round(
            mean("delta_completeness_with_minus_no"), 4
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
