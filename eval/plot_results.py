#!/usr/bin/env python3
"""
从 eval/results/*.json 生成论文用对比图（柱状 + 折线）。
依赖：matplotlib（见 requirements.txt）

用法（项目根目录）：
  .venv/bin/python eval/plot_results.py
  .venv/bin/python eval/plot_results.py --in eval/results/latest.json --out-dir eval/results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pick_cjk_font() -> str | None:
    from matplotlib import font_manager

    prefer = (
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
    )
    names = set()
    for f in font_manager.fontManager.ttflist:
        try:
            names.add(f.name)
        except Exception:
            continue
    for p in prefer:
        if p in names:
            return p
    return None


def _setup_matplotlib(use_zh: bool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    if use_zh:
        cjk = _pick_cjk_font()
        if cjk:
            plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        else:
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "eval/results/latest.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results/figures")
    ap.add_argument(
        "--zh",
        action="store_true",
        help="图内中文标签（需本机已安装黑体/雅黑/苹方等；否则请加 --zh 仍无中文时请去掉 --zh 使用英文图）",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_zh = args.zh and _pick_cjk_font() is not None
    if args.zh and not use_zh:
        print("警告: 未检测到中文字体，已改用英文标签出图。", file=sys.stderr)

    plt = _setup_matplotlib(use_zh=use_zh)

    # ---------- 图1：总体柱状对比 ----------
    m_ret = data.get("mean_retrieval_term_recall", 0)
    m_ar = data.get("mean_answer_term_recall_with_rag", 0)
    m_an = data.get("mean_answer_term_recall_no_rag", 0)

    if use_zh:
        bar_labels = ["检索片段\n期望词召回", "回答期望词召回\n（有 RAG）", "回答期望词召回\n（无 RAG）"]
        bar_title = "RAG 消融：总体指标"
        y_label = "召回率（0～1）"
        line_title = "逐病例：有 / 无 RAG 回答召回"
        y_line = "回答期望词召回率"
        x_line = "病例 ID"
        leg_r, leg_n = "有 RAG", "无 RAG（基线）"
    else:
        bar_labels = [
            "Retrieval\n(term recall)",
            "Answer\n(w/ RAG)",
            "Answer\n(w/o RAG)",
        ]
        bar_title = "RAG ablation (aggregate)"
        y_label = "Recall (0–1)"
        line_title = "Per-case answer term recall"
        y_line = "Answer term recall"
        x_line = "Case ID (gold set)"
        leg_r, leg_n = "With RAG", "Baseline (no retrieval)"

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    vals = [m_ret, m_ar, m_an]
    colors = ["#5B8FF9", "#61DDAA", "#F6BD16"]
    bars = ax1.bar(bar_labels, vals, color=colors, edgecolor="#333", linewidth=0.6)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel(y_label)
    ax1.set_title(bar_title)
    for b, v in zip(bars, vals):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig1.tight_layout()
    bar_path = out_dir / "ablation_bar.png"
    fig1.savefig(bar_path, dpi=160)
    plt.close(fig1)

    # ---------- 图2：逐病例折线 ----------
    cases = data.get("per_case") or []
    ids = [c.get("id", str(i)) for i, c in enumerate(cases)]
    y_r = [c.get("answer_term_recall_with_rag", 0) for c in cases]
    y_n = [c.get("answer_term_recall_no_rag", 0) for c in cases]

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    x = range(len(ids))
    ax2.plot(x, y_r, "o-", color="#61DDAA", label=leg_r, linewidth=2, markersize=6)
    ax2.plot(x, y_n, "s--", color="#F6903D", label=leg_n, linewidth=2, markersize=6)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(ids, rotation=35, ha="right", fontsize=9)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_ylabel(y_line)
    ax2.set_xlabel(x_line)
    ax2.set_title(line_title)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()
    line_path = out_dir / "per_case_line.png"
    fig2.savefig(line_path, dpi=160)
    plt.close(fig2)

    print(f"已生成:\n  {bar_path}\n  {line_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
