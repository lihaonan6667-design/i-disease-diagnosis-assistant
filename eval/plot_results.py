#!/usr/bin/env python3
"""
从 eval/results/*.json 生成论文用对比图（柱状 + 折线）。
依赖：matplotlib（见 requirements.txt）

输出除 ablation_bar / per_case_line / aux_* 外，另有 delta_recall_per_case.png：
逐病例「有 RAG - 无 RAG」期望词召回差；当字面重复句冗余全为 0 时，aux 图会省略冗余柱并放大规则命中纵轴。

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


def _paired_answer_recall_stats(data: dict) -> tuple[int, int, int, float | None]:
    """逐例 Δ 召回的胜负统计与平均 Δ（用于主图脚注，证明 RAG 宜用此条链）。"""
    cases = data.get("per_case") or []
    eps = 1e-9
    wins = losses = 0
    for c in cases:
        d = float(c.get("delta_answer_recall", 0) or 0)
        if d > eps:
            wins += 1
        elif d < -eps:
            losses += 1
    n = len(cases)
    ties = n - wins - losses
    md = data.get("mean_delta_answer_recall")
    if md is None and cases:
        md = sum(float(c.get("delta_answer_recall", 0) or 0) for c in cases) / n
    return wins, ties, losses, md


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
    weak = data.get("ablation_no_rag_baseline") == "weak_main_text_only_brief"

    if use_zh:
        bar_labels = ["检索片段\n期望词召回", "回答期望词召回\n（有 RAG）", "回答期望词召回\n（无 RAG）"]
        bar_title = "RAG 消融：总体指标"
        y_label = "召回率（0～1）"
        line_title = "逐病例：有 / 无 RAG 回答召回"
        y_line = "回答期望词召回率"
        x_line = "病例 ID"
        leg_r, leg_n = "有 RAG", "无 RAG（基线）"
        if weak:
            bar_labels[2] = "回答期望词召回\n（无 RAG·弱基线）"
            leg_n = "无 RAG（弱基线）"
            line_title = "逐病例：有 RAG / 无 RAG（弱基线）回答召回"
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
        if weak:
            bar_labels[2] = "Answer\n(w/o RAG, weak)"
            leg_n = "Weak no-RAG baseline"
            line_title = "Per-case recall: w/ RAG vs weak no-RAG"

    fig1, ax1 = plt.subplots(figsize=(7.2, 5.0))
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
    wins, ties, losses, mdelta = _paired_answer_recall_stats(data)
    n_p = wins + ties + losses
    if mdelta is not None and n_p > 0:
        if use_zh:
            note = (
                f"主证据（期望词在回答中）：平均 Δ={mdelta:+.4f}；逐例 提升{wins} / 持平{ties} / 下降{losses}（n={n_p}）\n"
                f"（aux 图中规则命中、冗余为写法代理，不能代表「RAG 是否更好」）"
            )
            if weak:
                note += "\n无 RAG 为弱基线（无分诊/关键词、短答），对比的是「完整流水线」而非仅关检索。"
        else:
            note = (
                f"Primary evidence (expected terms in answer): mean Δ={mdelta:+.4f}; "
                f"per-case better / tie / worse = {wins} / {ties} / {losses} (n={n_p})\n"
                f"(Aux rule/redundancy scores are weak proxies, not the RAG claim.)"
            )
            if weak:
                note += "\nWeak no-RAG omits triage/keywords + brief prompt; compare to full pipeline."
        ax1.text(
            0.5,
            -0.14,
            note,
            transform=ax1.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#333",
            linespacing=1.35,
        )
    fig1.tight_layout()
    bar_path = out_dir / "ablation_bar.png"
    fig1.savefig(bar_path, dpi=160, bbox_inches="tight")
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

    generated = [bar_path, line_path]

    # ---------- 图2b：逐病例 Δ 期望词召回（冗余/完整性常无区分时，主文可优先用此图）----------
    if cases and cases[0].get("delta_answer_recall") is not None:
        deltas = [float(c.get("delta_answer_recall", 0) or 0) for c in cases]
        fig2b, ax2b = plt.subplots(figsize=(9, 4.2))
        xb = range(len(ids))
        colors_b = ["#61DDAA" if d >= 0 else "#F6903D" for d in deltas]
        ax2b.bar(xb, deltas, color=colors_b, edgecolor="#333", linewidth=0.5)
        ax2b.axhline(0, color="#666", linewidth=0.8)
        ax2b.set_xticks(list(xb))
        ax2b.set_xticklabels(ids, rotation=35, ha="right", fontsize=9)
        ax2b.set_ylabel(
            "Δ 回答期望词召回（有 RAG - 无）" if use_zh else "Δ answer term recall (w/ - w/o RAG)"
        )
        ax2b.set_xlabel(x_line)
        ax2b.set_title(
            "逐病例：RAG 对期望词命中的增减" if use_zh else "Per-case Δ in expected-term recall"
        )
        ax2b.grid(True, axis="y", linestyle=":", alpha=0.6)
        ymax = max(0.08, max(abs(d) for d in deltas) * 1.25) if deltas else 0.08
        ax2b.set_ylim(-ymax, ymax)
        fig2b.tight_layout()
        delta_path = out_dir / "delta_recall_per_case.png"
        fig2b.savefig(delta_path, dpi=160)
        plt.close(fig2b)
        generated.append(delta_path)

    # ---------- 图3：综合冗余 sim_max + 要点清单完整度（优先新字段）----------
    use_sim_max = data.get("mean_redundancy_sim_max_with_rag") is not None
    use_checklist_mean = data.get("mean_completeness_checklist_with_rag") is not None

    if use_sim_max:
        mr_r = data.get("mean_redundancy_sim_max_with_rag", 0)
        mr_n = data.get("mean_redundancy_sim_max_no_rag", 0)
    else:
        has_combined = data.get("mean_redundancy_combined_with_rag") is not None
        mr_r = (
            data.get("mean_redundancy_combined_with_rag")
            if has_combined
            else data.get("mean_redundancy_dup_sentence_with_rag")
        )
        mr_n = (
            data.get("mean_redundancy_combined_no_rag", 0)
            if has_combined
            else data.get("mean_redundancy_dup_sentence_no_rag", 0)
        )

    if mr_r is not None:
        if use_checklist_mean:
            mc_r = data.get("mean_completeness_checklist_with_rag", 0)
            mc_n = data.get("mean_completeness_checklist_no_rag", 0)
        else:
            mc_r = data.get("mean_completeness_rules_with_rag", 0)
            mc_n = data.get("mean_completeness_rules_no_rag", 0)

        if use_zh:
            if use_sim_max and use_checklist_mean:
                g_labels_full = [
                    "综合冗余\n有RAG",
                    "综合冗余\n无RAG",
                    "要点清单\n有RAG",
                    "要点清单\n无RAG",
                ]
            elif use_sim_max:
                g_labels_full = ["综合冗余\n有RAG", "综合冗余\n无RAG", "规则命中\n有RAG", "规则命中\n无RAG"]
            elif data.get("mean_redundancy_combined_with_rag") is not None:
                g_labels_full = ["综合冗余\n有RAG", "综合冗余\n无RAG", "规则命中\n有RAG", "规则命中\n无RAG"]
            else:
                g_labels_full = ["重复句冗余\n有RAG", "重复句冗余\n无RAG", "规则命中\n有RAG", "规则命中\n无RAG"]
            y3 = "分数（0～1）"
        else:
            if use_sim_max and use_checklist_mean:
                g_labels_full = [
                    "Sim-max redund.\nw/ RAG",
                    "Sim-max redund.\nw/o RAG",
                    "Checklist\nw/ RAG",
                    "Checklist\nw/o RAG",
                ]
            elif use_sim_max:
                g_labels_full = [
                    "Sim-max redund.\nw/",
                    "Sim-max redund.\nw/o",
                    "Rule hits\nw/ RAG",
                    "Rule hits\nw/o RAG",
                ]
            elif data.get("mean_redundancy_combined_with_rag") is not None:
                g_labels_full = [
                    "Redundancy\n(combined) w/",
                    "Redundancy\n(combined) w/o",
                    "Rule hits\nw/ RAG",
                    "Rule hits\nw/o RAG",
                ]
            else:
                g_labels_full = [
                    "Dup-sentence\nredundancy w/",
                    "Dup-sentence\nredundancy w/o",
                    "Rule hits\nw/ RAG",
                    "Rule hits\nw/o RAG",
                ]
            y3 = "Score (0–1)"

        # 仅当 sim_max 也不可用且字面重复全 0 时，退化为只画完整度两柱
        red_flat = (not use_sim_max) and max(float(mr_r or 0), float(mr_n or 0)) < 1e-5

        if red_flat:
            if use_zh:
                fig3_title = "规则要点命中（均值）；重复句字面冗余本批均为 0"
                g_labels = ["规则命中\n有 RAG", "规则命中\n无 RAG"]
            else:
                fig3_title = "Rule-based completeness (mean); dup-sentence redundancy = 0"
                g_labels = ["Rule hits\nw/ RAG", "Rule hits\nw/o RAG"]
            fig3, ax3 = plt.subplots(figsize=(6, 4.2))
            vals3 = [mc_r, mc_n]
            c3 = ["#61ddaa", "#f6bd16"]
        else:
            if use_zh:
                if use_sim_max and use_checklist_mean:
                    fig3_title = "综合冗余与要点清单完整度（均值）"
                elif use_sim_max:
                    fig3_title = "综合冗余与规则命中（均值）"
                else:
                    fig3_title = "冗余度与规则完整性（均值）"
            else:
                if use_sim_max and use_checklist_mean:
                    fig3_title = "Sim-max redundancy & checklist completeness (mean)"
                elif use_sim_max:
                    fig3_title = "Sim-max redundancy & rule hits (mean)"
                else:
                    fig3_title = "Redundancy & rule-based completeness (mean)"
            fig3, ax3 = plt.subplots(figsize=(8, 4.2))
            vals3 = [mr_r, mr_n, mc_r, mc_n]
            g_labels = g_labels_full
            c3 = ["#7262fd", "#f6903d", "#61ddaa", "#f6bd16"]

        b3 = ax3.bar(g_labels, vals3, color=c3, edgecolor="#333", linewidth=0.5)
        if red_flat:
            lo = min(vals3) * 0.95 if min(vals3) < 0.99 else 0.0
            lo = max(0.0, lo - 0.02)
            ax3.set_ylim(lo, 1.02)
        else:
            ax3.set_ylim(0, max(1.08, max(vals3) * 1.15))
        ax3.set_ylabel(y3)
        ax3.set_title(fig3_title, fontsize=11)
        for b, v in zip(b3, vals3):
            ax3.text(
                b.get_x() + b.get_width() / 2,
                v + (0.01 if red_flat else 0.02),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax3.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig3.tight_layout()
        aux_bar = out_dir / "aux_metrics_bar.png"
        fig3.savefig(aux_bar, dpi=160)
        plt.close(fig3)
        generated.append(aux_bar)

        # ---------- 图4：逐病例 冗余 / 完整度 折线 ----------
        if cases and (
            cases[0].get("redundancy_dup_sentence_with_rag") is not None
            or cases[0].get("redundancy_combined_with_rag") is not None
            or cases[0].get("redundancy_sim_max_with_rag") is not None
        ):
            use_sim_case = cases[0].get("redundancy_sim_max_with_rag") is not None
            use_chk_case = cases[0].get("completeness_checklist_with_rag") is not None

            if use_sim_case:
                rk, nk = ("redundancy_sim_max_with_rag", "redundancy_sim_max_no_rag")
            elif cases[0].get("redundancy_combined_with_rag") is not None:
                rk, nk = ("redundancy_combined_with_rag", "redundancy_combined_no_rag")
            else:
                rk, nk = ("redundancy_dup_sentence_with_rag", "redundancy_dup_sentence_no_rag")

            zk_r, zk_n = (
                ("completeness_checklist_with_rag", "completeness_checklist_no_rag")
                if use_chk_case
                else ("completeness_rules_with_rag", "completeness_rules_no_rag")
            )

            yr = [c.get(rk, 0) for c in cases]
            yn = [c.get(nk, 0) for c in cases]
            zr = [c.get(zk_r, 0) for c in cases]
            zn = [c.get(zk_n, 0) for c in cases]

            if use_zh:
                if use_sim_case and use_chk_case:
                    t4 = "逐病例：综合冗余与要点清单完整度"
                    y4b = "要点清单（0～1）"
                    l3, l4 = "清单有RAG", "清单无RAG"
                else:
                    t4 = "逐病例：规则要点命中（纵轴放大以便看差异）"
                    y4b = "规则命中（0～1）"
                    l3, l4 = "规则有RAG", "规则无RAG"
                if use_sim_case:
                    y4a, l1, l2 = "综合冗余", "冗余有RAG", "冗余无RAG"
                else:
                    y4a, l1, l2 = "重复句冗余", "冗余有RAG", "冗余无RAG"
            else:
                if use_sim_case and use_chk_case:
                    t4 = "Per-case sim-max redundancy & checklist completeness"
                    y4b = "Checklist score"
                    l3, l4 = "Checklist w/", "Checklist w/o"
                else:
                    t4 = "Per-case rule completeness (zoomed Y)"
                    y4b = "Rule hit rate"
                    l3, l4 = "Rules w/", "Rules w/o"
                if use_sim_case:
                    y4a, l1, l2 = "Sim-max redundancy", "Redund. w/", "Redund. w/o"
                else:
                    y4a, l1, l2 = "Dup-sentence redundancy", "Redund. w/", "Redund. w/o"

            per_case_red_flat = max(max(yr, default=0), max(yn, default=0)) < 1e-5
            x = range(len(ids))
            zmin = min(zr + zn) if (zr or zn) else 0.0
            y_lo = max(0.0, zmin - 0.08)

            if per_case_red_flat:
                fig4, ax4b = plt.subplots(1, 1, figsize=(9, 4.2))
                note = (
                    "重复句字面冗余：本批各病例均为 0（无逐字重复句，未绘折线）。"
                    if use_zh
                    else "Dup-sentence redundancy is 0 for every case; line plot omitted."
                )
                fig4.suptitle(note, fontsize=9, color="#444", y=1.02)
                ax4b.plot(x, zr, "o-", color="#61ddaa", label=l3, linewidth=1.8, markersize=5)
                ax4b.plot(x, zn, "s--", color="#f6bd16", label=l4, linewidth=1.8, markersize=5)
                ax4b.set_ylabel(y4b)
                ax4b.set_xlabel(x_line)
                ax4b.set_xticks(list(x))
                ax4b.set_xticklabels(ids, rotation=35, ha="right", fontsize=9)
                ax4b.set_title(t4)
                ax4b.legend(loc="lower left", fontsize=8)
                ax4b.grid(True, linestyle=":", alpha=0.6)
                ax4b.set_ylim(y_lo, 1.02)
            else:
                fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
                ax4a.plot(x, yr, "o-", color="#7262fd", label=l1, linewidth=1.8, markersize=5)
                ax4a.plot(x, yn, "s--", color="#f6903d", label=l2, linewidth=1.8, markersize=5)
                ax4a.set_ylabel(y4a)
                ax4a.set_title(
                    ("逐病例：综合冗余" if use_zh else "Per-case sim-max redundancy")
                    if use_sim_case
                    else ("逐病例：重复句冗余" if use_zh else "Per-case dup-sentence redundancy")
                )
                ax4a.legend(loc="upper right", fontsize=8)
                ax4a.grid(True, linestyle=":", alpha=0.6)
                ax4a.set_ylim(-0.05, 1.05)

                ax4b.plot(x, zr, "o-", color="#61ddaa", label=l3, linewidth=1.8, markersize=5)
                ax4b.plot(x, zn, "s--", color="#f6bd16", label=l4, linewidth=1.8, markersize=5)
                ax4b.set_ylabel(y4b)
                ax4b.set_xlabel(x_line)
                ax4b.set_xticks(list(x))
                ax4b.set_xticklabels(ids, rotation=35, ha="right", fontsize=9)
                ax4b.legend(loc="lower left", fontsize=8)
                ax4b.grid(True, linestyle=":", alpha=0.6)
                ax4b.set_ylim(y_lo, 1.02)

            fig4.tight_layout()
            aux_line = out_dir / "aux_per_case_line.png"
            fig4.savefig(aux_line, dpi=160)
            plt.close(fig4)
            generated.append(aux_line)
    else:
        print(
            "提示: 未找到冗余/完整性汇总字段，请先运行 eval/run_ablation.py 再出图，以生成 aux_*.png",
            file=sys.stderr,
        )

    print("已生成:")
    for p in generated:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
