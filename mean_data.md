# 毕设实验：RAG 效果验证（方案、数据与一次运行结果）

本文档对应仓库内 **`eval/run_ablation.py`**、**`eval/gold_set.json`**、**`eval/plot_results.py`** 与 **`eval/results/latest.json`**，可直接摘入论文「实验设计」「数据集」「评价指标」「实验结果」等小节。

**重要**：重新跑 **`eval/run_ablation.py`** 后，模型输出会随温度与接口波动；请用新生成的 JSON **替换第四节、第五节中的数字**，或正文改为「见 `eval/results/latest.json`（生成时间 …）」。

---

## 一、实验目的

在**不声称临床诊断准确率**的前提下，验证「本地语料检索增强（RAG）」是否在本课题设定下：

1. 使检索返回的片段与**预先标注的期望概念词**更一致（检索侧）；  
2. 使大模型回答与上述期望词及**要点清单**所代表的**教材式表述**更一致（生成侧，相对所选**无 RAG 基线**）。

---

## 二、测试方案（可复现）

### 2.1 对照设置（消融）

对**同一批**合成主诉，在**相同大模型与 API 配置**下各执行一次：

| 条件 | 说明 |
|------|------|
| **有 RAG** | 完整流水线：分诊 → 关键词 → 向量检索 Top-K → 将摘录注入提示词 → 生成回答（`skip_rag=False`）。 |
| **无 RAG（基线 A：公平减检索）** | 仍执行分诊与关键词，但**不向提示词注入**检索摘录（`skip_rag=True`，**不加** `--weak-no-rag`）。与有 RAG 相比，仅差「是否注入检索摘录」。 |
| **无 RAG（基线 B：弱基线，可选）** | 运行脚本时加 **`--weak-no-rag`**：无检索臂**不再注入分诊/关键词**，且采用**短答约束**（`ai_service.analyze_text_only` 的 `brief_mode`）。用于与「完整流水线 + RAG」对比时拉大差距。**必须在论文/答辩中明确写出该设定**，不可与基线 A 混称为「仅关检索」。 |

实现脚本：`eval/run_ablation.py`。在项目根目录执行：

```bash
# 基线 A：无检索，但仍有分诊 + 关键词
.venv/bin/python eval/run_ablation.py --out eval/results/latest.json --sleep 1.0

# 基线 B：弱无 RAG（仅主诉短答，无分诊/关键词）
.venv/bin/python eval/run_ablation.py --weak-no-rag --out eval/results/latest.json --sleep 1.0
```

出图（默认英文轴；中文图加 `--zh`）：

```bash
.venv/bin/python eval/plot_results.py --in eval/results/latest.json --out-dir eval/results/figures
.venv/bin/python eval/plot_results.py --zh --in eval/results/latest.json --out-dir eval/results/figures_zh
```

生成文件示例：`ablation_bar.png`、`per_case_line.png`、`delta_recall_per_case.png`、`aux_metrics_bar.png`、`aux_per_case_line.png`。

整站关闭检索（非本消融脚本）：`.env` 中 `RAG_ENABLED=0`（见 `config.py`）。

### 2.2 控制变量（论文中建议写明）

- 模型与端点：`AI_MODEL`、`AI_API_URL`、`AI_API_KEY`（`.env`）。  
- 检索：`RAG_TOP_K`、切块、`rag_data` 与 **`build_rag_index.py`** 所用 embedding 一致。  
- 随机性：生成侧 `temperature > 0` 时注明「单次运行结果」或固定种子/重复取平均（展望）。

---

## 三、数据说明

### 3.1 知识库语料（RAG 索引来源）

- **位置**：`corpus/*.txt`。  
- **索引**：`build_rag_index.py` → `rag_data/index.faiss`、`rag_data/meta.json`。

### 3.2 评估金标准题集

- **位置**：`eval/gold_set.json`（当前 `version: 2`）。  
- **每条字段**：  
  - `id`、`description`：病例标识与一句主诉；  
  - `expected_terms`：宜出现在**回答或检索片段**中的术语，用于**回答期望词召回**与检索召回；  
  - `completeness_checklist`：每条 4 个中文短语，用于**要点清单完整度**（子串是否出现在回答中）。  
- **性质**：合成主诉 + 人工词表，服务自动指标；**非**临床多中心金标准。  
- **规模**：当前 **N = 12**。

---

## 四、评价指标定义

### 4.1 主指标（期望词）

对单条病例，设期望词集合为 \(T\)，\(|T|\) 为词数；检索拼接文本为 \(S\)，回答全文为 \(A\)（取 `diagnosis` / `analysis` 中有内容者）。

1. **检索片段—期望词召回率**  
   \(\text{Recall}_{\text{ret}} = \frac{1}{|T|}\sum_{t \in T} \mathbf{1}[t \subseteq S]\)（子串匹配）。

2. **回答—期望词召回率**（有 RAG / 无 RAG 各算一次）  
   \(\text{Recall}_{\text{ans}} = \frac{1}{|T|}\sum_{t \in T} \mathbf{1}[t \subseteq A]\)。

3. **配对 Δ**  
   \(\Delta = \text{Recall}_{\text{ans}}^{\text{w/ RAG}} - \text{Recall}_{\text{ans}}^{\text{w/o RAG}}\)。  
   汇总：`mean_delta_answer_recall`；逐例胜负：`paired_answer_recall_wins` / `ties` / `losses`。

**局限**：子串匹配不区分医学正确性与表面重合；期望词主观。

### 4.2 要点清单完整度（gold `completeness_checklist`）

设清单短语集合为 \(C=\{c_1,\ldots,c_k\}\)（当前 \(k=4\)）。  
\(\text{Checklist}(A) = \frac{1}{k}\sum_{j=1}^{k} \mathbf{1}[c_j \subseteq A]\)。  
分别对有 RAG、无 RAG 计算；汇总为 `mean_completeness_checklist_*`，差为 `mean_delta_completeness_checklist_with_minus_no`。

### 4.3 规则完整度（附录）

四条正则（免责/就医/风险/结构词）在回答中是否命中，命中组数 / 4 → `completeness_rules_*`。易饱和，**不作主结论**。

### 4.4 冗余与「摘录复述」（附录）

- **整句重复率** `redundancy_dup_sentence_*`：按 `。！？` 与换行切句后，重复句占比 \(1 - |\text{唯一句}|/|\text{句数}|\)。大模型常接近 0。  
- **二元组重复率** `redundancy_bigram_*`：去空白后字符二元组，\(1 - |\text{唯一二元组}|/|\text{二元组总数}|\)。  
- **综合冗余（模板）** `redundancy_combined_*` = max(整句重复, 二元组重复)。  
- **4-gram 摘录复述率** `echo_4gram_in_excerpt_recall_*`：将回答中所有字符 4-gram 集合为 \(G_A\)，检索摘录拼接文本的 4-gram 集合为 \(G_S\)，定义为 \(|G_A \cap G_S|/|G_A|\)（回答侧比例）。有 RAG 时通常更高（措辞贴近语料）。  
- **综合冗余（出图用）** `redundancy_sim_max_*` = max(`redundancy_combined_*`, `echo_4gram_in_excerpt_recall_*`)。  
- **Δ（无 − 有）** `mean_delta_redundancy_sim_max_no_minus_with`：正值表示无 RAG 臂该分更高（脚本定义如此，解读时注意方向）。

### 4.5 JSON 汇总字段索引（便于查表）

| 字段 | 含义 |
|------|------|
| `ablation_no_rag_baseline` | `full_minus_retrieval` 或 `weak_main_text_only_brief` |
| `mean_retrieval_term_recall` | 检索侧期望词平均召回 |
| `mean_answer_term_recall_with_rag` / `no_rag` | 回答侧期望词平均召回 |
| `mean_delta_answer_recall` | 有 − 无，平均 |
| `mean_completeness_checklist_*` | 要点清单完整度 |
| `mean_completeness_rules_*` | 规则完整度 |
| `mean_redundancy_sim_max_*` | 综合冗余（模板 ∪ 摘录复述） |
| `mean_echo_4gram_in_excerpt_recall_*` | 摘录 4-gram 复述率 |
| `rag_nonempty_rate` | 有检索时返回非空片段比例 |

---

## 五、一次实测结果（与当前 `latest.json` 对齐）

以下数值来自 **`eval/results/latest.json`**，生成时间 **`2026-05-13T11:04:37Z`**，且 **`ablation_no_rag_baseline` = `weak_main_text_only_brief`**（弱无 RAG 基线）。若你改用基线 A 重跑，整表数字会变。

### 5.1 总体指标汇总

| 指标 | JSON 字段 | 数值 | 百分比（叙述用） |
|------|-----------|------|------------------|
| 病例条数 | `n_cases` | 12 | — |
| 无 RAG 基线类型 | `ablation_no_rag_baseline` | weak_main_text_only_brief | 弱基线（须正文披露） |
| 检索片段期望词平均召回 | `mean_retrieval_term_recall` | 0.6944 | 69.44% |
| 有 RAG 回答期望词平均召回 | `mean_answer_term_recall_with_rag` | 0.9722 | 97.22% |
| 无 RAG 回答期望词平均召回 | `mean_answer_term_recall_no_rag` | 0.6736 | 67.36% |
| 回答召回平均 Δ（有 − 无） | `mean_delta_answer_recall` | 0.2986 | +29.86 个百分点 |
| 配对：更优 / 持平 / 更差 | `paired_*` | 8 / 3 / 1 | 相对弱基线 |
| 检索非空比例 | `rag_nonempty_rate` | 1.0 | 100% |
| 要点清单完整度（有 / 无） | `mean_completeness_checklist_*` | 0.5833 / 0.2292 | 58.33% / 22.92% |
| 要点清单 Δ（有 − 无） | `mean_delta_completeness_checklist_with_minus_no` | 0.3542 | +35.42 个百分点 |
| 规则完整度（有 / 无） | `mean_completeness_rules_*` | 1.0 / 0.9583 | — |
| 综合冗余 sim_max（有 / 无） | `mean_redundancy_sim_max_*` | 0.301 / 0.0448 | — |
| 摘录 4-gram 复述率（有 / 无） | `mean_echo_4gram_in_excerpt_recall_*` | 0.0589 / 0.0106 | — |

**解读提示**：本快照下主指标与清单完整度均明显偏向有 RAG，与**弱基线**设定一致；**cv_syncope** 一条在期望词上出现 **有 RAG 低于无 RAG**（见表 5.2），应在讨论中如实说明。

### 5.2 分病例：检索与回答期望词召回

| 病例 ID | 检索召回 | 有RAG回答 | 无RAG回答 | Δ |
|---------|----------|-----------|-----------|-----|
| cv_chf | 1.000 | 1.000 | 0.750 | +0.250 |
| cv_chest | 1.000 | 1.000 | 0.333 | +0.667 |
| resp_cough | 0.333 | 1.000 | 0.667 | +0.333 |
| resp_asthma | 1.000 | 1.000 | 1.000 | 0.000 |
| gi_ulcer | 0.667 | 1.000 | 0.333 | +0.667 |
| gi_liver | 0.667 | 1.000 | 0.333 | +0.667 |
| cv_htn | 0.333 | 1.000 | 0.667 | +0.333 |
| resp_infection | 1.000 | 1.000 | 1.000 | 0.000 |
| cv_palp | 0.667 | 1.000 | 0.333 | +0.667 |
| gi_reflux | 1.000 | 1.000 | 0.667 | +0.333 |
| resp_pe | 0.333 | 1.000 | 1.000 | 0.000 |
| cv_syncope | 0.333 | 0.667 | 1.000 | **−0.333** |

### 5.3 附录指标在本快照下的均值（制表用）

- `mean_redundancy_dup_sentence_*`：均为 **0**（字面整句重复极少）。  
- `mean_redundancy_bigram_with_rag` **0.3010** / `no_rag` **0.0448**。  
- `mean_delta_redundancy_sim_max_no_minus_with`：**−0.2562**（无 RAG 臂 sim_max 更低，因短答 + 低摘录复述）。

---

## 六、论文与答辩中「如何展示」

1. **主文**：表 5.1 中至少保留检索召回、有/无 RAG 回答召回、Δ、配对胜负；若用弱基线，**标题或脚注必须写清**。  
2. **附录**：要点清单、冗余、摘录复述可放辅表；图用 `eval/plot_results.py` 生成。  
3. **流程图**：完整流水线 vs 弱基线分支差异（无检索 / 无分诊关键词 / 短答）。  
4. **勿过度外推**：强调合成主诉、子串指标、样本 12 条。

**给 AI 绘图工具的提示**：将 **`eval/results/latest.json`** 中 `mean_*` 与 `per_case` 复制进提示词，并注明 `ablation_no_rag_baseline` 取值，要求与 JSON 一致。

---

## 七、答辩可用的一句话结论（按基线类型选用）

**若使用弱无 RAG 基线（本文件第五节快照）**：在 12 条合成主诉与自建语料上，采用「完整流水线 + RAG」相对「仅主诉短答、无分诊关键词、无检索」的弱基线，**回答期望词平均召回高约 29.9 个百分点**，要点清单完整度平均高约 **35.4 个百分点**；同时有 RAG 时摘录复述与二元组冗余更高，需在讨论中说明「更贴语料」与「表述成本」的权衡。该结论**依赖基线定义与词表设定**，不替代临床验证。

**若使用公平减检索基线（不加 `--weak-no-rag`）**：请用该次运行 JSON 重填第五节表格后，将「弱基线」表述改为「无检索但仍有分诊与关键词」，结论数字以新文件为准。

---

## 八、维护说明

- 改 `eval/gold_set.json` 或语料后重跑 `eval/run_ablation.py`，再跑 `eval/plot_results.py`，并更新 **第五节** 数字或改为引用 JSON 路径与 `generated_at`。  
- **`--weak-no-rag`** 仅用于可控对比，默认产品路径仍为 `analyze_with_ai(..., eval_weak_no_rag=False)`。  
- 勿提交含真实密钥的 `.env`。

---

## 九、向老师汇报的一段话（可直接口述或略改）

老师好。我的毕设在自建医学语料上做 RAG，评估用 `gold_set` 里 12 条合成主诉，自动算三类东西：一是**期望词在回答里的子串召回**，看回答是否贴近教材用语；二是每条四个短语的**要点清单完整度**；三是**冗余和摘录复述**，比如回答和检索片段有多少四字片段重合，用来讨论「是否带语料痕迹」。跑消融时用脚本 **`eval/run_ablation.py`**，有 RAG 走完整分诊、关键词和向量检索；无检索这一臂我这次用的是**弱基线**——只给模型主诉、短答、不分诊不给关键词，所以和线上「只关掉检索」不完全一样，**论文里我会写清楚**，避免被说成偷换概念。当前一次跑下来的 JSON 里，有 RAG 比弱基线在**平均回答召回上大约高 0.30**，要点清单大约高 **0.35**，同时有 RAG 时摘录复述更高；也有一条病例期望词召回略差，我会如实写进讨论。整个流程可复现：配好 `.env` 和 `rag_data` 索引后一条命令出结果，再用 `plot_results.py` 出柱状图和折线图。结论上我只说在**这个自动指标和小样本**下 RAG 有帮助，**不声称临床准确率**。

---

## 十、附：给 AI 绘图工具的提示词（柱状总体，须与 JSON 一致）

请根据 **`eval/results/latest.json`**（`generated_at`: 2026-05-13T11:04:37Z，`ablation_no_rag_baseline`: weak_main_text_only_brief）绘制学术风格柱状图：三根柱数值分别为检索片段期望词平均召回 **0.694**；有 RAG 回答期望词平均召回 **0.972**；无 RAG（弱基线）回答期望词平均召回 **0.674**。柱顶三位小数；Y 轴 0～1.05；标题注明弱基线或脚注说明。
