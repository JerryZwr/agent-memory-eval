# 发现（findings）

本文分两部分。

- **第一部分：评测 AI 记忆与 Agent 时容易犯的错。** 每一条都给出两样东西：本仓库里的一个反例，也就是真实运行中撞上的、或者差点撞上的情况；以及一个把它固定下来的测试用例。
- **第二部分：live-v1 的实测数据**（F1–F8）。第一部分引用的数字都出自这里。

数字来自 2026-10-19 至 10-20 的 live 运行。本地结果文件在 `runs/`，不入库。公开摘要在 [`results/live-v1/summary.json`](../results/live-v1/summary.json)，内容是逐题结局与判分，以及每个阶段文件的 records_sha256，不含数据集原文。公开版报告见 [GitHub Pages](https://jerryzwr.github.io/agent-memory-eval/live/live-v1/)。

**怎么读第一部分。** 这里的错都有一个共同点：犯了它不会报错，只会得到一个格式正常、看起来合理的数字（[METHODOLOGY](METHODOLOGY.md) P1）。所以每一条后面都跟着一个测试：当框架犯这个错时，这个测试会失败。

## 第一部分 · 评测 AI 记忆与 Agent 时容易犯的错

### 题目与数据

**1. 不设对照组，只看绝对分数**
- 后果：60% 看起来像是记忆系统的功劳。实际上，什么都不存的 null 在 A 类就有 27%（评分员 v1）到 47%（v2）。原因是 110 题里有 77 道选择题，可以凭题面和选项猜。见 [F1](#f1)。
- 测试：[`test_demo.py`](../tests/unit/test_demo.py) `::test_bm25_clearly_beats_null_on_questions_with_evidence`。报告里每个 arm 都和 null 配对比较；没有 null 时，报告写明"无法给出相对对照组的结论"。

**2. 拿全部题目去问，不按证据窗口筛题**
- 后果：只灌入了前 90 天的语料，却去问证据在第 200 天的题，必然答错，结果被稀释。更危险的是证据跨越窗口边界的题（B 类）：检索成功了，但材料残缺，看起来和"排序缺陷"一模一样。live-v1 只提问 A 类 110 题和 D 类 78 题。见[运行概况](#run)。
- 测试：[`test_windows.py`](../tests/unit/test_windows.py) `::test_b_is_excluded_unless_explicitly_included`、`::test_default_run_asks_only_scored_classes`；证据位置恢复不了的题归为 E 类，见 `::test_unrecoverable_is_e`。

**3. 用"返回非空"判定检索成功**
- 后果：对记忆系统来说，提问本身就可能被存成记忆。下一次检索返回了提问者自己的问题，结果非空，探针亮绿灯（METHODOLOGY 1.12）。
- 测试：[`test_breakdown_and_baseline.py`](../tests/unit/test_breakdown_and_baseline.py) `::test_empty_retrieval_is_a_miss_and_non_empty_is_not_automatically_a_hit`；[`test_probes.py`](../tests/unit/test_probes.py) `::test_non_empty_but_wrong_retrieval_fails_the_corpus_probe`。

**4. 相信组件"正常返回"就是正常工作**
- 后果：本仓库实际撞上了三次，三次都没有报错。
  - mem0 解析不了抽取回复时，只写一行日志，然后存入空记忆。结果是一个空库，看起来却和正常灌入的库一样。
  - DeepSeek 的思考模式会静默忽略 `temperature`（[D-009](decisions.md)）。
  - 常用的 BM25 库 `rank_bm25`：一个词恰好出现在一半文档里时，idf 为 0；超过一半时为负。在小语料上，这会让检索静默失效。
- 测试：[`test_mem0_arm.py`](../tests/unit/test_mem0_arm.py) `::test_replies_without_memories_for_every_batch_are_refused`、`::test_extraction_outage_surfaces_as_infra_error_not_a_silent_empty_store`；[`test_arms.py`](../tests/unit/test_arms.py) `::test_term_in_half_or_more_documents_still_scores`（自己实现 Lucene BM25，用手算值核对）。

### 结局与判分

**5. 把"没交付"和"环境坏了"算成答错**
- 后果：被测系统替基础设施背锅，而"根本不交付"这个更严重的问题被淹没在正确率里。agent-v1 在 20 个任务里有 11 个跑满预算也没调用 `final_answer`，这是"不收敛"，不是"答错"。见 [F6](#f6)。
- 交付的判定还要防另一个方向：用交付动作发出的一句"收到，稍后汇报"不是答案。
- 测试：
  - [`test_outcome_and_scoring.py`](../tests/unit/test_outcome_and_scoring.py) `::test_infra_failure_wins_over_any_text`、`::test_acknowledgement_is_undelivered`。
  - [`test_d1_delivery.py`](../tests/agent/test_d1_delivery.py) `::test_an_acknowledgement_through_final_answer_is_not_the_answer`。
  - [`test_d5_fault_injection.py`](../tests/agent/test_d5_fault_injection.py) `::test_a_bug_in_the_agent_is_agent_error_not_env_failure`。
  - 同一文件的 `::test_calling_a_nonexistent_tool_is_the_agents_fault_not_the_environments`：调用不存在的工具，是 Agent 自己的错，不是环境故障。

**6. 评分员调用失败时返回 False**
- 后果：一次超时被静默记成 WRONG，事后无法分辨。live-v1 中 DeepSeek 在 JSON 模式下两次返回无效 JSON，两次都重试成功，并记录在账本里。见 [F7](#f7)。
- 测试：[`test_llm_judge.py`](../tests/unit/test_llm_judge.py) `::test_call_failure_is_a_judge_error_not_wrong`、`::test_unusable_replies_are_judge_errors`；计分时拒绝缺少判分的条目，见 [`test_outcome_and_scoring.py`](../tests/unit/test_outcome_and_scoring.py) `::test_refuses_delivered_without_verdict`。

**7. 只用一个评分员口径就下结论**
- 后果：同一份答案，换一个评分员提示词，D 类的结论从"记忆系统显著更不容易幻觉"变成"没有差异"。见 [F3](#f3)。两个口径还各自违反自己写下的规则，方向相反：null 在 v2 下多出的 22 题，恰好等于 v2 违反规则的次数。见 [F4](#f4)。
- 测试：[`test_live_pipeline.py`](../tests/unit/test_live_pipeline.py) `::test_v2_rejudges_the_same_answers`（让提示词成为唯一变量）；[`test_judge_agreement.py`](../tests/unit/test_judge_agreement.py) `::test_different_answers_are_refused`、`::test_rule_check_flags_verdicts_that_contradict_the_rules`。
- 注意：这仍然不是人工校准。本项目没有人工标注，所以不报告评分员准确率。见[局限](#limits)。

**8. 选择题的答案从长文本里"搜"出一个字母**
- 后果：在 "A good guess is C" 这样的回复里，冠词 "A" 会被当成所选的选项。静默取第一个字母，就会把这道题判成选了 A。
- 测试：[`test_judging.py`](../tests/unit/test_judging.py) `::test_single_letter_dug_out_of_text_is_flagged`、`::test_several_letters_yield_no_letter_and_a_flag`。这类情况打上 `ambiguous_extraction` 标记单独统计，而不是静默取第一个。

### 统计与结论

**9. 小样本、不配对，就说"提升了 X%"**
- 后果：agent-v2 把端到端成功率从 4/17 提到 7/17，看起来提升了 17 个百分点。但结论不一致的配对只有 3 个，McNemar 精确检验 p = 0.25。不一致的配对少于 6 个时，任何结果都达不到 0.05。见 [F6](#f6)。mem0 比 bm25 高 6–7 个百分点，同样不显著。见 [F2](#f2)。
- 测试：[`test_stats.py`](../tests/unit/test_stats.py) `::test_power_table_from_methodology`（把"6 个才可能显著"写成测试）；[`test_compare.py`](../tests/unit/test_compare.py) `::test_without_significance_it_never_says_better`（结论措辞由代码生成，不显著时不会出现"更好"）。

**10. 只看总分；把性质不同的题混在一起检验**
- 后果：把 top_k 从 5 改成 10 后，D 类总分不变，但实际上 5 题变对、5 题变错，正好互相抵消。A、D 两类合并检验时 p = 0.052，把 A 类上显著的效果（p = 0.013）稀释掉了。见 [F8](#f8)。
- 测试：[`test_breakdown_and_baseline.py`](../tests/unit/test_breakdown_and_baseline.py) `::test_all_four_statuses`、`::test_config_change_produces_a_real_diff`；语料或判分口径不同时拒绝 diff，见 `::test_different_corpus_is_refused_unless_explicitly_allowed`、`::test_different_judge_version_is_refused`。

**11. 统计函数没有和已知答案核对**
- 后果：错一点的 Wilson 区间不会报错，只会输出一个看起来正常的区间。
- 测试：[`test_stats.py`](../tests/unit/test_stats.py)。用 Newcombe (1998) 的 Wilson 例子、教科书上的 κ 例子、McNemar 的 2 × 0.5ⁿ 表，逐个核对。

### 环境

**12. 探针只查"能连上"；上游坏了，下游记"失败"**
- 后果：只读故障时读一直正常，只做 `SELECT 1` 的探针会一路亮绿灯。而把"没测"记成"失败"，会让人去查一个不存在的问题。
- 测试：
  - [`test_probes.py`](../tests/unit/test_probes.py) `::test_memory_probe_writes_to_a_scratch_namespace_only`：检查"可写"，并且写入临时命名空间。
  - 同一文件的 `::test_memory_outage_marks_corpus_not_run_and_never_calls_it`。
  - [`test_live_pipeline.py`](../tests/unit/test_live_pipeline.py) `::test_red_post_probe_marks_the_round_not_measured`。
- live-v1 三个 arm 的前置、灌入后、后置探针全部为绿，所以整轮记为 measured。见[运行概况](#run)。

### Agent

**13. 离线的脚本化测试全绿，就以为 Agent 没问题**
- 后果：真实模型暴露了两个缺陷，六种脚本化策略都没有模拟到其中一个：路由提示里带了选择题选项，模型在路由这一步直接答题。另一个缺陷（不停换措辞检索）正是 `stubborn_searcher` 模拟的行为。见 [F6](#f6)。
- 做法：离线测试和真实批量评测共用同一套判据 [`agent/checks.py`](../src/agentmemeval/agent/checks.py)，真实运行时同一个检查能直接抓住。
- 测试：[`test_policy_coverage.py`](../tests/agent/test_policy_coverage.py) `::test_defective_policy_is_caught_by_its_check`，确保每种缺陷策略都有一个检查会失败。

**14. 看路由标签，不看路由之后的行为**
- 后果：标签是 memory，但随后调用的是 `web_search`，端到端也可能碰巧"答对"。
- 测试：[`test_d3_routing.py`](../tests/agent/test_d3_routing.py) `::test_right_label_wrong_behaviour_is_caught`、`::test_the_first_routing_decision_is_the_one_scored`、`::test_expected_labels_come_from_the_declared_rule`（期望标签取自 Agent 自己声明的规则，不取评测者的主观判断）。

**15. 不检查观测数据本身**
- 后果：耗时拆解和轨迹断言都建立在 trace 上。一个缺 span 或 span 没关闭的 trace，会让"工具耗时 1.8 ms"这种数字失去意义。
- 测试：[`test_d8_trace_integrity.py`](../tests/agent/test_d8_trace_integrity.py) `::test_integrity_check_reports_a_deliberately_broken_trace`。故意构造缺 span、未关闭、越出父 span、孤儿 span 四种坏 trace，确认检查能报出来。

### 工程

**16. 把数字手抄进 README**
- 后果：重新发布结果之后，README 首屏的数字就过期了。
- 测试：README 的结果表和结果图由 `agentmemeval readme` 从 [`summary.json`](../results/live-v1/summary.json) 生成。[`test_headline.py`](../tests/unit/test_headline.py) `::test_committed_readme_and_figure_match_the_published_summary` 在 CI 里检查两者一致，手改一个数字就会失败。

## 第二部分 · live-v1 实测数据

<a id="run"></a>

### 运行概况

| 项 | 值 | 来源 |
|---|---|---|
| 语料 | EverMemBench 004 批（revision `a6b210a3`），前 90 天：3404 条消息 | `corpus_id = emb-004@a1607d39c0:2025-01-09..2025-05-14` |
| 提问 | A 类 110 题 + D 类 78 题 = 188 题（B、C、E 类不提问，METHODOLOGY 1.3） | 各 arm 的 `judge.json` |
| 被测 arm | null（对照组）、bm25（本地）、mem0（本地开源版，向量检索） | `runs/live-v1/*/run.json` |
| 答题 / 评分员 | `deepseek-flash` / `deepseek-v4-pro`，都关闭思考模式；评分员口径 v1 与 v2 判同一份答案 | 结果文件元数据 |
| 环境探针 | 三个 arm 的前置、灌入后、后置探针全部为绿，整轮状态 measured | `runs/live-v1/*/probes.json` |
| 费用 | live-v1 账本共 1576 次付费调用，¥5.61（上限 ¥30 / 4000 次）；mem0 灌入占 ¥3.25。top_k 实验另用一本独立账本（见 F8） | `runs/live-v1/usage.jsonl` |

样本量说明：A 类 n = 110、D 类 n = 78，区间宽度约 ±9 个百分点；arm 之间只用配对的 McNemar 精确检验比较（METHODOLOGY 1.7）。

<a id="f1"></a>

### F1 · 两个检索 arm 都显著高于对照组，且在两个评分员口径下都成立（A 类）

| 口径 | null | bm25 | mem0 | bm25 vs null | mem0 vs null |
|---|---|---|---|---|---|
| v1 | 30/110 = 27% [20%, 36%] | 59/110 = 54% [44%, 63%] | 66/110 = 60% [51%, 69%] | 只有 bm25 对 35 / 只有 null 对 6，p < 0.001 | 38 / 2，p < 0.001 |
| v2 | 52/110 = 47% [38%, 57%] | 67/110 = 61% [52%, 70%] | 75/110 = 68% [59%, 76%] | 29 / 14，p = 0.032 | 32 / 9，p < 0.001 |

- 结论：在本语料、本配置下，有检索比没检索答得好，这一点**不依赖评分员口径**。
- null 的 27% 或 47% 不是 0：A 类 110 题里有 77 道选择题，可以凭题面和选项猜，这正是需要对照组的原因（METHODOLOGY 1.1）。
- 来源：`runs/live-v1/{null,bm25,mem0}/judge.json`、`runs/live-v1/v2/*/judge.json`；配对检验由 `analysis/compare.py` 计算，见报告“与 null 对照组的配对比较”。

<a id="f2"></a>

### F2 · mem0 与 bm25 之间未观察到显著差异

- A 类，mem0 vs bm25：v1 下只有 mem0 对 15 题、只有 bm25 对 8 题，p = 0.210；v2 下 16 / 8，p = 0.152。点估计 mem0 高 6–7 个百分点，但**不能说 mem0 更好**。
- 附加的局限：mem0 本次只用了向量检索（混合检索的依赖未安装，D-011）；EverOS 没有做（D-009）。
- 来源：同 F1。

<a id="f3"></a>

### F3 · D 类（无证据题）的结论随评分员口径翻转

| 口径 | null | bm25 | mem0 | 记忆 arm vs null |
|---|---|---|---|---|
| v1 | 47/78 = 60% [49%, 70%] | 62/78 = 79% [69%, 87%] | 65/78 = 83% [74%, 90%] | 都显著“更好”（p ≤ 0.001） |
| v2 | 78/78 = 100% [95%, 100%] | 78/78 = 100% [95%, 100%] | 77/78 = 99% [93%, 100%] | 未观察到显著差异（不一致的配对只有 0–1 个） |

- 在 v1 下读报告，会得出“记忆系统更不容易幻觉”的结论；在 v2 下这个差异不存在。差异的来源是**评分员**，不是被测系统：v1 把 null 的 31 个正确拒答判成了 WRONG（见 F4）。
- 这就是 METHODOLOGY 1.5 描述的问题：否定类标准答案带了背景（“X 没参与，那段时间他在做 Y”），宽松的评分员把背景也当成必须复述的内容。
- 来源：同 F1；规则一致性检查见 F4。

<a id="f4"></a>

### F4 · 两个评分员口径各自违反自己的规则，方向相反

同一份答案用两个口径判分，提示词是唯一变量（rejudge 复制了答案阶段文件，records_sha256 一致）。

| arm | 两口径一致率 | Cohen's κ [95% bootstrap] | v1：无证据题纯拒答却判 WRONG | v2：可回答题拒答且不含标准答案却判 CORRECT |
|---|---|---|---|---|
| null | 58/111 = 52% [43%, 61%] | 0.15 [0.07, 0.25] | 31 | 22 |
| bm25 | 87/111 = 78% [70%, 85%] | 0.37 [0.20, 0.54] | 16 | 8 |
| mem0 | 88/111 = 79% [71%, 86%] | 0.35 [0.16, 0.53] | 12 | 9 |

- v2 新增的拒答规则（规则 4）本应只适用于否定类标准答案，模型把它推广到了可回答的题。对 null 来说，v2 把 A 类正确数从 30 推到 52，**多出的 22 题恰好等于 v2 违反规则 1 的次数**。
- 这些计数来自**按评分员规则的自动一致性检查**，不是人工标注（D-010）。本项目没有人工校准，因此不报告任何“评分员准确率”或人工 κ（D-009）。
- 按 METHODOLOGY 1.6，**没有**在这批样本上继续改 v2；若做 v3，需要并行报告并用留出样本评估。
- 对结论的影响：F1 的排序（检索 arm > null）在两个口径下都成立；绝对分数和 D 类结论则依赖口径，读数时要带着口径一起读。
- 来源：`analysis/judge_agreement.py` 对 `runs/live-v1/*` 与 `runs/live-v1/v2/*` 的计算，见报告“评分员口径对比”。

<a id="f5"></a>

### F5 · 查询形态：把选项拼进查询提高了证据命中率

同一批 110 道 A 类题，用三种查询各检索一次（只检索，不写入，不调用 LLM）：

| arm（判据） | ① 只用题干 | ② 题干 + 全部选项 | ③ 自动短查询 |
|---|---|---|---|
| bm25（命中证据消息） | 74/110 = 67% [58%, 75%] | 99/110 = 90% [83%, 94%] | 32/110 = 29% [21%, 38%] |
| mem0（命中证据当天） | 80/110 = 73% [64%, 80%] | 97/110 = 88% [81%, 93%] | 54/110 = 49% [40%, 58%] |

- 空结果率三种形态都是 0/110。
- “长选项会拖垮检索”的假设在这两个 arm 上**没有得到支持**；反而是截短的查询明显变差。选项文本往往与证据原文共享专有名词，这可能是原因（未验证）。
- 两个 arm 的命中判据不同（mem0 存的是抽取后的记忆，只能按日期判），两行之间不能直接比较。
- ③ 是自动截取的实词，不是人工改写（D-010）。
- 来源：`runs/live-v1/{bm25,mem0}/query_shape.json`。

<a id="f6"></a>

### F6 · Agent 层：真实模型暴露了两个脚本化策略没有暴露的缺陷

同一个 Agent、同一套判据（`agent/checks.py`），20 个任务（13 道记忆题 + 4 道常识题 + 3 句闲聊），真实时钟，流式调用测 TTFT。

| 提示词版本 | 交付率 | 预算触发率 | 端到端成功率（有标准答案的 17 题） | 路由判据未通过 |
|---|---|---|---|---|
| agent-v1 | 9/20 = 45% [26%, 66%] | 11/20 = 55% [34%, 74%] | 4/17 = 24% [10%, 47%] | 6 |
| agent-v2 | 14/20 = 70% [48%, 85%] | 6/20 = 30% [15%, 52%] | 7/17 = 41% [22%, 64%] | 4 |

- 缺陷 1：路由提示里带了选择题的选项，模型在 5 道选择题上**直接答题**（输出 “b”“c”）而不是给出路由类别；路由判据抓住了它。脚本化策略不读选项，所以离线测试暴露不了。
- 缺陷 2：模型在 11/20 个任务上不断换措辞调用 `memory_search`，直到同一工具连续调用的预算把它停下，从不调用 `final_answer`：这正是 `stubborn_searcher` 模拟的行为，在真实模型身上出现了。
- agent-v2 修复（路由提示只给题干；要求最多检索两次后作答）后：逐题配对，3 题由错变对、0 题由对变错，McNemar 精确检验 p = 0.25。不一致的配对只有 3 个，少于 6 个时任何结果都达不到 0.05，所以只能说**未观察到显著差异**，不能说“端到端成功率提升了 17 个百分点”。
- 修复同时引入了新问题：2 道记忆题被路由到了 chitchat。如实列出。
- 耗时（真实时钟，平均每任务）：agent-v1 总 3532 ms，其中 LLM 3530 ms、工具 1.8 ms（bm25 在本地）、框架约 0.2 ms；平均 TTFT 700 ms。在这个配置下瓶颈完全在 LLM。
- 来源：`runs/live-v1/agent/live.json`（agent-v1）、`runs/live-v1/agent/live-agent-v2.json`（agent-v2）。

<a id="f7"></a>

### F7 · 费用结构

| 用途 | 调用次数 | 费用 |
|---|---|---|
| mem0 抽取（252 批 + 重试） | 254 | ¥3.25 |
| 答题（3 个 arm × 188 题） | 564 | ¥0.75 |
| 判分 v1 / v2（只判开放题；相同请求走缓存） | 304 / 304 | ¥0.75 / ¥0.64 |
| Agent（两个提示词版本，各 20 个任务） | 140 | ¥0.21 |
| 探针 | 10 | ¥0.01 |

- 约三分之二的输入 token 命中了服务端缓存（mem0 的抽取提示词前缀固定），按缓存价计费。
- 重试 2 次（DeepSeek 在 JSON 模式下两次返回的不是有效 JSON，第 3 次成功），全部记录在账本里并打印了警告；没有失败。
- 来源：`runs/live-v1/usage.jsonl`（`agentmemeval usage --ledger ...`）。

<a id="f8"></a>

### F8 · 冻结基线后改一个配置：top_k 从 5 改为 10 的逐题 diff

把 bm25（top_k = 5，评分员 v1）冻结为基线 `baselines/live-v1-bm25.json`，只改 top_k = 10 重跑同一批 188 题（同一语料、同一题集、同一评分员；`arm_config_hash` 从 `c14294558d90` 变为 `b622c3a1d329`），逐题对比（METHODOLOGY 1.10）：

| 类别 | n | fixed | regressed | unchanged | 精确 McNemar |
|---|---|---|---|---|---|
| A 类 | 110 | 14 | 3 | 93 | p = 0.013 |
| D 类 | 78 | 5 | 5 | 68 | p = 1.000 |
| 合计 | 188 | 19 | 8 | 161 | p = 0.052 |

- A 类：top_k = 10 下 bm25 为 70/110 = 64% [54%, 72%]（基线 59/110 = 54% [44%, 63%]），配对检验显著。
- D 类：总分不变（62/78），但有 10 道题的结局互换（5 题变对、5 题变错）。**只看总分会以为“没影响”**，逐题 diff 才看得到这种抵消。
- 合并 A、D 两类时 p = 0.052，达不到 0.05：把性质不同的两类题混在一起，会稀释 A 类上真实存在的效果（METHODOLOGY 1.8）。
- 代价：答题提示更长，这次重跑 301 次调用、¥0.57（独立账本，上限 ¥3）。
- 来源：`baselines/live-v1-bm25.json`、`runs/live-v1-k10/bm25/judge.json`、`runs/live-v1-k10/usage.jsonl`；`agentmemeval baseline diff --run runs/live-v1-k10/bm25 --baseline baselines/live-v1-bm25.json`。

<a id="limits"></a>

### 局限

- 评分员**未经人工校准**；本文所有正确率都是某个评分员口径下的正确率。F4 表明口径本身有系统性偏差。
- 只有一个真实被测记忆系统，且 mem0 只用了向量检索。
- 单次运行：温度 0、关闭思考模式，但服务端仍可能有非确定性，没有做重复运行。
- 只用了 004 批前 90 天、A + D 类题；结论不外推到全量 EverMemBench。
