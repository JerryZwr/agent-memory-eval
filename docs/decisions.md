# 决策记录（decisions）

按时间追加，不改写旧条目。每条写明：日期、决定了什么、为什么、影响哪些已有结果。
改动 [METHODOLOGY.md](METHODOLOGY.md) 的定义时必须在这里记一条（见 CLAUDE.md §2.5）。

---

## D-001 · 2026-10-18 · 仓库许可证：MIT

- **决定**：代码以 MIT 许可发布（PLAN §3 第 2 项的推荐选项）。
- **为什么**：作品集项目，希望别人能自由阅读、复用；MIT 最简单、最常见。
- **影响**：无已有结果。作者如需改用其他许可证，替换 `LICENSE` 和 `pyproject.toml` 的 `license` 字段即可。

## D-002 · 2026-10-18 · EverMemBench 数据集许可证核实

**核实到的事实**（2026-10-18 通过 HF / GitHub API 查询）：

| 项 | 结论 | 来源 |
|---|---|---|
| HF 数据集 | `EverMind-AI/EverMemBench-Dynamic`，未设门槛（`gated: false`），无需登录 | https://huggingface.co/api/datasets/EverMind-AI/EverMemBench-Dynamic |
| 数据许可证 | dataset card 声明 `license: apache-2.0` | 同上，`cardData.license` |
| 许可证何时加上 | commit `a6b210a3`（2026-07-16，"Set license to apache-2.0"）。**更早的 revision 没有声明许可证** | https://huggingface.co/api/datasets/EverMind-AI/EverMemBench-Dynamic/commits/main |
| 最后一次数据变更 | commit `69106a31`（2026-02-25）；之后只改了 README | 同上 |
| 额外条款 | dataset card 里没有找到关于再分发、缓存、引用、非商用的额外条款；只要求引用论文（BibTeX） | dataset card |
| GitHub 评测代码 | https://github.com/EverMind-AI/EverMemBench **没有 LICENSE 文件**，API 返回 `license: null` | https://api.github.com/repos/EverMind-AI/EverMemBench |
| 论文 | *Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues*，arXiv:2602.01313，KDD 2026 | https://arxiv.org/abs/2602.01313 |

**决定**：

1. **下载脚本固定 revision 为 `a6b210a32248e841967b7b64a64281d2ff3f669d`**：这是第一个声明了许可证的 revision；仓库名带 "Dynamic"、数据文件被重新上传过，不固定 revision 会让同一个 `corpus_id` 背后的数据悄悄变化。脚本在下载前检查该 revision 的许可证字段，不是 `apache-2.0` 就拒绝下载；下载后写 `MANIFEST.json`（revision、许可证、每个文件的 sha256）。
2. **可以在本地缓存数据**（`data/raw/`，已 gitignore）。Apache-2.0 允许复制和再分发，但**本仓库仍然不提交原始数据**（CLAUDE.md §2.3 的项目规则比许可证更严）：仓库小、来源唯一，避免承担再分发时保留 NOTICE 等义务。
3. **报告里可以引用个别原题原文**，同时注明出处（数据集名称 + revision + Apache-2.0 + 论文引用）。默认仍以题号为主，只在逐题明细和失败归因示例里引用原文。
4. **不复制、不改写 EverMemBench GitHub 仓库里的评测代码**：它没有声明许可证，默认保留所有权利。本项目只按论文和数据格式独立实现。

**影响**：无已有结果。PLAN §3 第 1 项据此关闭。

**引用**（报告和 README 中使用）：

```bibtex
@inproceedings{hu2026evermembench,
  title     = {Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues},
  author    = {Chuanrui Hu and Tong Li and Xingze Gao and Hongda Chen and Yi Bai and Dannong Xu and Tianwei Lin and Xiaohong Li and Yunyun Han and Jian Pei and Yafeng Deng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2026},
  eprint    = {2602.01313},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2602.01313}
}
```

## D-003 · 2026-10-18 · 数据集实际结构（待作者确认数据来源，M1 前决定）

M0 核实许可证时顺带确认了数据结构。**这条不改 METHODOLOGY 的定义**，只记录事实，并提出 M1 要定的数据来源。

**事实**（基于 revision `a6b210a3` 的文件，计数用本地脚本核对过）：

| | 当前格式 `0N/`（N = 1–5，5 个 topic） | 旧格式 `dataset/NNN/`（004、005、010、011、016） |
|---|---|---|
| 文件 | `dialogue.json`、`qa_0N.json` | `dialogue_en.json`、`qa_NNN.json` |
| 题量 | 共 2400 题（与论文摘要一致），每个 topic 467–493 题 | 每批约 610–633 题；**004 批正好是 626 题，其中 389 道选择题** |
| 证据位置 | 每题有 `R`：`[{date, group, message_index}]`，`message_index` 形如 `"1, 4-6"` | **没有 `R`** |
| 无证据题（F_HL） | **没有** | 有（004 批 78 道） |
| 题型 | 由题号前缀给出：F_SH / F_MH / F_TP / MA_C / MA_P / MA_U / P_Style / P_Skill / P_Title | 同左，外加 F_HL |

- PLAN 和 METHODOLOGY 里"626 道题 / 389 道选择题"对应的是**旧格式 004 批**，不是整个基准。
- 004 批的 548 道非 F_HL 题中，**445 道能按归一化后的题干（合并空白、忽略大小写）在 `01/qa_01.json` 中找到同题**，从而恢复 `R`。这正是 METHODOLOGY 1.3 预想的"按归一化题目文本 join 恢复证据位置"；恢复不了的约 103 道归 E 类。

**待作者决定**（推荐选项在前）：

1. **推荐**：评测语料用旧格式 004 批（含 D 类反幻觉题），证据位置从 `01/` 按题干 join 恢复，恢复不了的归 E 类。与现有 PLAN / METHODOLOGY 的表述完全一致。
   - M1 实现加载器前要先核实：`dataset/004/dialogue_en.json` 和 `01/dialogue.json` 的日期、群、消息顺序是否一致。不一致时，`R` 里的 `message_index` 不能直接套到 004 的对话上，需要另议。
2. 备选：用当前格式 `01/`（证据位置齐全），但这样**没有 D 类题**，反幻觉只能靠合成 fixture 和 Agent 层来测。

**影响**：无已有结果。PLAN M5 第 3 项里"A 类 + D 类约 50 道题"依赖这个决定。

## D-004 · 2026-10-18 · 评测语料定为 004 批，证据位置按题干从 01 批恢复（D-003 的决定）

- **决定**：作者选择 D-003 的推荐选项：语料用 `dataset/004/`（`dialogue_en.json` + `qa_004.json`），证据位置从 `01/qa_01.json` 的 `R` 字段恢复。
- **核实了两份对话能否对齐**（M1 实现加载器前，用本地脚本逐条比对，revision `a6b210a3`）：
  - 两份对话日期集合相同（254 天）；每个（日期, 群）的消息条数相同，共 10222 条；按顺序逐条比对，**时间戳全部相同**。
  - 差异只涉及人名：`speaker` 字段有 4 对改名，共 762 条消息；正文有 811 条消息不同，按词做 diff 后，每一处改动都只涉及大写开头的词，看样本是人名拼写的统一或姓名语序的调换（如 `Xin-jie Li` → `Xinjie Li`、`Zhou Lizhen` → `Lizhen Zhou`）。没有增删消息。
  - 01 的 `message_index` 是从 1 开始的连续整数；同一（日期, 群）内时间戳没有重复；`R` 的 3301 条引用都能解析，且全部指向存在的消息。
  - 结论：01 的对话是 004 的人名修订版，`R` 可以映射到 004 的对话上。
- **join 规则**（实现在 `data/loaders.py`，有单元测试）：
  1. 题目按**归一化题干**（合并空白、忽略大小写）匹配；两边归一化后的题干都没有重复。
  2. 匹配上之后还要求**题型前缀相同、归一化后的标准答案相同**，否则不采用该 `R`，归 E 类并记下原因。理由：答案变了，说明这道题被修订过，原来的证据不一定还成立（P2：拒绝胜过猜测）。
  3. `R` 里的每条引用 `(日期, 群, message_index)` 先在 01 的对话里找到那条消息，再用 **(日期, 群, 时间戳)** 找到 004 里对应的消息，**不按位置对应**。找不到或不唯一时，整道题归 E 类。
  4. F_HL 题是无证据题（D 类），不需要 `R`。
- **结果**：004 批 626 题 = 444 题恢复了证据位置 + 78 题 F_HL（无证据）+ 104 题无法恢复（E 类）。无法恢复的 104 题中，103 题在 01 里找不到同题，1 题标准答案不同。
- **另一个发现**：`qa_004.json` 的 `metadata` 写 `single_hop: 50`，实际 F_SH 题只有 49 道。加载器以实际题目为准，并在加载报告里记一条警告。
- **影响**：无已有结果。PLAN M5 第 3 项的"A 类 + D 类约 50 道题"从这 444 + 78 题里按窗口筛选。

## D-005 · 2026-10-19 · 发布的报告不单独做敏感词扫描

- **决定**：不把本地敏感词清单放进 GitHub Actions secret，也不在发布到 GitHub Pages 前再扫描一遍生成的报告。数据集在结果文件里的名称定为 `emb-<批次>`（如 `emb-004`）。
- **为什么**：Pages 上的报告只由 CI 用**已提交的内容**（代码、模板、合成语料）离线生成，这些输入在提交时都已经过 pre-commit 的敏感词检查，生成的报告不会引入仓库里没有的文本。多维护一份云端清单和扫描步骤，增加的复杂度和泄露面（清单本身也是敏感信息）都大于收益。
- **这个简化成立的前提**（后续里程碑必须遵守）：
  1. 发布流程不读取任何未提交的输入（本地 `runs/`、`data/raw/`、`.env` 等）。
  2. M5 的真实运行结果如果要对外展示，先把要展示的结果文件提交进仓库（经过钩子），报告只引用已提交的文件。
- **影响**：M3 的 `pages.yml` 不需要敏感词扫描步骤。无已有结果。

## D-006 · 2026-10-19 · “检索命中证据”的两种判据并列报告

- **背景**：METHODOLOGY 1.8 的 2×2 用到“检索是否命中证据”，但没有给出精确判据；PLAN M2 写的是“以证据日期判定”；METHODOLOGY 1.12 要求判据是“命中证据”而不是“返回非空”。
- **决定**：实现两种判据，报告里**并列**给出，不选其一覆盖另一个：
  - `message`（严格）：检索结果里有一条是证据消息本身。需要 arm 返回来源消息 id（bm25 可以）。
  - `date`（宽松，即 PLAN 的口径）：检索结果里有一条来自证据所在的那一天。对只返回“抽取后的记忆 + 日期”的 arm（如 M5 的 mem0）也适用。
  - 两者都按证据判定；“返回非空”在任何判据下都不算命中（有单元测试）。
- **为什么**：只用日期判据会把“同一天的无关消息”算成命中，在每天消息很多的真实语料上偏宽；只用消息判据则 mem0 一类的 arm 无法计算。并列报告让读者看到两种口径是否给出同一个结论。
- **这不是定义改动**：METHODOLOGY 的结局分类、窗口分类、统计方法都没有变，只是把一个未写明的判据写明。
- **相关的小改动**：合成语料把 F_MH_Mini_001 的第二条证据移到另一个群，使“跨群”这一证据范围在离线 demo 里有样本（`corpus_id` 随之改变；此前没有冻结过基于它的结果）。

## D-007 · 2026-10-19 · pytest 插件放在 `src/pytest_agenteval/`，用 entry point 加载

- **决定**：插件是独立的顶层包 `pytest_agenteval`，放在 `src/` 下与 `agentmemeval` 并列，由 uv_build 一起打包（`module-name = ["agentmemeval", "pytest_agenteval"]`），通过 `pytest11` entry point 自动加载。CLAUDE.md §4 的架构图里它画在仓库根目录。
- **为什么**：放在 `src/` 下才能被打包和安装，entry point 才能生效（METHODOLOGY 3.1 允许 `-p` 或 entry point 两种加载方式）；放在根目录则只能靠 `pythonpath` + `-p` 加载，安装后的包里也不会有它。它仍然是独立的包目录，只依赖 `agentmemeval` 的公开模块。
- **另外**：`tests/eval/` 里的逐题用例是测量而不是门禁（被测系统答错时用例失败），所以不在默认 `testpaths` 里，需要显式运行 `uv run pytest tests/eval`。默认的 `uv run pytest` 只跑框架自测（`tests/unit`、`tests/plugin`），必须全绿。
- **影响**：无已有结果。

## D-008 · 2026-10-19 · Agent 层结局判定的细则

METHODOLOGY 1.4 / 2.1 定义了 delivered / undelivered / env_failure / agent_error，下面是实现时补充的细则，**不改变定义**，只把没写明的情况写明（`agent/core.py`，均有测试）：

1. **确认语不结束运行**：通过 `final_answer` 发出的确认或占位语（“收到，稍后汇报”）不算答案，循环继续，给真正的答案留出机会。之后在预算内来了真实答案 → `delivered`；始终只有确认 → `undelivered`（acknowledgement）。这对应 1.4 的“交付动作产出的也不一定是答案”和“最后一条可能只是确认”，两个方向都有用例。
2. **工具故障**：工具超时、服务报错（基础设施）或返回格式不符（malformed）之后，Agent 没能恢复就结束 → `env_failure`；重试成功 → 按答案正常判定，并记一条警告（重试不能静默，3.4）。
3. **调用不存在的工具是 Agent 自己的错**：记为 `unknown_tool`，不算环境故障，最终按 `undelivered` 等正常规则判定。
4. **LLM 端点故障** → `env_failure`；**Agent 自身代码或策略抛出异常** → `agent_error`，两者分开计数。
5. **路由**：以第一次路由决定为准；期望标签来自 Agent 自己声明的规则（`ROUTING_RULE`：需要历史对话的问题必须走 memory），不依赖评测者的主观判断。
6. **耗时**：批量评测默认用虚拟时钟按固定成本推进（`SimulatedLatency`），结果文件元数据记录 `clock: simulated` 及各项成本，报告在耗时旁标注“模拟耗时”。`--real-clock` 改为测量真实耗时，此时不伪造 TTFT（非流式调用记为空）。

**影响**：无已有结果。

## D-009 · 2026-10-19 · M5 真实运行的配置（作者确认）

作者确认的选项（括号内为理由）：

- **LLM 提供方**：DeepSeek 开放平台（OpenAI 兼容接口）。答题与 Agent 用 `deepseek-flash`，评分员用强一档的 `deepseek-v4-pro`（避免评分员偏向同模型的答案）。两者都**关闭思考模式**：DeepSeek 默认开启思考模式，而思考模式下 `temperature` 被静默忽略；关闭后 `temperature=0` 才生效。结果文件里的模型名带 `+thinking=off`。（模型名与参数行为于 2026-10-19 按 api-docs.deepseek.com 核实；旧名 `deepseek-chat` / `deepseek-reasoner` 已停用。）
- **费用上限**：¥30 / 4000 次调用，所有 live 组件共用一本账（`usage.jsonl`），每次调用前检查，超出即 `BudgetExceededError` 停止；按官方**高峰价**计费（保守）。先用 5 题试跑测单题成本。
- **mem0**：本地开源版（mem0ai 2.2.1）。mem0 自带的 DeepSeek 适配器默认用已停用的模型名、且无法关闭思考模式（已在源码核实），所以把它的 LLM 换成走同一本账、同一缓存的适配器；embedding 用本地 Ollama `nomic-embed-text`（DeepSeek 没有 embedding 接口）；向量库为运行目录内的本地 Qdrant，历史库 `history.db` 也放在运行目录（默认的 `~/.mem0/history.db` 会被所有运行共用）；关闭 mem0 遥测。每个（日期, 群）调用一次 `add()`。mem0 存的是抽取后的记忆，不是消息，因此“命中证据消息”判据对它不适用（报告标为不适用），按日期判据判定（D-006）。
- **EverOS 不做**。影响：真实被测系统只有 mem0 一个，展示不了两个真实系统之间的配对比较，也少了“独立核验出题方自家系统”的角度；方法论（对照组、证据窗口、配对检验、2×2 归因、评分员 v1/v2 并行）不受影响。写进报告局限。
- **窗口**：前 90 天，提问全部 A 类 + D 类（110 + 78 = 188 题），不再抽子集（试跑除外）。
- **评分员人工校准暂不做**。报告注明“评分员未经人工校准”（METHODOLOGY 4.2 允许）。**不模拟、不估计人工 κ**：报告里的数字必须来自实际结果（CLAUDE.md §2.4），且凭经验取值等于引入外部指标（§2.1）。可以做、且不需要人的是：评分员 v1 与 v2 的一致性（明确标为口径之间的对比）、评分员稳定性（翻转率）。校准工具在作者决定标注时再做。
- **探针**：live 运行在灌入前跑 LLM 与记忆服务写入探针（写入**临时命名空间**，不碰被测库），灌入后跑语料探针并选定一道“金丝雀”题，运行结束再用同一道题复查；探针调用绕过缓存（缓存的回复证明不了可达）。

**影响**：无已有结果。

## D-010 · 2026-10-19 · 查询形态实验的短查询、公开版报告、评分员规则一致性检查

- **查询形态实验（METHODOLOGY 1.9）的第三种查询**：方法定义要求“人工改写的短查询”。本项目用**自动截取的题干前 8 个实词**代替，报告和结果文件里都明确写“自动，不是人工改写”。理由：人工改写 188 道题需要作者本人逐题完成；由 AI 改写再称为“人工”则是编造。这是对定义的**替代实现**，不是改定义；作者愿意时可以补一份人工改写的查询文件，作为第四种形态并行报告。
- **公开版报告（`report --public`）**：保留所有统计表、探针状态和费用，逐题明细只列结局、判分和检索是否命中，**不含题目、标准答案、模型答案和检索到的对话原文**。理由：完整明细相当于再分发数据集的大部分内容，超出 D-002“引用个别原题”的范围；也不含本机路径。
- **评分员规则一致性检查**：没有人工标注时，仍然可以机械地检查评分员的判定是否违反它自己写下的规则，例如“可回答的题，答案是拒答且不含标准答案，却判 CORRECT”。报告里标为“按规则的自动一致性检查，不是人工标注”，只用来定位可疑判定，不用来计算评分员准确率。
- **不根据这批样本修改 v2**：试跑中已看到 v2 把一道可回答题的拒答判成 CORRECT（规则 4 只适用于否定类标准答案）。按 METHODOLOGY 1.6，不在同一批样本上继续调 prompt；问题如实报告。若以后做 v3，必须并行报告并用没参与改写的留出样本评估。

**影响**：无已有结果。
