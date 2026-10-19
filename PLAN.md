# PLAN —— agent-memory-eval 项目计划

> 方法定义见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)，工作守则见 [CLAUDE.md](CLAUDE.md)。
> 按里程碑顺序推进。每个里程碑完成后勾选状态，并对照"完成标准"逐项自检。

---

## 0. 目标与非目标

**目标**：一个面试官能在 5 分钟内看懂、在 2 分钟内跑起来的评测作品，展示：
- 记忆层评测：对照组设计、证据窗口筛题、结局分类、评分员校准、统计显著性、失败归因；
- Agent 层评测：交付、收敛、路由、回退、故障注入、轨迹断言、反幻觉、trace 完整性、耗时拆解；
- 测试工程：pytest 插件二次开发、Allure 报告、CI、GitHub Pages。

**非目标**：
- 不做一个好用的 Agent 或记忆系统。Agent 刻意做得很小。
- 不追求排行榜式的"谁分数最高"。重点是**结论是否可信、失败能否归因**。
- 不做压测。耗时拆解只用于说明"会拆、会看"。
- 不跑全量 626 道题。真实运行只用一个子集，控制费用。

---

## 1. 里程碑总览

| 里程碑 | 内容 | 预估工时 | 是否需要 key / 花钱 | 状态 |
|---|---|---|---|---|
| M0 | 仓库骨架、工具链、CI 雏形、数据集许可证核实 | 3–4 h | 否 | ✅ |
| M1 | 离线核心流水线：数据模型、合成语料、null/bm25、ScriptedLLM、判分、结局分类、demo | 8–10 h | 否 | ✅ |
| M2 | 分析与报告：证据窗口筛题、按维度拆分、2×2、统计、评测报告 HTML | 6–8 h | 否 | ✅ |
| M3 | pytest 插件 + Allure + GitHub Pages | 8–10 h | 否 | ⬜ |
| M4 | 极简 Agent + 追踪 + Agent 层 9 个测试维度 | 10–12 h | 否 | ⬜ |
| M5 | 真实运行：真实 LLM、mem0（可选 EverOS）、评分员校准、查询形态实验 | 8–12 h | **是**（需预算确认） | ⬜ |
| M6 | 打磨：README 首屏、findings、图表、徽章 | 4–6 h | 否 | ⬜ |

合计约 50–60 小时。M0–M4 全部离线，**不需要任何 key**，CI 能完整跑通；只有 M5 花钱。

---

## 2. 里程碑详情

### M0 · 仓库骨架

任务：
1. `uv init`，src layout，包名 `agentmemeval`，console script `agentmemeval`。
2. 依赖分组：运行依赖、`dev`（pytest、pytest-asyncio、allure-pytest、ruff、pre-commit）、`live`（openai、mem0ai 等，可选安装）。
3. `pyproject.toml`：pytest 配置（`--strict-markers`、注册 marker：unit / agent / live / slow、`asyncio_mode=auto`）、ruff 配置。
4. `.gitignore`：`.env`、`.denylist.local`、`reports/`、`runs/`、`data/raw/`、缓存目录。
5. pre-commit：ruff、gitleaks，以及一个本地钩子 `scripts/check_denylist.py`（读取 `.denylist.local`，命中即拒绝提交；文件不存在时跳过并给出提示）。文件格式：每行一条规则，大小写不敏感的子串匹配；`#` 开头的行是注释，空行忽略；`re:` 开头的行是 Python 正则（大小写不敏感）。命中时报告文件名、行号和规则序号，**不要在输出里打印规则原文**（敏感词本身也是敏感信息）。用一个临时的假清单写测试，不要读真实的 `.denylist.local`。
6. `.env.example`：列出 live 需要的环境变量名（不填值）。
7. `.github/workflows/ci.yml`：push 和 PR 时运行 ruff + `uv run pytest`（离线）。
8. LICENSE（建议 MIT）、README 占位。
9. **核实 EverMemBench 数据集的许可证**（HF 页面 `EverMind-AI/EverMemBench-Dynamic` 和 GitHub 仓库），把结论写进 `docs/decisions.md`。根据许可证决定：下载脚本是否可以缓存数据、报告里能否引用原题原文。
10. `scripts/download_data.py`：从 HF 下载到 `data/raw/`（gitignored）。

完成标准：
- [x] 新克隆的仓库执行 `uv sync && uv run pytest` 通过（此时可以只有一个冒烟测试）
- [x] CI 是绿的（run 39112389219，Ubuntu 3.11/3.12 + Windows 3.12）
- [x] 提交一个含假密钥的文件会被 pre-commit 拦下
- [x] `docs/decisions.md` 记录了数据集许可证的结论（D-002；数据结构见 D-003）

### M1 · 离线核心流水线

任务：
1. `data/`：数据模型（消息、天、数据集、题目、检索结果、答案、判分结果）；EverMemBench 格式加载器；**合成 fixture**（`tests/fixtures/mini_corpus/`：约 3 天、2 个群、40–60 条消息、12–16 道题，覆盖单跳、多跳、时间计算、选择题、无证据题，并给出每道题的证据日期）。
2. `arms/`：统一接口 `add(dataset, ...)` / `search(query, top_k, ...)`；实现 `null`、`bm25`。
3. `llm/`：`LLM` 抽象接口；`ScriptedLLM`（按策略和规则产出确定性回复；答题时能"读"检索结果里的关键词，用来模拟真实行为）。
4. `pipeline/`：四个阶段分别落盘，结果文件带 METHODOLOGY 1.2 规定的元数据。
5. `judge/`：选择题确定性判分，带 `ambiguous_extraction` 标记；开放题评分员接口，离线用 `ScriptedJudge`（规则匹配）。
6. `analysis/outcome.py`：结局三分类（METHODOLOGY 1.4）。
7. `agentmemeval demo`：合成语料 → null 和 bm25 两个 arm → 四个阶段 → 在终端打印结果表。

完成标准：
- [x] `uv run agentmemeval demo` 离线运行，2 分钟内结束（`tests/unit/test_demo.py` 断言 < 120 s）
- [x] demo 结果中，bm25 在有证据的题上明显高于 null（`test_demo.py` 从落盘的结果文件计算并断言）
- [x] 单元测试覆盖：加载器、两个 arm、选择题提取（含歧义情况）、结局分类、元数据落盘
- [x] 删除中间阶段的任意一个结果文件后，可以只重跑该阶段（`test_pipeline.py` 对四个阶段逐一验证）

### M2 · 分析与报告

任务：
1. `analysis/windows.py`：证据窗口 A–F 分类（METHODOLOGY 1.3），`--days N` 选窗口；所需天数超过已灌入天数时报错。
2. `analysis/breakdown.py`：按题型、证据范围拆分；检索 × 判分 2×2（检索是否命中证据，以 fixture 里的证据日期判定）。
3. `analysis/stats.py`：Wilson 区间、bootstrap、McNemar 精确检验、Cohen's κ（带 bootstrap 区间）。**用教科书或标准库的已知数值做单元测试。**
4. `analysis/baseline.py`：冻结基线、逐题 diff 的四种结局、拒绝不可比的比较（METHODOLOGY 1.10）。
5. `report/`：Jinja2 评测报告，包含总表（带区间和分母）、按题型拆分、2×2、逐题明细（可折叠）、运行元数据。只用静态 HTML（CSS 内联，图表可以用纯 SVG 或从 CDN 加载的轻量图表库）。
6. `agentmemeval report --run <dir>`。

完成标准：
- [x] stats 的每个函数都有与已知数值核对的单元测试（Newcombe 1998 的 Wilson 例子、Wikipedia 的 κ 例子、McNemar 的 2×0.5ⁿ 表；`tests/unit/test_stats.py`）
- [x] B 类题默认被排除，传显式参数才纳入（`QuestionSpec.include_b`；`tests/unit/test_windows.py`）
- [x] 基线 diff 的四种结局都有测试；`corpus_id` 不同时拒绝 diff（`tests/unit/test_breakdown_and_baseline.py`）
- [x] 报告里每个比率都带分母和 95% 区间；样本 < 30 的地方有"仅供定性参考"标注（`tests/unit/test_report.py` 逐个检查）
- [x] demo 运行后能生成报告并在浏览器中打开（demo 自动写 `report.html`；`--open` 用浏览器打开）

### M3 · pytest 插件 + Allure + GitHub Pages

任务：
1. `pytest_agenteval/`：METHODOLOGY 3.1 的全部功能；`probes/` 实现前置和后置探针（离线时用假服务）。
2. 插件自测（`tests/plugin/`，用 pytester）。
3. Allure 接入（METHODOLOGY 3.2）：epic/feature/story、severity、动态标题、附件、步骤、`environment.properties`、`categories.json`。
4. 记忆层的逐题用例：`tests/eval/test_memory_qa.py`，用 `--eval-qids` 参数化，失败时附上检索结果和评分员理由。
5. `.github/workflows/pages.yml`：main 分支推送后：
   - 离线运行全部测试 → allure-results；
   - 从 `gh-pages` 分支取回上一次的 `history/`，生成带趋势的 Allure 报告；
   - 运行 demo 并生成评测报告；
   - 把两份报告部署到 GitHub Pages（`/allure/` 和 `/report/`）。
   具体用哪个 Action，以写的时候的官方文档为准，不要凭记忆。
6. README 加 CI 状态徽章和两个报告的链接。

完成标准：
- [ ] 插件自测覆盖：探针红时跳过、后置探针红时整轮 not_measured、diff 四种结局、`--live` 开关
- [ ] 不带 `--live` 时，所有 live 用例都是 skipped，并给出原因
- [ ] Allure 报告里能按 epic/feature 浏览，失败用例带附件，失败按 categories 分类
- [ ] GitHub Pages 上能打开两份报告，Allure 的趋势图在第二次 CI 运行后出现

### M4 · 极简 Agent 与 Agent 层评测

任务：
1. `agent/`：路由、循环、三个工具、预算（METHODOLOGY 2.1）；ScriptedLLM 的 6 种策略（含 `acknowledger`）。
2. `tracing/`：span 追踪器，JSON 导出（METHODOLOGY 2.2）。ScriptedLLM 要能模拟流式输出，以便测 TTFT。
3. `tests/agent/`：9 个测试维度各一个模块（METHODOLOGY 2.3），每个用例都接入 Allure（步骤 + trace 附件）。
4. 工具故障注入：一个可配置的故障包装器（超时 / 空结果 / 异常 / 格式错误），用 pytest 参数化覆盖。
5. Agent 批量评测：`agentmemeval agent-eval`，输出 METHODOLOGY 2.4 的指标；评测报告加一节 Agent 指标。

完成标准：
- [ ] 9 个维度各有至少 2 个用例，且全部离线通过
- [ ] 每种 ScriptedLLM 策略都至少被一个用例"抓住"：用例在该策略下给出预期的失败判定
- [ ] trace 完整性检查本身有反例测试：故意构造一个缺 span 的 trace，检查能报出来
- [ ] 评测报告里的 Agent 指标一节包含：交付率、预算触发率、工具调用分布、路由混淆矩阵、耗时三段拆解

### M5 · 真实运行（需要作者确认预算后才开始）

开始前，由作者确认：模型选择、调用次数上限、预计费用。

任务：
1. `OpenAICompatLLM`（答题、评分员、Agent 共用；支持 OpenRouter 等兼容端点）。
2. `mem0` arm（按官方文档实现；本地或云端由作者决定）。`everos` arm 可选：先读它的 QUICKSTART，评估部署成本后再决定是否做。
3. 子集运行：1 个群、前约 30 天、窗口内 A 类 + D 类约 50 道题；arm 为 null / bm25 / mem0（/ everos）；判分口径 v1 和 v2 都跑。
4. 评分员校准（METHODOLOGY 1.6）：导出盲标表 → 作者标注 → 计算 κ、混淆矩阵、翻转率 → 逐条归因。**标注工作由作者完成，AI 不得代标。**
5. 查询形态敏感性实验（METHODOLOGY 1.9）。
6. 真实模型下跑一次 Agent 批量评测（小规模，例如 20 道题）。
7. 冻结一份基线 `baselines/live-v1.json`，并演示一次"改配置（例如检索 top_k）→ 逐题 diff"。
8. `docs/findings.md`：只写实际跑出来的结论，每条都附上数据来源文件。

完成标准：
- [ ] 费用在上限内，实际调用次数记录在运行元数据里
- [ ] 报告里所有数字都能追溯到 `runs/` 下的文件
- [ ] 评分员校准报告包含 κ（带区间）、混淆矩阵、翻转率、不一致样本的逐条归因
- [ ] arm 之间的比较都用了配对检验，结论措辞符合 METHODOLOGY 1.7

### M6 · 打磨

任务：
1. README 首屏：一句话定位、结果总表（带区间）、一张图、两个报告链接、`uv sync && uv run agentmemeval demo` 快速开始。中文为主，附英文摘要。
2. 架构图（mermaid）。
3. `docs/findings.md` 整理成"评测 AI 记忆与 Agent 时容易犯的错"：每一条配一个本仓库里的反例或测试用例。
4. 徽章：CI 状态、Python 版本、License；覆盖率徽章可选。
5. 用一个全新的 Windows 环境（或 GitHub Codespaces）完整走一遍 README 的快速开始。

完成标准：
- [ ] 一个从没见过本项目的人，照着 README 能在 5 分钟内跑出 demo 报告
- [ ] README 里的每个结论都链接到报告或 findings 的对应位置

---

## 3. 需要作者决策的事项

| # | 事项 | 时机 | 推荐 |
|---|---|---|---|
| 1 | 数据集许可证是否允许缓存数据、引用原题 | M0 | 以许可证为准；不确定时只引用题号，不贴原文 |
| 2 | 仓库开源许可证 | M0 | MIT |
| 3 | M5 的模型、调用上限、预算 | M5 开始前 | 答题和 Agent 用便宜的模型，评分员用稍强一档的模型 |
| 4 | mem0 用本地还是云端；是否做 EverOS | M5 | 先 mem0；EverOS 看部署成本 |
| 5 | 第二标注人 | M5 | 有的话最好；没有就自身隔周重标并注明 |

## 4. 风险

| 风险 | 应对 |
|---|---|
| 范围膨胀（Agent 越做越复杂） | Agent 只服务于"展示测试维度"；新功能如果不对应 METHODOLOGY 2.3 的某个维度，就不做 |
| 真实运行花费超预期 | 调用上限 + 结果缓存；先在 5 道题上试跑，确认单题成本 |
| 外部服务 API 变动 | live 代码隔离在 `arms/` 和 `llm/` 里；离线测试不依赖它们 |
| 数据集许可证限制 | 用合成 fixture 完成全部离线演示；真实数据只在本地使用 |
| "看起来合理的错误数字" | 每个统计和分析函数都用已知答案的单元测试核对；报告数字只从结果文件生成 |
