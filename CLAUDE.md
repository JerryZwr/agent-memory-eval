# CLAUDE.md —— agent-memory-eval 工作守则

本文件给在本仓库里工作的 AI 助手看，也给作者自己看。动手之前读完本文件，再读：
- [PLAN.md](PLAN.md)：做什么、按什么顺序做、做到什么程度算完成；
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md)：评测方法的定义（**这是本项目的核心，不要自行改动定义**）。

---

## 1. 项目是什么

一个**开源的 AI 记忆系统与 Agent 评测框架**，作为测试开发方向的个人作品。它展示两件事：

1. **记忆层评测**：在公开基准 EverMemBench（多人群聊长期记忆）上，对比几个记忆系统和一个无记忆对照组，并且**证明结论可信**（评分员校准、统计显著性、失败归因）。
2. **Agent 层评测**：用一个**刻意做得很小**的工具调用 Agent，展示会测 Agent 的哪些方面：是否交付、是否收敛、路由是否正确、工具故障时的行为、轨迹断言、trace 完整性与耗时拆解。

工程上用 **pytest + Allure** 组织测试，用 GitHub Actions 做 CI，用 GitHub Pages 发布报告。

**目标读者是面试官。** 每个设计都要能回答"这体现了什么测试能力"。宁可小而完整，不要大而半成品。

---

## 2. 硬性规则（违反任何一条都要停下来问作者）

### 2.1 信息隔离
- 作者曾在实习中做过类似的评测。**本仓库不得包含实习单位的任何信息**：公司名、产品名、内部系统名、内部代码、内部数据、内部指标数值、缺陷编号、同事姓名。
- 方法论是作者自己的经验，可以写；**具体发现和数字不可以**。
- 作者如果粘贴了看起来来自内部的内容，先问"这段能公开吗"，再使用。
- 仓库根目录下的 `.denylist.local`（已 gitignore，由作者本地维护）是敏感词清单，由 pre-commit 钩子检查。**不要读取并复述它的内容，也不要把它提交进仓库。**

### 2.2 密钥与费用
- 任何 API key 只能放在 `.env`（已 gitignore）或 GitHub Actions secrets 里。代码里只读环境变量，提交 `.env.example` 作为模板。
- **默认的所有测试和 demo 都不能需要任何 key，也不能联网。** 真实调用 LLM 或记忆服务的代码，只能通过 `live` marker 或 `--live` 参数显式开启。
- 真实运行必须有**费用上限**：调用次数上限写在配置里，超出就报错退出，不允许静默继续。
- pre-commit 钩子里有 gitleaks，不要绕过（不加 `--no-verify`）。

### 2.3 数据
- **不提交 EverMemBench 的原始数据集**，用下载脚本获取（先核实数据集许可证，见 PLAN.md 的 M0）。
- demo 和 CI 只用 `tests/fixtures/` 里**自己编写的合成小语料**（几天、两三个群、几十条消息、十几道题，覆盖所有题型和"无证据"题）。

### 2.4 不编造
- 不确定外部库或服务的 API（mem0、EverOS、Allure、Langfuse 等）时，**去读文档或已安装的源码**，不要凭记忆写。读不到就说读不到。
- 报告里的每个数字都必须来自本仓库实际跑出的结果文件。不手写数字，不"示意"数据。
- 结论与证据不符时，修正结论，不要修饰证据。

### 2.5 方法定义受保护
- `docs/METHODOLOGY.md` 里的定义（结局分类、证据窗口分类、评分员口径、统计方法、基线对比语义）是评测可信性的基础。**改定义之前先问作者**，改了要记到 `docs/decisions.md`（日期、改了什么、为什么、影响哪些已有结果）。
- 改评分员 prompt 等于改分数口径：新口径**并行报告**，不覆盖旧口径。

---

## 3. 技术约定

| 项 | 约定 |
|---|---|
| 语言 / 版本 | Python 3.11+ |
| 包管理 | `uv`（`uv sync`、`uv run ...`）。不要用裸 `python` / `pip`：作者的 Windows 上 `python` 可能指向 Microsoft Store 的占位程序 |
| 代码布局 | `src/agentmemeval/`（src layout），可执行入口是 console script `agentmemeval` |
| 风格 | ruff（lint + format）；类型标注；dataclass 或 pydantic 做数据模型 |
| 测试 | pytest；`asyncio_mode=auto`；marker 用 `--strict-markers` 注册：`unit`、`agent`、`live`、`slow` |
| 报告 | Allure（allure-pytest）出测试报告；Jinja2 生成静态评测报告 HTML |
| 平台 | 作者在 Windows + PowerShell 上开发。写跨平台的 Python 脚本，**不要依赖 make 或 bash**；路径用 `pathlib` |
| Allure CLI | 需要 Java。本地生成报告是可选的；CI 里用 GitHub Action 生成和发布 |
| 编码 | 所有文件读写显式指定 `encoding="utf-8"`；控制台输出注意 Windows GBK 问题 |

常用命令（M0 完成后应该都能用）：

```powershell
uv sync
uv run pytest                                  # 全部离线测试，无 key 可跑
uv run pytest --alluredir=reports/allure-results
uv run agentmemeval demo                       # 离线 demo：合成语料 + 假 LLM，2 分钟内跑完
uv run agentmemeval report --run <run_dir>     # 生成评测报告 HTML
uv run pytest -m live --live                   # 真实调用，需要 .env，有费用
```

---

## 4. 架构速览（细节见 PLAN.md）

```
src/agentmemeval/
  data/        数据模型、加载器（EverMemBench 格式 + 合成 fixture）
  arms/        被测记忆系统的适配器：null（对照）、bm25（本地）、mem0 / everos（live）
  llm/         LLM 抽象：ScriptedLLM（确定性，测试用）、OpenAICompatLLM（live）
  pipeline/    ingest → search → answer → judge 四阶段，带结果落盘
  judge/       选择题确定性判分、开放题 LLM 评分员（v1 / v2）、校准工具
  analysis/    结局分类、证据窗口筛题、按题型拆分、检索 × 判分 2×2、stats
  agent/       极简工具调用 Agent、工具、预算、路由
  tracing/     轻量 span 追踪器（JSON 导出；Langfuse 导出可选）
  report/      Jinja2 评测报告
  probes/      环境探针（前置 / 后置）
  cli.py
pytest_agenteval/   pytest 插件（独立包目录，用 -p 或 entry point 加载）
tests/
  unit/        纯函数和组件的单元测试
  agent/       Agent 层评测用例（用 ScriptedLLM，离线）
  plugin/      插件自测（pytester）
  live/        真实调用（默认跳过）
  fixtures/    合成语料与题目
docs/          METHODOLOGY.md、decisions.md、findings.md
```

---

## 5. 工作方式

1. **按 PLAN.md 的里程碑顺序做**，不跳步。每个里程碑结束时对照它的"完成标准"自检，并告诉作者哪些通过、哪些没有。
2. **先写测试或先写验收条件**，再写实现。评测框架本身出错的代价是"给出一个看起来合理的错误数字"，这比崩溃更糟，所以框架代码的测试覆盖要高于普通项目。
3. **每个产出数字的环节都要能回答"它是怎么算出来的"**：结果文件里记录语料标识、配置哈希、评分员版本、时间窗口等元数据；不同元数据的结果拒绝直接比较。
4. **小步提交**，提交信息说明"为什么"，不只说"做了什么"。
5. 遇到需要作者决策的事（花钱、改方法定义、公开数据），停下来问，并给出推荐选项。
6. 每完成一个里程碑，更新 PLAN.md 里的状态勾选框。
