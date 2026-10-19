# agent-memory-eval

[![CI](https://github.com/JerryZwr/agent-memory-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/JerryZwr/agent-memory-eval/actions/workflows/ci.yml)
[![Pages](https://github.com/JerryZwr/agent-memory-eval/actions/workflows/pages.yml/badge.svg)](https://github.com/JerryZwr/agent-memory-eval/actions/workflows/pages.yml)

**在线报告**（CI 每次推送后用合成语料离线生成）：
[评测报告](https://jerryzwr.github.io/agent-memory-eval/report/) ·
[Allure 测试报告](https://jerryzwr.github.io/agent-memory-eval/allure/)

> 🚧 建设中（已完成 M0–M3；下一步 M4 极简 Agent 与 Agent 层评测）。完整计划见 [PLAN.md](PLAN.md)，评测方法见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)。

一个开源的 AI 记忆系统与 Agent 评测框架：在公开基准 EverMemBench 上对比记忆系统与无记忆对照组，并用评分员校准、配对统计检验和失败归因证明结论可信；同时用一个刻意做小的工具调用 Agent，展示 Agent 层的测试维度。

*An evaluation framework for AI memory systems and tool-calling agents, focused on whether the conclusions are trustworthy (control arm, judge calibration, paired significance tests, failure attribution) rather than on leaderboard scores.*

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。默认测试全部离线，不需要任何 API key。

```powershell
uv sync
uv run agentmemeval demo --open   # 离线 demo：合成语料，null 对照组 vs bm25，几秒内跑完并打开报告
uv run pytest                     # 框架自测（门禁，必须全绿）
uv run pytest tests/eval          # 记忆层逐题评测（测量：答错的题会失败）
uv run pre-commit install         # 开发者：启用提交前检查
```

demo 把每次运行的四个阶段（ingest → search → answer → judge）分别写到 `runs/demo-<时间>/<arm>/` 下，
每个文件都带语料标识、配置哈希、模型与评分员版本、git commit 等元数据。
删掉任意一个阶段文件后执行 `uv run agentmemeval run --run-dir <目录>`，只会重跑那个阶段。

评测报告（`report.html`）里每个比率都带分母和 95% Wilson 区间，arm 之间用配对的 McNemar 精确检验比较，
并按证据窗口、题型、证据范围和“检索 × 判分 2×2”拆分。也可以单独生成：`uv run agentmemeval report --run <目录>`。
基线与逐题回归：`uv run agentmemeval baseline freeze --run <目录> --name <名字>`，`uv run agentmemeval baseline diff --run <目录> --baseline baselines/<名字>.json`。

真实数据（可选，不需要 key）：`uv run --extra data python scripts/download_data.py`，
下载到 `data/raw/`（不入库），许可证与版本固定见 [docs/decisions.md](docs/decisions.md)。

## License

代码以 [MIT](LICENSE) 许可发布。EverMemBench 数据集不随本仓库分发，其许可见 [docs/decisions.md](docs/decisions.md)。
