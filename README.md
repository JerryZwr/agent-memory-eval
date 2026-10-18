# agent-memory-eval

> 🚧 建设中（当前进度：M1 离线核心流水线）。完整计划见 [PLAN.md](PLAN.md)，评测方法见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)。

一个开源的 AI 记忆系统与 Agent 评测框架：在公开基准 EverMemBench 上对比记忆系统与无记忆对照组，并用评分员校准、配对统计检验和失败归因证明结论可信；同时用一个刻意做小的工具调用 Agent，展示 Agent 层的测试维度。

*An evaluation framework for AI memory systems and tool-calling agents, focused on whether the conclusions are trustworthy (control arm, judge calibration, paired significance tests, failure attribution) rather than on leaderboard scores.*

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。默认测试全部离线，不需要任何 API key。

```powershell
uv sync
uv run agentmemeval demo    # 离线 demo：合成语料，null 对照组 vs bm25，几秒内跑完
uv run pytest
uv run pre-commit install   # 开发者：启用提交前检查
```

demo 把每次运行的四个阶段（ingest → search → answer → judge）分别写到 `runs/demo-<时间>/<arm>/` 下，
每个文件都带语料标识、配置哈希、模型与评分员版本、git commit 等元数据。
删掉任意一个阶段文件后执行 `uv run agentmemeval run --run-dir <目录>`，只会重跑那个阶段。

真实数据（可选，不需要 key）：`uv run --extra data python scripts/download_data.py`，
下载到 `data/raw/`（不入库），许可证与版本固定见 [docs/decisions.md](docs/decisions.md)。

## License

代码以 [MIT](LICENSE) 许可发布。EverMemBench 数据集不随本仓库分发，其许可见 [docs/decisions.md](docs/decisions.md)。
