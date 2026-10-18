# agent-memory-eval

> 🚧 建设中（当前进度：M0 仓库骨架）。完整计划见 [PLAN.md](PLAN.md)，评测方法见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)。

一个开源的 AI 记忆系统与 Agent 评测框架：在公开基准 EverMemBench 上对比记忆系统与无记忆对照组，并用评分员校准、配对统计检验和失败归因证明结论可信；同时用一个刻意做小的工具调用 Agent，展示 Agent 层的测试维度。

*An evaluation framework for AI memory systems and tool-calling agents, focused on whether the conclusions are trustworthy (control arm, judge calibration, paired significance tests, failure attribution) rather than on leaderboard scores.*

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。默认测试全部离线，不需要任何 API key。

```powershell
uv sync
uv run pytest
uv run pre-commit install   # 开发者：启用提交前检查
```

## License

代码以 [MIT](LICENSE) 许可发布。EverMemBench 数据集不随本仓库分发，其许可见 [docs/decisions.md](docs/decisions.md)。
