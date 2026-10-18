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
