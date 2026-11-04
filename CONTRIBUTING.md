# Contributing to learn-vLLM

## 项目概览

使用 Python 3.12+、PyTorch 和 Transformers，推荐通过 uv 管理环境。`src/learn_vllm/` 存放实现，`scripts/` 存放辅助脚本，`reference/config.json` 定义参考模型和验证用例。

## 开发验证

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest
```

在项目根目录运行参考生成脚本：

```bash
uv run python scripts/build_reference.py
```

默认加载 Qwen3-0.6B，将固定用例的 logits 和 greedy 生成结果写入 `reference/outputs.pt`（会覆盖旧文件，不纳入 Git）。使用本地模型时，将 `reference/config.json` 的 `model.id` 改为完整模型目录的绝对路径；仅使用本地文件或缓存时，在命令前加 `HF_HUB_OFFLINE=1`。配置细节见 [CHANGELOG.md](CHANGELOG.md)。

生成参考产物后，使用集成测试将本实现的单请求 greedy 生成结果与其比较。
默认 pytest 会排除 `integration` 测试，避免下载模型权重：

```bash
uv run pytest -m integration tests/test_generation_reference.py
```

## 文档索引

- [ROADMAP.md](ROADMAP.md)：实现阶段切分与规划。
- [CHANGELOG.md](CHANGELOG.md)：版本变化、配置与使用限制。
