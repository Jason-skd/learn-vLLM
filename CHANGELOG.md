# Changelog

## 0be68a9 — Transformers 参考基线

- 新增 `scripts/build_reference.py`，使用 Transformers 生成参考张量；运行入口见 [CONTRIBUTING.md](CONTRIBUTING.md#开发验证)。
- `reference/config.json` 定义模型来源（`model.id`、`revision`）、精度（`dtype`）、设备（`device`）、是否添加特殊 token、最大生成长度、EOS 停止规则及输入用例。默认使用 `Qwen/Qwen3-0.6B`、`main`、CPU 和 float32。
- `model.id` 支持 Hugging Face 仓库 ID 或完整本地模型目录。目录需包含配置、权重和 tokenizer 文件；本地目录不通过 `revision` 选择版本，但该字段仍须保留。相对路径以运行时工作目录为基准。
- 设置 `HF_HUB_OFFLINE=1` 可禁止联网，缺少本地文件或缓存时会报错。默认 `revision: main` 随上游变化；需要固定基线时应指定模型提交 ID。
- 输出 `reference/outputs.pt` 包含输入 token、attention mask、prefill logits、逐步生成 logits、生成 token 和文本、停止原因，以及配置和运行环境信息。每次运行覆盖输出，该文件已加入 Git 忽略规则。
- 参考生成使用 eager attention、greedy 解码且不启用 KV Cache。用例列表不能为空，ID 必须唯一，`max_new_tokens` 必须大于零。
- 已通过 Ruff 检查和格式检查；实际模型生成验收待完成，本地离线试运行因缺少模型缓存而未完成。
