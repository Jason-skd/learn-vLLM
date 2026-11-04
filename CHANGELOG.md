# Changelog

## de2e3fb — Qwen3 模型前向

- 新增 `src/learn_vllm/model.py`，使用 PyTorch 组装 Attention、Decoder 层、模型骨干和词表输出层。传入 Transformers 的 `Qwen3Config` 构造 `Qwen3ForCausalLM`，调用 `model(input_ids)` 返回全部输入位置的 logits，形状为 `[1, L, vocab_size]`。
- 模型需要先加载权重再进行推理。可使用 Transformers 加载官方模型，将其 `state_dict()` 通过 `load_state_dict(..., strict=True)` 加载到本实现；配置启用 `tie_word_embeddings` 时，输入 embedding 与输出头共享参数。推理时使用 `eval()` 和 `torch.inference_mode()`。
- 当前仅支持非空单请求、无 padding、完整上下文前向；不提供生成循环或 KV Cache。配置仅支持默认 RoPE、full attention、SiLU 和零 attention dropout，超出范围时会报错。
- 新增 pytest 测试，覆盖中间层及 logits 对齐、权重共享、因果性和不支持的输入与配置。全量运行命令见 [CONTRIBUTING.md](CONTRIBUTING.md#开发验证)；测试使用小型随机模型，无需下载官方权重。
- 参考基线实际运行验收已完成。使用本地 Qwen3-0.6B 官方权重，在 CPU、float32 下验证英文、中文和代码三个固定用例，完整前向 logits 与保存的参考张量逐元素一致；此项真实权重验证不包含在默认 pytest 测试中。

## 0be68a9 — Transformers 参考基线

- 新增 `scripts/build_reference.py`，使用 Transformers 生成参考张量；运行入口见 [CONTRIBUTING.md](CONTRIBUTING.md#开发验证)。
- `reference/config.json` 定义模型来源（`model.id`、`revision`）、精度（`dtype`）、设备（`device`）、是否添加特殊 token、最大生成长度、EOS 停止规则及输入用例。默认使用 `Qwen/Qwen3-0.6B`、`main`、CPU 和 float32。
- `model.id` 支持 Hugging Face 仓库 ID 或完整本地模型目录。目录需包含配置、权重和 tokenizer 文件；本地目录不通过 `revision` 选择版本，但该字段仍须保留。相对路径以运行时工作目录为基准。
- 设置 `HF_HUB_OFFLINE=1` 可禁止联网，缺少本地文件或缓存时会报错。默认 `revision: main` 随上游变化；需要固定基线时应指定模型提交 ID。
- 输出 `reference/outputs.pt` 包含输入 token、attention mask、prefill logits、逐步生成 logits、生成 token 和文本、停止原因，以及配置和运行环境信息。每次运行覆盖输出，该文件已加入 Git 忽略规则。
- 参考生成使用 eager attention、greedy 解码且不启用 KV Cache。用例列表不能为空，ID 必须唯一，`max_new_tokens` 必须大于零。
- 已通过 Ruff 检查和格式检查；实际模型生成验收待完成，本地离线试运行因缺少模型缓存而未完成。
