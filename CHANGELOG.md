# Changelog

## 8b36b25 — 连续单请求 KV Cache

- 新增 `KVCache`，按层保存连续 K/V 缓冲区；`Qwen3ForCausalLM.create_kv_cache(max_seq_len)` 按模型的层数、KV 头数、头维度、dtype 和设备创建单请求缓存。
- `Qwen3ForCausalLM.forward` 和 `greedy_generate` 新增可选 `cache`。传入空缓存时，首轮处理完整 prompt，后续只处理新 token，并从缓存长度续接 RoPE 位置；不传缓存仍沿用每步重算完整上下文。
- 缓存容量必须覆盖 prompt 与后续处理 token 的总数，超过 `max_seq_len` 时抛出 `ValueError`。当前仅支持非空、无 padding 的单请求和连续缓存；不同请求的批处理、分页、回收及动态调度尚未实现。
- 新增测试验证 cached prefill / decode logits 与完整前向一致，`greedy_generate` 的 cached 调用序列和生成结果正确。

## c0dfa4a — KV Cache 参考基线

- `scripts/build_reference.py` 现在为每个固定用例同时生成整句重算（`full`）和 Transformers `DynamicCache` 增量解码（`cached`）两套参考结果。参考产物升级为 schema 2，并记录支持的生成模式与缓存实现。
- 集成测试按 `full` 和 `cached` 分别执行，本实现的每条生成路径只与同模式的参考结果比较；生成的 token、文本和停止原因仍要求完全一致。
- schema 1 产物不再兼容。升级后必须重新运行 `uv run python scripts/build_reference.py`，否则集成测试会拒绝加载旧产物。

## 269f03d — 参考生成集成测试

- 将 `scripts/compare_generation.py` 迁移为 `tests/test_generation_reference.py`，按 `reference/config.json` 的用例逐个报告 tokenizer、logits、生成 token、停止原因和文本的对比结果。
- 参考用例标记为 `integration`，默认 `uv run pytest` 不加载真实模型权重；生成 `reference/outputs.pt` 后使用 `uv run pytest -m integration tests/test_generation_reference.py` 执行验收。缺少参考产物时测试会跳过并提示先运行 `scripts/build_reference.py`。
- `scripts/build_reference.py` 继续作为参考产物生成入口。模型精度、设备或生成配置与参考产物不一致时，集成测试会要求重新生成产物。

## 448e0dc — 单请求 greedy 生成

- 新增 `src/learn_vllm/generation.py` 的 `greedy_generate`。传入单请求 `input_ids`、`max_new_tokens` 和可选 EOS token 集合，返回 prefill logits、生成 token、逐步 logits 与停止原因。EOS token 会保留在生成结果中。
- 当前生成每步重新执行完整上下文前向；仅支持非空、无 padding 的单请求和 greedy 解码，尚不支持 KV Cache。默认 pytest 使用桩模型验证 EOS、长度上限和逐步执行，不下载官方权重。
- 新增 `scripts/compare_generation.py`，使用 `reference/config.json` 的 tokenizer 与模型设置加载本实现，并与 `reference/outputs.pt` 比较生成 token、logits、停止原因和文本。模型配置与参考产物中的生成设置不一致时会提示重新生成产物。

```bash
uv run python scripts/compare_generation.py
```

- 使用 Qwen3-0.6B 本地权重在 CPU、float32 下完成英文、中文和代码三个固定用例验收；生成结果及 logits 均与参考产物对齐。

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
