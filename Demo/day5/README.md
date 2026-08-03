# Day 5 · 分词器（Tokenization）与嵌入（Embedding）

掌握分词器与嵌入概念。使用 **tiktoken**（GPT-4 分词器）和 **HuggingFace transformers**（BERT 分词器）对中英文进行编码/解码，观察 token 消耗差异，理解大模型上下文长度限制的底层原因。

## 环境依赖

```
pip install tiktoken transformers torch matplotlib numpy
```

> 本实验**不调用 API**，纯本地分词 + 可视化，无需 `.env` 中的 `OPENAI_API_KEY`。
> 首次运行 `transformers` 会自动从 HuggingFace 下载 `bert-base-chinese` 和 `bert-base-uncased` 权重（约 400MB），需联网。

## 文件

| 文件 | 内容 | 验收方式 |
|---|---|---|
| `tokenizer_lab.py` | tiktoken vs HuggingFace 分词对比 + Token 估算函数 + 注意力热力图 + O(N²) 复杂度曲线 | 脚本运行成功，生成 2 张图 |
| `notes_tokenizer.md` | BPE/WordPiece/SentencePiece 对比、Token vs Embedding、上下文限制底层原因、面试速记卡 | — |
| `results/attention_heatmap.png` | 运行后自动生成的注意力权重热力图 | 颜色深浅代表注意力权重 |
| `results/token_complexity.png` | 运行后自动生成的 O(N²) 复杂度曲线 | log-log 图上斜率=2 |

## 运行

```bash
cd c:\MachineLearning\LLM\Demo
python day5\tokenizer_lab.py
```

> 脚本会在 **当前工作目录** 下创建 `results/` 文件夹并保存图片。建议在 `Demo/` 根目录运行，图片会生成到 `Demo/results/`。

## 实验矩阵

| 编号 | 实验 | 内容 | 预期输出 |
|---|---|---|---|
| ① | tiktoken (GPT-4 BPE) | 中/英/混合文本编码 + 解码，批量消耗对比 | 中文 token 数 > 英文 |
| ② | HuggingFace (BERT WordPiece) | 中英文 WordPiece 分词 + `##` 子词标记 + `__call__` 张量输出 | 中文 1 字 1 token，英文有 `##ing` |
| ③ | Token 估算函数 | `estimate_tokens(text, model)` 通用估算 | 1000 字中文 ≈ 1500-2000 token |
| ④ | 注意力热力图 + 复杂度曲线 | 模拟 N×N 注意力矩阵 + N² 增长曲线 | 直观看到 O(N²) 爆炸 |

## 三大分词算法对比

| 算法 | 核心机制 | 代表模型 |
|---|---|---|
| **BPE** | 按频率合并最高频字节对 | GPT 系列 |
| **WordPiece** | 按语言模型似然合并，`##` 标记词内子词 | BERT 系列 |
| **SentencePiece** | 语言无关，Unicode 序列，不依赖空格 | T5 / LLaMA / Qwen |

## 核心认知：上下文长度为什么有限制？

1. **Self-Attention 是 O(N²)**：注意力矩阵 N×N，N 翻倍计算量变 4 倍。8K vs 128 上下文，自注意力计算量放大 4096 倍。
2. **KV Cache 线性增长**：长上下文需缓存所有历史 token 的 K/V，显存随 N 线性增长。
3. **语言差异**：中文 1 字 ≈ 1-2 token，英文 1 词 ≈ 1.3 token，同样窗口能装的中文比英文少。

## 验收清单

| 验收项 | 标准 | 如何判断 |
|---|---|---|
| 代码运行成功 | 无报错跑完四个实验 | 终端打印 "实验完成！" |
| tiktoken 对比 | 中英文 token 数差异明显 | 中文 token/字符比 > 英文 |
| BERT 加载成功 | `bert-base-chinese` 正常分词 | 输出 `['我', '喜', '欢', ...]` 字级别 |
| 估算函数 | `estimate_tokens` 可对任意文本估 token | 1000 字中文给出合理估值 |
| 热力图 | `results/attention_heatmap.png` 生成 | 每行和≈1，颜色有差异 |
| 复杂度曲线 | `results/token_complexity.png` 生成 | log-log 图上斜率=2 |
| 口头解释 | 能说出"上下文限制的底层原因" | 见 `notes_tokenizer.md` 第六节 Q6 |

## 扩展练习

1. **对比 GPT-4o 编码**：用 `tiktoken.get_encoding("o200k_base")` 对比 `cl100k_base`，观察新版对中文 token 数的压缩效果。
2. **加载 SentencePiece 模型**：用 `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")` 看国产模型的中文分词效率。
3. **估算 API 成本**：写一个函数，输入文本 + 模型，输出 token 数 + 对应的 OpenAI API 价格（参考官网定价表）。
4. **可视化 BPE 合并过程**：手动实现一个简化版 BPE 训练，画出词表大小随合并步数增长的曲线。
