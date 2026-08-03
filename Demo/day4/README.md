# Day 4 · Transformer 核心机制：自注意力与多头注意力

理解 Transformer 的核心机制 —— **注意力（Attention）**，为后续微调打基础。用 PyTorch 手动实现单头 / 多头自注意力，并用热力图可视化注意力权重。

## 环境依赖

```
pip install torch matplotlib numpy
```

> 本实验**不调用 API**，纯本地 PyTorch 计算，无需 `.env` 中的 `OPENAI_API_KEY`。

## 文件

| 文件 | 内容 | 验收方式 |
|---|---|---|
| `attention_lab.py` | 单头 / 多头自注意力实现 + 热力图可视化 + 数学性质验证 | 脚本运行成功，生成 2 张热力图 |
| `notes_attention.md` | QKV 理论、多头作用、缩放因子、口头解释要点、扩展思考 | — |
| `results/single_head_attention.png` | 运行后自动生成的单头注意力热力图 | 打开图片查看权重分布 |
| `results/multi_head_attention.png` | 运行后自动生成的多头注意力热力图（4 头并排） | 4 个子图关注模式不同 |

## 运行

```bash
cd c:\MachineLearning\LLM\Demo
python day4\attention_lab.py
```

> 脚本会在 **当前工作目录** 下创建 `results/` 文件夹并保存图片。建议在 `Demo/` 根目录运行，图片会生成到 `Demo/results/`。

## 实验矩阵

| 编号 | 实验 | 内容 | 预期输出 |
|---|---|---|---|
| ① | 单头自注意力 | 手动实现 QKV 全流程，`torch.bmm` 算注意力 | 输出 `(1, 8, 64)`，注意力矩阵 `(1, 8, 8)`，行和=1 |
| ② | 多头自注意力 | 4 头并行，每个头独立学关注模式 | 输出 `(1, 8, 64)`，注意力矩阵 `(1, 4, 8, 8)` |
| ③ | 数学性质验证 | 行和=1、权重∈[0,1]、输出形状=输入形状 | 三个 ✅ |

## 核心公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

| 符号 | 含义 | 类比 |
|---|---|---|
| Q (Query) | 我在找什么 | 搜索词 |
| K (Key) | 我有什么标签 | 网页标题 |
| V (Value) | 我的实际内容 | 网页正文 |

## 核心原理

- **为什么需要注意力**：RNN 串行慢、长距离依赖弱。注意力让任意两点直接连接（距离=1），全图并行，根据内容动态路由。
- **为什么多头**：单头只能学一种关注模式。多头在 `d_model` 的不同子空间并行学习语法 / 语义 / 位置 / 指代等多维关系，最后 Concat + Linear 融合。**参数量不变**，只是重塑使用方式。
- **为什么除以 √d_k**：点积随 `d_k` 增大而变大，softmax 进入饱和区梯度消失。缩放把方差稳定到 1，保证梯度稳定。

## 验收清单

| 验收项 | 标准 | 如何判断 |
|---|---|---|
| 代码运行成功 | 无报错跑完三个实验 | 终端打印 "🏁 实验完成！" |
| 单头热力图 | `results/single_head_attention.png` 生成 | 打开图片，颜色分布有差异 |
| 多头热力图 | `results/multi_head_attention.png` 生成 | 4 个子图关注模式明显不同 |
| 数学性质 | 行和=1、权重∈[0,1]、形状一致 | 三个 ✅ |
| 口头解释 | 能解释"为什么需要注意力"和"多头作用" | 见 `notes_attention.md` 第六节 |

## 扩展练习

1. **加 Causal Mask**：在 softmax 前把上三角设为 `-inf`，模拟 GPT 解码器的因果注意力。
2. **加位置编码**：给输入加上 sinusoidal 位置编码，观察注意力分布的变化。
3. **对比 `nn.MultiheadAttention`**：用 PyTorch 内置实现替换手写版本，验证输出是否接近。
4. **接入真实 token**：用 `transformers` 的 tokenizer 把一句话编码成向量，再喂入自注意力，观察语义相关的 token 是否互相关注。
