# Day 3 · 进阶提示技巧：CoT 与分支提示

掌握进阶提示技巧：**思维链（Chain-of-Thought）** 与 **分支提示（Tree of Thoughts 简化版）**，并通过对比实验验证其效果。

## 环境依赖

与 day1/day2 共用根目录 `.env`：

```
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

依赖：`openai`、`python-dotenv`（day1 已安装）。

## 文件

| 文件 | 内容 | 验收方式 |
|---|---|---|
| `cot_lab.py` | Zero-shot / Zero-shot CoT / Few-shot CoT / 分支提示 四种策略对比 | 脚本自动对比 ✅/❌ + 生成 markdown 报告 |
| `notes_cot.md` | CoT/ToT 理论、实验设计、模型观察、实用建议 | — |
| `results/cot_experiment_report.md` | 运行后自动生成的完整实验报告 | 打开文件检查 |

## 运行

```bash
cd c:\MachineLearning\LLM\Demo
python day3\cot_lab.py
```

## 实验矩阵

| 编号 | 方法 | 提示策略 | 预期效果 |
|---|---|---|---|
| ① | Zero-shot 直接回答 | 仅给问题，要求直接输出答案 | 简单题正确，复杂题可能出错 |
| ② | Zero-shot CoT | 问题 + "让我们一步一步思考" | 激活推理链，准确率提升 |
| ③ | Few-shot CoT | 2 个带推理过程的示例 + 问题 | 推理格式最规范，准确率最高 |
| ④ | 分支提示（ToT 简化） | 假设不同场景，生成多条路径 | 展示多角度推理能力 |

## 核心原理

- **CoT 有效的原因**：分解复杂度（多步拆单步）、中间表示（推理步骤作工作记忆）、概率路径（推理 token 为最终答案构建更精确的条件概率）。
- **Few-shot CoT 最稳**：示例锚定输出结构，`#### 数字` 格式被严格模仿，便于下游解析。
- **分支提示（ToT 简化）**：对同一问题生成多条推理路径（正向 / 逆向 / 边界），交叉验证提高置信度。

## 验收清单

| 验收项 | 标准 | 如何判断 |
|---|---|---|
| CoT 输出推理步骤 | Zero-shot CoT 和 Few-shot CoT 输出包含分步推理 | 检查是否出现 "1. … 2. … 3. …" |
| 最终答案正确 | 提取的数字 == 正确答案 | 脚本自动打印 ✅/❌ |
| 三种方式自动对比 | 汇总表列出所有方法结果和对错 | 终端打印 + markdown 报告 |
| 分支提示 | 输出包含路径 A/B/C 三条推理 | 检查输出中是否有三个路径标记 |
| 进阶题体现差异 | 进阶题中 Zero-shot 可能出错，CoT 正确 | 对比同一题不同方法结果 |
| 报告保存 | `results/cot_experiment_report.md` 生成且内容完整 | 打开文件检查 |

## 扩展练习

替换 `cot_lab.py` 中的 `HARD_Q` 为"火车过桥"问题，可更明显体现 CoT 优势：

```python
HARD_Q = (
    "一列火车长200米，以每秒20米的速度通过一座长800米的桥。"
    "从火车头进入桥到火车尾离开桥，共需多少秒？"
)
CORRECT_HARD = "50"  # (200+800)/20 = 50秒
```
