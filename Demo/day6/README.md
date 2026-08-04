# Day 6 · 大模型核心参数与现象：上下文窗口、幻觉、温度

深入理解大模型推理时的核心参数与典型现象：**Temperature**、**幻觉（Hallucination）**、**上下文窗口（Context Window）**、**Frequency Penalty**，并通过对比实验验证它们的行为。

## 环境依赖

与 day1/day2/day3 共用根目录 `.env`：

```
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

依赖：`openai`、`python-dotenv`（day1 已安装）；可选 `matplotlib`（用于绘图扩展）。

## 文件

| 文件 | 内容 | 验收方式 |
|---|---|---|
| `llm_params_lab.py` | Temperature 多样性 / 幻觉检测 / 上下文窗口 / Frequency Penalty 四组实验 | 脚本自动统计 + 生成 markdown 报告 |
| `notes_params.md` | 参数理论、幻觉防范、上下文窗口原理、参数速查表 | — |
| `results/exp1_temperature.md` | 运行后自动生成的 Temperature 对比报告 | 打开文件检查 |
| `results/exp2_hallucination.md` | 运行后自动生成的幻觉检测报告 | 打开文件检查 |
| `results/temperature_diversity.png` | 可选：Temperature vs 多样性柱状图 | 需 matplotlib |

## 运行

```bash
cd c:\MachineLearning\LLM\Demo
python day6\llm_params_lab.py
```

## 实验矩阵

| 编号 | 实验 | 方法 | 预期效果 |
|---|---|---|---|
| ① | Temperature 对比 | 同一 prompt 在 T=0/0.7/1.5 下各跑 10 次 | T↑ -> 多样性↑，验证"温度越高越随机" |
| ② | 幻觉检测 | 5 个易触发幻觉的 prompt（T=0 可复现） | 至少识别 3 个幻觉实例 |
| ③ | 上下文窗口 | 不同长度输入观察响应时间与截断 | 直观感受 Token 数限制 |
| ④ | Frequency Penalty | 同一描述任务设置 0/1.0/2.0 | 字符唯一率随惩罚值上升 |

## 核心原理

- **Temperature 数学本质**：`P(token_i) = softmax(logit_i / T)`。T↓ 分布尖锐（确定性↑），T↑ 分布平坦（随机性↑）。
- **幻觉成因**：模型是"概率续写器"而非"知识检索器"，面对虚构实体倾向"编造合理答案"而非拒绝。
- **上下文窗口**：限制的是 Token 数（非字数），底层源于 Self-Attention 的 O(N²) 复杂度与 KV Cache 线性增长。
- **Frequency Penalty**：对已出现 token 施加惩罚，正值防止"复读机"现象。

## 验收清单

| 验收项 | 标准 | 如何判断 |
|---|---|---|
| Temperature=0 | 10 次输出完全相同（多样性极低） | 脚本自动统计 |
| Temperature=0.7 | 输出有明显变化但保持连贯 | 多样性≈80-100% |
| Temperature=1.5 | 输出高度随机，可能不通顺 | 多样性=100% |
| 幻觉识别 | 至少识别出 3 个幻觉实例 | 检查报告中 [可能产生幻觉] 条目 |
| 规律总结 | 能说出"温度越高->分布越平->采样越随机->多样性越高" | 报告总结部分 |
| 上下文窗口 | 理解 Token 数限制与 O(N²) 复杂度 | 口头解释 |
| 报告保存 | `results/exp1_temperature.md` 与 `results/exp2_hallucination.md` 生成 | 打开文件检查 |

## 扩展练习

替换 `llm_params_lab.py` 中 `experiment_temperature` 的 prompt 为"请用三个词描述秋天"，可更明显体现 T=1.5 下的跳跃感；或在 `experiment_hallucination` 中追加"请介绍 2025 年图灵奖得主李雷的获奖理由"等新场景，观察模型拒绝率。
