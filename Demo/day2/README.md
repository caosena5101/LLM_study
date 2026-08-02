# Day 2 · 提示工程基础

系统学习提示工程三大核心技巧：**角色设定 / 结构化输出 / 少样本提示**，并通过实验验证其效果。

## 环境依赖

与 day1 共用根目录 `.env`：

```
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

依赖：`openai`、`python-dotenv`（day1 已安装）。

## 文件

| 文件 | 实验 | 验收方式 |
|---|---|---|
| `exp1_role_prompting.py` | 角色设定对比（租房纠纷） | 自动检测是否引用法条 + 分步建议 |
| `exp2_structured_output.py` | 结构化输出（个人信息抽取为 JSON） | 脚本自动 `json.loads` 校验字段完整性 |
| `exp3_few_shot.py` | 少样本提示（情感分类） | 对比零样本与少样本输出格式一致性 |

## 运行

```bash
cd c:\MachineLearning\LLM\Demo
python day2\exp1_role_prompting.py
python day2\exp2_structured_output.py
python day2\exp3_few_shot.py
```

## 核心原理

- **角色设定有效**：角色指令在注意力机制中提高特定领域 token 权重（知识路由），改变后续 token 的条件概率分布（分布偏移），并隐含负约束如"法律顾问不用网络流行语"（行为锚定）。
- **结构化输出**：提供完整 JSON Schema + 明确禁止额外解释 + 低温度（≤0.2）减少格式漂移。
- **少样本提示**：3~5 个示例足够，需覆盖所有类别；利用近因效应把最接近目标的示例放最后。

## 最佳实践

1. 角色越具体越好（"15年民事租赁律师" > "律师"），可在角色描述中嵌入输出格式要求。
2. 结构化输出需提供完整 Schema，明确禁止额外解释，使用低温度。
3. 少样本示例需覆盖所有类别，assistant 输出格式精确，模型会严格模仿。
