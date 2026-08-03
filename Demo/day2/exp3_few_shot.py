"""
实验三：少样本提示（Few-Shot Prompting）

任务：情感分类（正面 / 负面 / 中性 / 混合）。
- 对照组：零样本，仅在 user 消息中给出待分类文本。
- 实验组：在 messages 中交替提供 3 个示例（user+assistant），覆盖所有类别，
  然后给出待分类文本。

关键：示例覆盖所有类别，assistant 输出精确格式，模型会严格模仿。
利用近因效应——把最接近目标类别（混合）的示例放在最后。

核心原理：
- 上下文学习（ICL）：模型不更新参数，仅通过 prompt 中的示例"学会"任务模式。
- 格式锚定：assistant 示例只输出类别名，模型严格模仿，输出格式稳定。
- 类别覆盖：示例必须覆盖所有可能类别，否则模型遇未见过类别会漂移。
- 近因效应：最后一个示例对当前输入影响最大，把最接近目标的示例放最后。
"""
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量并初始化 DeepSeek 兼容客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
# 模型名兜底为 deepseek-v4-flash
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 三个示例，覆盖全部 4 个类别（最后一个示例为"混合"，利用近因效应）
# 每个元组：(用户文本, 期望的 assistant 输出标签)
# 关键：assistant 输出必须是精确的"类别名"——模型会严格模仿此格式
FEW_SHOT_EXAMPLES = [
    ("这家餐厅菜品很棒，服务也很周到，下次还会来！", "正面"),
    ("收到的快递外包装破损严重，里面的东西也碎了，客服态度还差。", "负面"),
    ("今天路过新开的便利店，进去看了一眼，东西不算多。", "中性"),
    # "混合"放在最后：近因效应使最后一个示例对当前输入影响最大，
    # 而测试集中恰好有"混合"样本，放最后有助于模型识别该类别
    ("屏幕显示效果很惊艳，但是续航实在太短了，又爱又恨。", "混合"),
]

# 待分类的测试文本（包含一个"混合"情感的样本，用于检验模型是否学会该类别）
TEST_TEXTS = [
    "刚买的耳机音质很好，但戴久了耳朵疼，退货又有点舍不得。",
    "今天的会议按时结束了，没有特别的内容。",
]


def classify_zero_shot(text: str) -> tuple[str, str]:
    """
    零样本对照组：仅给出待分类文本，并简单说明标签集合。返回 (标签, 来源)。

    预期：格式可能不稳定（可能输出多余解释），且对"混合"类别识别较弱。
    """
    system = "你是一个情感分类器。请将文本分类为：正面 / 负面 / 中性 / 混合。只输出类别名，不要解释。"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.0,  # 0 温度保证分类结果可复现
        max_tokens=512,
    )
    return _extract_label(resp.choices[0].message)


def classify_few_shot(text: str) -> tuple[str, str]:
    """
    少样本实验组：在 messages 中提供示例，再给出待分类文本。返回 (标签, 来源)。

    构造方式：user/assistant 交替排列示例，最后追加待分类的 user 文本。
    模型会延续示例建立的"模式"，直接输出类别名。
    """
    messages = []
    # 把每个示例按 user→assistant 交替追加进 messages
    for u, a in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    # 最后追加待分类文本——模型会延续上文模式，输出一个类别名
    messages.append({"role": "user", "content": text})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return _extract_label(resp.choices[0].message)


def _extract_label(message) -> tuple[str, str]:
    """
    从 message 中提取分类标签。

    DeepSeek V4 默认开启 thinking 模式，最终答案可能在 content，
    也可能因 token 用尽而只出现在 reasoning_content。这里优先取 content；
    若 content 为空或不合法，则从 reasoning_content 中抓取最后一个出现的合法标签。
    返回 (标签, 来源标记)。
    """
    # 优先取 content（thinking 模式下最终回答在此）
    content = (message.content or "").strip()
    if is_valid_label(content):
        return content, "content"

    # 兜底：content 为空或不合法时，从推理过程抓最后一个合法标签
    # 因为推理过程中模型可能列举多个类别，最后提到的往往是结论
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    # 在推理过程中按"正面/负面/中性/混合"顺序找最后一次出现的类别
    matches = re.findall(r"(正面|负面|中性|混合)", reasoning)
    if matches:
        return matches[-1], "reasoning(兜底)"

    return content or "(空)", "无法提取"


def is_valid_label(s: str) -> bool:
    """判断字符串是否为合法的类别标签。"""
    return s in {"正面", "负面", "中性", "混合"}


if __name__ == "__main__":
    print(f"🤖 模型: {MODEL}")
    print(f"📚 少样本示例数: {len(FEW_SHOT_EXAMPLES)} (覆盖 正面/负面/中性/混合)\n")

    # 对每个测试文本，分别用零样本和少样本两种方式分类，并对比
    for text in TEST_TEXTS:
        print("=" * 60)
        print(f"📝 待分类: {text}")
        z, z_src = classify_zero_shot(text)
        f, f_src = classify_few_shot(text)
        # 格式合法性检查：少样本应严格输出类别名（合法），零样本可能不合法
        print(f"  零样本: {z!r:12} ({z_src})  格式合法={'✅' if is_valid_label(z) else '❌'}")
        print(f"  少样本: {f!r:12} ({f_src})  格式合法={'✅' if is_valid_label(f) else '❌'}")

    print("\n📋 验收：少样本输出应严格遵循示例格式（仅类别名），"
          "并能识别'混合'情感；零样本格式可能不稳定。")
