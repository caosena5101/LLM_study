"""
实验三：少样本提示（Few-Shot Prompting）

任务：情感分类（正面 / 负面 / 中性 / 混合）。
- 对照组：零样本，仅在 user 消息中给出待分类文本。
- 实验组：在 messages 中交替提供 3 个示例（user+assistant），覆盖所有类别，
  然后给出待分类文本。

关键：示例覆盖所有类别，assistant 输出精确格式，模型会严格模仿。
利用近因效应——把最接近目标类别（混合）的示例放在最后。
"""
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 三个示例，覆盖全部 4 个类别（最后一个示例为"混合"，利用近因效应）
FEW_SHOT_EXAMPLES = [
    ("这家餐厅菜品很棒，服务也很周到，下次还会来！", "正面"),
    ("收到的快递外包装破损严重，里面的东西也碎了，客服态度还差。", "负面"),
    ("今天路过新开的便利店，进去看了一眼，东西不算多。", "中性"),
    ("屏幕显示效果很惊艳，但是续航实在太短了，又爱又恨。", "混合"),
]

# 待分类的测试文本（包含一个"混合"情感的样本，用于检验模型是否学会该类别）
TEST_TEXTS = [
    "刚买的耳机音质很好，但戴久了耳朵疼，退货又有点舍不得。",
    "今天的会议按时结束了，没有特别的内容。",
]


def classify_zero_shot(text: str) -> tuple[str, str]:
    """零样本对照组：仅给出待分类文本，并简单说明标签集合。返回 (标签, 来源)"""
    system = "你是一个情感分类器。请将文本分类为：正面 / 负面 / 中性 / 混合。只输出类别名，不要解释。"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return _extract_label(resp.choices[0].message)


def classify_few_shot(text: str) -> tuple[str, str]:
    """少样本实验组：在 messages 中提供示例，再给出待分类文本。返回 (标签, 来源)"""
    messages = []
    for u, a in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
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
    content = (message.content or "").strip()
    if is_valid_label(content):
        return content, "content"

    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    # 在推理过程中按"正面/负面/中性/混合"顺序找最后一次出现的类别
    matches = re.findall(r"(正面|负面|中性|混合)", reasoning)
    if matches:
        return matches[-1], "reasoning(兜底)"

    return content or "(空)", "无法提取"


def is_valid_label(s: str) -> bool:
    return s in {"正面", "负面", "中性", "混合"}


if __name__ == "__main__":
    print(f"🤖 模型: {MODEL}")
    print(f"📚 少样本示例数: {len(FEW_SHOT_EXAMPLES)} (覆盖 正面/负面/中性/混合)\n")

    for text in TEST_TEXTS:
        print("=" * 60)
        print(f"📝 待分类: {text}")
        z, z_src = classify_zero_shot(text)
        f, f_src = classify_few_shot(text)
        print(f"  零样本: {z!r:12} ({z_src})  格式合法={'✅' if is_valid_label(z) else '❌'}")
        print(f"  少样本: {f!r:12} ({f_src})  格式合法={'✅' if is_valid_label(f) else '❌'}")

    print("\n📋 验收：少样本输出应严格遵循示例格式（仅类别名），"
          "并能识别'混合'情感；零样本格式可能不稳定。")
