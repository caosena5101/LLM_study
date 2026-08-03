"""
实验一：角色设定对比（Role Prompting）

对照组：无角色设定，直接询问租房纠纷问题。
实验组：在 system 消息中设定"15年民事法律顾问"，要求引用法条、分步建议。

预期：实验组回答包含具体法条引用与分步维权路径；对照组泛泛而谈。

核心原理：
- 知识路由：角色指令在注意力机制中提高特定领域 token 权重，激活相关专业知识。
- 分布偏移：角色改变后续 token 的条件概率分布，使回答更专业、更聚焦。
- 行为锚定：角色描述中隐含负约束（如"避免网络流行语"），限制不期望的输出风格。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 环境变量并初始化 DeepSeek 兼容客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
# 模型名兜底为 deepseek-v4-flash
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 统一的测试问题：租房纠纷（典型的民事法律场景，便于检验角色设定的效果）
QUESTION = "我租的房子水管漏水已经两周，房东一直不修，我该怎么办？"


def ask(question: str, system_prompt: str | None = None, temperature: float = 0.7) -> tuple[str, str]:
    """
    单次对话，返回 (最终回答, 推理过程)。

    参数:
        question:      用户问题
        system_prompt: 角色设定；为 None 时即"无角色对照组"
        temperature:   采样温度，0.7 为通用默认值
    """
    messages = []
    # 仅当提供 system_prompt 时才插入 system 消息（对照组无此消息）
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    msg = resp.choices[0].message
    # thinking 模式的推理过程（标准 SDK 不识别此字段，用 getattr 安全取）
    reasoning = getattr(msg, "reasoning_content", None) or ""
    # 最终回答
    content = msg.content or ""
    # 兜底：thinking 用尽 token 导致 content 为空时，展示推理过程
    if not content:
        content = f"[最终回答为空，展示推理过程]\n{reasoning}"
    return content, reasoning


def has_article_citation(text: str) -> bool:
    """
    粗略判断是否引用了法条（如"《民法典》第xxx条"、"合同法第x条"等）。
    同时支持阿拉伯数字（第712条）和中文数字（第七百一十二条）。
    """
    import re
    # 中文数字字符集：覆盖常见写法，兼容大写形式（壹贰叁…）
    num = r"(?:\d+|[一二三四五六七八九十百千零壹贰叁肆伍陆柒捌玖拾佰仟]+)"
    # 两种形式：
    #   形式1：《某法》...第x条  （引用具体法律名称）
    #   形式2：第x条              （仅引用条文号）
    pattern = rf"《[^》]+》[^，。；\n]*第\s*{num}\s*条|第\s*{num}\s*条"
    return bool(re.search(pattern, text))


def has_step_advice(text: str) -> bool:
    """
    粗略判断是否给出分步建议（出现 1./2./3. 或 第一步/第二步 等）。
    使用 re.M 多行模式，使 ^ 能匹配每行行首。
    """
    import re
    return bool(re.search(r"(第一步|第二步|第三步|第四步|^\s*[1-9][.、)]|首先|其次|然后|最后)", text, re.M))


def evaluate(label: str, answer: str) -> None:
    """对单条回答做自动验收：是否引用法条 + 是否给出分步建议。"""
    cite = has_article_citation(answer)
    step = has_step_advice(answer)
    print(f"[{label}] 引用法条={'✅' if cite else '❌'}  分步建议={'✅' if step else '❌'}")


if __name__ == "__main__":
    print(f"🤖 模型: {MODEL}\n问题: {QUESTION}\n")

    # ============ 对照组：无角色设定 ============
    # 不给 system 消息，模型以默认身份回答，预期泛泛而谈、缺乏专业引用
    print("=" * 60)
    print("【对照组 · 无角色设定】")
    print("=" * 60)
    ans_ctrl, _ = ask(QUESTION, system_prompt=None)
    print(ans_ctrl)
    evaluate("对照组", ans_ctrl)

    # ============ 实验组：15年民事法律顾问 ============
    # 角色越具体越好："15年民事法律顾问" > "律师"
    # 在角色描述中嵌入输出格式要求（引用法条 + 分步建议），实现行为锚定
    ROLE_PROMPT = (
        "你是一位拥有15年执业经验的民事法律顾问，专精房屋租赁纠纷。"
        "回答时请：1) 引用具体法律法规及条文；"
        "2) 给出清晰的分步维权建议；"
        "3) 语言专业、客观，避免网络流行语。"
    )
    print("\n" + "=" * 60)
    print("【实验组 · 15年民事法律顾问】")
    print("=" * 60)
    ans_exp, _ = ask(QUESTION, system_prompt=ROLE_PROMPT)
    print(ans_exp)
    evaluate("实验组", ans_exp)

    print("\n📋 验收：实验组应同时出现法条引用与分步建议，对照组则泛泛而谈。")
