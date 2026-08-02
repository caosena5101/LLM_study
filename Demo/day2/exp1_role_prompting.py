"""
实验一：角色设定对比（Role Prompting）

对照组：无角色设定，直接询问租房纠纷问题。
实验组：在 system 消息中设定"15年民事法律顾问"，要求引用法条、分步建议。

预期：实验组回答包含具体法条引用与分步维权路径；对照组泛泛而谈。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

QUESTION = "我租的房子水管漏水已经两周，房东一直不修，我该怎么办？"


def ask(question: str, system_prompt: str | None = None, temperature: float = 0.7) -> tuple[str, str]:
    """单次对话，返回 (最终回答, 推理过程)"""
    messages = []
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
    reasoning = getattr(msg, "reasoning_content", None) or ""
    content = msg.content or ""
    if not content:
        content = f"[最终回答为空，展示推理过程]\n{reasoning}"
    return content, reasoning


def has_article_citation(text: str) -> bool:
    """粗略判断是否引用了法条（如"《民法典》第xxx条"、"合同法第x条"等）。
    同时支持阿拉伯数字（第712条）和中文数字（第七百一十二条）。"""
    import re
    # 中文数字：一二三四五六七八九十百千零壹贰叁肆伍陆柒捌玖拾佰仟
    num = r"(?:\d+|[一二三四五六七八九十百千零壹贰叁肆伍陆柒捌玖拾佰仟]+)"
    # 形式1：《某法》第x条  形式2：第x条
    pattern = rf"《[^》]+》[^，。；\n]*第\s*{num}\s*条|第\s*{num}\s*条"
    return bool(re.search(pattern, text))


def has_step_advice(text: str) -> bool:
    """粗略判断是否给出分步建议（出现 1./2./3. 或 第一步/第二步 等）"""
    import re
    return bool(re.search(r"(第一步|第二步|第三步|第四步|^\s*[1-9][.、)]|首先|其次|然后|最后)", text, re.M))


def evaluate(label: str, answer: str) -> None:
    cite = has_article_citation(answer)
    step = has_step_advice(answer)
    print(f"[{label}] 引用法条={'✅' if cite else '❌'}  分步建议={'✅' if step else '❌'}")


if __name__ == "__main__":
    print(f"🤖 模型: {MODEL}\n问题: {QUESTION}\n")

    # 对照组：无角色
    print("=" * 60)
    print("【对照组 · 无角色设定】")
    print("=" * 60)
    ans_ctrl, _ = ask(QUESTION, system_prompt=None)
    print(ans_ctrl)
    evaluate("对照组", ans_ctrl)

    # 实验组：15年民事法律顾问
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
