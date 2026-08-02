"""
实验二：结构化输出（Format Control）

从一段非结构化文本中提取个人信息，要求模型严格输出符合 JSON Schema 的纯 JSON。
设置 temperature=0.1 减少格式漂移，脚本自动校验输出是否为合法 JSON 且字段完整。
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 待抽取的原始文本
RAW_TEXT = (
    "张三，男，1990年5月12日出生，手机号 138-0000-1234，"
    "邮箱 zhangsan@example.com，目前居住在北京市海淀区中关村大街1号，"
    "在某某科技有限公司担任后端工程师，入职时间 2021年3月。"
)

# 期望的 JSON Schema（用于在 prompt 中描述，并用于校验）
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "姓名"},
        "gender": {"type": "string", "description": "性别"},
        "birthday": {"type": "string", "description": "出生日期 YYYY-MM-DD"},
        "phone": {"type": "string", "description": "手机号"},
        "email": {"type": "string", "description": "邮箱"},
        "address": {"type": "string", "description": "居住地址"},
        "company": {"type": "string", "description": "所在公司"},
        "position": {"type": "string", "description": "职位"},
        "hire_date": {"type": "string", "description": "入职时间 YYYY-MM"},
    },
    "required": ["name", "gender", "birthday", "phone", "email",
                 "address", "company", "position", "hire_date"],
}

SYSTEM_PROMPT = (
    "你是一个信息抽取助手。请从用户提供的文本中提取个人信息，"
    "并严格输出符合以下 JSON Schema 的纯 JSON 对象。\n"
    "要求：\n"
    "1) 只输出 JSON，不要任何解释、前后缀或 markdown 代码块标记；\n"
    "2) 日期统一格式化：birthday 为 YYYY-MM-DD，hire_date 为 YYYY-MM；\n"
    "3) 缺失字段用 null 填充。\n\n"
    f"JSON Schema:\n{json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2)}"
)


def extract() -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RAW_TEXT},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


def validate(raw_output: str) -> bool:
    """校验输出：能解析为 JSON 且包含全部 required 字段"""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return False

    missing = [k for k in JSON_SCHEMA["required"] if k not in data]
    if missing:
        print(f"❌ 缺失字段: {missing}")
        return False

    print("✅ JSON 解析成功，所有字段完整：")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return True


if __name__ == "__main__":
    print(f"🤖 模型: {MODEL}")
    print(f"📥 原始文本: {RAW_TEXT}\n")

    raw = extract()
    print("📤 模型原始输出：")
    print(raw)
    print("\n" + "-" * 60)
    print("🔍 校验结果：")
    ok = validate(raw)
    print("\n📋 验收：", "通过 ✅" if ok else "未通过 ❌")
