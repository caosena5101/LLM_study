"""
实验二：结构化输出（Format Control）

从一段非结构化文本中提取个人信息，要求模型严格输出符合 JSON Schema 的纯 JSON。
设置 temperature=0.1 减少格式漂移，脚本自动校验输出是否为合法 JSON 且字段完整。

核心原理：
- Schema 约束：在 prompt 中给出完整 JSON Schema，明确字段名、类型、含义，模型有"模板"可循。
- 格式锁定：要求"只输出 JSON，无任何解释/前后缀/markdown 标记"，降低格式漂移概率。
- 低温度：temperature≤0.2 时采样集中在高概率 token，格式稳定性显著提升。
- 自动校验：脚本用 json.loads 解析 + required 字段检查，机器可判，不依赖人眼。
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量并初始化 DeepSeek 兼容客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 待抽取的原始文本：一段非结构化的中文个人信息描述
RAW_TEXT = (
    "张三，男，1990年5月12日出生，手机号 138-0000-1234，"
    "邮箱 zhangsan@example.com，目前居住在北京市海淀区中关村大街1号，"
    "在某某科技有限公司担任后端工程师，入职时间 2021年3月。"
)

# 期望的 JSON Schema：
#   - 既用于在 prompt 中向模型描述目标结构（让模型有"模板"可循）
#   - 又用于在脚本侧校验输出（required 字段是否齐全）
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
    # required：所有字段都必须出现，缺失则校验失败
    "required": ["name", "gender", "birthday", "phone", "email",
                 "address", "company", "position", "hire_date"],
}

# System Prompt：把约束写清楚，是结构化输出成功的关键
SYSTEM_PROMPT = (
    "你是一个信息抽取助手。请从用户提供的文本中提取个人信息，"
    "并严格输出符合以下 JSON Schema 的纯 JSON 对象。\n"
    "要求：\n"
    # 1) 只输出 JSON 本体——禁止任何解释文字、禁止 markdown 代码块标记（```json）
    "1) 只输出 JSON，不要任何解释、前后缀或 markdown 代码块标记；\n"
    # 2) 日期统一格式化，便于下游程序解析
    "2) 日期统一格式化：birthday 为 YYYY-MM-DD，hire_date 为 YYYY-MM；\n"
    # 3) 缺失字段用 null 填充，保证字段完整（不漏键）
    "3) 缺失字段用 null 填充。\n\n"
    # 把 Schema 序列化进 prompt（ensure_ascii=False 保留中文可读性）
    f"JSON Schema:\n{json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2)}"
)


def extract() -> str:
    """调用模型抽取信息，返回原始输出字符串（尚未校验）。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RAW_TEXT},
        ],
        # 低温度是结构化输出的关键：减少格式漂移，提高 JSON 合法率
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


def validate(raw_output: str) -> bool:
    """
    校验输出：能解析为 JSON 且包含全部 required 字段。

    两层校验：
    1) json.loads 能解析 → 格式合法
    2) required 字段全部存在 → 内容完整
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        # 格式错误：模型可能输出了多余文字或 markdown 标记
        print(f"❌ JSON 解析失败: {e}")
        return False

    # 检查 required 字段是否齐全
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

    # 第一步：调用模型抽取
    raw = extract()
    print("📤 模型原始输出：")
    print(raw)
    print("\n" + "-" * 60)

    # 第二步：自动校验（机器可判，无需人眼检查）
    print("🔍 校验结果：")
    ok = validate(raw)
    print("\n📋 验收：", "通过 ✅" if ok else "未通过 ❌")
