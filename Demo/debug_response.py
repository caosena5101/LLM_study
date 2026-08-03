"""
调试脚本：观察 DeepSeek V4 Flash 完整响应对象的结构。

用途：理解 OpenAI 兼容接口返回的 response 对象层级，以及
     DeepSeek V4 特有的 reasoning_content 字段（thinking 模式的推理过程）。
运行：python debug_response.py
"""
# os：读取环境变量（API Key、Base URL、模型名）
import os
# dotenv：从同目录 .env 文件加载环境变量到 os.environ
from dotenv import load_dotenv
# OpenAI：官方 SDK，兼容 DeepSeek API（通过 base_url 指向 DeepSeek 网关）
from openai import OpenAI

# 加载 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
load_dotenv()

# 初始化客户端：DeepSeek 兼容 OpenAI 协议，只需把 base_url 指向 https://api.deepseek.com
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
# 模型名兜底为 deepseek-v4-flash（若 .env 未设置 OPENAI_MODEL 则用此默认值）
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# 发起一次最简单的对话：system 设定角色，user 给出任务
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        # system 消息：设定模型身份/风格，影响后续所有 token 的条件概率分布
        {"role": "system", "content": "你是一个知识渊博且风趣的AI助手。"},
        # user 消息：本次具体的任务指令
        {"role": "user", "content": "用一句话给一款新上市的智能手表写个Slogan。"}
    ],
    # temperature=0.7：中等采样温度，平衡稳定性与创意（0 最确定，>1 更发散）
    temperature=0.7,
    # max_tokens=300：限制输出长度，避免 thinking 模式耗尽 token 后无最终回答
    max_tokens=300
)

# 逐层打印响应对象，帮助理解 OpenAI 响应结构：
# response → choices[] → choice.message → content / reasoning_content
print("=== Full response object ===")
print(response)
print("\n=== choices[0] ===")
print(response.choices[0])
print("\n=== message ===")
print(response.choices[0].message)
print("\n=== content ===")
# repr() 显示字符串原始形式（含引号、换行符），便于看清空白字符
print(repr(response.choices[0].message.content))

# reasoning_content 是 DeepSeek V4 thinking 模式特有的字段：
#   - 模型先生成推理过程（reasoning_content）
#   - 再生成最终回答（content）
# 标准 OpenAI SDK 不识别此字段，故用 getattr 安全读取（不存在时返回默认值）
print("\n=== reasoning_content ===")
rc = getattr(response.choices[0].message, "reasoning_content", "<no attr>")
print(repr(rc))
