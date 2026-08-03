"""
Day 1 实验：首次调用大模型 API，观察 temperature 对输出的影响。

核心知识点：
1. OpenAI 兼容协议：DeepSeek 通过相同的 SDK 与接口形态提供服务，只需替换 base_url。
2. thinking 模式：DeepSeek V4 默认开启，输出分两段——reasoning_content（推理）+ content（回答）。
3. temperature：控制采样随机性。低温度→确定性强、严谨；高温度→发散性强、有创意。

运行：python day1/first_api.py
"""
import os
# 从 .env 加载环境变量（避免把密钥硬编码进源码，便于多环境切换）
from dotenv import load_dotenv
# OpenAI 官方 SDK，DeepSeek 兼容其 Chat Completions 接口
from openai import OpenAI

# ============ 1. 加载环境变量并初始化客户端 ============
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),       # 鉴权密钥
    base_url=os.getenv("OPENAI_BASE_URL")      # 接入点：https://api.deepseek.com
)
# 模型名兜底：.env 未配置时默认使用 deepseek-v4-flash
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")


# ============ 2. 核心对话函数 ============
def chat_with_model(user_prompt, system_prompt="你是一个知识渊博且风趣的AI助手。", temperature=0.7, max_tokens=2048):
    """
    调用大模型 API 进行单次对话。

    DeepSeek V4 默认开启 thinking 模式，输出会先进入 reasoning_content，
    再生成最终 content。这里同时返回两者以便观察。

    参数:
        user_prompt:  用户输入的任务指令
        system_prompt: 设定模型角色/风格，影响后续 token 的条件概率分布
        temperature:  采样温度，0 最确定，>1 更随机发散
        max_tokens:   输出 token 上限，防止 thinking 耗尽预算后无最终回答

    返回: (最终回答, 推理过程, finish_reason)
    """
    # messages 是对话上下文，按 role 交替组织：system → user → assistant → ...
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        msg = response.choices[0].message
        # reasoning_content：thinking 模式的推理过程（标准 SDK 不识别，用 getattr 安全取）
        reasoning = getattr(msg, "reasoning_content", None) or ""
        # content：模型最终给出的回答
        content = msg.content or ""
        # 若最终回答为空（thinking 用尽了 token 预算），则回退展示推理过程，避免返回空串
        result = content if content else f"[最终回答为空，展示推理过程]\n{reasoning}"
        # finish_reason: stop=正常结束, length=达到 max_tokens 截断, content_filter=被过滤
        return result, reasoning, response.choices[0].finish_reason
    except Exception as e:
        # 网络异常、鉴权失败、限流等都走这里，保证脚本不崩
        return f"调用API时发生错误: {e}", "", "error"


# ============ 3. 验收测试：观察 temperature 带来的差异 ============
if __name__ == "__main__":
    # 统一的测试 prompt，便于横向对比两种温度下的输出差异
    test_prompt = "用一句话给一款新上市的智能手表写个Slogan。"

    print(f"🤖 当前使用模型: {MODEL}\n")

    # ---- 测试低温度（严谨、确定性强）----
    # T=0.2 时模型几乎总选最高概率 token，输出稳定、可复现，适合事实/结构化任务
    print("【低温度 T=0.2】:")
    answer, reasoning, finish = chat_with_model(test_prompt, temperature=0.2)
    if reasoning:
        # 只展示推理过程前 200 字，避免刷屏
        print(f"(思考过程 finish_reason={finish})：{reasoning[:200]}...")
    print(answer)
    print("-" * 40)

    # ---- 测试高温度（创意、发散性强）----
    # T=1.2 时采样分布更平坦，低概率 token 也可能被选中，输出多样、有创意
    print("【高温度 T=1.2】:")
    answer, reasoning, finish = chat_with_model(test_prompt, temperature=1.2)
    if reasoning:
        print(f"(思考过程 finish_reason={finish})：{reasoning[:200]}...")
    print(answer)
