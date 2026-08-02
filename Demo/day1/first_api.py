import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载环境变量并初始化客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 2. 核心对话函数
def chat_with_model(user_prompt, system_prompt="你是一个知识渊博且风趣的AI助手。", temperature=0.7, max_tokens=2048):
    """
    调用大模型API进行单次对话
    DeepSeek V4 默认开启 thinking 模式，输出会先进入 reasoning_content，
    再生成最终 content。这里同时返回两者以便观察。
    """
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
        reasoning = getattr(msg, "reasoning_content", None) or ""
        content = msg.content or ""
        # 若最终回答为空（thinking 用尽了 token），则回退展示推理过程
        result = content if content else f"[最终回答为空，展示推理过程]\n{reasoning}"
        return result, reasoning, response.choices[0].finish_reason
    except Exception as e:
        return f"调用API时发生错误: {e}", "", "error"

# 3. 验收测试：观察 temperature 带来的差异
if __name__ == "__main__":
    test_prompt = "用一句话给一款新上市的智能手表写个Slogan。"
    
    print(f"🤖 当前使用模型: {MODEL}\n")
    
    # 测试低温度（严谨、确定性强）
    print("【低温度 T=0.2】:")
    answer, reasoning, finish = chat_with_model(test_prompt, temperature=0.2)
    if reasoning:
        print(f"(思考过程 finish_reason={finish})：{reasoning[:200]}...")
    print(answer)
    print("-" * 40)
    
    # 测试高温度（创意、发散性强）
    print("【高温度 T=1.2】:")
    answer, reasoning, finish = chat_with_model(test_prompt, temperature=1.2)
    if reasoning:
        print(f"(思考过程 finish_reason={finish})：{reasoning[:200]}...")
    print(answer)
