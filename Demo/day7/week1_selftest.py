"""
第一周复习自测工具
============================================================================
Day 7 核心脚本：第一周学习成果自测与巩固，包含两个工具：

  1) API 成本计算器（验证第 ④ 题理解）
     - 内置 4 个典型场景（短对话 / 长文摘要 / 代码生成 / CoT 推理）
     - 支持交互式输入自定义 token 数
     - 同时输出人民币与近似美元金额
     - 可选 --api 调用真实 API 验证 usage 字段

  2) 交互式自测问答（共 5 题，满分 25 分）
     - 题目覆盖 Day1-Day6 核心知识点：注意力 / CoT / 温度 / 成本 / 幻觉
     - 每题先回答，再展示参考要点，由用户自评 0-5 分
     - 自动累计得分并按行动决策树给出复习建议
     - 自动生成 markdown 报告到 results/selftest_report.md

【为什么这个工具重要】
  复习不是"再看一遍笔记"，而是"主动检索"。认知科学表明主动检索
  (active recall) 比被动重读有效 2-3 倍。本脚本把"看笔记"变成
  "逼自己输出 + 对照标准答案 + 量化打分 + 定位盲区"，让复习可度量。

【使用方式】
  python day7/week1_selftest.py        # 菜单模式
  python day7/week1_selftest.py --cost  # 直接进入成本计算器
  python day7/week1_selftest.py --test  # 直接进入自测
  python day7/week1_selftest.py --api   # 调用真实 API 验证 usage 字段

依赖：仅用标准库；--api 模式额外需要 openai + python-dotenv（day1 已装）
"""

# ============================================================
# 模块导入说明
# ============================================================
# os      : 跨平台路径拼接（os.path）、环境变量读取（os.getenv）、目录创建
# sys     : 读取命令行参数（sys.argv）、检测当前控制台编码
# io      : TextIOWrapper 用于把 stdout/stderr 重包装成 UTF-8 流
# datetime: 在 markdown 报告中记录自测时间戳，便于复盘追踪进步
import os
import sys
import io
from datetime import datetime

# ------------------------------------------------------------
# Windows 控制台编码修复
# ------------------------------------------------------------
# 【为什么要这么做】
# Windows 中文系统控制台默认用 GBK 编码，而本脚本输出含 emoji（💰📋💵）、
# 数学符号（√、↑、→）、表格框线字符（╔═╗║╠╚）等 GBK 无法表示的内容，
# 直接 print 会抛 UnicodeEncodeError 让脚本崩溃。
# 【修复方法】
# 用 io.TextIOWrapper 把 stdout/stderr 强制重包装为 UTF-8 编码流，
# errors="replace" 表示遇到无法编码的字符用 ? 替代而不抛异常，
# 这样即使输出含特殊字符也不会中断脚本。
# 【兼容性】
# Linux/macOS 默认就是 UTF-8，sys.stdout.encoding.lower() == "utf-8"
# 条件不成立，这段代码对它们无副作用。
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 结果目录：与 day6 风格一致，把生成物放在 day7/results/ 下
# __file__ 是当前脚本路径，abspath 转绝对路径，dirname 取目录，
# 这样无论从哪个 cwd 运行都能正确定位到 day7/results/。
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ============================================================
# 工具 1: API 成本计算器（验证第 ④ 题理解）
# ============================================================
# 默认单价参考 DeepSeek-V4 Flash 公开定价（元/千 token）。
# 输入与输出单价不同是 LLM 计费核心特征：输出通常贵 2-4 倍，
# 原因是生成 token 需逐个自回归前向传播，而输入只需一次前向。
DEFAULT_INPUT_PRICE = 0.0001   # 元/千 token
DEFAULT_OUTPUT_PRICE = 0.0002   # 元/千 token
USD_TO_CNY = 7.2               # 汇率近似值，用于换算美元


def calculate_api_cost(input_tokens: int, output_tokens: int,
                       input_price: float = DEFAULT_INPUT_PRICE,
                       output_price: float = DEFAULT_OUTPUT_PRICE) -> dict:
    """
    计算 API 调用成本。

    【计费原理】
    LLM 按 token 数量计费（不是按字数/字符数），且输入与输出分开计价：
      总费用 = 输入 tokens × 输入单价 + 输出 tokens × 输出单价
    输出单价通常是输入的 2-4 倍，因为生成 token 需逐个自回归前向传播，
    而输入只需一次前向（prefill）。

    Args:
        input_tokens: 输入 token 数（prompt 消耗）
        output_tokens: 输出 token 数（completion 消耗）
        input_price:  输入单价（元/千 token），默认 DeepSeek-V4 Flash 参考价
        output_price: 输出单价（元/千 token）

    Returns:
        详细费用字典，包含输入/输出/总计的人民币与近似美元金额。
        所有金额保留 6 位小数，避免小额调用显示为 0。
    """
    # 注意 "千 token" 单位：单价是"元/千token"，所以要先除以 1000 再乘单价
    # 例如：80 tokens × 0.0001 元/千token = 80/1000 × 0.0001 = 0.000008 元
    input_cost = input_tokens / 1000 * input_price
    output_cost = output_tokens / 1000 * output_price
    total_cost = input_cost + output_cost

    # round(..., 6) 保留 6 位小数：单次调用费用往往极小（<0.01 元），
    # 保留足够位数才能在报告中看到差异，便于对比不同场景成本。
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price": input_price,
        "output_price": output_price,
        "input_cost_yuan": round(input_cost, 6),
        "output_cost_yuan": round(output_cost, 6),
        "total_cost_yuan": round(total_cost, 6),
        # 美元近似值：用人民币除以汇率，方便与国际定价对比
        "total_cost_usd_approx": round(total_cost / USD_TO_CNY, 6),
    }


def demo_cost_calculator() -> list:
    """
    演示不同场景的成本，并支持交互式自定义计算。

    【设计意图】
    通过 4 个典型场景让用户直观感受"不同任务 token 消耗差异巨大"：
    - 短对话 vs 长文摘要：输入长度差 40 倍，成本差约 16 倍
    - 代码生成 vs CoT 推理：输出长度决定成本（输出单价更贵）
    再让用户自己输入 token 数计算，强化"按 token 计费"的直觉。

    Returns:
        场景结果列表 [(场景名, 费用字典), ...]，供报告生成使用。
    """
    print("=" * 60)
    print("💰 API 成本计算器演示")
    print("=" * 60)
    print(f"参考单价: 输入 ¥{DEFAULT_INPUT_PRICE}/千token, "
          f"输出 ¥{DEFAULT_OUTPUT_PRICE}/千token")
    print(f"汇率近似: 1 USD = {USD_TO_CNY} CNY\n")

    # 四个典型场景，对应 notes_review.md 中的成本表
    # 【token 数估算依据】
    # 中文 1 字 ≈ 1-2 tokens（BPE 常把一个汉字切成 1-2 个子词）
    # 英文 1 词 ≈ 1.3 tokens（常见词 1 个 token，生僻词拆成多个）
    # 因此：
    #   - 50 字中文 ≈ 80 tokens（含 system prompt 开销）
    #   - 2000 字中文 ≈ 3500 tokens
    #   - 500 词英文 ≈ 650 tokens
    #   - 100 字提问 + 500 字回答 ≈ 180 + 900 tokens
    scenarios = [
        ("短对话(中文50字)",            80,   120),
        ("长文摘要(中文2000字)",        3500, 800),
        ("代码生成(英文500词)",         650,  1200),
        ("CoT推理(中文100字问+500字答)", 180,  900),
    ]

    results = []
    for name, inp, out in scenarios:
        r = calculate_api_cost(inp, out)
        results.append((name, r))
        print(f"\n📋 {name}")
        print(f"   输入: {inp} tokens → ¥{r['input_cost_yuan']:.4f}")
        print(f"   输出: {out} tokens → ¥{r['output_cost_yuan']:.4f}")
        print(f"   💵 总计: ¥{r['total_cost_yuan']:.4f} "
              f"(≈${r['total_cost_usd_approx']:.4f})")

    # 互动计算：让用户自己输入 token 数，强化"按 token 计费"的直觉
    # 这一步把"看演示"变成"自己算"，符合 active recall 学习法
    print("\n" + "-" * 60)
    print("自定义计算（直接回车跳过）")
    try:
        inp_str = input("输入你的 token 数(输入): ").strip()
        out_str = input("输入你的 token 数(输出): ").strip()
        # 空输入兜底为 0，避免 int("") 抛 ValueError
        inp = int(inp_str) if inp_str else 0
        out = int(out_str) if out_str else 0
        if inp > 0 or out > 0:
            r = calculate_api_cost(inp, out)
            print(f"\n   输入: {inp} tokens → ¥{r['input_cost_yuan']:.4f}")
            print(f"   输出: {out} tokens → ¥{r['output_cost_yuan']:.4f}")
            print(f"   💵 总计: ¥{r['total_cost_yuan']:.4f} "
                  f"(≈${r['total_cost_usd_approx']:.4f})")
            results.append((f"自定义({inp}+{out})", r))
        else:
            print("   跳过自定义计算")
    except ValueError:
        # 用户输入非数字时的兜底，不让脚本崩溃
        print("   输入无效，跳过自定义计算")

    return results


def verify_with_real_api() -> dict:
    """
    调用真实 API 验证 usage 字段，让用户直观看到一次调用消耗多少 token。

    【为什么需要这一步】
    前面的成本计算器用的是"估算 token 数"，而真实 API 返回的
    usage 字段是"精确 token 数"。对比两者可以验证估算是否准确，
    也让用户亲眼看到 API 响应里确实有 usage 字段——这是计费的依据。

    这是第 ④ 题的"实证"环节：从 API 响应里亲眼看到
    usage.prompt_tokens / usage.completion_tokens / usage.total_tokens。
    """
    print("\n" + "=" * 60)
    print("🔬 真实 API 调用验证 usage 字段")
    print("=" * 60)

    # 延迟导入：openai 和 dotenv 只有在 --api 模式才需要，
    # 放在函数内导入可以避免在纯自测模式下因缺少依赖而报错
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError:
        print("❌ 需要 openai + python-dotenv，请先安装（day1 已装）")
        return {}

    # 从 .env 加载配置：API Key、Base URL、模型名
    # DeepSeek 兼容 OpenAI 协议，所以可以直接用 OpenAI SDK
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
    if not api_key:
        print("❌ 未配置 OPENAI_API_KEY，请检查 .env")
        return {}

    client = OpenAI(api_key=api_key, base_url=base_url)
    # 用一个简单 prompt，便于观察 token 消耗
    prompt = "请用一句话解释什么是注意力机制。"
    print(f"模型: {model}")
    print(f"Prompt: {prompt}\n")

    try:
        # temperature=0 让输出确定可复现，max_tokens=200 限制输出长度控制成本
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
    except Exception as e:
        # API 调用可能因网络、鉴权、限流等失败，捕获异常不让脚本崩溃
        print(f"❌ API 调用失败: {e}")
        return {}

    # 提取回复内容和 usage 统计
    # choices[0].message.content 是模型生成的文本
    # usage 对象包含 prompt_tokens（输入）、completion_tokens（输出）、total_tokens
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    print(f"回复: {content}\n")
    print("📊 usage 字段:")
    print(f"   prompt_tokens     = {usage.prompt_tokens}")
    print(f"   completion_tokens = {usage.completion_tokens}")
    print(f"   total_tokens      = {usage.total_tokens}")

    # 用真实 token 数算一次成本，对比估算值与实际值
    r = calculate_api_cost(usage.prompt_tokens, usage.completion_tokens)
    print(f"\n💵 按真实 token 数计算:")
    print(f"   输入: {usage.prompt_tokens} tokens → ¥{r['input_cost_yuan']:.6f}")
    print(f"   输出: {usage.completion_tokens} tokens → ¥{r['output_cost_yuan']:.6f}")
    print(f"   总计: ¥{r['total_cost_yuan']:.6f} (≈${r['total_cost_usd_approx']:.6f})")
    return {
        "prompt": prompt,
        "response": content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "cost": r,
    }
# ============================================================
# 工具 2: 交互式自测
# ============================================================
# 【自测题库设计】
# 5 道题对应 Day1-Day6 的核心知识点，每题 5 分，满分 25。
# 每道题包含 5 个字段：
#   - id: 题号，用于关联复习指引表 REVIEW_GUIDE
#   - topic: 主题，用于报告展示
#   - related_day: 关联的 Day，提示用户该题对应哪天的学习
#   - question: 题目文本，引导用户回答
#   - key_points: 评分锚点列表，用户自评时对照，命中 1 个得 1 分（最多 5 分）
#   - one_liner: 一句话速记，帮助形成长期记忆
SELFTEST_QUESTIONS = [
    {
        "id": 1,
        "topic": "注意力机制",
        "related_day": "Day 4",
        "question": "什么是注意力机制？请提到 QKV 和缩放因子。",
        "key_points": [
            "Q(Query)、K(Key)、V(Value) 三元组",
            "softmax(QK^T / √d_k) × V",
            "√d_k 缩放防止梯度消失",
            "解决长距离依赖 / 支持并行",
            "多头注意力学习不同关注模式",
        ],
        "one_liner": "注意力 = 用 Q 和 K 算相关性权重，对 V 加权求和，实现任意位置间的直接信息交互。",
    },
    {
        "id": 2,
        "topic": "CoT 原理",
        "related_day": "Day 3",
        "question": "解释 CoT 提示的原理，至少说出两个原因。",
        "key_points": [
            "分解复杂度：多步拆成单步",
            "工作记忆：推理 token 作中间表示",
            "概率路径：为答案构建条件概率",
            "Few-shot CoT 优于 Zero-shot CoT",
        ],
        "one_liner": "CoT 通过生成中间推理步骤，将复杂问题分解为简单子问题，同时为模型提供工作记忆。",
    },
    {
        "id": 3,
        "topic": "Temperature",
        "related_day": "Day 6",
        "question": "温度参数如何影响输出？请从数学角度解释。",
        "key_points": [
            "P = softmax(logit / T)",
            "T→0: 分布尖锐，确定性输出",
            "T>1: 分布平坦，随机性增加",
            "实际应用：代码用低温，创意用高温",
        ],
        "one_liner": "温度是 softmax 的锐度旋钮：低温=确定性，高温=多样性，本质是对 logits 做除法改变概率分布形状。",
    },
    {
        "id": 4,
        "topic": "API 成本计算",
        "related_day": "Day 1",
        "question": "如何计算 API 调用成本？中英文 token 密度有何差异？",
        "key_points": [
            "按 token 计费，不是按字数",
            "输入和输出单价不同（输出更贵）",
            "中文 1 字 ≈ 1-2 tokens",
            "英文 1 词 ≈ 1.3 tokens",
            "费用 = 输入 tokens × 单价 + 输出 tokens × 单价",
        ],
        "one_liner": "费用 = Σ(token 数 × 单价)，输入输出分开计价，中文比英文费 token。",
    },
    {
        "id": 5,
        "topic": "幻觉成因",
        "related_day": "Day 6",
        "question": "为什么大模型会产生幻觉？如何防范？",
        "key_points": [
            "LLM 是概率续写器，非知识检索器",
            "不知道自己的知识边界",
            "softmax 使输出总是显得自信",
            "防范：RAG、引用来源、降温、拒绝指令",
        ],
        "one_liner": "幻觉是 LLM 作为概率语言模型的固有特性：优化文本流畅性而非事实准确性，不知道自己不知道。",
    },
]

# 针对性复习指引表：卡壳题目 -> (复习资源, 重点看)
# 【设计意图】
# 自测后如果有题目得分 < 3（卡壳），脚本会自动查这张表，
# 告诉用户该去看哪些资料、重点看什么，实现"测-学闭环"。
# key 是题号，value 是 (复习资源, 重点内容) 元组。
REVIEW_GUIDE = {
    1: ("Day4 笔记 + 3Blue1Brown 视频", "QKV 计算过程、缩放因子直觉"),
    2: ("Day3 笔记 + Wei et al. 论文摘要", "三个原理、Few-shot vs Zero-shot"),
    3: ("Day6 笔记 + 实验代码输出", "softmax 公式、温度 vs 分布图"),
    4: ("Day1 笔记 + 官方定价页", "输入/输出单价差异、token 估算"),
    5: ("Day6 笔记 + 幻觉实验报告", "5 种幻觉类型、防范策略"),
}


def _read_score(prompt: str) -> int:
    """
    安全读取 0-5 的整数评分，对非法输入循环提示。

    【为什么单独抽一个函数】
    input() 的原始返回是字符串，需要转 int；用户可能输入字母、空值、
    超范围数字等。这里统一兜底，避免主流程被 try/except 污染，也保证
    评分逻辑一致（0-5 整数，其他一律重新提示）。
    """
    while True:
        try:
            val = int(input(prompt).strip())
            if 0 <= val <= 5:
                return val
            print("   请输入 0-5 之间的整数")
        except ValueError:
            # 用户输入非数字（如字母、空串）时 int() 抛 ValueError
            print("   请输入数字")


def run_selftest() -> dict:
    """
    交互式自测流程：每题先回答 → 看参考要点 → 自评打分 → 累计。

    【流程设计】
    每道题分 4 步：
      1) 展示题目，等用户按回车（逼自己先想答案，再看参考）
      2) 展示参考要点 + 一句话速记
      3) 用户自评 0-5 分（对照 key_points 命中数打分）
      4) 累计得分，记录是否卡壳（<3 分）
    全部答完后按行动决策树判定通过/部分通过/未通过，并列出复习指引。

    Returns:
        自测结果字典，供报告生成使用。包含总分、百分比、判定、每题详情。
    """
    print("\n" + "=" * 60)
    print("📝 第一周自测（共 5 题，每题 5 分，满分 25 分）")
    print("=" * 60)
    print("规则: 先口头/书面回答，再按回车查看参考答案\n")

    total_score = 0      # 累计得分
    details = []         # 每题详情列表，供报告生成

    for q in SELFTEST_QUESTIONS:
        print(f"\n{'─' * 50}")
        print(f"❓ 第{q['id']}题 [{q['topic']} · 关联 {q['related_day']}]")
        print(f"   {q['question']}")
        print(f"{'─' * 50}")

        # 用户先回答（不收集内容，只做"逼自己输出"的仪式感）
        # 【为什么用 input() 而不收集内容】
        # active recall 的关键是"先回忆再看答案"，input() 起暂停作用，
        # 强制用户先在脑中/口头/纸上组织答案，再按回车看参考要点。
        input("   👉 回答完毕后按回车查看答案...")

        # 展示参考要点：用户对照自己答案，命中 1 个得 1 分
        print(f"\n   📋 参考要点（命中 1 个得 1 分，最多 5 分）:")
        for i, point in enumerate(q["key_points"], 1):
            print(f"      {i}. {point}")
        # 一句话速记帮助形成长期记忆，便于日后快速回忆
        print(f"\n   🔑 一句话速记: {q['one_liner']}")

        # 用户自评打分，_read_score 保证输入合法
        score = _read_score(f"\n   自评得分 (0-5): ")
        total_score += score
        # 得分 < 3 视为卡壳，需要针对性复习（对应行动决策树）
        stuck = score < 3
        details.append({
            "id": q["id"],
            "topic": q["topic"],
            "related_day": q["related_day"],
            "score": score,
            "stuck": stuck,
        })
        print(f"   累计: {total_score}/25")

    # 结果判定：对应 notes_review.md 的行动决策树
    #   >=18 (72%): 通过，可进入第二周
    #   14-17: 部分通过，重学薄弱题后重测
    #   <14: 未通过，系统重学 Day1-Day6，3 天后重试
    percent = total_score / 25 * 100
    print(f"\n{'═' * 60}")
    print(f"🏁 自测结果: {total_score}/25 ({percent:.0f}%)")
    print(f"{'═' * 60}")

    if total_score >= 18:
        verdict = "✅ 通过！可以进入第二周学习。"
        # 即使总分达标，个别题 <3 分仍建议针对性复习
        if any(d["stuck"] for d in details):
            verdict += "（个别题目 <3 分，建议针对性复习）"
    elif total_score >= 14:
        verdict = "⚠️ 部分通过。请针对薄弱题目复习后重新自测。"
    else:
        verdict = "❌ 未通过。建议系统复习 Day1-Day6 笔记，3 天后重试。"
    print(verdict)

    # 列出卡壳题目及复习指引，形成"测-学闭环"
    stuck_list = [d for d in details if d["stuck"]]
    if stuck_list:
        print(f"\n📌 需复习的题目:")
        for d in stuck_list:
            res, focus = REVIEW_GUIDE[d["id"]]
            print(f"   第{d['id']}题 [{d['topic']}]: {res}")
            print(f"      重点: {focus}")

    return {
        "total_score": total_score,
        "max_score": 25,
        "percent": round(percent, 1),
        "verdict": verdict,
        "details": details,
    }


# ============================================================
# 报告生成
# ============================================================
def save_report(selftest_result: dict = None, cost_scenarios: list = None,
                api_result: dict = None) -> str:
    """
    把自测与成本计算结果写入 markdown 报告，便于复盘追踪进步。

    【为什么生成报告】
    自测是一次性的，但进步需要长期追踪。把每次自测结果存成 markdown，
    可以对比不同时间的得分变化，看到自己的进步轨迹。报告分四部分：
      一、自测结果（总分 + 每题得分 + 卡壳标记 + 复习指引）
      二、API 成本计算（各场景费用对比表）
      三、真实 API usage 验证（实际 token 消耗与费用）
      四、复习心态提醒（固定文案，强化成长型思维）

    Args:
        selftest_result: run_selftest() 的返回值，None 表示本次未自测
        cost_scenarios: demo_cost_calculator() 的返回值，None 表示未算成本
        api_result: verify_with_real_api() 的返回值，None 表示未调 API

    Returns:
        报告文件绝对路径。
    """
    # 确保结果目录存在，exist_ok=True 避免目录已存在时报错
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 时间戳用于追踪自测历史，对比不同时间的进步
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 用列表收集 markdown 行，最后 join 成完整文本，比字符串拼接高效
    lines = [
        f"# 第一周复习自测报告",
        f"",
        f"> 生成时间: {ts}",
        f"",
        f"## 一、自测结果",
        f"",
    ]

    # --- 第一部分：自测结果 ---
    if selftest_result:
        sr = selftest_result
        lines += [
            f"- **总分**: {sr['total_score']}/{sr['max_score']} ({sr['percent']}%)",
            f"- **判定**: {sr['verdict']}",
            f"",
            f"| 题号 | 主题 | 关联 Day | 得分(/5) | 是否卡壳 |",
            f"|---|---|---|---|---|",
        ]
        for d in sr["details"]:
            # 卡壳标记用 emoji 直观区分，便于扫读
            stuck_mark = "⚠️ 是" if d["stuck"] else "✅ 否"
            lines.append(
                f"| {d['id']} | {d['topic']} | {d['related_day']} | {d['score']} | {stuck_mark} |"
            )
        lines += [f""]

        # 针对性复习指引：只列出卡壳题目，避免信息过载
        stuck_list = [d for d in sr["details"] if d["stuck"]]
        if stuck_list:
            lines += [
                f"### 针对性复习指引",
                f"",
                f"| 题号 | 主题 | 复习资源 | 重点看 |",
                f"|---|---|---|---|",
            ]
            for d in stuck_list:
                res, focus = REVIEW_GUIDE[d["id"]]
                lines.append(f"| {d['id']} | {d['topic']} | {res} | {focus} |")
            lines += [f""]
    else:
        lines += [f"_(本次未运行自测)_", f""]

    # --- 第二部分：API 成本计算 ---
    lines += [f"## 二、API 成本计算", f""]
    if cost_scenarios:
        lines += [
            f"| 场景 | 输入 tokens | 输出 tokens | 输入费(¥) | 输出费(¥) | 总费(¥) | 总费($) |",
            f"|---|---|---|---|---|---|---|",
        ]
        for name, r in cost_scenarios:
            lines.append(
                f"| {name} | {r['input_tokens']} | {r['output_tokens']} | "
                f"{r['input_cost_yuan']:.4f} | {r['output_cost_yuan']:.4f} | "
                f"{r['total_cost_yuan']:.4f} | {r['total_cost_usd_approx']:.4f} |"
            )
        lines += [f""]
    else:
        lines += [f"_(本次未运行成本计算器)_", f""]

    # --- 第三部分：真实 API usage 验证 ---
    lines += [f"## 三、真实 API usage 验证", f""]
    if api_result:
        u = api_result["usage"]
        c = api_result["cost"]
        lines += [
            f"- **Prompt**: {api_result['prompt']}",
            f"- **回复**: {api_result['response']}",
            f"- **prompt_tokens**: {u['prompt_tokens']}",
            f"- **completion_tokens**: {u['completion_tokens']}",
            f"- **total_tokens**: {u['total_tokens']}",
            f"- **实际费用**: ¥{c['total_cost_yuan']:.6f} (≈${c['total_cost_usd_approx']:.6f})",
            f"",
        ]
    else:
        lines += [f"_(本次未调用真实 API，使用 --api 可启用)_", f""]

    # --- 第四部分：复习心态提醒（固定文案） ---
    lines += [
        f"## 四、复习心态提醒",
        f"",
        f"> 自测的目的不是『证明自己都会』，而是『精准定位不会的地方』。",
        f"> 卡壳是好事情——它告诉你下一步该学什么。70% 通过率意味着允许 30% 的知识盲区，",
        f"> 这些盲区正是第二周之前需要填补的。",
    ]

    # 写入文件，encoding="utf-8" 保证中文与 emoji 正确保存
    path = os.path.join(RESULTS_DIR, "selftest_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 报告已保存: {path}")
    return path


# ============================================================
# 主入口
# ============================================================
def main():
    """
    主入口：解析命令行参数，分发到对应工具，最后生成报告。

    【命令行设计】
    支持两种使用方式：
      1) 带参数直达某个工具：--cost / --test / --api
      2) 无参数进入菜单模式，交互选择
    所有模式结束后都会调用 save_report 生成 markdown 报告。
    """
    # sys.argv[1:] 是命令行参数（去掉脚本名），set() 转集合便于 in 判断
    args = set(sys.argv[1:])

    # --- 命令行参数直达某个工具 ---
    if "--cost" in args:
        cost_scenarios = demo_cost_calculator()
        save_report(None, cost_scenarios, None)
        return
    if "--test" in args:
        result = run_selftest()
        save_report(result, None, None)
        return
    if "--api" in args:
        api_result = verify_with_real_api()
        save_report(None, None, api_result)
        return

    # --- 菜单模式：展示功能选项让用户选择 ---
    print("╔══════════════════════════════════════════════════╗")
    print("║       第一周复习自测工具                         ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  1. API 成本计算器                               ║")
    print("║  2. 交互式自测                                    ║")
    print("║  3. 真实 API 验证 usage 字段                       ║")
    print("║  4. 全部运行（成本 + 自测 + 报告）                ║")
    print("╚══════════════════════════════════════════════════╝")

    choice = input("\n选择功能 (1/2/3/4): ").strip()

    # 根据选择分发到对应工具，每个分支都生成报告
    if choice == "1":
        cost_scenarios = demo_cost_calculator()
        save_report(None, cost_scenarios, None)
    elif choice == "2":
        result = run_selftest()
        save_report(result, None, None)
    elif choice == "3":
        api_result = verify_with_real_api()
        save_report(None, None, api_result)
    elif choice == "4":
        # 全流程：成本计算 → 自测 → 真实 API 验证 → 综合报告
        print("\n>>> 步骤 1/3: API 成本计算器")
        cost_scenarios = demo_cost_calculator()
        print("\n>>> 步骤 2/3: 交互式自测")
        result = run_selftest()
        print("\n>>> 步骤 3/3: 真实 API 验证（可选，失败不影响报告）")
        api_result = verify_with_real_api()
        save_report(result, cost_scenarios, api_result)
    else:
        # 无效输入兜底为自测，保证用户总能拿到一份报告
        print("无效选择，默认运行自测")
        result = run_selftest()
        save_report(result, None, None)


# 标准的 Python 脚本入口：仅在被直接运行时执行 main()，
# 被 import 时不执行（避免导入即触发交互）
if __name__ == "__main__":
    main()
