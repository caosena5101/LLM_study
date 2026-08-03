"""
进阶提示技巧实验：Zero-shot / Zero-shot CoT / Few-shot CoT / 分支提示（ToT 简化）

任务：在数学应用题上对比四种提示策略的推理质量与准确率。
- ① Zero-shot 直接回答：仅给问题，要求直接输出数字。
- ② Zero-shot CoT：问题 + "让我们一步一步思考"。
- ③ Few-shot CoT：2 个带完整推理过程的示例 + 问题，并用 "#### 数字" 锚定答案格式。
- ④ 分支提示（ToT 简化）：从正向 / 逆向 / 边界三个角度分别推理，再综合得出答案。

模型：DeepSeek V4 Flash（默认开启 thinking，已对 reasoning_content 做兜底提取）。
运行：python day3/cot_lab.py

核心原理：
- CoT 有效原因：① 分解复杂度（多步拆单步，每步搜索空间更小）
                ② 中间表示（推理步骤作"工作记忆"，防止信息丢失）
                ③ 概率路径（推理 token 为最终答案构建更精确的条件概率）
- Few-shot CoT 最稳：示例锚定输出结构（#### 数字），下游解析最可靠。
- 分支提示（ToT 简化）：多路径交叉验证，提高高难度/歧义题的置信度。
"""
import os
import re
import time
# datetime：用于在报告中记录实验时间戳
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ============ 初始化 ============
# 加载 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
load_dotenv()

# DeepSeek 兼容 OpenAI 协议，只需把 base_url 指向 https://api.deepseek.com
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
# 模型名兜底为 deepseek-v4-flash
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 结果输出目录：放在 day3 下（基于本文件位置计算），避免污染 day2
# exist_ok=True：目录已存在时不报错
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def call_llm(messages: list, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """
    调用 DeepSeek V4 Flash，并对 thinking 模式做 reasoning_content 兜底。

    DeepSeek V4 默认开启 thinking：先生成 reasoning_content（推理），
    再生成 content（最终回答）。若 max_tokens 不足以容纳两段，
    可能只输出 reasoning_content 而 content 为空——此时回退取推理过程。
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        # 优先取 content（thinking 模式下最终回答在此）
        content = (msg.content or "").strip()
        if content:
            return content
        # 兜底：content 为空时（thinking 用尽 token），取推理过程
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        return reasoning or "(空)"
    except Exception as e:
        # 网络异常、鉴权失败、限流等统一返回错误字符串，保证脚本不崩
        return f"[ERROR] {type(e).__name__}: {e}"


def extract_final_answer(text: str) -> str:
    """
    从模型输出中提取最终数字答案。

    按优先级依次尝试多种正则，匹配到即返回；全部不匹配则兜底取最后一个独立数字。
    支持匹配: "#### 11"、"答案是11个"、"答案：11"、"还剩 11 个"、"结果是 11" 等。
    """
    # 按优先级排列：格式越规范（如 #### 11）越靠前，越能避免误匹配
    patterns = [
        r'####\s*(\d+)',                              # Few-shot CoT 锚定的格式：#### 11
        r'(?:最终)?答案[是为：:\s]*(\d+)',            # "答案是11" / "最终答案: 11"
        r'还剩\s*(\d+)\s*个',                          # "还剩 11 个"
        r'结果是\s*(\d+)',                             # "结果是 11"
        r'(\d+)\s*个(?:苹果|个)?[。.]?\s*$',           # 末尾的 "11 个苹果。"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    # 兜底：取文本中最后一个独立数字（\b 为词边界，避免匹配日期/编号中的子串）
    nums = re.findall(r'\b(\d+)\b', text)
    return nums[-1] if nums else "未提取到"


def save_result(filename: str, content: str):
    """将实验报告写入 results 目录，使用 utf-8 编码以支持中文与 emoji。"""
    path = os.path.join(RESULT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📄 已保存 → {path}")


# ============================================================
# 题目定义
# ============================================================
# 简单题：用户指定的苹果问题，所有方法都应答对，用于验证流程
SIMPLE_Q = "小明有10个苹果，买了3个，吃了2个，还剩几个？"
# 进阶题：水池进出水问题，需多步推理（净注入速率 + 剩余容量），用于体现 CoT 优势
# 解题：净注入 3-1=2 吨/小时，还需 50-10=40 吨，故需 40/2=20 小时
HARD_Q = (
    "一个水池有进水管和出水管。进水管每小时注入3吨水，"
    "出水管每小时排出1吨水。水池容量为50吨，当前已有10吨水。"
    "如果同时打开进水管和出水管，多少小时后水池恰好注满？"
)
# 正确答案（字符串形式，便于与提取结果直接比较）
CORRECT_SIMPLE = "11"
CORRECT_HARD = "20"


# ============================================================
# 方法一：Zero-shot 直接回答
# ============================================================
# 仅给问题，要求直接输出数字，不要求推理过程
# 预期：简单题能答对，复杂题可能跳步出错
def method_zeroshot(question: str) -> str:
    messages = [
        {"role": "user", "content": f"{question}\n请直接给出最终答案，只需一个数字。"}
    ]
    # 低温度：要求确定性输出，减少格式漂移
    return call_llm(messages, temperature=0.1)


# ============================================================
# 方法二：Zero-shot CoT
# ============================================================
# 在问题后追加"让我们一步一步思考"——无需示例即可激活推理链
# 论文：Kojima et al., "Large Language Models are Zero-Shot Reasoners"
def method_zeroshot_cot(question: str) -> str:
    messages = [
        {"role": "user", "content": f"{question}\n\n让我们一步一步思考。"}
    ]
    # 略高温度：允许推理过程有一定发散，但不过分
    return call_llm(messages, temperature=0.3)


# ============================================================
# 方法三：Few-shot CoT（2 个带推理过程的示例）
# ============================================================
# 提供 2 个带完整推理过程的示例，并用 "#### 数字" 锚定答案格式
# 论文：Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in LLMs", NeurIPS 2022
# 关键：assistant 示例展示分步推理 + 统一答案格式，模型严格模仿
def method_fewshot_cot(question: str) -> str:
    messages = [
        {
            "role": "system",
            # system 中固定输出格式，便于下游正则提取（#### 数字）
            "content": "你是一个数学推理助手。请一步一步展示推理过程，最后用 \"#### 数字\" 给出最终答案。"
        },
        # --- 示例 1：加减混合运算 ---
        {"role": "user", "content": "小红有5颗糖，妈妈又给了她4颗，她分给同学3颗，还剩几颗？"},
        {
            "role": "assistant",
            "content": (
                "让我们一步一步思考：\n"
                "1. 小红一开始有 5 颗糖。\n"
                "2. 妈妈给了她 4 颗，所以现在有 5 + 4 = 9 颗。\n"
                "3. 她分给同学 3 颗，所以剩下 9 - 3 = 6 颗。\n\n"
                "#### 6"
            ),
        },
        # --- 示例 2：与目标题结构相似（先减后加）---
        {"role": "user", "content": "书架上有20本书，借走了7本，又新购入5本，现在有几本？"},
        {
            "role": "assistant",
            "content": (
                "让我们一步一步思考：\n"
                "1. 书架上原有 20 本书。\n"
                "2. 借走了 7 本，剩下 20 - 7 = 13 本。\n"
                "3. 又新购入 5 本，现在有 13 + 5 = 18 本。\n\n"
                "#### 18"
            ),
        },
        # --- 待解答：模型延续示例模式，输出推理链 + #### 数字 ---
        {"role": "user", "content": question},
    ]
    # 低温度：示例已锚定格式，低温度保证输出稳定
    return call_llm(messages, temperature=0.2)


# ============================================================
# 方法四：分支提示（Tree of Thoughts 简化版）
# ============================================================
# 对同一问题生成多条推理路径（正向/逆向/边界），交叉验证后综合
# 论文：Yao et al., "Tree of Thoughts: Deliberate Problem Solving with LLMs", NeurIPS 2023
# 简化：不做 BFS/DFS 搜索与打分剪枝，仅用单次 prompt 让模型并行走三条路径
def method_branch(question: str) -> str:
    """
    生成 3 条不同假设/角度的推理路径，然后让模型综合判断。
    """
    branch_prompt = (
        f"{question}\n\n"
        "请你从以下三个不同角度分别推理，然后综合得出最终答案：\n\n"
        # 路径A：常规正向计算——按题目叙述顺序逐步算
        "【路径A - 正向逐步计算】\n"
        "按照题目叙述的顺序，一步步计算。\n\n"
        # 路径B：逆向验证——假设答案，反代检验是否满足全部条件
        "【路径B - 逆向验证】\n"
        "假设一个答案，反向代入验证是否满足题目所有条件。\n\n"
        # 路径C：边界检查——排查隐含条件、单位一致性、操作顺序等易错点
        "【路径C - 极端/边界检验】\n"
        "检查是否存在边界情况或容易忽略的细节（如：是否同时操作、单位是否一致等）。\n\n"
        # 综合三条路径给出最终答案，沿用 #### 数字 格式便于提取
        "最后综合三条路径，用 \"#### 数字\" 给出最终答案。"
    )

    messages = [
        {"role": "system", "content": "你是一个严谨的数学分析师，擅长多角度验证。"},
        {"role": "user", "content": branch_prompt},
    ]
    # 略高温度：鼓励多角度思考；max_tokens 加大以容纳三条路径的较长输出
    return call_llm(messages, temperature=0.4, max_tokens=2000)


# ============================================================
# 对比实验主逻辑
# ============================================================
def run_comparison(question: str, correct_answer: str, label: str) -> dict:
    """
    对单道题依次跑四种方法，自动提取答案并与正确答案比对。

    返回: {方法名: {output, answer, correct}} 字典
    """
    print(f"\n{'━' * 60}")
    print(f"📐 题目 [{label}]: {question}")
    print(f"   正确答案: {correct_answer}")
    print(f"{'━' * 60}")

    results = {}

    # --- 方法 1: Zero-shot ---
    print("\n🔹 [1/4] Zero-shot 直接回答...")
    r1 = method_zeroshot(question)
    a1 = extract_final_answer(r1)
    results["Zero-shot"] = {"output": r1, "answer": a1, "correct": a1 == correct_answer}
    # 截断显示前 120 字，避免刷屏
    print(f"   输出: {r1.strip()[:120]}")
    print(f"   提取答案: {a1}  {'✅' if a1 == correct_answer else '❌'}")
    # 请求间隔，避免触发限流
    time.sleep(0.5)

    # --- 方法 2: Zero-shot CoT ---
    print("\n🔹 [2/4] Zero-shot CoT...")
    r2 = method_zeroshot_cot(question)
    a2 = extract_final_answer(r2)
    results["Zero-shot CoT"] = {"output": r2, "answer": a2, "correct": a2 == correct_answer}
    # 推理链较长，显示前 300 字
    print(f"   推理链:\n{r2.strip()[:300]}")
    print(f"   提取答案: {a2}  {'✅' if a2 == correct_answer else '❌'}")
    time.sleep(0.5)

    # --- 方法 3: Few-shot CoT ---
    print("\n🔹 [3/4] Few-shot CoT (2 示例)...")
    r3 = method_fewshot_cot(question)
    a3 = extract_final_answer(r3)
    results["Few-shot CoT"] = {"output": r3, "answer": a3, "correct": a3 == correct_answer}
    print(f"   推理链:\n{r3.strip()[:300]}")
    print(f"   提取答案: {a3}  {'✅' if a3 == correct_answer else '❌'}")
    time.sleep(0.5)

    # --- 方法 4: 分支提示 ---
    print("\n🔹 [4/4] 分支提示 (ToT 简化)...")
    r4 = method_branch(question)
    a4 = extract_final_answer(r4)
    results["分支提示"] = {"output": r4, "answer": a4, "correct": a4 == correct_answer}
    # 分支输出最长，显示前 400 字
    print(f"   多路径推理:\n{r4.strip()[:400]}")
    print(f"   提取答案: {a4}  {'✅' if a4 == correct_answer else '❌'}")

    return results


def print_summary_table(all_results: dict):
    """打印汇总对比表：四种方法在简单题/进阶题上的答案与对错。"""
    print(f"\n\n{'═' * 60}")
    print("📊 汇总对比表")
    print(f"{'═' * 60}")
    # 表头：方法 | 简单题答案 | 简单题对错 | 进阶题答案 | 进阶题对错
    header = f"{'方法':<18} {'简单题答案':<12} {'简单题':^6} {'进阶题答案':<12} {'进阶题':^6}"
    print(header)
    print("─" * 60)
    for method in ["Zero-shot", "Zero-shot CoT", "Few-shot CoT", "分支提示"]:
        s = all_results["simple"][method]
        h = all_results["hard"][method]
        print(
            f"{method:<18} {s['answer']:<12} {'✅' if s['correct'] else '❌':^6} "
            f"{h['answer']:<12} {'✅' if h['correct'] else '❌':^6}"
        )
    print("─" * 60)


def save_full_report(all_results: dict):
    """保存完整实验报告（Markdown）：含每题每种方法的原始输出与汇总表。"""
    report_lines = [
        "# CoT 与分支提示实验报告（DeepSeek V4 Flash）",
        "",
        # 报告元信息：时间与模型，便于复现
        f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 模型: {MODEL}",
        "",
        "## 题目",
        f"- **简单题**: {SIMPLE_Q}（正确答案: {CORRECT_SIMPLE}）",
        f"- **进阶题**: {HARD_Q}（正确答案: {CORRECT_HARD}）",
        "",
    ]

    # 每道题的详细结果：方法名 + 对错 + 提取答案 + 原始输出（代码块）
    for q_label, q_key in [("简单题", "simple"), ("进阶题", "hard")]:
        report_lines.append(f"\n## {q_label}详细结果\n")
        for method, data in all_results[q_key].items():
            status = "✅ 正确" if data["correct"] else "❌ 错误"
            report_lines.append(f"### {method} [{status}]")
            report_lines.append(f"提取答案: {data['answer']}")
            report_lines.append("```")
            report_lines.append(data["output"])
            report_lines.append("```")
            report_lines.append("")

    # 汇总表：Markdown 表格形式，便于在 GitHub/IDE 预览
    report_lines.append("\n## 汇总\n")
    report_lines.append("| 方法 | 简单题 | 进阶题 |")
    report_lines.append("|------|--------|--------|")
    for method in ["Zero-shot", "Zero-shot CoT", "Few-shot CoT", "分支提示"]:
        s = "✅" if all_results["simple"][method]["correct"] else "❌"
        h = "✅" if all_results["hard"][method]["correct"] else "❌"
        report_lines.append(f"| {method} | {s} | {h} |")

    content = "\n".join(report_lines)
    save_result("cot_experiment_report.md", content)


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  进阶提示实验：CoT / Few-shot CoT / 分支提示            ║")
    print("║  模型: DeepSeek V4 Flash                                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 计时起点
    start = time.time()
    all_results = {}

    # 简单题：验证流程，所有方法都应答对
    all_results["simple"] = run_comparison(SIMPLE_Q, CORRECT_SIMPLE, "简单题")

    # 进阶题：多步推理，用于体现 CoT 相对 Zero-shot 的优势
    all_results["hard"] = run_comparison(HARD_Q, CORRECT_HARD, "进阶题")

    # 汇总：终端打印对比表
    print_summary_table(all_results)

    # 保存：生成 markdown 报告到 results 目录
    save_full_report(all_results)

    elapsed = time.time() - start
    print(f"\n🏁 实验完成！总耗时: {elapsed:.1f}s")
