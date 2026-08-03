"""
进阶提示技巧实验：Zero-shot / Zero-shot CoT / Few-shot CoT / 分支提示（ToT 简化）

任务：在数学应用题上对比四种提示策略的推理质量与准确率。
- ① Zero-shot 直接回答：仅给问题，要求直接输出数字。
- ② Zero-shot CoT：问题 + "让我们一步一步思考"。
- ③ Few-shot CoT：2 个带完整推理过程的示例 + 问题，并用 "#### 数字" 锚定答案格式。
- ④ 分支提示（ToT 简化）：从正向 / 逆向 / 边界三个角度分别推理，再综合得出答案。

模型：DeepSeek V4 Flash（默认开启 thinking，已对 reasoning_content 做兜底提取）。
运行：python day3/cot_lab.py
"""
import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ============ 初始化 ============
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 结果输出目录（放在 day3 下，避免污染 day2）
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def call_llm(messages: list, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """调用 DeepSeek V4 Flash，并对 thinking 模式做 reasoning_content 兜底。"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        if content:
            return content
        # content 为空时（thinking 模式下 token 用尽），兜底取 reasoning_content
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        return reasoning or "(空)"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def extract_final_answer(text: str) -> str:
    """
    从模型输出中提取最终数字答案。
    支持匹配: "#### 11"、"答案是11个"、"答案：11"、"还剩 11 个"、"结果是 11" 等。
    """
    patterns = [
        r'####\s*(\d+)',
        r'(?:最终)?答案[是为：:\s]*(\d+)',
        r'还剩\s*(\d+)\s*个',
        r'结果是\s*(\d+)',
        r'(\d+)\s*个(?:苹果|个)?[。.]?\s*$',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    # 兜底：取文本中最后一个独立数字
    nums = re.findall(r'\b(\d+)\b', text)
    return nums[-1] if nums else "未提取到"


def save_result(filename: str, content: str):
    path = os.path.join(RESULT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📄 已保存 → {path}")


# ============================================================
# 题目定义
# ============================================================
SIMPLE_Q = "小明有10个苹果，买了3个，吃了2个，还剩几个？"
HARD_Q = (
    "一个水池有进水管和出水管。进水管每小时注入3吨水，"
    "出水管每小时排出1吨水。水池容量为50吨，当前已有10吨水。"
    "如果同时打开进水管和出水管，多少小时后水池恰好注满？"
)
CORRECT_SIMPLE = "11"
CORRECT_HARD = "20"


# ============================================================
# 方法一：Zero-shot 直接回答
# ============================================================
def method_zeroshot(question: str) -> str:
    messages = [
        {"role": "user", "content": f"{question}\n请直接给出最终答案，只需一个数字。"}
    ]
    return call_llm(messages, temperature=0.1)


# ============================================================
# 方法二：Zero-shot CoT
# ============================================================
def method_zeroshot_cot(question: str) -> str:
    messages = [
        {"role": "user", "content": f"{question}\n\n让我们一步一步思考。"}
    ]
    return call_llm(messages, temperature=0.3)


# ============================================================
# 方法三：Few-shot CoT（2 个带推理过程的示例）
# ============================================================
def method_fewshot_cot(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是一个数学推理助手。请一步一步展示推理过程，最后用 \"#### 数字\" 给出最终答案。"
        },
        # --- 示例 1 ---
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
        # --- 示例 2 ---
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
        # --- 待解答 ---
        {"role": "user", "content": question},
    ]
    return call_llm(messages, temperature=0.2)


# ============================================================
# 方法四：分支提示（Tree of Thoughts 简化版）
# ============================================================
def method_branch(question: str) -> str:
    """
    生成 3 条不同假设/角度的推理路径，然后让模型综合判断。
    """
    branch_prompt = (
        f"{question}\n\n"
        "请你从以下三个不同角度分别推理，然后综合得出最终答案：\n\n"
        "【路径A - 正向逐步计算】\n"
        "按照题目叙述的顺序，一步步计算。\n\n"
        "【路径B - 逆向验证】\n"
        "假设一个答案，反向代入验证是否满足题目所有条件。\n\n"
        "【路径C - 极端/边界检验】\n"
        "检查是否存在边界情况或容易忽略的细节（如：是否同时操作、单位是否一致等）。\n\n"
        "最后综合三条路径，用 \"#### 数字\" 给出最终答案。"
    )

    messages = [
        {"role": "system", "content": "你是一个严谨的数学分析师，擅长多角度验证。"},
        {"role": "user", "content": branch_prompt},
    ]
    return call_llm(messages, temperature=0.4, max_tokens=2000)


# ============================================================
# 对比实验主逻辑
# ============================================================
def run_comparison(question: str, correct_answer: str, label: str) -> dict:
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
    print(f"   输出: {r1.strip()[:120]}")
    print(f"   提取答案: {a1}  {'✅' if a1 == correct_answer else '❌'}")
    time.sleep(0.5)

    # --- 方法 2: Zero-shot CoT ---
    print("\n🔹 [2/4] Zero-shot CoT...")
    r2 = method_zeroshot_cot(question)
    a2 = extract_final_answer(r2)
    results["Zero-shot CoT"] = {"output": r2, "answer": a2, "correct": a2 == correct_answer}
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
    print(f"   多路径推理:\n{r4.strip()[:400]}")
    print(f"   提取答案: {a4}  {'✅' if a4 == correct_answer else '❌'}")

    return results


def print_summary_table(all_results: dict):
    """打印汇总对比表"""
    print(f"\n\n{'═' * 60}")
    print("📊 汇总对比表")
    print(f"{'═' * 60}")
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
    """保存完整实验报告（Markdown）"""
    report_lines = [
        "# CoT 与分支提示实验报告（DeepSeek V4 Flash）",
        "",
        f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 模型: {MODEL}",
        "",
        "## 题目",
        f"- **简单题**: {SIMPLE_Q}（正确答案: {CORRECT_SIMPLE}）",
        f"- **进阶题**: {HARD_Q}（正确答案: {CORRECT_HARD}）",
        "",
    ]

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

    # 汇总表
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

    start = time.time()
    all_results = {}

    # 简单题
    all_results["simple"] = run_comparison(SIMPLE_Q, CORRECT_SIMPLE, "简单题")

    # 进阶题
    all_results["hard"] = run_comparison(HARD_Q, CORRECT_HARD, "进阶题")

    # 汇总
    print_summary_table(all_results)

    # 保存报告
    save_full_report(all_results)

    elapsed = time.time() - start
    print(f"\n🏁 实验完成！总耗时: {elapsed:.1f}s")
