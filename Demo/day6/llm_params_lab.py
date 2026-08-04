"""
大模型核心参数实验：Temperature / 幻觉检测 / 上下文窗口 / Frequency Penalty
============================================================================
本脚本通过四个实验帮助理解大模型推理时的核心参数与典型现象：
  1) Temperature 对比：同一 prompt 在 T=0 / 0.7 / 1.5 下各跑 10 次，
     统计多样性比率，验证"温度越高 -> 分布越平 -> 采样越随机"。
  2) 幻觉检测：用 10 个易触发幻觉的 prompt（虚构人物/书籍/奖项/技术/历史事件
     + 半真半假），观察模型是否编造，并至少识别出 3 个幻觉实例。
  3) 上下文窗口感知：构造不同长度输入，观察响应时间与截断/报错行为。
  4) Frequency Penalty：对同一描述任务设置 0 / 1.0 / 2.0，观察字符唯一率变化。

【为什么这个实验重要】
  大模型并非"确定性输出一个答案"，而是每步对词表做概率分布再采样。
  temperature / top_p / frequency_penalty 等参数控制"如何采样"，
  理解它们才能解释：为什么同一 prompt 多次结果不同、为什么会"编造"、
  为什么长输入会被截断——这些都是工程实践中最高频的坑。

【大模型采样的本质】
  生成式模型每一步的流程：
    1) 输入当前已生成序列 -> 模型前向传播 -> 最后一层输出 logits（V 维向量，V=词表大小）
    2) logits 经 softmax 转成概率分布 P(token_1), ..., P(token_V)
    3) 按 P 采样下一个 token（greedy / temperature / top_p / top_k 等策略）
    4) 把新 token 拼回序列，重复直到 EOS 或达到 max_tokens
  本脚本涉及的参数都作用在第 2、3 步：
    - temperature：在 softmax 前对 logits 缩放，改变分布尖锐度
    - top_p：在采样前裁剪候选集，只保留累积概率达 p 的最小集合
    - frequency_penalty：在采样前对已出现 token 的 logits 减分，抑制重复
    - max_tokens：硬性限制第 4 步的循环上界

模型：DeepSeek V4 Flash（默认开启 thinking，已对 reasoning_content 做兜底提取）。
运行：python day6/llm_params_lab.py
依赖：openai / python-dotenv（day1 已安装）；可选 matplotlib（用于绘图扩展）
"""

# ============================================================
# 模块导入说明
# ============================================================
# os      : 读取环境变量、构造跨平台路径、创建结果目录
# sys/io  : 检测并修复 Windows 控制台编码（GBK -> UTF-8）
# time    : sleep 避免触发 API 限流；计时观察上下文长度对延迟的影响
# datetime: 在 markdown 报告中记录实验时间戳，便于复盘
# dotenv  : 从 .env 文件加载 API Key 等敏感配置，避免硬编码进代码
# OpenAI  : 官方 SDK，DeepSeek 兼容 OpenAI 协议，可直接复用同一客户端
import os
import sys
import io
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# Windows 控制台编码修复
# ------------------------------------------------------------
# Windows 中文系统控制台默认用 GBK 编码，而模型输出可能包含 GBK 无法
# 表示的字符（emoji、生僻字、BPE 切碎的 UTF-8 字节片段等），会抛
# UnicodeEncodeError。这里把 stdout/stderr 强制重包装成 UTF-8：
#   encoding="utf-8"   -> 用 UTF-8 编码输出
#   errors="replace"   -> 遇到无法编码的字符用 ? 替代而不抛异常
# Linux/macOS 默认就是 UTF-8，这段代码对它们无副作用。
# 不修复的话，实验一打印含 emoji/生僻字的诗句时会直接崩溃。
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ============ 初始化 ============
# load_dotenv() 会从当前目录及父目录查找 .env 文件，把其中 KEY=VALUE
# 加载到 os.environ。本脚本与 day1/day2/day3 共用根目录 .env，避免
# 在多处重复配置 API Key（密钥管理最佳实践：集中存放、不入库）。
load_dotenv()

# DeepSeek 兼容 OpenAI 协议：API 路径、请求体、响应格式都与 OpenAI 一致，
# 只需把 base_url 从 https://api.openai.com 改成 https://api.deepseek.com。
# 这样我们复用官方 OpenAI SDK 即可，无需安装 DeepSeek 专用客户端。
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
# 模型名兜底为 deepseek-v4-flash：万一 .env 没配 OPENAI_MODEL 也能跑
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

# 结果输出目录：放在 day6 下（基于本文件位置计算），避免污染其他 day
# exist_ok=True：目录已存在时不报错（幂等，方便重复运行）
# 用 os.path.abspath(__file__) 而非相对路径，确保从任意目录运行都能正确写入
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def call_llm(prompt: str, temperature: float = 0.7, top_p: float = 1.0,
             frequency_penalty: float = 0.0, max_tokens: int = 300,
             thinking: bool = True) -> str:
    """
    通用调用封装：调用 DeepSeek V4 Flash，并对 thinking 模式做 reasoning_content 兜底。

    参数机制详解：
      temperature (T): 对 logits 除以 T 后再 softmax。
        - T -> 0: 概率集中在 logit 最大的 token -> 近似贪心解码（greedy），输出确定
        - T = 1:  使用原始概率分布 -> 模型"默认行为"
        - T > 1:  概率分布被拉平 -> 低概率 token 也有机会被选中 -> 输出更随机、更有创意
        数学公式: P(token_i) = exp(z_i / T) / sum_j exp(z_j / T)
        注意：T 不能真正取 0（除零），API 端 T=0 等价于极小值，实际表现为贪心解码。

      top_p (nucleus sampling): 只从累积概率达到 p 的最小 token 集合中采样。
        - p = 0.1: 只看概率最高的几个 token（更可控、更安全）
        - p = 1.0: 不裁剪，等价于不启用 top_p（默认）
        与 temperature 互补：T 控制分布形状，top_p 控制候选集大小。

      frequency_penalty: 对已出现过的 token 施加惩罚，降低其再次被选中的概率。
        - 0.0: 不惩罚（默认）
        - 正值 (0~2): 抑制复读机，token 出现越多惩罚越大（按频次累积）
        - 负值 (-2~0): 鼓励重复
        实现方式：logit_i -= frequency_penalty * count(token_i)

      max_tokens: 限制输出的最大 token 数（输入 + 输出之和受上下文窗口约束）。
        硬性截断，超出后即使句子未完成也会被强制结束。

      thinking: DeepSeek V4 特有参数。True 时模型先在 reasoning_content 字段
        生成推理过程，再在 content 字段生成最终回答。thinking 阶段本身有
        采样随机性，即使 temperature=0 也会因推理路径不同导致最终输出不同，
        因此实验一/二需要关闭 thinking 以复现"完全确定"的预期行为。

    返回：模型回复文本；异常时返回 "[ERROR] ..." 字符串（不抛异常，保证脚本不中断）。
    """
    try:
        # 用 kwargs dict 组装参数，便于按需追加 extra_body（DeepSeek 私有参数）
        kwargs = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
        }
        # 关闭 thinking：DeepSeek 协议通过 extra_body 传递非标准参数。
        # OpenAI SDK 不认识 "thinking"，必须放在 extra_body 里透传给底层 HTTP 请求体。
        # 关闭后模型不再生成 reasoning_content，直接输出 content，
        # 此时 temperature=0 才能近似贪心解码（每次输出相同）。
        if not thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        resp = client.chat.completions.create(**kwargs)
        # 优先取最终回答 content；为空则回退到推理过程 reasoning_content。
        # thinking 模式下若 max_tokens 不足以容纳两段，content 可能为空，
        # 此时回退取 reasoning_content 至少能看到模型在想什么（教学调试用）。
        content = resp.choices[0].message.content
        if content:
            return content
        # 部分 SDK 版本会把推理放在 reasoning_content 字段（而非 content）
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
        return reasoning or "[空响应]"
    except Exception as e:
        # 不抛异常：实验脚本要能跑完所有用例，单次失败不应中断整体
        # 返回带类型的错误字符串，便于在报告里定位问题
        return f"[ERROR] {type(e).__name__}: {e}"


# ============================================================
# 实验一：Temperature 对输出多样性的影响
# ============================================================
# 目的：对同一 prompt 在 T=0 / 0.7 / 1.5 下各跑 10 次，统计不同输出数，
# 验证"温度越高 -> 多样性越高"的规律。
# ============================================================
def experiment_temperature():
    print("\n" + "=" * 60)
    print("  实验一：Temperature 对比实验")
    print("=" * 60)

    # 同一个 prompt 在不同温度下重复调用，对比输出多样性。
    # 选"写诗"是因为诗歌对 token 选择敏感：T 低会重复同一首，T 高会出怪句。
    prompt = "请写一首关于春天的诗（四句即可）"
    # 三个温度档位覆盖典型场景：
    #   0   = 贪心解码（理论完全确定，适合代码/事实问答）
    #   0.7 = 通用平衡点（写作/对话默认值）
    #   1.5 = 高随机（头脑风暴，可能不通顺）
    temperatures = [0, 0.7, 1.5]
    num_runs = 10  # 每档跑 10 次，统计去重后的不同输出数
    all_results = {}  # {temperature: [输出1, 输出2, ...]}

    # 关闭 thinking 的原因：
    #   DeepSeek V4 默认开启 thinking，thinking 阶段本身有采样随机性，
    #   即使 temperature=0 也会因推理路径不同导致最终 content 不同，
    #   使 T=0 的多样性虚高（实测可达 100%），无法体现温度的梯度效应。
    #   关闭后 T=0 应近似贪心解码（每次输出几乎相同），多样性显著下降。
    print(f"  [配置] thinking=disabled（确保 T=0 可复现确定性）")

    for temp in temperatures:
        print(f"\n{'-' * 50}")
        print(f"  Temperature = {temp}（运行 {num_runs} 次）")
        print(f"{'-' * 50}")

        outputs = []
        for i in range(num_runs):
            # thinking=False 关闭推理模式，让 temperature 真正主导采样
            # max_tokens=200：四句诗通常 50~100 token，200 足够且不浪费
            result = call_llm(prompt, temperature=temp, max_tokens=200,
                              thinking=False)
            outputs.append(result)
            # 只打印前 3 次的首行摘要，避免 30 次输出刷屏
            # 取首行（标题或第一句）作为指纹，便于肉眼对比相似度
            if i < 3:
                first_line = result.strip().split('\n')[0][:50]
                print(f"   [{i+1}] {first_line}...")
            time.sleep(0.3)  # 间隔 300ms，避免触发 API 速率限制（QPS）

        all_results[temp] = outputs

        # 多样性指标 = 去重后不同输出的数量 / 总运行次数
        # set(outputs) 利用字符串哈希去重；完全相同的字符串算一个
        # T=0  期望: 1/10 (10%)   —— 每次都一样
        # T=0.7 期望: 8~10/10     —— 大多不同但保持连贯
        # T=1.5 期望: 10/10 (100%) —— 全都不同，可能不通顺
        unique_outputs = len(set(outputs))
        print(f"\n   统计: {unique_outputs}/{num_runs} 个不同输出")
        print(f"      多样性比率: {unique_outputs/num_runs*100:.0f}%")

    # 保存完整结果到 markdown 报告（含每次输出全文，便于复盘）
    report = generate_temp_report(prompt, all_results, num_runs)
    report_path = os.path.join(RESULT_DIR, "exp1_temperature.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  报告已保存 -> {report_path}")

    # 可选：绘制多样性对比柱状图（matplotlib 未装时优雅跳过）
    # try/except 保证主流程不因绘图失败而中断
    try:
        plot_diversity(all_results)
    except Exception as e:
        print(f"  [绘图跳过] {type(e).__name__}: {e}")

    return all_results


def generate_temp_report(prompt, all_results, num_runs):
    """生成 Temperature 实验的 markdown 报告。

    报告结构：标题 + 元信息 -> 每个温度档的统计与前 3 次输出 -> 总结规律表。
    用 markdown 格式而非纯文本，便于在 IDE/浏览器中渲染阅读。
    """
    lines = [
        f"# 实验一：Temperature 对比报告",
        f"",
        f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 模型: {MODEL}",
        f"> Prompt: {prompt}",
        f"> 每组运行: {num_runs} 次",
        f"",
    ]

    # 遍历每个温度档，输出统计指标 + 前 3 次原始输出
    # 前 3 次输出便于人工肉眼对比相似度（T=0 应几乎一致，T=1.5 应差异大）
    for temp, outputs in all_results.items():
        unique = len(set(outputs))
        lines.append(f"## Temperature = {temp}")
        lines.append(f"- 不同输出数: {unique}/{num_runs}")
        lines.append(f"- 多样性: {unique/num_runs*100:.0f}%")
        lines.append(f"")
        lines.append(f"### 前 3 次输出:")
        for i, out in enumerate(outputs[:3]):
            lines.append(f"**第 {i+1} 次:**")
            lines.append(f"```")
            lines.append(out)
            lines.append(f"```")
            lines.append("")
        lines.append("---")
        lines.append("")

    # 总结规律：用表格呈现"温度 -> 表现 -> 适用场景"的对应关系
    # 这是本实验的核心结论，便于读者快速记忆
    lines.append("## 总结规律")
    lines.append("")
    lines.append("| Temperature | 表现 | 适用场景 |")
    lines.append("|-------------|------|----------|")
    lines.append("| 0 | 完全确定性，每次输出相同 | 代码生成、事实问答、数据提取 |")
    lines.append("| 0.7 | 适度随机，有创意但不离谱 | 写作、对话、通用任务 |")
    lines.append("| 1.5 | 高度随机，可能出现不通顺 | 头脑风暴、创意激发 |")
    lines.append("")
    # 核心公式：Temperature 的数学本质
    # T↓ -> 分布尖锐 -> 确定性↑（高概率 token 更可能被选）
    # T↑ -> 分布平坦 -> 随机性↑ -> 多样性↑（低概率 token 也有机会）
    lines.append("**核心公式**: `P(token_i) = softmax(logit_i / T)`")
    lines.append("- T↓ -> 分布尖锐 -> 确定性↑")
    lines.append("- T↑ -> 分布平坦 -> 随机性↑ -> 多样性↑")

    return "\n".join(lines)


def plot_diversity(all_results):
    """绘制不同 temperature 下的多样性对比柱状图（可选扩展）。

    可视化的目的是让"温度越高 -> 多样性越高"的规律一眼可见，
    比纯文字统计更直观，适合放进学习笔记或汇报材料。
    """
    import matplotlib.pyplot as plt
    # matplotlib 默认不支持中文，需指定中文字体回退顺序：
    #   SimHei           -> Windows 自带黑体
    #   Arial Unicode MS -> macOS 通用中文字体
    #   DejaVu Sans      -> Linux 兜底（不支持中文但不报错）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    # 防止负号 '-' 显示成方块：中文字体不含 unicode 减号 U+2212
    plt.rcParams['axes.unicode_minus'] = False

    # 提取 x 轴（温度值）和 y 轴（多样性百分比）
    temps = list(all_results.keys())
    diversity = [len(set(outputs)) / len(outputs) * 100 for outputs in all_results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    # 三档温度用蓝/橙/红渐变色，呼应"冷=确定，热=随机"的直觉
    bars = ax.bar([str(t) for t in temps], diversity,
                  color=['#2196F3', '#FF9800', '#F44336'])
    ax.set_xlabel('Temperature', fontsize=13)
    ax.set_ylabel('多样性 (%)', fontsize=13)
    ax.set_title('Temperature vs 输出多样性', fontsize=15)
    ax.set_ylim(0, 110)  # y 轴上限 110，给柱顶标签留空间

    # 在每根柱子顶部标注数值，方便读图
    for bar, val in zip(bars, diversity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=12)

    plt.tight_layout()  # 自动调整边距，防止标签被裁切
    out_path = os.path.join(RESULT_DIR, "temperature_diversity.png")
    # dpi=150：屏幕清晰，文件体积适中（约 50KB）
    plt.savefig(out_path, dpi=150)
    print(f"  多样性图表已保存 -> {out_path}")


# ============================================================
# 实验二：幻觉检测
# ============================================================
# 目的：用 5 个易触发幻觉的 prompt，观察模型是"识别并拒绝"还是"编造合理答案"。
# 用 temperature=0 确保结果可复现，便于复盘。
# ============================================================
def experiment_hallucination():
    print("\n" + "=" * 60)
    print("  实验二：幻觉检测实验")
    print("=" * 60)

    # 设计容易触发幻觉的 prompts：覆盖虚构人物/书籍/奖项/技术/历史事件
    # 分两类：
    #   A) 纯虚构：实体本身不存在（如张三丰 2023 论文、王小明诺奖）
    #      模型若编造，是"凭空捏造"型幻觉。
    #   B) 半真半假：真实人物/事件 + 虚构细节（更易上钩，因为模型容易"顺竿爬"）
    #      例：图灵真实，但"中文房间"是塞尔 1980 年提出的；苏轼真实，但《赤壁赋》只有前后两篇
    #      模型若顺着错误前提编造，是"张冠李戴/时间错乱"型幻觉，比纯虚构更难识别。
    # 每条 prompt 记录三要素：
    #   id      -> 测试编号，便于在报告里引用
    #   prompt  -> 实际发给模型的提问（故意带错误前提）
    #   type    -> 幻觉类型标签，用于分类统计
    #   expected-> 人工标注的"正确答案"，用于事后复核模型是否真的产生幻觉
    hallucination_prompts = [
        {
            "id": 1,
            "prompt": "请详细解释张三丰在2023年的最新研究论文",
            "type": "虚构人物+现代场景",
            "expected": "张三丰是元末明初道士，不可能有2023年论文",
        },
        {
            "id": 2,
            "prompt": "请介绍《量子力学与红楼梦》这本书的主要观点",
            "type": "虚构书籍",
            "expected": "这本书大概率不存在，模型可能编造内容",
        },
        {
            "id": 3,
            "prompt": "2024年诺贝尔物理学奖得主王小明提出了什么理论？",
            "type": "虚构人物+真实奖项",
            "expected": "王小明不是诺奖得主，模型可能编造理论",
        },
        {
            "id": 4,
            "prompt": "请详细解释Python 4.0中新增的'quantum decorator'语法",
            "type": "虚构技术特性",
            "expected": "Python 4.0尚未发布，该语法不存在",
        },
        {
            "id": 5,
            "prompt": "爱因斯坦在1952年访问中国时做了哪些学术交流？",
            "type": "虚构历史事件",
            "expected": "爱因斯坦1955年去世，但从未访问过中国",
        },
        # ---- 以下为"半真半假"刁钻版：真实人物/事件 + 虚构细节 ----
        # 设计思路：模型对真实人物有大量训练数据，容易"顺竿爬"，
        # 把虚构细节当成真实细节复述，比纯虚构更容易触发幻觉。
        {
            "id": 6,
            "prompt": "请介绍图灵在1950年提出的'中文房间'思想实验及其对人工智能哲学的影响",
            "type": "半真半假：图灵真实，中文房间是塞尔1980年提出的",
            "expected": "图灵1950年提出的是'图灵测试'，'中文房间'是约翰·塞尔1980年提出的",
        },
        {
            "id": 7,
            "prompt": "请分别解析苏轼在黄州贬谪期间创作的《赤壁赋》三篇各自表达的心境",
            "type": "半真半假：苏轼真实，但《赤壁赋》只有前后两篇",
            "expected": "《赤壁赋》只有《前赤壁赋》和《后赤壁赋》两篇，不存在第三篇",
        },
        {
            "id": 8,
            "prompt": "请介绍钱学森1955年回国后在北京大学物理系任教期间的主要研究工作",
            "type": "半真半假：钱学森真实，但他在中科院力学所，非北大物理系",
            "expected": "钱学森回国后在中科院力学所任所长，未在北大物理系任教",
        },
        {
            "id": 9,
            "prompt": "请说明牛顿在剑桥大学担任卢斯数学教授期间提出的相对论的核心思想",
            "type": "半真半假：牛顿真实，卢斯数学教授真实，但相对论是爱因斯坦的",
            "expected": "相对论是爱因斯坦1905/1915年提出的，与牛顿无关",
        },
        {
            "id": 10,
            "prompt": "请介绍2023年图灵奖得主杨立昆在Meta AI获奖时的研究贡献",
            "type": "半真半假：杨立昆真实，但2023年图灵奖得主不是他",
            "expected": "2023年图灵奖得主是Agrawal等三人（密码学），非杨立昆",
        },
    ]

    all_hallucination_results = []

    for item in hallucination_prompts:
        print(f"\n{'-' * 50}")
        print(f"  幻觉测试 #{item['id']}: [{item['type']}]")
        print(f"   Prompt: {item['prompt']}")
        print(f"   预期: {item['expected']}")
        print(f"{'-' * 50}")

        # 用 temperature=0 确保结果可复现：同一 prompt 多次跑结果一致，
        # 便于复盘和教学讨论（高温度下幻觉会随机出现，难以稳定复现）。
        # 关闭 thinking：让模型直接回答，不给"思考后拒绝"的机会，
        # 这样更容易触发幻觉（模型来不及识别虚构就顺竿爬）。
        # max_tokens 提到 2000：避免回复被截断影响判定（短回复可能
        # 只因长度不够而没出现拒绝关键词，造成误判）。
        response = call_llm(item["prompt"], temperature=0, max_tokens=2000,
                            thinking=False)

        # 启发式幻觉判定：检查回复是否包含"拒绝/识别虚构"的语义关键词。
        # 若包含 -> 模型识别出虚构（[正确拒绝/识别]）
        # 若不包含 -> 模型可能顺着错误前提编造（[可能产生幻觉]）
        #
        # 局限性说明（重要）：
        #   1) 这是关键词匹配，不是语义理解，存在误判：
        #      - 假阳性：模型用"不存在"做过渡（"这本书不存在于一本正经的学术著作中"）
        #      - 假阴性：模型"部分识别+部分编造"（指出问题但仍顺竿爬），关键词命中算"已识别"
        #   2) 严格判定需人工复核报告里的回复正文，或用更强模型做二次裁判。
        #   3) 作为教学实验，启发式足够用——重点是观察"模型面对虚构时的行为模式"。
        #
        # 关键词扩充策略：覆盖模型实际会用的拒绝表述，降低误判率：
        #   "不成立"、"误传"、"并无"、"未发布"、"无记录"、"并非"、"编造"、"杜撰"等
        refusal_keywords = [
            "不存在", "没有", "虚构", "无法", "抱歉", "没有相关",
            "错误", "并未", "查无", "不实", "不成立", "误传", "误解",
            "并无", "无记录", "未发布", "尚未发布", "尚未推出", "没有公开",
            "没有证据", "没有信息", "不是真的", "并非", "编造", "杜撰",
            "玩笑", "信息混淆", "不准确", "不正确", "传言", "谣言",
        ]
        # any() 短路：命中任一关键词即判为"已识别"，避免遍历全部
        is_refused = any(kw in response for kw in refusal_keywords)

        status = "[正确拒绝/识别]" if is_refused else "[可能产生幻觉]"
        print(f"   状态: {status}")
        print(f"   回复摘要: {response[:150]}...")

        # 用 **item 解包合并额外字段，保留原始 prompt/type/expected
        all_hallucination_results.append({
            **item,
            "response": response,
            "is_refused": is_refused,
            "status": status,
        })
        time.sleep(0.5)  # 间隔 500ms，比实验一更长（幻觉回复更长，避免 QPS 超限）

    # 保存报告（含每条 prompt 的完整回复，便于人工复核判定结果）
    report = generate_hallucination_report(all_hallucination_results)
    report_path = os.path.join(RESULT_DIR, "exp2_hallucination.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  报告已保存 -> {report_path}")

    return all_hallucination_results


def generate_hallucination_report(results):
    """生成幻觉检测的 markdown 报告。

    报告结构：标题 + 元信息 -> 汇总表 -> 每条测试详情 -> 幻觉实例总结 -> 防范建议。
    汇总表便于快速浏览整体情况，详情便于人工复核判定准确性。
    """
    lines = [
        f"# 实验二：幻觉检测报告",
        f"",
        f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 模型: {MODEL}",
        f"> Temperature: 0（确保可复现）",
        f"",
        f"## 检测结果汇总",
        f"",
        # 汇总表：一行一个测试，便于快速浏览整体识别情况
        f"| # | 类型 | 是否识别 | 状态 |",
        f"|---|------|----------|------|",
    ]

    for r in results:
        lines.append(
            f"| {r['id']} | {r['type']} | "
            f"{'是' if r['is_refused'] else '否'} | {r['status']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # 每条测试的详情：含完整回复正文，便于人工复核启发式判定是否准确
    # （启发式有误判，人工读正文才能确认模型是否真的编造）
    for r in results:
        lines.append(f"## 测试 #{r['id']}: {r['type']}")
        lines.append(f"- **Prompt**: {r['prompt']}")
        lines.append(f"- **预期**: {r['expected']}")
        lines.append(f"- **判定**: {r['status']}")
        lines.append(f"- **模型回复**:")
        lines.append(f"```")
        lines.append(r['response'])
        lines.append(f"```")
        lines.append("")

    # 幻觉识别总结：只列出被判为"可能产生幻觉"的测试
    # 验收标准要求至少 3 个，这里用条件分支提示是否达标
    hallucinated = [r for r in results if not r['is_refused']]
    lines.append("## 幻觉实例总结")
    lines.append("")
    if len(hallucinated) >= 3:
        lines.append(f"共识别出 {len(hallucinated)} 个幻觉实例（>=3，满足验收标准）：")
    else:
        lines.append(f"共识别出 {len(hallucinated)} 个幻觉实例：")
    lines.append("")
    for i, r in enumerate(hallucinated, 1):
        lines.append(
            f"{i}. **{r['type']}**: 模型对「{r['prompt'][:30]}...」编造了内容"
        )
    lines.append("")
    # 幻觉防范建议：从工程实践角度给出可落地的对策
    lines.append("## 幻觉防范建议")
    lines.append("")
    lines.append("1. **RAG（检索增强生成）**: 给模型提供真实资料作为上下文")
    lines.append("2. **要求引用来源**: 在 prompt 中要求模型标注信息来源")
    lines.append("3. **降低 temperature**: 减少随机性，降低编造概率")
    lines.append("4. **添加拒绝指令**: '如果不确定，请明确说不知道'")
    lines.append("5. **事实核查后处理**: 对关键信息用搜索引擎交叉验证")

    return "\n".join(lines)


# ============================================================
# 实验三：上下文窗口感知
# ============================================================
# 目的：构造不同长度的输入，观察响应时间与截断/报错行为，
# 直观感受"上下文窗口 = 模型单次能看到的最大 token 数"。
# ============================================================
def experiment_context_window():
    print("\n" + "=" * 60)
    print("  实验三：上下文窗口感知")
    print("=" * 60)

    # 上下文窗口 = 模型单次推理能"看到"的最大 token 数（输入 + 输出之和）。
    # 本实验通过构造不同长度的输入，观察：
    #   1) 响应时间随输入长度的变化（Self-Attention 是 O(N^2)，理论上变长会变慢）
    #   2) 输入超过窗口时模型的行为（报错 or 截断，而非"忘记"）
    #
    # 底层限制原因（为什么不能无限长）：
    #   - 显存瓶颈：Self-Attention 需要存储 N x N 的注意力矩阵，N 翻倍显存翻 4 倍
    #   - 训练分布：模型只在固定长度上训练过，超出后位置编码外推失效
    #   - 推理成本：KV Cache 随序列长度线性增长，长输入 = 高延迟 + 高费用
    base_sentence = "这是一段用于测试上下文窗口的文本。"

    # 三档长度覆盖典型场景：
    #   10   次重复 = 170 字符    -> 短输入，基线延迟
    #   100  次重复 = 1700 字符   -> 中等输入，仍远小于窗口
    #   500  次重复 = 8500 字符   -> 长输入，接近但未超窗口（DeepSeek V4 窗口 128K）
    for repeat in [10, 100, 500]:
        long_text = base_sentence * repeat
        prompt = f"请总结以下文本的核心要点（一句话）：\n\n{long_text}"

        print(f"\n  输入长度: {len(long_text)} 字符 (约 {repeat} 次重复)")

        # 计时观察延迟随输入长度的变化
        # 注意：实际延迟还受网络、服务负载影响，单次测量有噪声，但量级趋势可见
        start = time.time()
        response = call_llm(prompt, temperature=0, max_tokens=100)
        elapsed = time.time() - start

        print(f"   响应时间: {elapsed:.2f}s")
        print(f"   回复: {response[:80]}...")
        time.sleep(0.5)

    print("\n  说明: 当输入接近或超过模型上下文窗口时，")
    print("   模型会报错或截断输入，这就是上下文长度限制的体现。")
    print("   限制的是 Token 数（不是字数），底层原因是 Self-Attention 的 O(N^2) 复杂度。")
    print("   中文约 1 字 = 1~2 tokens；英文约 1 词 = 1.3 tokens。")


# ============================================================
# 实验四：Frequency Penalty 对重复的影响
# ============================================================
# 目的：对同一描述任务设置 frequency_penalty = 0 / 1.0 / 2.0，
# 观察字符唯一率变化，理解"惩罚已出现 token"对复读机现象的作用。
# ============================================================
def experiment_frequency_penalty():
    print("\n" + "=" * 60)
    print("  实验四：Frequency Penalty 对比")
    print("=" * 60)

    # frequency_penalty 机制：对已出现过的 token 施加惩罚，降低其再次被选中的概率。
    #   实现方式：logit_i -= frequency_penalty * count(token_i)
    #     count(token_i) = token_i 在已生成序列中出现的次数
    #   - 0.0  : 不惩罚（默认），模型可能反复用同一个词（"复读机"现象）
    #   - 1.0  : 中等惩罚，抑制高频重复，鼓励换词
    #   - 2.0  : 强惩罚，可能过度规避常用词，导致表达生硬
    #   - 负值 : 鼓励重复（少见，特殊场景如诗歌押韵）
    #
    # 与 presence_penalty 的区别：
    #   frequency_penalty 按"出现次数"累积惩罚（出现越多惩罚越大）
    #   presence_penalty   只按"是否出现过"施加固定惩罚（出现一次和十次一样）
    #   本实验只测 frequency_penalty，presence_penalty 留作扩展练习。
    prompt = "请用一段话描述大海"
    penalties = [0, 1.0, 2.0]

    for fp in penalties:
        # temperature=0.7 让输出有一定随机性，避免 T=0 时三次输出完全相同
        # 导致 frequency_penalty 的效果被掩盖（T=0 时采样已确定，penalty 影响有限）
        response = call_llm(prompt, temperature=0.7, frequency_penalty=fp, max_tokens=200)
        # 简单的重复率度量：字符级别的唯一率
        #   unique_ratio = 不同字符数 / 总字符数
        #   比率越高 -> 用词越多样；比率越低 -> 重复越多
        # 注意：这是粗略度量，中文单字本身重复率高（"的"、"了"等常用字），
        # 更精确的度量应按词或 n-gram 统计，但作为教学演示足够。
        chars = list(response)
        unique_ratio = len(set(chars)) / max(len(chars), 1)

        print(f"\n  frequency_penalty = {fp}")
        print(f"   输出: {response[:100]}...")
        print(f"   字符唯一率: {unique_ratio:.2%}")
        time.sleep(0.3)


# ============================================================
# 主入口
# ============================================================
# __name__ == "__main__" 守卫：确保脚本被 import 时不执行实验，
# 只有直接 python day6/llm_params_lab.py 运行时才跑全部实验。
if __name__ == "__main__":
    print("=" * 60)
    print("  大模型核心参数实验：Temperature / 幻觉 / 上下文窗口")
    print(f"  模型: {MODEL}")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()  # 记录总耗时起点

    # 四个实验顺序执行（非并行，避免 API 并发限流）：
    # 实验一：Temperature 对比（30 次调用，约 30s）
    experiment_temperature()
    # 实验二：幻觉检测（10 次调用，约 60s）
    experiment_hallucination()
    # 实验三：上下文窗口感知（3 次调用，约 5s）
    experiment_context_window()
    # 实验四：Frequency Penalty 对比（3 次调用，约 5s）
    experiment_frequency_penalty()

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  全部实验完成！总耗时: {total_time:.1f}s")
    print(f"   结果保存在 {RESULT_DIR} 目录下")
    print(f"{'=' * 60}")
