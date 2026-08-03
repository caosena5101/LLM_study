"""
分词器与嵌入实验：tiktoken vs HuggingFace Transformers
============================================================
本脚本通过四个实验帮助理解分词器（Tokenization）与嵌入（Embedding）：
  1) 使用 tiktoken 观察 GPT-4 分词与中英文 Token 消耗差异
  2) 使用 HuggingFace 加载 BERT 中文分词器，对比 WordPiece 分词结果
  3) 编写通用 Token 估算函数（任意文本 / 任意模型）
  4) 可视化自注意力权重热力图（模拟），理解 O(N²) 复杂度来源

【为什么这个实验重要】
  大模型本质是做矩阵乘法的神经网络，它"看不懂文字，只懂数字"。
  分词器（Tokenizer）就是连接人类语言与机器计算的桥梁：
    文本 --[切分]--> Token 序列 --[查词典]--> 数字 ID --[Embedding 层]--> 高维向量
  上下文窗口限制的是 Token 数（不是字数），所以理解 token 化过程，
  才能真正理解"为什么模型有上下文长度限制"以及"为什么中文比英文更费 token"。

运行: python tokenizer_lab.py
依赖: tiktoken / transformers / torch / matplotlib / numpy
"""

# ============================================================
# 模块导入说明
# ============================================================
# tiktoken          : OpenAI 开源分词器库，GPT-4 / GPT-3.5 使用的 BPE 分词
#                     BPE = Byte Pair Encoding，按频率合并最高频字节对
# transformers      : HuggingFace 生态，加载 BERT 等模型的 WordPiece 分词器
#                     WordPiece 是 BPE 的变体，按"语言模型似然最大化"合并子词
# AutoTokenizer     : transformers 的通用分词器入口，按模型名自动选择实现
#                     不用关心底层是 BERTWordPieceTokenizer 还是别的具体类
# matplotlib        : 绘制注意力热力图（把抽象的权重矩阵变成直观的颜色图）
# numpy             : 生成模拟注意力矩阵 + 归一化（模拟 softmax 效果）
# os / sys / io     : 文件系统操作 + 强制 UTF-8 输出（Windows 控制台兼容）
import os
import sys
import io

# ------------------------------------------------------------
# Windows 控制台编码修复
# ------------------------------------------------------------
# Windows 中文系统控制台默认用 GBK 编码，而 tiktoken 解码出的 token 可能
# 包含 GBK 无法表示的字符（如某些 BPE 切碎的 UTF-8 字节片段），会抛
# UnicodeEncodeError。这里把 stdout/stderr 强制重包装成 UTF-8，
# errors="replace" 表示遇到无法编码的字符用 ? 替代而不报错。
# Linux/macOS 默认就是 UTF-8，这段代码对它们无副作用。
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tiktoken
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# matplotlib 全局配置
# ------------------------------------------------------------
# font.sans-serif: 指定中文字体回退顺序（matplotlib 找不到字体时会按顺序尝试）：
#   SimHei           -> Windows 自带黑体（中文显示用）
#   Arial Unicode MS -> macOS 通用中文字体
#   DejaVu Sans      -> Linux 常见无字体时的兜底（不支持中文但不会报错）
# axes.unicode_minus: 防止负号 '-' 显示成方块
#   （中文字体里没有 ASCII 减号字形，matplotlib 默认会用 unicode 减号 U+2212，
#    中文字体不含此字符，开启此选项强制用 ASCII '-'）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 在当前工作目录下创建 results/ 用于保存图片
# exist_ok=True: 目录已存在时不报错（幂等，方便重复运行）
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 实验一：使用 tiktoken 观察 GPT-4 分词与 Token 消耗
# ============================================================
# 目的：直观感受 BPE 分词的粒度，并量化"中文比英文更费 token"这一现象。
# 这是理解上下文长度限制的基础——同样语义，不同语言占用的 token 数差很多。
# ============================================================
def experiment_tiktoken():
    print("=" * 60)
    print("实验一：GPT-4 分词器 (tiktoken / BPE)")
    print("=" * 60)

    # tiktoken 用"编码名"标识不同的 BPE 词表：
    #   cl100k_base  -> GPT-4 / GPT-3.5-turbo / text-embedding-ada-002 共用
    #   p50k_base    -> GPT-3 (text-davinci-003 等)
    #   r50k_base    -> GPT-3 早期 / GPT-2
    #   o200k_base   -> GPT-4o（对中文/日文做了优化，token 数更少）
    # get_encoding 返回一个 Encoding 对象，提供 encode / decode 方法。
    enc = tiktoken.get_encoding("cl100k_base")

    # 三组测试文本：纯中文、纯英文、中英混合
    # 中英混合用来观察 BPE 在同一句子里如何切换处理策略
    texts = {
        "中文": "人工智能正在改变世界，大模型上下文长度限制的本质是Token数量。",
        "英文": "Artificial intelligence is changing the world. Context window limits tokens.",
        "中英混合": "Hello世界！今天吃龙虾eat food。",
    }

    for lang, text in texts.items():
        # encode: 文本 -> token id 列表（整数列表）
        # 这一步内部做了：BPE 切分 -> 查词表 -> 输出 ID 序列
        tokens = enc.encode(text)

        # 逐个 token 解码回字符串，观察 BPE 把文本切成了什么粒度
        # 注意：单个 token 解码可能得到乱码（如 '\ufffd'），因为 BPE 是按字节
        # 合并的，一个汉字的 UTF-8 编码（3 字节）可能被切到 2-3 个 token 里，
        # 单独解码其中一个 token 拿不到完整字符。这是 BPE 字节级分词的副作用。
        token_strs = [enc.decode([t]) for t in tokens]

        print(f"\n[{lang}] 原始文本: {text}")
        # "比" = token数 / 字符数，反映该语言的 token 密度
        # 中文通常 0.8~1.5（1 字 ≈ 1-2 token），英文通常 0.2~0.3（1 词 ≈ 1.3 token）
        print(f"  字符数: {len(text)} | Token数: {len(tokens)} | 比: {len(tokens)/len(text):.2f}")
        print(f"  分词详情: {token_strs}")

    # 批量计算不同长度文本的消耗：直观感受中文比英文"更贵"
    # 通过对比不同长度的"测试文本"和"Test text"，观察 token 数随长度的线性关系
    # 以及中英文 token 比值是否稳定（理论上应趋于一个常数）
    print("\n批量消耗估算 (中文 vs 英文):")
    print(f"  {'长度':<10}{'中文Tokens':<15}{'英文Tokens':<15}{'中文/英文'}")
    for length in [10, 50, 100, 500]:
        cn_text = "测试文本" * length  # 4 字 × length
        en_text = "Test text " * length  # 10 字符（含空格）× length
        cn_tokens = len(enc.encode(cn_text))
        en_tokens = len(enc.encode(en_text))
        ratio = cn_tokens / en_tokens if en_tokens else 0
        print(f"  {length:<10}{cn_tokens:<15}{en_tokens:<15}{ratio:.2f}")
        # 预期：中文/英文比值稳定在 1.5 左右
        # → 同样的"语义长度"，中文消耗的 token 数约为英文的 1.5 倍
        # → 8K 上下文窗口，能装的中文比英文少约 1/3

    print("\n观察: 同样的语义长度，中文消耗的 Token 数通常多于英文，")
    print("      这就是为什么相同上下文窗口，能容纳的中文比英文少。")


# ============================================================
# 实验二：使用 HuggingFace 加载 BERT 分词器对比
# ============================================================
# 目的：对比另一种主流分词算法 WordPiece（BERT 系）与 BPE（GPT 系）的差异。
# 重点观察：
#   1) BERT 中文分词是"字级别"（1 字 = 1 token），与 tiktoken 不同
#   2) 英文 WordPiece 用 ## 前缀标记"词内子词"，控制词表规模
#   3) tokenizer() 直接返回模型可用的张量（含特殊 token 和 attention_mask）
# ============================================================
def experiment_huggingface():
    print("\n" + "=" * 60)
    print("实验二：BERT 中文分词器 (HuggingFace / WordPiece)")
    print("=" * 60)

    # bert-base-chinese 使用 WordPiece 算法，词表约 21128 个 token
    # 中文场景下基本是"字级别"分词，每个汉字 = 1 个 token
    #   （因为中文 BERT 训练时把每个汉字当作一个词表条目）
    # 英文场景下会用 ## 标记词内子词，如 playing -> [play, ##ing]
    #   （## 表示"这个词片接在前一个词片后面，不是新词的开头"）
    # 首次调用会从 HuggingFace Hub 下载词表文件（约 1MB），缓存到 ~/.cache/huggingface
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # ---- 中文示例：观察字级别分词 ----
    cn_text = "我喜欢机器学习"
    # tokenize: 文本 -> token 字符串列表（还不是 ID）
    cn_tokens = tokenizer.tokenize(cn_text)
    # convert_tokens_to_ids: token 字符串 -> 词表 ID
    # 这两步分开是为了方便观察中间结果，实际可用 tokenizer(text) 一步到位
    cn_ids = tokenizer.convert_tokens_to_ids(cn_tokens)

    print(f"\n[中文] 原始文本: {cn_text}")
    print(f"  WordPiece分词: {cn_tokens}")
    print(f"  映射数字ID:    {cn_ids}")
    print(f"  Token总数:     {len(cn_ids)}")
    print("  观察: BERT 中文分词器采用字级别分词，每个汉字 = 1 个独立 token。")
    # 对比：tiktoken 把"人工智能"切成 ['人','工','智','能'] 等字节碎片，
    #       BERT 直接切成 ['人','工','智','能'] 整字 —— 中文 token 效率更高

    # ---- 英文示例：观察 ## 子词标记 ----
    # 用 bert-base-uncased（英文版）才能看到 ## 标记
    # uncased = 不区分大小写（"I" 和 "i" 都映射到同一个 token）
    print()
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 选 "tokenization" 和 "embeddings" 这种长词，确保会被 WordPiece 拆分
    # （高频短词如 "love" 不会被拆，因为它们整体就在词表里）
    en_text = "I love tokenization and embeddings."
    en_tokens = en_tokenizer.tokenize(en_text)
    en_ids = en_tokenizer.convert_tokens_to_ids(en_tokens)

    print(f"[英文] 原始文本: {en_text}")
    print(f"  WordPiece分词: {en_tokens}")
    print(f"  映射数字ID:    {en_ids}")
    print(f"  Token总数:     {len(en_ids)}")
    print("  观察: 英文 WordPiece 用 ## 标记词内子词，如 'tokenization' -> ['token', '##ization']，")
    print("        'embeddings' -> ['em', '##bed', '##dings']，控制词表规模。")
    # ## 的意义：让模型知道 "##ization" 不是独立词，而是接在前面词片后的后缀
    # 这样词表只需存 ~3 万条子词，就能组合出几十万英文词，解决 OOV 问题

    # ---- 直接调用 __call__ 拿到模型输入张量 ----
    # 实际喂给模型时不会用 tokenize + convert_tokens_to_ids 两步，
    # 而是直接 tokenizer(text)，一次返回所有模型需要的输入：
    #   input_ids       : token ID 序列（已加特殊 token）
    #   attention_mask  : 哪些位置是真实 token（1）vs padding（0）
    #   token_type_ids  : 句子对任务才用，单句全是 0
    print("\n直接调用 tokenizer() 返回模型输入:")
    encoded = tokenizer(
        cn_text,
        padding=True,        # padding=True: 短序列补齐到 max_length（用 0 补）
        truncation=True,     # truncation=True: 超长序列截断到 max_length
        max_length=32,       # 最大长度 32（BERT 默认 512，这里小一点便于观察）
        return_tensors="pt", # "pt" = 返回 PyTorch 张量；"tf" = TensorFlow；不传 = list
    )
    print(f"  input_ids:      {encoded['input_ids'].tolist()}")
    print(f"  attention_mask: {encoded['attention_mask'].tolist()}")
    print(f"  解码回文本:      {tokenizer.decode(encoded['input_ids'][0])}")
    print("  注意首尾的 [CLS] (101) 和 [SEP] (102) 是 BERT 自动加的特殊 token。")
    # [CLS] 放句首，其最终隐藏向量用于分类任务（如情感分析）
    # [SEP] 放句尾，分隔不同句子（句对任务如 NLI）
    # 这些特殊 token 的 ID 在词表里是预留的固定编号


# ============================================================
# 实验三：编写通用 Token 估算函数
# ============================================================
# 目的：写一个能在实际工程里直接用的工具函数。
# 应用场景：
#   - 调 OpenAI API 前预估 token 数，判断是否超上下文窗口
#   - 估算 prompt 成本（OpenAI 按 token 计费）
#   - 评估长文档是否需要分块（chunking）喂给模型
# ============================================================
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    估算任意文本在指定模型下的 Token 数量。

    Args:
        text:  待估算文本
        model: 目标模型名 ("gpt-4", "gpt-3.5-turbo", "gpt-4o" 等)
               未识别的模型名会回退到 cl100k_base 编码。

    Returns:
        Token 数量

    设计说明：
        - 不同 OpenAI 模型用不同 BPE 编码（cl100k_base / o200k_base / p50k_base），
          同一段文本在不同模型下 token 数会略有差异。
        - 用 encoding_for_model 按模型名查官方对应编码，保证估算准确。
        - 未识别模型名（如自定义模型）回退到 cl100k_base，这是 GPT-4 系列编码，
          对大多数现代 OpenAI 模型都是合理近似。
        - 此函数只适用于 OpenAI 系列模型；估算国产模型（Qwen/文心）需用
          对应的 tokenizer（如 Qwen2Tokenizer），不能直接用 tiktoken。
    """
    try:
        # encoding_for_model: 按模型名查 OpenAI 官方对应的 BPE 编码
        # 内部维护了一张 模型名 -> 编码名 的映射表
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # 未识别模型 -> 回退到 cl100k_base（GPT-4 系列）
        # 这是合理的默认值，因为 cl100k_base 是目前最通用的 OpenAI 编码
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def experiment_estimate():
    print("\n" + "=" * 60)
    print("实验三：Token 估算函数测试")
    print("=" * 60)

    # 测试用例覆盖：中文 / 不同模型 / 英文 / 中英混合
    # 重点关注：
    #   1) 同一段中文在 gpt-4 和 gpt-3.5-turbo 下 token 数是否相同
    #      （预期相同，因为两者都用 cl100k_base）
    #   2) 中英文 token/字符比的差异
    test_cases = [
        ("大语言模型的上下文窗口限制的是Token的数量，而不是字符数。", "gpt-4"),
        ("大语言模型的上下文窗口限制的是Token的数量，而不是字符数。", "gpt-3.5-turbo"),
        ("The quick brown fox jumps over the lazy dog.", "gpt-4"),
        ("Hello世界！今天吃龙虾eat food。", "gpt-4"),
    ]

    for text, model in test_cases:
        n = estimate_tokens(text, model)
        print(f"\n  模型: {model}")
        print(f"  文本: {text}")
        # 比 = token数 / 字符数，反映 token 密度
        # 中文 0.8~1.5，英文 0.2~0.3，中英混合介于两者之间
        print(f"  字符数: {len(text)} | 估算Token数: {n} | 比: {n/len(text):.2f}")

    # 模拟一个真实场景：估算一篇约 800 字中文文章的 token 数
    # 16 字 × 50 = 800 字，接近一篇短文长度
    sample_article = "人工智能是当前最热门的技术领域。" * 50
    article_tokens = estimate_tokens(sample_article)
    print(f"\n  场景: 800字中文文章估算")
    print(f"  字符数: {len(sample_article)} | 估算Token数: {article_tokens}")
    # 8K 上下文窗口 = 8192 token，估算能装几篇此类文章
    # 这是实际工程里判断"要不要分块"的关键依据
    print(f"  推论: 8K 上下文窗口约能容纳 {8000 // article_tokens} 篇此类文章")


# ============================================================
# 实验四：可视化自注意力权重热力图（模拟）
# ============================================================
# 目的：把抽象的"注意力矩阵"变成直观的颜色图，并量化 O(N²) 复杂度。
# 这一步是理解"为什么上下文长度有限制"的视觉化关键：
#   - 注意力矩阵大小 = N × N，N 翻倍则矩阵面积变 4 倍
#   - 显存和计算量随 N 平方增长，这就是上下文窗口的硬约束
# ============================================================
def visualize_attention():
    print("\n" + "=" * 60)
    print("实验四：自注意力权重热力图可视化（模拟 O(N²) 复杂度）")
    print("=" * 60)

    # 模拟一个句子分词后的 token 序列
    # [CLS] 和 [SEP] 是 BERT 风格的特殊 token，真实模型里也会有
    tokens = ["[CLS]", "我", "喜欢", "机器", "学习", "[SEP]"]
    n = len(tokens)

    # 模拟自注意力权重矩阵（实际由模型 QK^T / √d_k + softmax 计算得出）
    # 真实公式：Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    # 这里用随机矩阵归一化来演示形状和可视化，不涉及真实 QKV 计算
    np.random.seed(42)  # 固定随机种子，保证每次运行结果一致（便于复现）
    attention_weights = np.random.rand(n, n)
    # 沿最后一维归一化，模拟 softmax 效果（每行和为 1）
    # 每行 = 某个 Query token 对所有 Key token 的注意力分布
    # 和为 1 表示"注意力总量"被分配到各个 Key 上，类似概率分布
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

    # ---- 绘制热力图 ----
    # 热力图用颜色深浅表示数值大小，是可视化矩阵的标准方式
    fig, ax = plt.subplots(figsize=(8, 6))
    # cmap='YlGnBu': 黄->绿->蓝 的颜色映射，数值越大颜色越深
    # interpolation='nearest': 不做插值，每个格子显示纯色（更清晰）
    im = ax.imshow(attention_weights, cmap='YlGnBu', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Attention Weight')

    # 坐标轴：列=被关注的Key，行=发起关注的Query
    # 行 i 的颜色分布 = 第 i 个 token 如何分配它的注意力给其他 token
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha='right')  # 旋转 45 度防止标签重叠
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Key (被关注的Token)')
    ax.set_ylabel('Query (发起关注的Token)')
    ax.set_title('Self-Attention Weights Heatmap (Simulated)')

    # 在每个格子中显示数值，便于精确读数
    # ha/va="center": 文本水平垂直居中
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()  # 自动调整子图间距，防止标签被裁切
    out_path = os.path.join(RESULTS_DIR, "attention_heatmap.png")
    # dpi=120: 提高分辨率（默认 100），打印更清晰
    # bbox_inches='tight': 保存时裁掉多余白边
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()  # 关闭图，释放内存（重要：不关会累积导致内存泄漏）
    print(f"热力图已保存: {out_path}")

    # ---- 用一张图直观展示 O(N²) 复杂度：N 增大时矩阵面积爆炸 ----
    # 这是本实验的核心结论：自注意力计算量与 token 数的平方成正比
    # 取从 8 到 8192 的典型上下文长度，看 N² 如何爆炸增长
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    pairs = [n * n for n in sizes]  # 注意力矩阵元素数 = N²

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, pairs, marker='o', linewidth=2, color='#d62728')
    # 用 log-log 坐标：O(N²) 在 log-log 图上是一条斜率为 2 的直线
    # 斜率 = 2 直接对应"平方"关系，是判断复杂度阶数的直观方法
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Token 数 N')
    ax.set_ylabel('Self-Attention 计算对数 N^2')
    ax.set_title('为什么上下文长度有限制？Self-Attention 的 O(N^2) 复杂度')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # 标注几个关键点，让"指数爆炸"更直观
    # 128 = 短上下文，1024 = 中等，8192 = GPT-4 标准 8K 窗口
    for n in [128, 1024, 8192]:
        idx = sizes.index(n)
        # xy: 箭头指向的数据点；xytext: 文本放置位置
        # n*1.5, pairs[idx]*0.5: 文本偏移到数据点右下方，避免遮挡
        ax.annotate(f'N={n}\nN^2={n*n:,}', xy=(n, pairs[idx]),
                    xytext=(n*1.5, pairs[idx]*0.5), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    out_path2 = os.path.join(RESULTS_DIR, "token_complexity.png")
    plt.savefig(out_path2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"复杂度曲线已保存: {out_path2}")

    print("\n观察:")
    print("  - 注意力矩阵大小 = N × N，N 翻倍则计算量变 4 倍。")
    print("  - 8K 上下文 vs 128 上下文，自注意力计算量放大 (8192/128)² ≈ 4096 倍。")
    print("  - 这就是为什么扩大上下文窗口需要更多显存 + 更长推理时间。")


# ============================================================
# 主入口
# ============================================================
# 按顺序执行四个实验：
#   1) tiktoken BPE 分词观察
#   2) HuggingFace WordPiece 对比
#   3) Token 估算函数测试
#   4) 注意力热力图 + 复杂度曲线
# 顺序有依赖：实验三的估算函数依赖实验一引入的 tiktoken；
#            实验四的可视化依赖前面建立的概念基础。
def main():
    experiment_tiktoken()
    experiment_huggingface()
    experiment_estimate()
    visualize_attention()
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


# __name__ == "__main__" 保证脚本被 import 时不会自动执行 main()
# 只有直接 python tokenizer_lab.py 运行时才跑
if __name__ == "__main__":
    main()
