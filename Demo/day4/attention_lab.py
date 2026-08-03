"""
Transformer 核心机制实验：自注意力层实现 + 热力图可视化
============================================================
本脚本通过三个实验帮助理解 Transformer 的核心机制：
  1) 手动实现单头自注意力，看清 QKV 计算的每一步
  2) 实现多头自注意力，对比不同头的关注模式
  3) 验证注意力矩阵的数学性质（行和=1、权重∈[0,1]、形状守恒）

运行: python attention_lab.py
依赖: torch / matplotlib / numpy
"""

# ============================================================
# 模块导入说明
# ============================================================
# torch                : PyTorch 主包，提供张量运算
# torch.nn             : 神经网络层（Linear、Module 等）
# torch.nn.functional  : 函数式接口（softmax 等无状态操作）
# matplotlib           : 绘制热力图
# numpy                : 数值辅助（本脚本中主要被 matplotlib 间接使用）
# os                   : 创建结果目录
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# matplotlib 全局配置
# ------------------------------------------------------------
# font.sans-serif: 指定中文字体回退顺序：
#   SimHei           -> Windows 自带黑体
#   Arial Unicode MS -> macOS 通用中文字体
#   DejaVu Sans      -> Linux 常见无字体时的兜底
# axes.unicode_minus: 防止负号 '-' 显示成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 在当前工作目录下创建 results/ 用于保存热力图
# exist_ok=True: 目录已存在时不报错（便于重复运行）
os.makedirs("results", exist_ok=True)


# ============================================================
# 1. 手动实现单头自注意力（不依赖 nn.MultiheadAttention）
# ============================================================
# 为什么不用 nn.MultiheadAttention？
#   -> 那是一个"黑盒"，会把 QKV 计算封装起来。
#   -> 学习阶段需要把每一步都拆开看，所以这里从零手写。
# ============================================================
class SingleHeadSelfAttention(nn.Module):
    """
    单头自注意力层的手动实现
    目的：让你看清 QKV 计算的每一步

    核心公式:
        Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V

    其中 Q、K、V 都是同一个输入 x 经过不同线性投影得到的，
    所以叫"自"注意力（Self-Attention）—— 自己关注自己。
    """
    def __init__(self, d_model: int):
        """
        参数:
            d_model: 输入/输出向量的维度（也叫嵌入维度 embedding dim）
                     例如 64 表示每个 token 用一个 64 维向量表示
        """
        super().__init__()
        self.d_model = d_model

        # ------------------------------------------------------------
        # 4 个可学习的线性投影矩阵（都是方阵 d_model x d_model）
        # ------------------------------------------------------------
        # bias=False: 注意力里的投影通常不加偏置，让计算更"纯粹"
        #   - W_q: 把 x 投影成 Query  "我在找什么信息"
        #   - W_k: 把 x 投影成 Key    "我有什么标签可被匹配"
        #   - W_v: 把 x 投影成 Value  "我实际提供的内容"
        #   - W_o: 输出投影，把加权求和后的 context 再做一次变换
        #         作用是增加表达能力，并让多头拼接后能"融合"回原维度
        # 为什么 Q/K/V 要用不同的矩阵？
        #   -> 同一输入从不同角度被"提问"。若三者共享矩阵，
        #      Q=K 时点积退化为自相似，表达力大幅下降。
        # ------------------------------------------------------------
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_attn=False):
        """
        前向传播：实现一次完整的单头自注意力

        Args:
            x:          输入张量，形状 (batch_size, seq_len, d_model)
            return_attn: 是否返回注意力权重矩阵（可视化/解释性时用）

        Returns:
            output:      输出张量，形状 (batch_size, seq_len, d_model)
            attn_weights: 注意力权重，形状 (batch_size, seq_len, seq_len)
                          仅当 return_attn=True 时返回
        """
        # 解包形状：B=batch_size, L=seq_len, d=d_model
        batch_size, seq_len, d_model = x.shape

        # ------------------------------------------------------------
        # Step 1: 线性投影得到 Q, K, V
        # ------------------------------------------------------------
        # nn.Linear 对最后一维做变换: (B, L, d) @ (d, d)^T -> (B, L, d)
        # 三个投影矩阵独立可学习，让模型学会"从不同角度看同一输入"
        Q = self.W_q(x)  # (B, L, d)  Query
        K = self.W_k(x)  # (B, L, d)  Key
        V = self.W_v(x)  # (B, L, d)  Value

        # ------------------------------------------------------------
        # Step 2: 计算注意力分数 = Q @ K^T
        # ------------------------------------------------------------
        # K.transpose(1, 2): 把 (B, L, d) -> (B, d, L)，交换"序列"和"维度"
        #   维度编号: 0=batch, 1=seq, 2=dim
        # torch.bmm: batched matrix multiply，只支持 3D
        #   (B, L, d) @ (B, d, L) -> (B, L, L)
        # scores[i, j, k] = 第 j 个 token 的 Query 与第 k 个 token 的 Key 的点积
        #   点积越大 -> 两者越"相关" -> 第 j 个 token 越应该关注第 k 个
        scores = torch.bmm(Q, K.transpose(1, 2))

        # ------------------------------------------------------------
        # Step 3: 缩放（除以 √d_k）
        # ------------------------------------------------------------
        # 为什么需要缩放？
        #   点积 = d_k 个乘积之和，当 d_k 较大时数值会很大。
        #   过大的值进入 softmax 后梯度趋近于 0（饱和区），训练不动。
        #   除以 √d_k 把方差稳定回 1，让 softmax 处于梯度友好的区间。
        # 这里 d_model == d_k（单头情况下没有切分）
        scores = scores / (d_model ** 0.5)

        # ------------------------------------------------------------
        # Step 4: Softmax 归一化
        # ------------------------------------------------------------
        # dim=-1: 沿最后一维（Key 维度）做 softmax
        #   -> 每一行的 Query 对所有 Key 的权重之和 = 1（概率分布）
        # attn_weights[b, i, j] = 第 i 个 token 对第 j 个 token 的关注程度
        #   ∈ [0, 1]，且 sum_j attn_weights[b, i, j] = 1
        attn_weights = F.softmax(scores, dim=-1)  # (B, L, L)

        # ------------------------------------------------------------
        # Step 5: 用注意力权重对 V 加权求和
        # ------------------------------------------------------------
        # (B, L, L) @ (B, L, d) -> (B, L, d)
        # 每个位置的输出 = 所有权重的 V 向量按权重加权和
        # 这就是"软检索"：不是硬选某一个 V，而是按相似度混合所有 V
        context = torch.bmm(attn_weights, V)

        # ------------------------------------------------------------
        # Step 6: 输出投影
        # ------------------------------------------------------------
        # 再做一次线性变换 W_o：
        #   1) 增加表达能力（多一层非线性映射前的线性组合）
        #   2) 在多头场景下，把拼接后的向量"融合"回原维度
        output = self.W_o(context)

        # 返回注意力权重供可视化/解释使用
        # 正常 forward 不返回中间结果，但教学/调试场景需要它
        if return_attn:
            return output, attn_weights
        return output


# ============================================================
# 2. 多头注意力（用于对比）
# ============================================================
# 多头的核心思想：
#   把 d_model 切成 h 个 d_k = d_model / h 的子空间，
#   每个子空间独立做一次注意力，让模型在不同"视角"上
#   同时学习不同的关注模式（语法、语义、位置、指代等）。
#
# 关键性质：参数量与单头相同！
#   单头: 4 个 (D, D) 矩阵 = 4D²
#   多头: 4 个 (D, D) 矩阵 = 4D²  （只是在计算时拆分到子空间）
#   所以多头不是"堆参数"，而是"重塑参数的使用方式"。
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力（Scaled Dot-Product Attention 的多头版本）"""

    def __init__(self, d_model: int, num_heads: int):
        """
        参数:
            d_model:    总嵌入维度
            num_heads:  头的个数（必须能整除 d_model）
        """
        super().__init__()
        # 整除约束：每个头的维度 d_k 必须是整数
        # 否则无法把 d_model 均匀切分给各个头
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        # d_k: 每个头的子空间维度，例如 d_model=64, num_heads=4 -> d_k=16
        self.d_k = d_model // num_heads

        # 投影矩阵与单头相同：仍然是 (d_model, d_model) 的方阵
        # 区别在 forward 中如何使用——拆成多头并行计算
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_attn=False):
        """
        前向传播：多头并行注意力

        Args / Returns 与单头版本一致，但 attn_weights 多了一个头维度:
            attn_weights: (B, H, L, L)
        """
        B, L, D = x.shape   # batch, seq_len, d_model
        H = self.num_heads
        d_k = self.d_k

        # ------------------------------------------------------------
        # Step 1: 投影 + 拆分为多头
        # ------------------------------------------------------------
        # 先做线性投影: (B, L, D) -> (B, L, D)
        # 再 view 成 (B, L, H, d_k): 把最后一维 D 拆成 (H, d_k)
        # 再 transpose(1, 2): 把 head 维度提到 batch 后面
        #   (B, L, H, d_k) -> (B, H, L, d_k)
        # 为什么要 transpose？
        #   -> 让"head"维度靠近 batch 维，便于对每个头独立做矩阵乘法
        #      （PyTorch 的 matmul 会自动对最后两维做矩阵乘）
        Q = self.W_q(x).view(B, L, H, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, d_k).transpose(1, 2)

        # ------------------------------------------------------------
        # Step 2: 每个头独立做注意力
        # ------------------------------------------------------------
        # torch.matmul 支持批量广播: (B, H, L, d_k) @ (B, H, d_k, L)
        #   -> (B, H, L, L)
        # transpose(-2, -1): 交换最后两维 (d_k, L) -> (L, d_k)
        # 缩放用 d_k 而不是 d_model，因为每个头实际计算维度是 d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # dim=-1: 沿 Key 维做 softmax，每个 (head, query) 行和为 1
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)

        # 加权求和: (B, H, L, L) @ (B, H, L, d_k) -> (B, H, L, d_k)
        context = torch.matmul(attn_weights, V)    # (B, H, L, d_k)

        # ------------------------------------------------------------
        # Step 3: 拼接所有头（Concat）+ 输出投影
        # ------------------------------------------------------------
        # transpose(1, 2): 把 head 维度换回去 (B, H, L, d_k) -> (B, L, H, d_k)
        # contiguous(): transpose 后内存不连续，view 前必须调用
        # view(B, L, D): 把 (H, d_k) 拼回 D，恢复原维度
        #   -> 即把 h 个头的输出首尾相连，得到 (B, L, D)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        # W_o: 把拼接后的多头信息"融合"回原维度
        # 这一步是多头能学到不同模式的关键——让模型学会如何混合各头
        output = self.W_o(context)

        if return_attn:
            return output, attn_weights
        return output


# ============================================================
# 3. 可视化函数
# ============================================================
# 把注意力权重矩阵画成热力图，直观看到"谁关注谁"
# ============================================================
def visualize_attention(attn_matrix, title, save_path, labels=None):
    """
    绘制单张注意力权重热力图

    Args:
        attn_matrix: 注意力权重，形状 (seq_len, seq_len)
                     行 = Query（发起关注的位置）
                     列 = Key   （被关注的位置）
        title:       图标题
        save_path:   保存路径
        labels:      坐标轴标签（如 ['T0', 'T1', ...]）
    """
    # 创建一张图，尺寸 8x7 英寸
    fig, ax = plt.subplots(figsize=(8, 7))

    # imshow: 把二维矩阵渲染成图像
    #   cmap='YlOrRd': 黄->橙->红 的颜色映射，红表示权重高
    #   vmin=0, vmax=1: 固定颜色范围（softmax 输出必在 [0,1]）
    #   detach(): 从计算图分离（绘图不需要梯度）
    #   .numpy(): 转 numpy 数组供 matplotlib 使用
    im = ax.imshow(attn_matrix.detach().numpy(), cmap='YlOrRd', vmin=0, vmax=1)

    # 设置坐标轴标签（如果提供了 labels）
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        # rotation=45: x 轴标签倾斜 45°，避免长标签重叠
        # ha='right': 标签右对齐到刻度
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    # ------------------------------------------------------------
    # 在每个格子里写上数值
    # ------------------------------------------------------------
    # 双层循环遍历矩阵每个元素
    # color: 权重 > 0.5 用白字（深色背景），否则用黑字（浅色背景）
    #   -> 保证可读性，避免黑字落在深红格子里看不清
    for i in range(attn_matrix.shape[0]):
        for j in range(attn_matrix.shape[1]):
            val = attn_matrix[i, j].item()  # .item(): 取 Python 标量
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    # 标题与轴标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Key (被关注的位置)', fontsize=12)
    ax.set_ylabel('Query (发起关注的位置)', fontsize=12)

    # 颜色条：显示颜色与数值的对应关系
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # tight_layout: 自动调整子图间距，避免标签被裁切
    # bbox_inches='tight': 保存时也按紧凑边界裁切
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图，释放内存（脚本里多次绘图时尤其重要）
    print(f"  📊 热力图已保存 → {save_path}")


# ============================================================
# 4. 主实验
# ============================================================
def main():
    # 装饰性的标题栏（用制表符画的双线框）
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Transformer 核心机制实验：自注意力 + 可视化             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 固定随机种子，保证每次运行结果一致（便于复现与对比）
    torch.manual_seed(42)

    # --- 参数设置 ---
    # d_model:   嵌入维度，设小一点便于可视化（实际模型通常 256/512/768）
    # seq_len:   序列长度（8 个 token，热力图 8x8 看得清）
    # num_heads: 多头数（必须整除 d_model，64/4=16 ✓）
    # batch_size: 单 batch 便于可视化（实际训练会更大）
    d_model = 64       # 嵌入维度（为了可视化清晰，设小一些）
    seq_len = 8        # 序列长度
    num_heads = 4      # 多头数
    batch_size = 1

    # 模拟输入：随机向量
    # 实际中这是词嵌入 + 位置编码的输出（这里简化，只用随机数）
    # 形状: (B, L, D) = (1, 8, 64)
    x = torch.randn(batch_size, seq_len, d_model)
    # 坐标轴标签：T0~T7 表示 8 个 token 的位置
    token_labels = [f'T{i}' for i in range(seq_len)]

    print(f"\n⚙️  配置: d_model={d_model}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"   输入形状: {x.shape}")

    # ========== 实验1: 单头自注意力 ==========
    print("\n" + "=" * 60)
    print("📋 实验1: 单头自注意力")
    print("=" * 60)

    # 实例化单头注意力层
    single_attn = SingleHeadSelfAttention(d_model)
    # eval(): 切换到评估模式（关闭 dropout 等。本例无 dropout，但保持习惯）
    single_attn.eval()

    # torch.no_grad(): 不构建计算图，节省内存与时间（推理时用）
    with torch.no_grad():
        out_single, attn_single = single_attn(x, return_attn=True)

    # 验证输出形状：应与输入相同 (B, L, D)
    print(f"   输出形状: {out_single.shape}")
    # 注意力矩阵形状: (B, L, L) = (1, 8, 8)
    print(f"   注意力矩阵形状: {attn_single.shape}")
    # 验证 softmax 性质：每行和应为 1
    print(f"   注意力权重行和（应全为1）: {attn_single[0].sum(dim=-1)}")

    # 可视化单头注意力权重（取 batch 0）
    visualize_attention(
        attn_single[0],
        'Single-Head Self-Attention Weights',
        'results/single_head_attention.png',
        token_labels
    )

    # ========== 实验2: 多头自注意力 ==========
    print("\n" + "=" * 60)
    print("📋 实验2: 多头自注意力（4头）")
    print("=" * 60)

    # 实例化多头注意力层
    multi_attn = MultiHeadSelfAttention(d_model, num_heads)
    multi_attn.eval()

    with torch.no_grad():
        out_multi, attn_multi = multi_attn(x, return_attn=True)

    print(f"   输出形状: {out_multi.shape}")
    # 注意力矩阵多了一个头维度: (B, H, L, L)
    print(f"   注意力矩阵形状: {attn_multi.shape} (batch, heads, seq, seq)")

    # ------------------------------------------------------------
    # 绘制每个头的热力图（4 个子图并排）
    # ------------------------------------------------------------
    # plt.subplots(1, num_heads): 1 行 num_heads 列的子图布局
    # figsize=(5*num_heads, 5): 每个子图 5 英寸宽，总宽 20 英寸
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))

    # 遍历每个头，分别绘制热力图
    for h in range(num_heads):
        # attn_multi[0, h]: 取 batch 0 的第 h 个头，形状 (L, L)
        im = axes[h].imshow(attn_multi[0, h].numpy(), cmap='YlOrRd', vmin=0, vmax=1)
        axes[h].set_title(f'Head {h}', fontsize=13)
        axes[h].set_xticks(range(seq_len))
        axes[h].set_yticks(range(seq_len))
        axes[h].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
        axes[h].set_yticklabels(token_labels, fontsize=8)

        # 在每个格子里写上数值（字体比单头图小一号，避免拥挤）
        for i in range(seq_len):
            for j in range(seq_len):
                val = attn_multi[0, h, i, j].item()
                color = 'white' if val > 0.5 else 'black'
                axes[h].text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=7)

    # suptitle: 整张图的总标题
    # y=1.02: 标题往上抬一点，避免和子图标题重叠
    fig.suptitle('Multi-Head Self-Attention (4 Heads)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/multi_head_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 多头热力图已保存 → results/multi_head_attention.png")

    # ========== 实验3: 验证注意力矩阵性质 ==========
    print("\n" + "=" * 60)
    print("📋 实验3: 验证注意力矩阵的数学性质")
    print("=" * 60)

    # 性质1: 每行之和 = 1（softmax 沿 Key 维归一化的结果）
    # torch.allclose: 允许浮点误差 (atol=1e-5)
    row_sums = attn_single[0].sum(dim=-1)
    print(f"   ✅ 每行之和 = 1（Softmax归一化）: {torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)}")

    # 性质2: 所有权重 ∈ [0, 1]（softmax 输出的概率分布性质）
    print(f"   ✅ 所有权重 ∈ [0, 1]: {(attn_single >= 0).all() and (attn_single <= 1).all()}")

    # 性质3: 输出形状 == 输入形状（注意力是"位置无关"的变换，不改变序列长度和维度）
    print(f"   ✅ 输出形状 == 输入形状: {out_single.shape == x.shape}")

    # ========== 总结输出 ==========
    print("\n" + "=" * 60)
    print("🏁 实验完成！")
    print("=" * 60)
    # 打印核心要点回顾，方便学习时直接对照终端输出
    print("""
📝 核心要点回顾:

1. QKV 机制:
   - Q (Query): "我在找什么信息"
   - K (Key):   "我包含什么信息标签"
   - V (Value): "我的实际内容"
   - Attention = softmax(QK^T / √d) × V

2. 为什么需要注意力:
   - 解决 RNN 的长距离依赖问题（任意两点直接连接）
   - 支持并行计算（不依赖序列顺序）
   - 动态路由：根据内容决定关注谁（而非固定窗口）

3. 多头注意力的作用:
   - 单头只能学一种关注模式
   - 多头在 d_model 的不同子空间中并行学习
   - 每个头可能关注: 语法、语义、位置、指代等不同维度
   - 最终 Concat + Linear 融合多维度信息

4. 缩放因子 √d_k 的作用:
   - 防止 QK^T 的值过大
   - 避免 softmax 进入梯度极小的饱和区
   - 保证训练时梯度稳定
""")


# ============================================================
# 脚本入口
# ============================================================
# __name__ == "__main__": 仅当直接运行本脚本时才执行 main()
# 被其他模块 import 时不会自动运行（便于复用 SingleHeadSelfAttention 等类）
if __name__ == "__main__":
    main()
