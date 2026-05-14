from __future__ import annotations

import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from typing import Callable
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from functools import partial

from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance

# ============================================================================
# 辅助函数 helpers
# ============================================================================

def exists(val):
    """判断值是否存在（不为 None）"""
    return val is not None

def default(val, d):
    """如果 val 存在则返回 val，否则返回默认值 d"""
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    """将值转换为指定长度的元组。如果 val 已经是元组则直接返回，否则重复 length 次"""
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    """
    使用 einops 将张量打包并返回打包后的结果。
    封装了 pack([x], pattern)，简化单张量打包操作。
    """
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    """
    使用 einops 将打包的张量解包，返回解包后的第一个张量。
    封装了 unpack(x, ps, pattern)[0]，简化单张量解包操作。
    """
    return unpack(x, ps, pattern)[0]

# ============================================================================
# 正弦余弦位置编码 sinusoidal positions
# ============================================================================

def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    """
    生成一维正弦余弦位置编码。

    输入:
        seq: 序列长度（位置数量）
        dim: 编码维度
        temperature: 温度系数，控制频率衰减速度（默认 10000）
        device: 设备（CPU/GPU）
        dtype: 数据类型

    输出:
        pos_emb: 形状为 (seq, dim) 的位置编码张量
    """
    # 位置索引: [0, 1, 2, ..., seq-1]
    n = torch.arange(seq, device = device)
    # 频率索引，归一化到 [0, 1]
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    # 频率: 1 / (temperature^omega)，不同维度对应不同频率
    omega = 1. / (temperature ** omega)

    # 外积: (seq, dim//2)，每个位置×每个频率
    n = n[:, None] * omega[None, :]
    # 拼接正弦和余弦，得到 (seq, dim)
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# ============================================================================
# 辅助类 helper classes
# ============================================================================

class Residual(Module):
    """
    残差连接包装器。
    将输入通过子模块 fn，然后将结果与原始输入相加（残差连接）。

    输入: x — 任意形状的张量
    输出: fn(x) + x — 形状与输入相同
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LayerNorm(Module):
    """
    自定义层归一化（Layer Normalization）。
    与官方 nn.LayerNorm 不同，不使用偏置的均值减法，仅使用缩放因子 gamma 和偏置 beta。

    输入:
        dim: 归一化的特征维度

    输入张量: x — 形状为 (..., dim) 的张量
    输出张量: 归一化后的张量，形状与输入相同
    """
    def __init__(self, dim):
        super().__init__()
        # 可学习的缩放参数，初始化为全 1
        self.gamma = nn.Parameter(torch.ones(dim))
        # 不可训练的偏置，注册为 buffer（不会被优化器更新）
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        # 沿最后一维进行层归一化
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class FeedForward(Module):
    """
    前馈网络（Feed Forward Network），带自适应层归一化支持。

    结构: LayerNorm → (可选的条件函数 cond_fn) → Linear → GELU → Dropout → Linear → Dropout

    输入:
        dim: 输入/输出特征维度
        mult: 内部隐藏层维度的倍数（默认 4，即 inner_dim = dim * 4）
        dropout: Dropout 概率

    前向传播:
        x: 输入张量，形状 (..., dim)
        cond_fn: 可选的条件函数，用于自适应层归一化（分类器自由引导中使用）
        输出: 形状与输入相同的张量
    """
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),   # 扩展维度
            nn.GELU(),                    # GELU 激活函数
            nn.Dropout(dropout),          # Dropout 正则化
            nn.Linear(inner_dim, dim),    # 投影回原始维度
            nn.Dropout(dropout)           # Dropout 正则化
        )
    def forward(self, x, cond_fn = None):
        x = self.norm(x)

        if exists(cond_fn):
            # 自适应层归一化：cond_fn 对归一化后的特征进行条件变换
            x = cond_fn(x)

        return self.net(x)

# ============================================================================
# MBConv（移动倒置瓶颈卷积）相关模块
# ============================================================================

class SqueezeExcitation(Module):
    """
    挤压-激励模块（Squeeze-and-Excitation）。

    通过对每个通道进行全局平均池化，学习通道间的注意力权重，
    实现对不同通道的自适应重新校准。

    输入:
        dim: 输入通道数
        shrinkage_rate: 压缩率，控制中间隐藏层的维度缩小比例（默认 0.25）

    前向传播:
        x: 形状为 (B, C, H, W) 的特征图
        输出: 形状相同的加权特征图，每个通道乘以学习到的 [0,1] 权重
    """
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),           # 全局平均池化 → (B, C)
            nn.Linear(dim, hidden_dim, bias = False),   # 压缩 → (B, hidden_dim)
            nn.SiLU(),                                   # SiLU 激活
            nn.Linear(hidden_dim, dim, bias = False),   # 扩展 → (B, C)
            nn.Sigmoid(),                                # 归一化到 [0, 1]
            Rearrange('b c -> b c 1 1')                 # 重塑为 (B, C, 1, 1) 以广播
        )

    def forward(self, x):
        # 按通道乘上注意力权重
        return x * self.gate(x)


class MBConvResidual(Module):
    """
    带 DropSample（随机深度）的 MBConv 残差连接。

    输入:
        fn: 内部的卷积模块
        dropout: 随机丢弃整个样本的概率（stochastic depth）

    前向传播:
        x: 输入特征图
        输出: fn(x) 经 dropsample 后 + x（残差连接）
    """
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)  # 随机深度正则化
        return out + x

class Dropsample(Module):
    """
    随机丢弃整个样本（Stochastic Depth / DropPath）。

    以 prob 的概率将整个样本的特征图置零，用于深层网络的正则化。
    训练时生效，评估时被禁用。

    输入:
        prob: 丢弃概率（0 表示不丢弃）
    """
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        # 概率为 0 或非训练模式时，直接返回
        if self.prob == 0. or (not self.training):
            return x

        # 为每个样本生成保留/丢弃掩码: (B, 1, 1, 1)
        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        # 除以 (1-prob) 保持期望值不变
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    """
    MobileNet 风格的倒置残差瓶颈卷积块（MBConv）。

    结构: 1×1 扩展卷积 → BN → GELU → 3×3 深度可分离卷积 → BN → GELU
          → 挤压激励模块 → 1×1 投影卷积 → BN
    当 dim_in == dim_out 且不下采样时，添加残差连接。

    输入:
        dim_in: 输入通道数
        dim_out: 输出通道数
        downsample: 是否进行 2 倍下采样（stride=2）
        expansion_rate: 中间通道扩展倍数（默认 4）
        shrinkage_rate: SE 模块的压缩率（默认 0.25）
        dropout: 随机深度丢弃概率

    输出:
        net: nn.Sequential 模块（或 MBConvResidual 包装后的模块）
    """
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),                                       # 1×1 扩展卷积
        nn.BatchNorm2d(hidden_dim),                                              # 批归一化
        nn.GELU(),                                                               # GELU 激活
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),  # 3×3 深度可分离卷积
        nn.BatchNorm2d(hidden_dim),                                              # 批归一化
        nn.GELU(),                                                               # GELU 激活
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),          # 挤压-激励注意力
        nn.Conv2d(hidden_dim, dim_out, 1),                                       # 1×1 投影卷积
        nn.BatchNorm2d(dim_out)                                                  # 批归一化
    )

    # 输入输出维度相同且不下采样时，使用残差连接
    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# ============================================================================
# 注意力相关类 attention related classes
# ============================================================================

class Attention(Module):
    """
    窗口多头自注意力（Window Multi-Head Self-Attention），带相对位置偏置和记忆 token。

    用于 MaxViT 中的 block-attention 和 grid-attention，支持窗口内的自注意力计算。

    输入:
        dim: 输入特征维度
        dim_head: 每个注意力头的维度（默认 32）
        dropout: 注意力 dropout 概率
        window_size: 窗口大小（默认 7×7）
        num_mem_kv: 记忆/寄存器 KV token 数量（默认 4），类似暖启动 token

    前向传播:
        x: 形状为 (batch, x_blocks, y_blocks, window_h, window_w, dim) 的张量
        输出: 形状相同的注意力输出张量
    """
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_mem_kv = 4
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head     # 注意力头数量
        self.scale = dim_head ** -0.5    # 缩放因子: 1/√d_k，防止内积过大

        # QKV 联合投影，输出 dim*3 然后切分成 Q、K、V
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # 可学习的记忆 KV token（寄存器），用于全局信息交互
        # 形状: (2 表示 K 和 V, heads, num_mem_kv, dim_head)
        self.mem_kv = nn.Parameter(torch.randn(2, self.heads, num_mem_kv, dim_head))

        # 注意力计算: Softmax → Dropout
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # --- 相对位置偏置 ---
        # 为每个注意力头和每对相对位置学习一个偏置值
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        # 构建相对位置索引表
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))           # (2, w, w)
        grid = rearrange(grid, 'c i j -> (i j) c')                              # (w*w, 2)，每个位置的 (x,y) 坐标
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')  # (w*w, w*w, 2)，成对相对坐标
        rel_pos += window_size - 1                                              # 偏移到非负范围 [0, 2*w-2]
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)   # 将二维相对坐标编码为一维索引

        # 注册为不可训练的 buffer
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        # x 形状: (batch, x_blocks, y_blocks, window_h, window_w, dim)
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # 将块维度展平: (b, x, y, w1, w2, d) → (b*x*y, w1*w2, d)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # QKV 投影并沿最后一维切分为 Q、K、V
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 拆分为多头: (b, n, h*d_head) → (b, h, n, d_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 缩放 Q
        q = q * self.scale

        # 记忆/寄存器 KV token：在每个注意力计算中添加可学习的全局 token
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = q.shape[0]),  self.mem_kv)
        num_mem = mk.shape[-2]

        # 将记忆 token 拼接到 K 和 V 前面
        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        # 计算注意力分数: Q · K^T
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 添加相对位置偏置
        bias = self.rel_pos_bias(self.rel_pos_indices)
        # 为记忆 token 在偏置前面填充 0（记忆 token 无位置偏置）
        bias = F.pad(bias, (0, 0, num_mem, 0), value = 0.)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # Softmax + Dropout 得到注意力权重
        attn = self.attend(sim)

        # 加权聚合 V: attn @ V
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并多头并恢复窗口形状: (b, h, w1*w2, d) → (b, w1, w2, h*d)
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # 输出投影
        out = self.to_out(out)
        # 恢复块维度: (b*x*y, w1, w2, d) → (b, x, y, w1, w2, d)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(Module):
    """
    MaxViT 视觉编码器：结合多轴注意力（block + grid attention）和 MBConv 的混合视觉 Transformer。

    核心思想是交替使用窗口自注意力（block attention）和网格自注意力（grid attention），
    以高效地捕捉局部和全局特征。使用 MBConv 作为卷积干和阶段前端。

    输入:
        num_classes: 分类类别数（用于最终的 MLP 头）
        dim: 基础特征维度
        depth: 元组，每个元素表示对应阶段的 Transformer 块层数
        dim_head: 每个注意力头的维度（默认 32）
        dim_conv_stem: 卷积干的输出通道数（默认等于 dim）
        window_size: 注意力窗口大小（默认 7）
        mbconv_expansion_rate: MBConv 扩展率（默认 4）
        mbconv_shrinkage_rate: MBConv 中 SE 模块压缩率（默认 0.25）
        dropout: Dropout 概率
        channels: 输入图像通道数（默认 3，RGB）

    前向传播:
        x: 图像张量，形状 (B, C, H, W)
        texts: 可选的文本列表，用于条件引导
        cond_fns: 可选的条件函数元组，用于分类器自由引导
        cond_drop_prob: 条件丢弃概率
        return_embeddings: 是否返回中间嵌入而非分类 logits
        输出: 分类 logits 或特征嵌入
    """
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # --- 卷积干 (convolutional stem) ---
        # 初始卷积层，将图像下采样 2 倍并提取低级特征
        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),  # 下采样 2×
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)          # 保持分辨率
        )

        # --- 阶段配置 ---
        num_stages = len(depth)

        # 每阶段维度翻倍: dim, 2*dim, 4*dim, ...
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)                       # 加入卷积干的维度
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))         # 各阶段的 (dim_in, dim_out)

        self.layers = ModuleList([])

        w = window_size   # 窗口大小

        # 记录每个块输入的条件隐藏维度（用于 TextConditioner 配置）
        cond_hidden_dims = []

        # --- 构建各阶段 ---
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    # 1. MBConv 块（每阶段第一个进行下采样）
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    # 2. Block Attention（窗口内自注意力）
                    # 将特征图切分为 w×w 的窗口
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),   # 恢复特征图形状

                    # 3. Grid Attention（网格自注意力）
                    # 另一种切分方式，与 Block Attention 互为补充
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),   # 恢复特征图形状
                )

                self.layers.append(block)

        embed_dim = dims[-1]             # 最终嵌入维度
        self.embed_dim = dims[-1]
        self.cond_hidden_dims = cond_hidden_dims

        # --- MLP 分类头 ---
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),        # 全局平均池化
            LayerNorm(embed_dim),                      # 层归一化
            nn.Linear(embed_dim, num_classes)          # 线性分类头
        )

    @beartype
    def forward(
        self,
        x,
        texts: list[str] | None = None,
        cond_fns: tuple[Callable, ...] | None = None,
        cond_drop_prob = 0.,
        return_embeddings = False
    ):
        """
        前向传播。

        输入:
            x: 图像张量 (B, C, H, W)
            texts: 文本条件列表（可选）
            cond_fns: 条件函数元组（可选），每个函数对应一个网络层的前自适应归一化
            cond_drop_prob: 条件丢弃概率
            return_embeddings: 若为 True，返回中间嵌入而非分类结果

        输出:
            分类 logits (B, num_classes) 或嵌入特征 (B, embed_dim, H', W')
        """
        # 卷积干提取初始特征
        x = self.conv_stem(x)

        # 将条件函数列表转换为迭代器
        cond_fns = iter(default(cond_fns, []))

        # 逐层前向传播，每层可选地应用条件函数
        for stage in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)   # 自适应条件变换（用于文本引导等）

            x = stage(x)

        if return_embeddings:
            return x

        # MLP 头输出分类结果
        return self.mlp_head(x)

# ============================================================================
# 通用 Transformer 注意力与编码器
# ============================================================================

class TransformerAttention(Module):
    """
    通用 Transformer 注意力层，支持交叉注意力、因果掩码和自适应层归一化。

    可用于：
    - 自注意力（context=None）：Q 和 KV 来自同一输入
    - 交叉注意力（context 存在）：Q 来自 x，KV 来自 context
    - 因果注意力（causal=True）：使用上三角掩码防止看到未来信息

    输入:
        dim: 输入/输出特征维度
        causal: 是否使用因果掩码（默认 False）
        dim_head: 每个注意力头的维度（默认 64）
        dim_context: 上下文维度（默认等于 dim）
        heads: 注意力头数量（默认 8）
        norm_context: 是否对上下文进行层归一化（默认 False）
        dropout: Dropout 概率

    前向传播:
        x: 查询输入 (B, N, dim)
        context: 键值上下文（可选），(B, N_ctx, dim_context)
        mask: 键的掩码 (B, N_ctx)，True 表示保留
        attn_bias: 注意力偏置 (heads, N_q, N_kv) 或广播兼容形状
        attn_mask: 注意力掩码 (B, heads, N_q, N_kv) 或广播兼容形状，True 表示保留
        cond_fn: 可选的条件函数，用于自适应层归一化
        输出: 注意力输出 (B, N, dim)
    """
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5          # 缩放因子: 1/√d_k
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        # Q 的层归一化
        self.norm = LayerNorm(dim)
        # 上下文的层归一化（可选）
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        # Q 投影: dim → inner_dim（所有头）
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # KV 投影: dim_context → dim_head * 2（共享 K、V）
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        attn_mask = None,
        cond_fn: Callable | None = None
    ):
        b = x.shape[0]

        # 对上下文进行归一化
        if exists(context):
            context = self.context_norm(context)

        # 若没有提供 context，则使用自注意力（KV 来自 x）
        kv_input = default(context, x)

        x = self.norm(x)

        # 自适应层归一化（用于分类器自由引导等条件控制）
        if exists(cond_fn):
            x = cond_fn(x)

        # Q 投影: (B, N, dim) → (B, N, inner_dim)
        q = self.to_q(x)
        # KV 投影并拆分
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # Q 拆分为多头: (B, N, inner_dim) → (B, heads, N, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 缩放 Q
        q = q * self.scale

        # 计算注意力分数: Q · K^T，K 保持 (B, N, dim_head) 格式用于高效计算
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # 应用各种掩码和偏置

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            # attn_mask 中 True 表示保留，False 的位置设为 -inf
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            # mask: (B, N_kv)，True 表示保留
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            # 上三角掩码：位置 i 只能看到位置 j ≤ i
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # Softmax + Dropout
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # 加权聚合: attn @ V
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # 合并多头: (B, heads, N, dim_head) → (B, N, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    """
    通用 Transformer 编码器，由多层 TransformerAttention + FeedForward 堆叠而成。

    每层结构: x = x + Attention(LN(x)) → x = x + FF(LN(x))

    输入:
        dim: 特征维度
        dim_head: 每个注意力头的维度（默认 64）
        heads: 注意力头数量（默认 8）
        depth: 层数（默认 6）
        attn_dropout: 注意力 dropout 概率
        ff_dropout: 前馈网络 dropout 概率

    前向传播:
        x: 输入序列 (B, N, dim)
        cond_fns: 可选的条件函数元组，长度为 depth*2（每层给 attn 和 ff 各一个）
        attn_mask: 注意力掩码
        输出: 编码后的序列 (B, N, dim)
    """
    @beartype
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            # 每层包含一个注意力模块和一个前馈模块
            self.layers.append(ModuleList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    @beartype
    def forward(
        self,
        x,
        cond_fns: tuple[Callable, ...] | None = None,
        attn_mask = None
    ):
        cond_fns = iter(default(cond_fns, []))

        for attn, ff in self.layers:
             # 注意力子层 + 残差连接
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns, None)) + x
             # 前馈子层 + 残差连接
             x = ff(x, cond_fn = next(cond_fns, None)) + x
        return x

# ============================================================================
# Token Learner 模块
# ============================================================================

class TokenLearner(Module):
    """
    Token Learner：从特征图中自适应地学习少量信息 token。

    参考论文: https://arxiv.org/abs/2106.11297
    使用 1.1 版本：用 MLP（两层带 GELU 的全连接）生成注意力图，
    对特征图进行加权池化，得到固定数量的输出 token。

    输入:
        dim: 输入特征通道数
        ff_mult: 中间维度扩展倍数（默认 2）
        num_output_tokens: 输出 token 数量（默认 8）
        num_layers: 注意力图生成网络的层数（默认 2）

    前向传播:
        x: 特征图 (B, C, H, W) 或任意前缀形状
        输出: 学习到的 token (B, num_output_tokens, C)
    """
    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        # 使用分组卷积生成注意力图：每个输出 token 对应一组独立的通道
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        # 打包批量维度: (..., C, H, W) → (B, C, H, W)
        x, ps = pack_one(x, '* c h w')
        # 复制通道 g 次以生成 g 组注意力图: (B, C, H, W) → (B, g*C, H, W)
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        # 生成注意力图: (B, g*C, H, W) → (B, g, H, W)
        attn = self.net(x)

        # 重塑为分组格式
        attn = rearrange(attn, 'b g h w -> b 1 g h w')           # (B, 1, g, H, W)
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)  # (B, C, g, H, W)

        # 用注意力图对特征进行加权空间池化: (B, C, g, H, W) → (B, C, g)
        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        # 恢复原始批量维度: (B, C, g) → (..., C, g)
        x = unpack_one(x, ps, '* c n')
        return x

# ============================================================================
# 机器人 Transformer（RT-1）
# ============================================================================

class RT1(Module):
    """
    RT-1 (Robotic Transformer 1)：机器人动作预测模型。

    架构流程:
    1. MaxViT 视觉编码器处理每帧图像 → 得到特征图
    2. Token Learner 将每帧的特征图压缩为少量（默认 8 个）token
    3. 将所有帧的 token 按时间顺序排列，加入正弦位置编码
    4. 因果 Transformer 处理时间序列 token
    5. 时间池化 → 线性投影 → 输出动作 logits

    输入:
        vit: MaxViT 实例，作为视觉骨干网络
        num_actions: 动作维度数（默认 11，对应机器人动作空间的维度）
        action_bins: 每个动作维度的离散化箱数（默认 256，将连续动作离散化）
        depth: Transformer 层数（默认 6）
        heads: Transformer 注意力头数（默认 8）
        dim_head: 每个注意力头的维度（默认 64）
        token_learner_ff_mult: Token Learner 中间层扩展倍数（默认 2）
        token_learner_num_layers: Token Learner 层数（默认 2）
        token_learner_num_output_tokens: 每帧学习到的 token 数（默认 8）
        cond_drop_prob: 文本条件丢弃概率，用于分类器自由引导（默认 0.2）
        use_attn_conditioner: 是否使用 AttentionTextConditioner（默认 False，使用 TextConditioner）
        conditioner_kwargs: 传递给条件器的额外参数
    """
    @beartype
    def __init__(
        self,
        *,
        vit: MaxViT,
        num_actions = 11,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict()
    ):
        super().__init__()
        self.vit = vit

        # ViT 中的阶段数（对应 cond_hidden_dims 的长度）
        self.num_vit_stages = len(vit.cond_hidden_dims)

        # 选择文本条件器类型
        conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner

        # 文本条件器：将文本嵌入映射为各层的条件向量
        # hidden_dims 包含 ViT 各阶段和 Transformer 各层的维度
        # hiddens_channel_first 标记哪些层处理的是通道优先（图像）格式
        self.conditioner = conditioner_klass(
            hidden_dims = (*tuple(vit.cond_hidden_dims), *((vit.embed_dim,) * depth * 2)),
            hiddens_channel_first = (*((True,) * self.num_vit_stages), *((False,) * depth * 2)),
            cond_drop_prob = cond_drop_prob,
            **conditioner_kwargs
        )

        # Token Learner：将每帧特征图压缩为少量 token
        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        # 因果 Transformer：处理时间序列 token
        self.transformer = Transformer(
            dim = vit.embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth
        )

        self.cond_drop_prob = cond_drop_prob

        # 动作预测头：LayerNorm → Linear → 重塑为 (num_actions, action_bins)
        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, num_actions * action_bins),
            Rearrange('... (a b) -> ... a b', b = action_bins)  # 每个动作维度对应 action_bins 个类别
        )

    @beartype
    def embed_texts(self, texts: list[str]):
        """
        将文本列表预嵌入为文本条件向量。

        输入:
            texts: 文本字符串列表

        输出:
            文本嵌入向量，可用于后续 forward 中的 text_embeds 参数
        """
        return self.conditioner.embed_texts(texts)

    @classifier_free_guidance
    @beartype
    def forward(
        self,
        video,
        texts: list[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = 0.
    ):
        """
        前向传播：输入视频和文本指令，输出机器人动作预测。

        使用 classifier_free_guidance 装饰器实现分类器自由引导，
        训练时同时处理条件和无条件两种路径。

        输入:
            video: 视频张量，形状 (B, C, F, H, W)
                   B=批量大小, C=通道数, F=帧数, H=高度, W=宽度
            texts: 文本指令列表（与 text_embeds 二选一）
            text_embeds: 预嵌入的文本向量（与 texts 二选一）
            cond_drop_prob: 条件丢弃概率

        输出:
            logits: 动作预测 logits，形状 (B, F, num_actions, action_bins)
                    每个帧、每个动作维度有 action_bins 个分类 logits
        """
        # 确保 texts 和 text_embeds 恰好提供一个
        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            num_texts = len(texts)
        elif exists(text_embeds):
            num_texts = text_embeds.shape[0]

        # 验证文本数量与视频批量大小一致
        assert num_texts == video.shape[0], f'you only passed in {num_texts} strings for guiding the robot actions, but received batch size of {video.shape[0]} videos'

        cond_kwargs = dict(texts = texts, text_embeds = text_embeds)

        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # F = 帧数
        frames, device = video.shape[2], video.device

        # 生成条件函数序列
        # repeat_batch: ViT 各阶段按帧数重复（每帧独立处理），Transformer 层不重复（共享）
        cond_fns, _ = self.conditioner(
            **cond_kwargs,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_vit_stages), *((1,) * self.transformer_depth * 2))
        )

        # 分离 ViT 和 Transformer 的条件函数
        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        # 将视频重塑为独立图像: (B, C, F, H, W) → (B*F, C, H, W)
        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')

        # 用 ViT 编码所有图像帧，返回特征图
        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True     # 返回嵌入而非分类结果
        )

        # 恢复帧维度: (B*F, C, H, W) → (B, F, C, H, W)
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        # Token Learner 压缩: (B, F, C, H, W) → (B, F, C, num_tokens)
        learned_tokens = self.token_learner(tokens)

        # 合并帧和 token 维度: (B, F, C, num_tokens) → (B, F*num_tokens, C)
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')

        # --- 因果注意力掩码 ---
        # 帧 i 只能看到帧 j ≤ i 的 token（机器人控制中不能看到未来帧）
        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        # 将帧级掩码扩展到 token 级: 每帧有 num_learned_tokens 个 token
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # --- 正弦位置编码 ---
        # 为每帧添加时间位置信息
        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)
        # 每帧的所有 token 共享相同的位置编码
        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

        # --- 因果 Transformer 处理 ---
        # attn_mask 取反: True=保留 → ~True=False 被 mask
        attended_tokens = self.transformer(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = ~attn_mask)

        # 时间池化：将同一帧的 token 平均: (B, F*num_tokens, D) → (B, F, D)
        pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)

        # 动作预测: (B, F, D) → (B, F, num_actions, action_bins)
        logits = self.to_logits(pooled)
        return logits
