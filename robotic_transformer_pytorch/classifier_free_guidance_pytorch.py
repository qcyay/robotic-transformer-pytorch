"""
Classifier-Free Guidance (CFG) 文本条件模块
============================================

实现论文 "Classifier-Free Diffusion Guidance" 中的核心思想，将其适配为通用的文本条件注入机制。
主要用于 RT-1 (Robotic Transformer) 等模型，在训练和推理时动态地将文本指令注入到网络各层。

核心组件:
    - classifier_free_guidance: 函数装饰器，为 forward 方法添加 CFG 推理逻辑
    - classifier_free_guidance_class_decorator: 类装饰器，自动为模型添加文本条件器
    - TextConditioner: 基于 FiLM 的文本条件器
    - AttentionTextConditioner: 基于交叉注意力的文本条件器
    - TextEmbeddingReturner: 直接返回文本嵌入（不做 FiLM/Attention 调制）
    - NullConditioner: 空条件器，用于无条件模型

CFG 原理:
    训练时: 以概率 cond_drop_prob 丢弃文本条件，使模型学会有/无文本指令的情况
    推理时: output = unconditioned + cond_scale * (conditioned - unconditioned)
            cond_scale > 1 时会放大条件信号的影响
"""

from __future__ import annotations

from collections import namedtuple
from functools import wraps, partial, cache

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from beartype.door import is_bearable
from beartype.typing import Callable, Tuple, List, Literal, Dict, Any

from inspect import signature

from classifier_free_guidance_pytorch.typing import typecheck, beartype_isinstance

from classifier_free_guidance_pytorch.t5 import T5Adapter
from classifier_free_guidance_pytorch.open_clip import OpenClipAdapter
from classifier_free_guidance_pytorch.attend import Attend
from classifier_free_guidance_pytorch.bge import BGEAdapter

# ==============================================================================
# 常量定义
# ==============================================================================

# 关键字名称常量，用于统一参数传递时的键名
COND_DROP_KEY_NAME = 'cond_drop_prob'        # 条件丢弃概率的键名

TEXTS_KEY_NAME = 'texts'                     # 文本列表的键名
TEXT_EMBEDS_KEY_NAME = 'text_embeds'         # 预嵌入文本向量的键名
TEXT_CONDITIONER_NAME = 'text_conditioner'   # 条件器属性的名称
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'     # 条件函数列表的键名

# 文本条件返回的命名元组：包含嵌入向量和注意力掩码
TextCondReturn = namedtuple('TextCondReturn', [
    'embed',  # 文本嵌入向量
    'mask'    # 注意力掩码（仅 AttentionTextConditioner 使用）
])

# ==============================================================================
# 通用辅助函数
# ==============================================================================

def exists(val):
    """判断值是否不为 None"""
    return val is not None

def is_empty(l):
    """判断列表是否为空"""
    return len(l) == 0

def default(*values):
    """返回第一个不为 None 的值，类似 || 运算符的短路逻辑"""
    for value in values:
        if exists(value):
            return value
    return None

def cast_tuple(val, length = 1):
    """将值转换为指定长度的元组，已是元组则直接返回"""
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    """将张量按 pattern 打包，返回打包后的张量（包装 einops pack）"""
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    """将打包的张量解包，返回解包后的张量（包装 einops unpack）"""
    return unpack(x, ps, pattern)[0]

def pack_one_with_inverse(x, pattern):
    """
    打包张量并返回逆操作函数，用于需要恢复原始形状的场景。
    返回 (packed_tensor, inverse_fn)，inverse_fn 可还原打包前的形状。
    """
    packed, packed_shape = pack_one(x, pattern)

    def inverse(x, inverse_pattern = None):
        return unpack_one(x, packed_shape, default(inverse_pattern, pattern))

    return packed, inverse

# ==============================================================================
# 张量操作工具函数
# ==============================================================================

def project(x, y):
    """
    将向量 x 投影到向量 y 上，分解为平行分量和正交分量。

    用于 CFG 推理中的 remove_parallel_component 选项：
    移除更新向量中与原始输出平行的分量，防止 CFG 过度放大。

    返回:
        (parallel, orthogonal): 平行分量和正交分量
    """
    # 将多余维度打包到 batch 维度，统一处理
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    # 使用双精度进行投影计算，确保数值稳定性
    x, y = x.double(), y.double()
    # 将 y 归一化为单位向量
    unit = F.normalize(y, dim = -1)

    # 计算平行分量: (x·ŷ) * ŷ — x 在 y 方向上的投影
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    # 正交分量: x - x_平行
    orthogonal = x - parallel

    # 恢复原始精度和形状
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

def prob_mask_like(shape, prob, device):
    """
    生成 Bernoulli 分布的布尔掩码。

    以概率 prob 为 True，概率 (1-prob) 为 False。
    用于实现条件丢弃（cond_drop）：
    - prob=1 时全部保留条件 → 全 True
    - prob=0 时全部丢弃条件 → 全 False
    """
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        # 均匀采样 [0,1)，小于 prob 的位置为 True
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# ==============================================================================
# Classifier-Free Guidance 核心函数装饰器
# ==============================================================================

@typecheck
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    """
    函数装饰器，为模型的 forward 方法添加 Classifier-Free Guidance 功能。

    装饰后的 forward 方法支持:
    1. 自动将文本转换为条件函数（通过 text_conditioner）
    2. 训练时按概率丢弃文本条件
    3. 推理时根据 cond_scale 参数调节条件信号的强度

    参数:
        fn: 被装饰的 forward 函数
        cond_drop_prob_keyname: 条件丢弃概率的参数名
        texts_key_name: 文本输入参数名
        text_embeds_key_name: 文本嵌入参数名
        cond_fns_keyname: 条件函数参数名
        text_conditioner_name: 模型上条件器属性的名称
    """
    # 检查原始 forward 函数的参数列表
    fn_params = signature(fn).parameters

    # 如果 forward 不接受 texts/text_embeds，则自动处理文本条件转换
    auto_handle_text_condition = texts_key_name not in fn_params and text_embeds_key_name not in fn_params

    @wraps(fn)
    def inner(
        self,
        *args,
        cond_scale: float = 1.,              # CFG 条件缩放因子，>1 时放大条件影响
        rescale_phi: float = 0.,             # 输出重缩放系数，防止 CFG 过饱和
        return_unconditioned: bool = False,  # 是否同时返回无条件预测（用于 CFG++）
        remove_parallel_component: bool = False,  # 是否移除平行分量
        keep_parallel_frac: float = 0.,      # 保留平行分量的比例（0=完全移除）
        # 分别控制有条件和无条件前向的参数（用于处理 Transformer 解码时的缓存）
        cfg_routed_kwargs: Dict[str, Tuple[Any, Any]] = dict(),
        **kwargs
    ):
        """
        内部包装函数，处理 CFG 推理逻辑。
        """

        @wraps(fn)
        def fn_maybe_with_text(self, *args, **kwargs):
            """
            可选的文本条件注入版本的前向传播。
            如果模型有 text_conditioner，则自动将 texts/text_embeds 转换为条件函数。
            """
            if auto_handle_text_condition:
                # 从 kwargs 中提取文本相关参数
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)

                # 确保 texts 和 text_embeds 不同时存在
                assert not (exists(texts) and exists(text_embeds))

                raw_text_cond = cond_fns = None

                # 获取模型上的文本条件器
                text_conditioner = getattr(self, text_conditioner_name, None)

                # 提取并验证条件丢弃概率
                cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None)
                assert not exists(cond_drop_prob) or 0. <= cond_drop_prob <= 1.

                # ---- 自动将文本转换为条件函数 ----

                if exists(texts) ^ exists(text_embeds):
                    # 验证 texts 类型
                    assert is_bearable(texts, List[str] | None), \
                        f'keyword `{texts_key_name}` must be a list of strings'

                    # 验证条件器存在且类型正确
                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), \
                        'text_conditioner must be set on your network with the correct hidden dimensions to be conditioned on'

                    # 构建条件器输入
                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds)

                    # 调用条件器生成条件函数和原始文本条件
                    cond_fns, raw_text_cond = text_conditioner(
                        **text_condition_input,
                        cond_drop_prob = cond_drop_prob
                    )

                elif isinstance(text_conditioner, NullConditioner):
                    # 空条件器：不丢弃任何东西
                    assert cond_drop_prob == 0., 'null conditioner has nothing to dropout'
                    cond_fns, raw_text_cond = text_conditioner()

                # 如果 forward 接受 cond_fns 参数，注入
                if 'cond_fns' in fn_params:
                    kwargs.update(cond_fns = cond_fns)

                # 如果 forward 接受 raw_text_cond 参数，注入
                if 'raw_text_cond' in fn_params:
                    kwargs.update(raw_text_cond = raw_text_cond)

            # 调用原始 forward
            return fn(self, *args, **kwargs)

        # ======================================================================
        # 主 CFG 逻辑
        # ======================================================================

        # ---- 训练模式：不支持条件缩放，直接前向传播 ----
        if self.training:
            assert cond_scale == 1, 'you cannot do condition scaling when in training mode'
            return fn_maybe_with_text(self, *args, **kwargs)

        # ---- 推理模式：执行 CFG ----
        assert cond_scale >= 1, 'invalid conditioning scale, must be greater or equal to 1'

        # 准备丢弃/不丢弃文本条件的 kwargs（分别用于有条件和无条件前向）
        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}  # 保留条件
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}     # 丢弃条件

        # 分离 cfg_routed_kwargs 到有条件和无条件两组
        fn_kwargs = {k: v[0] for k, v in cfg_routed_kwargs.items()}
        null_fn_kwargs = {k: v[1] for k, v in cfg_routed_kwargs.items()}

        # ---- 有条件前向传播 ----
        outputs = fn_maybe_with_text(self, *args, **fn_kwargs, **kwargs_without_cond_dropout)

        # cond_scale == 1 时直接返回有条件输出
        if cond_scale == 1:
            return outputs

        logits, *rest = cast_tuple(outputs)

        # ---- 无条件前向传播（丢弃文本条件） ----
        null_outputs = fn_maybe_with_text(self, *args, **null_fn_kwargs, **kwargs_with_cond_dropout)

        null_logits, *null_rest = cast_tuple(null_outputs)

        # 将多返回值中的辅助输出配对
        zipped_rest = tuple(zip(rest, null_rest))

        # ---- CFG 公式: output = logits + (cond_scale - 1) * (logits - null_logits) ----
        # update = 条件信号的方向（有条件 - 无条件）
        update = logits - null_logits

        # 可选：移除平行分量，防止 CFG 过度放大特定方向
        if remove_parallel_component:
            update_parallel, update_orthog = project(update, logits)
            update = update_orthog + update_parallel * keep_parallel_frac

        # 按 scale 缩放条件信号
        scaled_logits = logits + update * (cond_scale - 1.)

        # ---- 输出重缩放 ----
        # 源自论文 https://arxiv.org/abs/2305.08891
        # 防止 CFG 过饱和，在像素空间和潜在空间都有效
        if rescale_phi <= 0:
            logit_output = scaled_logits
        else:
            # 计算除 batch 和最后一维外的所有维度
            dims = tuple(range(1, logits.ndim - 1))
            # 将 scaled_logits 的标准差缩放到与原始 logits 一致
            rescaled_logits = scaled_logits * (
                logits.std(dim = dims, keepdim = True) /
                scaled_logits.std(dim = dims, keepdim = True)
            )
            # phi 控制重缩放的强度（插值）
            logit_output = rescaled_logits * rescale_phi + scaled_logits * (1. - rescale_phi)

        # ---- 可选返回无条件预测（用于 CFG++） ----
        # CFG++ 论文: https://arxiv.org/abs/2406.08070
        output = logit_output

        if return_unconditioned:
            output = (output, null_logits)

        # ---- 处理多返回值：将辅助输出与主输出合并 ----
        if is_empty(zipped_rest):
            return output

        return (output, *zipped_rest)

    return inner

# ==============================================================================
# 类装饰器：自动为模型添加 CFG 文本条件功能
# ==============================================================================

@typecheck
def classifier_free_guidance_class_decorator(
    orig_class,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    """
    类装饰器，自动为 nn.Module 子类添加文本条件器和 CFG 功能。

    功能:
    1. 修改 __init__，自动创建 text_conditioner
    2. 用 classifier_free_guidance 装饰 forward 方法
    3. 添加 embed_texts 和 max_cond_text_len 属性

    参数:
        orig_class: 被装饰的模型类（必须是 nn.Module 的子类）
    """
    assert issubclass(orig_class, Module)

    # ==========================================================================
    # 装饰 __init__：自动创建 text_conditioner
    # ==========================================================================

    orig_init = orig_class.__init__

    @wraps(orig_init)
    @typecheck
    def __init__(
        self,
        *args,
        # 文本条件类型：FiLM 调制 / 交叉注意力 / 原始嵌入 / 无条件
        text_condition_type: Literal['film', 'attention', 'null', 'raw'] = 'film',
        text_condition_model_types: Tuple[str, ...] = ('t5',),  # 文本编码器类型
        text_condition_hidden_dims: Tuple[int, ...],             # 各层隐藏维度
        text_condition_cond_drop_prob: float,                    # 条件丢弃概率
        **kwargs
    ):
        # 调用原始 __init__
        orig_init(self, *args, **kwargs)

        # 根据类型选择条件器
        if text_condition_type == 'film':
            condition_klass = TextConditioner
        elif text_condition_type == 'attention':
            condition_klass = AttentionTextConditioner
        elif text_condition_type == 'raw':
            condition_klass = TextEmbeddingReturner
        else:
            condition_klass = NullConditioner

        # 创建文本条件器实例
        self.text_conditioner = condition_klass(
            model_types = text_condition_model_types,
            hidden_dims = text_condition_hidden_dims,
            cond_drop_prob = text_condition_cond_drop_prob
        )

    # 替换原始 __init__
    orig_class.__init__ = __init__

    # ==========================================================================
    # 装饰 forward：添加 CFG 功能
    # ==========================================================================

    decorated_forward = classifier_free_guidance(
        orig_class.forward,
        cond_drop_prob_keyname = cond_drop_prob_keyname,
        texts_key_name = texts_key_name,
        text_embeds_key_name = text_embeds_key_name,
        cond_fns_keyname = cond_fns_keyname,
        text_conditioner_name = text_conditioner_name
    )

    orig_class.forward = decorated_forward

    # ==========================================================================
    # 添加辅助方法
    # ==========================================================================

    @typecheck
    def embed_texts(self, texts: List[str]):
        """将文本列表编码为嵌入向量，委托给 text_conditioner"""
        return self.text_conditioner.embed_texts(texts)

    @property
    @cache
    def max_cond_text_len(self):
        """所有文本模型的最大文本长度之和（结果被缓存）"""
        total_cond_text_len = sum([
            text_model.max_text_len for text_model in self.text_conditioner.text_models
        ])
        return total_cond_text_len

    # 仅在原始类没有这些属性时添加（避免覆盖）
    if not hasattr(orig_class, 'max_cond_text_len'):
        orig_class.max_cond_text_len = max_cond_text_len

    if not hasattr(orig_class, 'embed_texts'):
        orig_class.embed_texts = embed_texts

    # 标记该类已被 CFG 装饰
    orig_class.__decorated_with_cfg = True
    return orig_class

# ==============================================================================
# 注意力模块
# ==============================================================================

class Attention(Module):
    """
    多头交叉/自注意力模块。支持可学习的空键值对（null key-value），
    用于无条件时的"空白"注意力。

    参数:
        dim: 输入/输出维度
        dim_head: 每个注意力头的维度
        heads: 注意力头数
        dim_context: 上下文（条件）维度，None 则为自注意力
        norm_context: 是否对上下文做 LayerNorm
        num_null_kv: 可学习空键值对的数量
        flash: 是否使用 Flash Attention
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_context = None,
        norm_context = False,
        num_null_kv = 0,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        # 缩放因子: 1/√d_k，用于防止点积过大
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # 如果没有指定上下文维度，默认为输入维度（自注意力）
        dim_context = default(dim_context, dim)

        # 对 query 输入做归一化
        self.norm = nn.LayerNorm(dim)
        # 对上下文（key/value 来源）做归一化（可选）
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attend = Attend(flash = flash)

        # 可学习的空键值对：用于无条件模式下的"无信息"注意力
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        # Q 投影：从输入到 Q（自注意力时 dim = dim_context）
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # KV 投影：从上下文到 K 和 V（合并为一个线性层，之后 chunk 分开）
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        # 输出投影
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None
    ):
        b = x.shape[0]

        # 对上下文做归一化
        if exists(context):
            context = self.context_norm(context)

        # 未提供上下文时，使用自注意力（kv_input = x）
        kv_input = default(context, x)

        # 对 query 输入做归一化
        x = self.norm(x)

        # 计算 Q, K, V
        # Q 来自归一化后的 x
        # K, V 来自 kv_input（上下文或 x 自身）
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        # 拼接可学习的空键值对（使模型能注意到"无信息"位置）
        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        # 扩展掩码以容纳空键值位置（空键值总是可见的，value=True）
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # 重塑 Q 为多头格式: (B, heads, seq_len, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 执行注意力计算
        out = self.attend(q, k, v, mask = mask)

        # 合并多头输出: (B, heads, seq_len, dim_head) → (B, seq_len, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ==============================================================================
# 通道格式适配器
# ==============================================================================

def rearrange_channel_last(fn):
    """
    包装条件函数，适配 (B, *, D) 格式的输入（通道在最后一维）。
    适用于 Transformer 层的隐藏状态。

    工作流程:
        (B, *, D) → pack 展平 → 调用 fn → unpack 恢复
    """
    @wraps(fn)
    def inner(hiddens):
        # 将除 batch 维度外的所有维度打包成一维（'b * d' 中 * 表示所有剩余维度）
        # 输入 hiddens 形状: (B, *, D) → 输出 hiddens 形状: (B, N, D)，其中 N = 所有非 batch 维度的元素总数
        hiddens, ps = pack_one(hiddens, 'b * d')  # 将除 batch 外的所有维度打包
        # 调用原始条件函数 fn，它期望输入形状 (B, N, D) 并返回相同形状
        conditioned = fn(hiddens)
        # 将输出解包，恢复原始的 (B, *, D) 形状
        return unpack_one(conditioned, ps, 'b * d')
    return inner

def rearrange_channel_first(fn):
    """
    包装条件函数，适配 (B, D, *) 格式的输入（通道在第一维，图像格式）。
    适用于 ViT 中间层的特征图。
    包装后得到的 wrapper_fn(cond_fn) 是一个新函数，它接受 (B, C, H, W) 形状的张量，并在内部自动转换为 (B, N, C) 传给原始 cond_fn，最后再恢复形状。

    工作流程:
        (B, D, H, W) → pack 展平 → transpose 到 (B, N, D) → 调用 fn → transpose 回 (B, D, N) → unpack 恢复
    """
    @wraps(fn)
    def inner(hiddens):
        # 1. 将除 batch 和通道外的所有空间维度“打包”成一维
        #    输入 hiddens 形状: (B, D, H, W) 或 (B, D, H, W, ...)
        #    输出 hiddens 形状: (B, D, N)，其中 N = H*W*...
        hiddens, ps = pack_one(hiddens, 'b d *')       # 打包空间维度
        # 2. 转换维度：将通道维度移到最后一维
        #    (B, D, N) → (B, N, D)
        hiddens = rearrange(hiddens, 'b d n -> b n d')  # 转为通道在最后的格式
        # 3. 调用原始的条件函数 fn，fn 的输入输出形状都是 (B, N, D)
        conditioned = fn(hiddens)
        # 4. 把维度换回来：(B, N, D) → (B, D, N)
        conditioned = rearrange(conditioned, 'b n d -> b d n')  # 转回通道在第一的格式
        # 5. 解包，恢复原来的空间形状（例如 (B, D, H, W)）
        return unpack_one(conditioned, ps, 'b d *')
    return inner

# ==============================================================================
# 条件注入模块
# ==============================================================================

class FiLM(Module):
    """
    FiLM (Feature-wise Linear Modulation) 条件调制模块。

    通过文本嵌入生成 scale 和 shift 参数，对隐藏层特征进行仿射变换:
        output = hidden * (scale + 1) + shift

    权重和偏置初始化为零，确保初始时条件注入为恒等变换。

    参数:
        dim: 条件嵌入的维度
        hidden_dim: 被调制的隐藏层特征维度
    """
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        # 两层 MLP: dim → hidden_dim*4 → hidden_dim*2（生成 scale 和 shift）
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )

        # 零初始化最后一层的权重和偏置
        # 这样初始输出为 (1 + 0) * x + 0 = x，即恒等变换
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens):
        """
        参数:
            conditions: (B, D_cond) 文本条件向量
            hiddens: (B, N, D_hidden) 被调制的隐藏层特征
        返回:
            modulated: (B, N, D_hidden) 调制后的特征
        """
        # 将条件向量映射为 scale 和 shift: (B, hidden_dim*2) → 各 (B, hidden_dim)
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert scale.shape[-1] == hiddens.shape[-1], \
            f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'

        # 扩展维度以匹配隐藏层: (B, D) → (B, 1, D)，便于广播
        scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))

        # FiLM 调制: 特征乘以 (1+scale) 再加 shift
        return hiddens * (scale + 1) + shift

class CrossAttention(Module):
    """
    交叉注意力条件注入模块。

    以隐藏层特征为 query，文本嵌入为 key/value，
    通过交叉注意力将文本信息融合到隐藏层中。
    使用残差连接: output = attention(hiddens, condition) + hiddens

    参数:
        dim: 文本嵌入的维度
        hidden_dim: 隐藏层特征的维度
        heads: 注意力头数
        dim_head: 每个注意力头的维度
        flash: 是否使用 Flash Attention
    """
    def __init__(
        self,
        dim,
        hidden_dim,
        heads = 8,
        dim_head = 64,
        flash = False
    ):
        super().__init__()
        self.attn = Attention(
            dim = hidden_dim,           # query 的维度
            dim_context = dim,          # key/value 来自文本嵌入
            norm_context = True,        # 对文本上下文做归一化
            num_null_kv = 1,            # 一个可学习的空键值对（用于无条件模式）
            dim_head = dim_head,
            heads = heads,
            flash = flash
        )

    def forward(
        self,
        condition,
        hiddens,
        mask = None
    ):
        # 残差连接：交叉注意力 + 原始输入
        return self.attn(hiddens, condition, mask = mask) + hiddens

# ==============================================================================
# 文本条件器配置
# ==============================================================================

# 支持的文本编码器类型及其适配器
CONDITION_CONFIG = dict(
    t5 = T5Adapter,       # Google T5 文本编码器
    clip = OpenClipAdapter,  # OpenAI CLIP 文本编码器
    bge = BGEAdapter       # BAAI BGE 文本编码器
)

MODEL_TYPES = CONDITION_CONFIG.keys()

# ==============================================================================
# 条件器基类
# ==============================================================================

class Conditioner(Module):
    """所有条件器的抽象基类，继承自 nn.Module"""
    pass

# ==============================================================================
# 空条件器 (NullConditioner) — 不注入任何文本条件
# ==============================================================================

class Identity(Module):
    """恒等模块，直接返回输入，不做任何变换"""
    def forward(self, t, *args, **kwargs):
        return t

class NullConditioner(Conditioner):
    """
    空条件器：生成一组恒等条件函数，不注入任何文本信息。

    用于不需要文本条件的模型，或作为占位符。
    所有条件函数都是 Identity()，前向传播时 t 保持不变。

    参数:
        hidden_dims: 各层隐藏维度的元组（仅用于确定条件函数的数量）
    """
    @typecheck
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        **kwargs
    ):
        super().__init__()
        # 为每个隐藏维度创建一个恒等条件函数
        num_null_conditioners = len(hidden_dims)
        self.cond_fns = tuple(Identity() for _ in range(num_null_conditioners))

        # 注册一个持久化的缓冲区参数，用于获取设备信息
        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        """返回模型所在的设备"""
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        """空条件器不支持文本嵌入，调用此方法会报错"""
        assert False, 'null conditioner cannot embed text'

    def forward(self, *args, **kwarg):
        """返回恒等条件函数和 None 文本条件"""
        return self.cond_fns, None

# ==============================================================================
# FiLM 文本条件器 (TextConditioner)
# ==============================================================================

class TextConditioner(Conditioner):
    """
    基于 FiLM 的文本条件器。

    工作流程:
    1. 使用文本编码器（T5/CLIP/BGE）将文本编码为嵌入向量
    2. 将所有编码器的嵌入拼接，通过 MLP stem 变换
    3. 训练时以 cond_drop_prob 概率替换为空嵌入（CFG 训练）
    4. 为每个目标隐藏维度创建一个 FiLM 条件函数
    5. 这些条件函数在前向传播时注入到网络各层

    参数:
        hidden_dims: 需要条件注入的各层隐藏维度元组
        model_types: 文本编码器类型（'t5', 'clip', 'bge'）
        model_names: 文本编码器的预训练模型名称
        cond_drop_prob: 训练时的条件丢弃概率
        hiddens_channel_first: 各层数据是否为通道优先格式（True=图像格式）
        text_embed_stem_dim_mult: MLP stem 输出维度的倍增因子
        text_embed_pad_value: 文本嵌入的填充值
    """
    @typecheck
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...], # 需要被条件调制的各个层的隐藏维度（例如 ViT 各阶段输出通道数、Transformer 每层的特征维度）
        model_types = 't5', # 文本编码器类型，支持 't5', 'bert' 等，可多个
        model_names = None, # 文本编码器的具体模型名称（如 't5-base'）
        cond_drop_prob = 0., # 条件丢弃概率（classifier-free guidance 中随机将文本替换为空嵌入）
        hiddens_channel_first = True, # 被调制的隐藏层是通道优先（True，图像特征）还是通道最后（False，序列 token）
        text_embed_stem_dim_mult = 2, # 文本嵌入投影 MLP 的扩展倍数
        text_embed_pad_value = 0. # 文本 padding 值
    ):
        super().__init__()
        # 标准化输入为元组
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 加载文本编码器
        text_models = []
        # 为每个 (model_type, model_name) 实例化对应的文本编码器（如 T5Encoder）
        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value)
            text_models.append(model)

        self.text_models = text_models # 保存文本编码器列表
        # 收集各编码器的潜在维度
        self.latent_dims = [model.dim_latent for model in text_models]

        self.conditioners = ModuleList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)  # 条件函数的总数
        # 标记每个隐藏层是通道优先（图像）还是通道最后（序列）
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns)

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.cond_drop_prob = cond_drop_prob

        # 总潜在维度 = 各编码器维度之和
        total_latent_dim = sum(self.latent_dims)
        self.dim_latent = total_latent_dim

        # MLP stem 输出维度：可配置的倍增
        mlp_stem_output_dim = total_latent_dim * text_embed_stem_dim_mult

        # MLP stem：将拼接的文本嵌入映射到更高维空间
        self.text_embed_stem_mlp = nn.Sequential(
            nn.Linear(total_latent_dim, mlp_stem_output_dim),
            nn.SiLU()
        )

        # 为每个目标隐藏维度创建 FiLM 条件模块，输入维度为 mlp_stem_output_dim，输出维度为 hidden_dim
        # FiLM 会根据文本嵌入计算缩放和偏置 (scale, shift)，用于调制特征
        for hidden_dim in hidden_dims:
            self.conditioners.append(FiLM(mlp_stem_output_dim, hidden_dim))

        # 可学习的空文本嵌入：用于 CFG 训练中的条件丢弃
        self.null_text_embed = nn.Parameter(torch.randn(total_latent_dim))

        # 注册持久化设备参数
        # TODO：搞清楚什么意思
        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        """返回模型所在的设备"""
        # TODO：搞清楚什么意思
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        """
        将文本列表编码为嵌入向量。

        依次调用所有文本编码器，然后将输出拼接。
        返回: (B, total_latent_dim) 的嵌入向量
        """
        device = self.device

        text_embeds = []
        for text_model in self.text_models:
            # 平均池化后的句子级嵌入 (B, dim_latent)
            text_embed = text_model.embed_text(texts)
            text_embeds.append(text_embed.to(device))

        # 在最后一维拼接所有编码器的输出
        return torch.cat(text_embeds, dim = -1)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None,
        repeat_batch = 1,               # 批量重复次数（用于 RT-1 等需要帧级条件的场景）
    ) -> Tuple[
        Tuple[Callable, ...],           # 条件函数元组
        TextCondReturn                  # 文本嵌入和掩码
    ]:
        """
        前向传播：将文本转换为条件函数列表。

        参数:
            texts: 文本字符串列表
            text_embeds: 预嵌入的文本向量（与 texts 二选一）
            cond_drop_prob: 条件丢弃概率
            repeat_batch: 批量重复次数，用于将条件向量扩展到更大 batch
                         例如 RT-1 中视频每帧都需要独立的条件

        返回:
            条件函数元组 + TextCondReturn(嵌入, 掩码)
            cond_fns: 一个元组，包含 num_condition_fns 个条件函数，每个函数接受一个特征张量 x 并返回调制后的 x
            TextCondReturn: 包含处理后的文本嵌入等信息的辅助返回对象
        """

        # 确保 texts 和 text_embeds 恰好提供一个
        assert exists(texts) ^ exists(text_embeds)

        # 确定条件丢弃概率
        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            # 推理时必须显式指定 cond_drop_prob
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        # 确定批量大小
        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        # 如果没有预嵌入向量，则从文本编码
        if not exists(text_embeds):
            # 尺寸为 (B, total_latent_dim)
            text_embeds = self.embed_texts(texts)

        # ---- CFG 条件丢弃 ----
        # 以概率 cond_drop_prob 将文本嵌入替换为空嵌入
        if cond_drop_prob > 0.:
            # 生成保留掩码: 1-cond_drop_prob 的概率为 True
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            # 扩展空嵌入以匹配 batch
            null_text_embeds = rearrange(self.null_text_embed, 'd -> 1 d')

            # 根据掩码选择：True → 保留文本嵌入, False → 使用空嵌入
            # 尺寸为 (B, total_latent_dim)
            text_embeds = torch.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds
            )

        # ---- MLP Stem 变换 ----
        # 将原始嵌入映射到更高维空间（类似 Guided Diffusion 中的处理）
        # 尺寸为 (B, mlp_stem_output_dim)
        text_embeds = self.text_embed_stem_mlp(text_embeds)

        # ---- 生成条件函数 ----
        # 将 repeat_batch 扩展为与条件函数数量等长的元组
        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        # 遍历每个目标隐藏层
        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(
            self.conditioners,
            self.hiddens_channel_first,
            repeat_batch
        ):
            # 按 repeat_batch 重复文本嵌入（例如 RT-1 中按帧数重复）
            # 尺寸为 (B * repeat_batch, mlp_stem_output_dim)
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)

            # 创建偏函数：将文本嵌入固定为 FiLM 的条件输入
            # 调用时只需传入 hiddens，自动使用 cond_text_embeds 作为条件
            cond_fn = partial(cond, cond_text_embeds)

            # 根据数据格式选择合适的包装器
            # 通道优先（图像格式）→ rearrange_channel_first
            # 通道最后（序列格式）→ rearrange_channel_last
            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        # 将文本嵌入向量以及可能需要的辅助信息打包返回
        return tuple(cond_fns), TextCondReturn(text_embeds, None)

# ==============================================================================
# 交叉注意力文本条件器 (AttentionTextConditioner)
# ==============================================================================

@typecheck
class AttentionTextConditioner(Conditioner):
    """
    基于交叉注意力的文本条件器。

    与 TextConditioner (FiLM) 的区别:
    - FiLM 通过 scale/shift 全局调制特征
    - 交叉注意力允许文本嵌入与隐藏层特征进行 token 级别的交互

    工作流程:
    1. 文本编码器生成嵌入序列（而非单一向量）
    2. 以隐藏层特征为 query，文本嵌入为 key/value 做交叉注意力
    3. 使用残差连接保留原始信息

    参数:
        hidden_dims: 需要条件注入的各层隐藏维度
        model_types: 文本编码器类型
        model_names: 预训练模型名称
        cond_drop_prob: 条件丢弃概率
        hiddens_channel_first: 各层数据是否为通道优先格式
        dim_latent: 统一的潜在维度
        attn_dim_head: 注意力头维度
        attn_heads: 注意力头数
        flash: 是否使用 Flash Attention
        text_embed_pad_value: 文本嵌入的填充值
    """
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        dim_latent = None,
        attn_dim_head = 64,
        attn_heads = 8,
        flash = True,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 加载文本编码器
        text_models = []
        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value)
            text_models.append(model)

        self.text_models = text_models

        # 将各编码器的嵌入投影到统一的潜在维度
        self.to_latent_dims = ModuleList([])

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))
        self.dim_latent = dim_latent

        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = ModuleList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns)

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.text_embed_pad_value = text_embed_pad_value
        self.cond_drop_prob = cond_drop_prob

        # 为每个目标隐藏维度创建交叉注意力条件模块
        for hidden_dim in hidden_dims:
            self.conditioners.append(CrossAttention(dim_latent, hidden_dim, flash = flash))

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        """返回模型所在的设备"""
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        """
        将文本编码为嵌入序列，并投影到统一维度。

        与 TextConditioner.embed_texts 的区别:
        - 返回的是序列嵌入 (B, seq_len, dim)，而非单一向量
        - 每个编码器的输出被线性投影到统一维度
        - 保留填充位置对应的掩码信息

        返回: (B, total_seq_len, dim_latent) 的嵌入序列
        """
        device = self.device

        text_embeds = []

        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            # return_text_encodings=True 返回完整序列嵌入而非池化后的向量
            text_embed = text_model.embed_text(texts, return_text_encodings = True)
            text_embed = text_embed.to(device)

            # 生成掩码：标记非填充位置
            mask = (text_embed != self.text_embed_pad_value).any(dim = -1)

            # 投影到统一维度，并填充无效位置
            text_embed = to_latent(text_embed)
            text_embed = text_embed.masked_fill(~mask[..., None], self.text_embed_pad_value)

            text_embeds.append(text_embed)

        # 在序列维度上拼接
        return torch.cat(text_embeds, dim = -2)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None,
        repeat_batch = 1,               # 批量重复次数（用于 RT-1 等场景）
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:
        """
        前向传播：将文本转换为交叉注意力条件函数列表。
        """

        assert exists(texts) or exists(text_embeds)

        # 如果同时提供 texts 和 text_embeds，text_embeds 优先
        # 这允许在第一个 epoch 后缓存文本嵌入以提高效率
        if exists(text_embeds) and exists(texts):
            texts = None

        # 确定条件丢弃概率
        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        # 确定批量大小
        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        # 编码文本（如未提供预嵌入）
        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        # 生成注意力掩码：标记非填充位置
        mask = (text_embeds != self.text_embed_pad_value).any(dim = -1)

        # ---- CFG 条件丢弃 ----
        # 丢弃条件时，整个掩码置为 False（所有位置都被视为填充）
        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            mask = mask & prob_keep_mask

        # ---- 生成条件函数 ----
        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(
            self.conditioners,
            self.hiddens_channel_first,
            repeat_batch
        ):
            # 按 repeat_batch 重复文本嵌入和掩码
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            cond_mask = repeat(mask, 'b ... -> (b r) ...', r = cond_repeat_batch) if exists(mask) else None

            # 创建偏函数：固定 text_embeds 和 mask
            cond_fn = partial(cond, cond_text_embeds, mask = cond_mask)

            # 选择合适的格式包装器
            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        return tuple(cond_fns), TextCondReturn(text_embeds, mask)

# ==============================================================================
# 原始文本嵌入返回器 (TextEmbeddingReturner)
# ==============================================================================

class TextEmbeddingReturner(Conditioner):
    """
    直接返回文本嵌入的条件器，不做 FiLM 调制或交叉注意力注入。

    用于模型需要直接使用原始文本嵌入的场景（text_condition_type='raw'）。
    条件函数为 Identity()，即将文本嵌入不做变换地透传。

    参数:
        dim_latent: 统一的潜在维度
        hidden_dims: 额外层的隐藏维度（每个对应一个 Identity 条件函数）
        model_types: 文本编码器类型
        model_names: 预训练模型名称
        model_kwargs: 各编码器的额外参数
        cond_drop_prob: 条件丢弃概率
        text_embed_pad_value: 文本嵌入的填充值
    """
    @typecheck
    def __init__(
        self,
        *,
        dim_latent = None,
        hidden_dims: Tuple[int, ...] = (),
        model_types = 't5',
        model_names = None,
        model_kwargs: dict = dict(),
        cond_drop_prob = 0.,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))
        model_kwargs = cast_tuple(model_kwargs, length = len(model_types))

        assert len(model_types) == len(model_names) == len(model_kwargs)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 加载文本编码器（支持传入额外参数）
        text_models = []
        for model_type, model_name, model_kwarg in zip(model_types, model_names, model_kwargs):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value, **model_kwarg)
            text_models.append(model)

        self.text_models = text_models
        self.text_embed_pad_value = text_embed_pad_value

        # 投影层：将各编码器输出映射到统一维度
        self.to_latent_dims = ModuleList([])

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))
        self.dim_latent = dim_latent

        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = ModuleList([])
        self.cond_drop_prob = cond_drop_prob

        # 条件函数为恒等变换（不做任何调制）
        for hidden_dim in hidden_dims:
            self.conditioners.append(nn.Identity())

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        """返回模型所在的设备"""
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        """
        将文本编码为嵌入序列，投影到统一维度。
        与 AttentionTextConditioner.embed_texts 的实现相同。

        返回: (B, total_seq_len, dim_latent) 的嵌入序列
        """
        device = self.device

        text_embeds = []

        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            text_embed = text_model.embed_text(texts, return_text_encodings = True)
            text_embed = text_embed.to(device)

            mask = (text_embed != self.text_embed_pad_value).any(dim = -1)

            text_embed = to_latent(text_embed)
            text_embed = text_embed.masked_fill(~mask[..., None], self.text_embed_pad_value)

            text_embeds.append(text_embed)

        return torch.cat(text_embeds, dim = -2)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:
        """
        前向传播：返回恒等条件函数和原始文本嵌入。

        条件函数不做任何变换（Identity），文本嵌入通过 mask 机制
        实现 CFG 丢弃。
        """

        assert exists(texts) ^ exists(text_embeds)

        # 确定条件丢弃概率
        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        # 确定批量大小
        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        # 编码文本
        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        # 生成掩码
        mask = (text_embeds != self.text_embed_pad_value).any(dim = -1)

        # CFG 条件丢弃：以概率置掩码为 False
        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            mask = mask & prob_keep_mask

        # 返回恒等条件函数（不做调制）+ 文本嵌入及掩码
        return tuple(self.conditioners), TextCondReturn(text_embeds, mask)
