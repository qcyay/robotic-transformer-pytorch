"""
T5 文本编码适配器
=================

封装 HuggingFace T5 模型，提供文本编码和嵌入提取功能。
用于 RT-1 等模型中将自然语言指令转换为固定维度的向量表示。

核心组件:
    - get_model_and_tokenizer: 单例模式加载 T5 模型和分词器，避免重复加载
    - get_encoded_dim: 获取 T5 模型的输出维度（无需完整加载模型）
    - t5_encode_text: 将文本编码为嵌入序列（独立函数，非类方法）
    - T5Adapter: T5 适配器类，提供 dim_latent、max_text_len 属性和 embed_text 方法

编码流程:
    文本 → Tokenizer → input_ids → T5 Encoder → last_hidden_state
                                                     ↓
                            return_text_encodings=True → 完整序列嵌入 (B, seq, dim)
                            return_text_encodings=False → 平均池化嵌入 (B, dim)
"""

from typing import List
from beartype import beartype

import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

# 关闭 HuggingFace 冗余日志，仅显示错误
transformers.logging.set_verbosity_error()

# ==============================================================================
# 通用辅助函数
# ==============================================================================

def exists(val):
    """判断值是否不为 None"""
    return val is not None

def default(val, d):
    """如果 val 不为 None 则返回 val，否则返回默认值 d"""
    return val if exists(val) else d

# ==============================================================================
# 常量配置
# ==============================================================================

# 最大文本长度（token 数），超出的部分将被截断
MAX_LENGTH = 256

# 默认的 T5 模型名称（v1.1 版本，不含预训练的语言模型头）
DEFAULT_T5_NAME = 'google/t5-v1_1-base'

# 全局配置缓存字典，实现单例模式
# 键: 模型名称, 值: dict(可选包含 model, tokenizer, config)
T5_CONFIGS = {}

# ==============================================================================
# 单例模式的全局加载函数
# ==============================================================================

def get_tokenizer(name):
    """
    根据模型名称加载 T5 分词器。

    参数:
        name: HuggingFace 模型名称（如 'google/t5-v1_1-base'）
    返回:
        T5Tokenizer 实例
    """
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

def get_model(name):
    """
    根据模型名称加载 T5 编码器模型（仅 Encoder 部分）。

    参数:
        name: HuggingFace 模型名称
    返回:
        T5EncoderModel 实例
    """
    # 从 Hugging Face 模型仓库中，将谷歌预训练好的一个仅包含编码器（encoder）部分的 T5 模型下载并加载到内存中，并将它赋值给变量 model。
    model = T5EncoderModel.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    """
    单例模式获取 T5 模型和分词器。

    首次调用时加载模型和分词器并缓存到全局字典 T5_CONFIGS 中，
    后续调用直接从缓存获取，避免重复加载占用显存。

    参数:
        name: HuggingFace 模型名称
    返回:
        (model, tokenizer) 元组
    """
    global T5_CONFIGS

    # 确保该模型名在缓存字典中有条目
    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    # 按需加载模型（首次加载后会被缓存）
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    # 按需加载分词器
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    """
    获取 T5 模型的输出嵌入维度（d_model），无需加载完整模型。

    优化策略（按优先级）:
    1. 如果未缓存任何信息 → 仅加载 config 获取 d_model（最轻量）
    2. 如果已缓存 config → 直接从 config 读取
    3. 如果已缓存 model → 从 model.config 读取

    参数:
        name: HuggingFace 模型名称
    返回:
        int: 模型的 d_model 维度
    """
    if name not in T5_CONFIGS:
        # 无缓存：仅加载配置（避免加载完整模型权重）
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        # 已有配置缓存
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        # 已有模型缓存，从模型中读取配置
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# ==============================================================================
# 文本编码
# ==============================================================================

def t5_encode_text(texts, name = DEFAULT_T5_NAME, output_device = None):
    """
    将文本列表编码为 T5 嵌入序列。

    这是一个独立函数（非类方法），可在不创建 T5Adapter 实例的情况下使用。

    参数:
        texts: 文本字符串列表
        name: T5 模型名称
        output_device: 输出张量的目标设备（None 则保持在模型设备上）
    返回:
        (encoded_text, attn_mask) 元组
        - encoded_text: (B, seq_len, d_model) 的嵌入序列
        - attn_mask: (B, seq_len) 的布尔注意力掩码
    """
    # 获取模型和分词器（单例模式）
    t5, tokenizer = get_model_and_tokenizer(name)

    # 自动将模型移至 GPU（如果可用）
    if torch.cuda.is_available():
        t5 = t5.cuda()

    # 获取模型当前所在的设备
    device = next(t5.parameters()).device

    # 使用分词器批量编码文本
    # padding='longest': 批量内对齐到最长序列
    # truncation=True: 超出 MAX_LENGTH 的序列被截断
    # encoded 是一个字典，通常包含以下键（具体取决于 tokenizer 类型）：
    # input_ids：形状为 (batch_size, sequence_length) 的 token ID 张量。
    # attention_mask：与 input_ids 同形状，1 表示真实 token，0 表示填充 token。
    # token_type_ids（某些模型如 BERT 有）：用于区分两个句子的段标识。
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",       # 返回 PyTorch 张量
        padding = 'longest',         # 填充到 batch 内最长序列
        max_length = MAX_LENGTH,     # 最大序列长度
        truncation = True            # 超出则截断
    )

    # 将 input_ids 和注意力掩码移至模型设备
    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    # 设置为评估模式（禁用 dropout 等训练特性）
    t5.eval()

    # 无梯度推理，节省显存
    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        # 提取最后一层的隐藏状态，脱离计算图
        # 形状：(batch_size, sequence_length, hidden_size)
        # batch_size：样本数量（比如你传入的句子数）。
        # sequence_length：每个句子的 token 序列长度（经过填充后）。
        # hidden_size：模型的隐藏层维度（T5-base 是 768，T5-large 是 1024 等）。
        encoded_text = output.last_hidden_state.detach()

    # 将注意力掩码转为布尔类型（True=有效token, False=填充token）
    attn_mask = attn_mask.bool()

    # 如果未指定输出设备，直接返回
    if not exists(output_device):
        return encoded_text, attn_mask

    # 将结果移至指定设备
    encoded_text.to(output_device)
    attn_mask.to(output_device)

    return encoded_text, attn_mask

# ==============================================================================
# T5 适配器类
# ==============================================================================

class T5Adapter():
    """
    T5 文本编码适配器。

    封装 T5 模型和分词器，为 CFG 条件模块提供统一的文本嵌入接口。
    支持两种输出模式:
    - 平均池化: 返回单一的句子级嵌入向量 (B, dim_latent)
    - 序列编码: 返回完整的 token 级嵌入序列 (B, seq_len, dim_latent)

    参数:
        name: T5 模型名称，默认 'google/t5-v1_1-base'
        text_embed_pad_value: 填充位置的值，用于掩码处理
    """
    def __init__(
        self,
        name,
        text_embed_pad_value = 0.
    ):
        # 使用默认模型名（如果未提供）
        name = default(name, DEFAULT_T5_NAME)

        # 单例模式加载模型和分词器
        t5, tokenizer = get_model_and_tokenizer(name)

        # 自动将模型移至 GPU
        if torch.cuda.is_available():
            t5 = t5.cuda()

        self.name = name
        self.t5 = t5
        self.tokenizer = tokenizer
        self.text_embed_pad_value = text_embed_pad_value

    @property
    def dim_latent(self):
        """
        T5 模型的潜在维度（d_model）。
        例如 t5-v1_1-base 为 768，t5-v1_1-large 为 1024。
        """
        return get_encoded_dim(self.name)

    @property
    def max_text_len(self):
        """最大文本长度（token 数），超出部分将被截断"""
        return MAX_LENGTH

    @torch.no_grad()
    @beartype
    def embed_text(
        self,
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):
        """
        将文本列表编码为嵌入向量。

        参数:
            texts: 文本字符串列表
            return_text_encodings: 是否返回完整序列嵌入
                - False: 返回平均池化后的句子级嵌入 (B, dim_latent)
                - True: 返回完整序列嵌入 (B, seq_len, dim_latent)
            output_device: 输出张量的目标设备
        返回:
            Tensor: 文本嵌入向量
        """
        # 获取模型设备
        device = next(self.t5.parameters()).device

        # 分词：文本 → token IDs
        # 将一批文本（texts）同时转换为模型可接受的输入格式
        # encoded 是一个字典，通常包含以下键（具体取决于 tokenizer 类型）：
        # input_ids：形状为 (batch_size, sequence_length) 的 token ID 张量。
        # attention_mask：与 input_ids 同形状，1 表示真实 token，0 表示填充 token。
        # token_type_ids（某些模型如 BERT 有）：用于区分两个句子的段标识。
        encoded = self.tokenizer.batch_encode_plus(
            texts, # 输入的文本列表
            return_tensors = "pt", # 返回 PyTorch 张量（"pt"）格式
            padding = 'longest', # 将该批次中所有序列填充到最长序列的长度
            max_length = MAX_LENGTH, # 设置序列的最大长度
            truncation = True # 允许截断
        )

        # 将输入移至模型设备
        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device)

        # 评估模式 + 无梯度推理
        self.t5.eval()

        with torch.no_grad():
            output = self.t5(input_ids = input_ids, attention_mask = attn_mask)
            # 提取最后一层的隐藏状态，脱离计算图
            # 形状：(batch_size, sequence_length, hidden_size)
            # batch_size：样本数量（比如你传入的句子数）。
            # sequence_length：每个句子的 token 序列长度（经过填充后）。
            # hidden_size：模型的隐藏层维度（T5-base 是 768，T5-large 是 1024 等）。
            encoded_text = output.last_hidden_state.detach()

        # 转为布尔掩码：True=有效token, False=PAD token
        attn_mask = attn_mask.bool()

        # 将填充位置的嵌入值设为 text_embed_pad_value（通常为 0）
        encoded_text.masked_fill_(~attn_mask[..., None], self.text_embed_pad_value)

        if not return_text_encodings:
            # ---- 平均池化 ----
            # 对所有有效 token 的嵌入求和
            numer = encoded_text.sum(dim = -2)          # (B, dim)
            # 统计每个样本的有效 token 数量
            denom = attn_mask.sum(dim = -1)[..., None]  # (B, 1)
            # 安全处理全填充样本（分母为 0）
            numer.masked_fill_(denom == 0, 0.)
            # 平均嵌入 = 有效 token 嵌入之和 / 有效 token 数量
            mean_encodings = numer / denom.clamp(min = 1e-3)
            return mean_encodings

        # 返回完整序列嵌入，并移至目标设备
        return encoded_text.to(output_device)
