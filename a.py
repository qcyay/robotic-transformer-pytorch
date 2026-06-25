from einops import pack, unpack, repeat, reduce, rearrange
import torch
from transformers import T5Tokenizer, T5EncoderModel, T5Config

from robotic_transformer_pytorch.robotic_transformer import pack_one

# video = torch.randn(1, 3, 16, 224, 224)
# video = rearrange(video, 'b c f h w -> b f c h w')
# images, packed_shape = pack_one(video, '* c h w')
# print(images.size())
# print(packed_shape)

# text_embeds = torch.randn(2, 3)
# null_text_embeds = torch.zeros(1, 3)
# prob_keep_mask = torch.tensor([[True], [False]])
# result = torch.where(prob_keep_mask, text_embeds, null_text_embeds)
# print(result)

# name = 'google/t5-v1_1-base'
# t5 = T5EncoderModel.from_pretrained(name)
# # print(t5)
# tokenizer = T5Tokenizer.from_pretrained(name)
# MAX_LENGTH = 256
# texts = [
#     "I really like to play football",
#     "I like to play basketball"
# ]
# # texts = ["我非常喜欢踢足球", "我喜欢打篮球"]
# encoded = tokenizer(
#             texts, # 输入的文本列表
#             return_tensors = "pt", # 返回 PyTorch 张量（"pt"）格式
#             padding = 'longest', # 将该批次中所有序列填充到最长序列的长度
#             max_length = MAX_LENGTH, # 设置序列的最大长度
#             truncation = True # 允许截断
#           )
# # print(encoded)
#
# input_ids, attn_mask = encoded.input_ids, encoded.attention_mask
# with torch.no_grad():
#     output = t5(input_ids=input_ids, attention_mask=attn_mask)
#     print(output)
#     # 提取最后一层的隐藏状态，脱离计算图
#     encoded_text = output.last_hidden_state.detach()
#     print(encoded_text.size())

# from transformers import T5Tokenizer, T5EncoderModel, T5Config
# DEFAULT_T5_NAME = 'google/t5-v1_1-base'
# name = DEFAULT_T5_NAME
# model = T5EncoderModel.from_pretrained(name)
# print(model.config)
# config = T5Config.from_pretrained(name)
# print(config)

a = torch.ones((5, 5)).triu(1)
print(a)