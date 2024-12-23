import torch
from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp16_triton, sageattn_varlen
import torch.nn.functional as F
from datetime import datetime
from einops import rearrange, repeat

import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func

torch.manual_seed(81627)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

batch_size = 1
head_num = 3
seq_len = 118824
# seq_len = 512
head_dim = 128

q = torch.randn((batch_size,  seq_len,head_num, head_dim), dtype=torch.bfloat16).cuda()
k = torch.randn((batch_size,  seq_len,head_num, head_dim), dtype=torch.bfloat16).cuda()
v = torch.randn((batch_size,  seq_len,head_num, head_dim), dtype=torch.bfloat16).cuda()


for i in range(3):
    # attn_output1 = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32+fp32")
    attn_output = sageattn_qk_int8_pv_fp16_triton(q.clone().transpose(1,2).contiguous(), k.clone().transpose(1,2).contiguous(), v.clone().transpose(1,2).contiguous(), tensor_layout="HND", is_causal=False)
    attn_output = attn_output.transpose(1,2)
    # attn_output1 = sageattn_varlen(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, is_causal=False)
print("attn_fp8: ", attn_output)

for i in range(1):
    # attn_output1 = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn_output1 = flash_attn_func(q, k, v)
print("attn_output",attn_output1)




attn_output = attn_output.to(torch.float32).reshape(-1)
attn_output1 = attn_output1.to(torch.float32).reshape(-1)

# print (attn_output.reshape(-1).shape)
# print (attn_output1.reshape(-1).shape)
a = torch.norm(attn_output)
b = torch.norm(attn_output1)
print(a)
print(b)

ab = torch.sum(torch.abs(torch.mul(attn_output1, attn_output)))
print(ab)


# output = cos_sim(attn_output.reshape(-1), attn_output1.reshape(-1))

print("finished: ", ab/(a*b))

print("finished")
