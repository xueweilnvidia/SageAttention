import torch
from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp16_triton, sageattn_varlen
import torch.nn.functional as F
from datetime import datetime
from einops import rearrange, repeat


torch.manual_seed(81626)

torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

batch_size = 3
head_num = 3
seq_len = 119056
# seq_len = 512
head_dim = 128

q = torch.randn((batch_size, head_num, seq_len, head_dim), dtype=torch.bfloat16).cuda()*10
k = torch.randn((batch_size, head_num, seq_len, head_dim), dtype=torch.bfloat16).cuda()*5
v = torch.randn((batch_size, head_num, seq_len, head_dim), dtype=torch.bfloat16).cuda()*5

for i in range(3):
    attn_output = F.scaled_dot_product_attention(q, k, v)


print("attn_output",rearrange(attn_output.permute(0, 2, 1, 3), "b s h d -> (b s) h d"))

q = q.permute(0, 2, 1, 3)
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)
print(q.shape)

q= rearrange(q, "b s h d -> (b s) h d")
k= rearrange(k, "b s h d -> (b s) h d")
v= rearrange(v, "b s h d -> (b s) h d")

print(q.shape)

cu_seqlen = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device="cuda"
        )
print("seqlen: ", cu_seqlen)
max_seqlen = seq_len


for i in range(1):
    # attn_output1 = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32+fp32")
    # attn_output1 = sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)
    attn_output1 = sageattn_varlen(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, is_causal=False)


print("attn_fp8: ", attn_output1)

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
