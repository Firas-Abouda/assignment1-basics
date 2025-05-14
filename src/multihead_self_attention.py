import torch
import torch.nn as nn
from einops import einsum ,rearrange
from jaxtyping import Float
from src.linear import Linear
from src.activation import DpAttention
from src.rope import RoPE


class MHSA(nn.Module) : 

    def __init__(self, d_model, num_heads, w_rope= False, max_seq_len = 0, theta = 0, device= None, dtype = None):
        '''
        Construct a Multi head self attention module.
        d_model: int Dimensionality of the Transformer block inputs
        num_heads: int Number of heads to use in multi-head self-attention
        max_seq_len int: Maximum sequence length to pre-cache if your implementation does that.
        theta float: RoPE parameter.
        '''
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = self.d_v = d_model // num_heads

        self.Wq = Linear(d_model, num_heads * self.d_k, device, dtype) # same as (d_model d_model) but d_k and d_v could change in future
        self.Wk = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.Wv = Linear(d_model, num_heads * self.d_v, device, dtype)

        self.Wo = Linear(d_model, num_heads * self.d_v, device, dtype)

        self.attention = DpAttention()
        
        self.heads = num_heads
        self.d_model = d_model

        self.w_rope = w_rope
        if w_rope : 
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)



    def forward(self, x, token_position = None):
        Q: Float[torch.Tensor, "... sequence_length hdk"]= self.Wq(x)
        K: Float[torch.Tensor, "... sequence_length hdk"] = self.Wk(x)
        V: Float[torch.Tensor, "... sequence_length hdv"] = self.Wv(x)


        Q = rearrange(Q, "... sequence_length (heads dk)-> ... heads sequence_length dk", heads = self.heads) # @inspect Q
        K = rearrange(K, "... sequence_length (heads dk)-> ... heads sequence_length dk", heads = self.heads)
        V = rearrange(V, "... sequence_length (heads dv)-> ... heads sequence_length dv", heads = self.heads)

        sequence_length = x.shape[-2]

        if self.w_rope: 
            if token_position is None :
                token_position = torch.arange(sequence_length, device = x.device , dtype = torch.int)
            Q = self.rope(Q, token_position)
            K = self.rope(K, token_position)


        causal_mask = torch.tril(torch.ones((sequence_length,sequence_length), device = x.device, dtype = torch.bool))
        mh: Float [torch.Tensor, "... heads sequence_length dv"] = self.attention(Q, K, V, causal_mask)
        mh = rearrange(mh, "... heads sequence_length dv -> ... sequence_length (heads dv)")

        mhsa = self.Wo(mh)
        return mhsa