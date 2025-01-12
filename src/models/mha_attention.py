import torch.nn as nn 
import torch
from typing import Tuple
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads:int, d_model:int, dropout_ratio: float=0.1):

        super(MultiHeadAttention,self).__init__()

        remainder = d_model % num_heads
        print(f"Remainder is: {remainder}")  # Debug print

        if remainder != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
   

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.scale_dim = math.sqrt(self.head_dim)
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.wo = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                v: torch.Tensor, mask:torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        bs = q.size()[0]

        # Linear projection and reshape to Multi Head attention
        Q = self.wq(q).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        K = self.wk(k).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = self.wv(v).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1))/self.scale_dim

        # apply mask if provided 
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output  = self.wo(context)
        return output, attention_weights




        





        
