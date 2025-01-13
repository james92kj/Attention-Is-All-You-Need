from .feed_forward import FeedForwardLayer
from .attention import MultiHeadAttention
from typing import Optional
import torch.nn as nn 
import torch 


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: int = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward_layer = FeedForwardLayer(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None): # the input to the forward function is embeddings 

        output, _ = self.mha_attention(x,x,x, mask)
        x = self.norm1(x + self.dropout(output))
        x = self.norm2(x + self.dropout(self.feed_forward_layer(x)))
        return x
    


class Encoder(nn.Module):

    def __init__(self, num_layer: int, d_model: int, d_ff: int, num_heads: int, dropout: int = 0.1):
        super(Encoder, self).__init__()

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layer)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.encoder_stack:
            x = layer(x)
        return self.norm(x) 



        

