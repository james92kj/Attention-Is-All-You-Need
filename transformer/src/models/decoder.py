from transformer.src.models.attention import MultiHeadAttention
from transformer.src.models.feed_forward import FeedForward
import torch.nn as nn 
import torch 


class DecoderBlock(nn.Module):

    def __init__(self,num_heads: int, d_model:int,  dropout: float, d_ff: int):
        super(DecoderBlock, self).__init__()

        self.self_attention_block = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.cross_attention_block = MultiHeadAttention(num_heads=num_heads,d_model=d_model, dropout=dropout)
        self.feed_forward_block = FeedForward(d_model=d_model,d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor):
        
        # masked multi head attention
        mask_multi_head_attention,_ = self.self_attention_block(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(mask_multi_head_attention))

        # multi head attention
        multi_head_attention,_ = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(multi_head_attention))

        # feed forward layer 
        feed_forward_layer_output = self.feed_forward_block(x)
        x = self.norm3(x + self.dropout(feed_forward_layer_output))

        return x

class Decoder(nn.Module):

    def __init__(self, num_layers: int, num_heads: int, d_model:int,  dropout: float, d_ff: int):
        super(Decoder, self).__init__()
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(num_heads, d_model,  dropout, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor):

        '''
            tgt_mask: Prohibits to look at all the future tokens
            src_mask: says which token are valid or not 
        '''

        for layer in self.decoder_blocks:
            x = layer(x, encoder_output, tgt_mask, src_mask)
    
        return self.norm(x)
    
        