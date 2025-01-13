from .encoder import Encoder
from .decoder import Decoder
from .embeddings import InputEmbedding, PositionalEncoding
from .projection import Projection
import torch.nn as nn
import torch


class Transformer(nn.Module):

    def __init__(self, 
                 encoder: Encoder, decoder: Decoder, 
                 src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, 
                 src_pos_embedding:PositionalEncoding, tgt_pos_embedding: PositionalEncoding,
                 projection:Projection):

        self.encoder = encoder 
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embeddings = tgt_embedding
        self.src_pos_embeddings = src_pos_embedding
        self.tgt_pos_embeddings = tgt_pos_embedding
        self.projection = projection

    def encode(self, x: torch.Tensor, src_mask: torch.Tensor):
        #(bs, seq_len) -> (bs, seq_len, d_model)
        x = self.src_embedding(x)
        x = self.src_pos_embeddings(x)
        return self.encoder(x, src_mask)


    def decode(self, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor):
        # (bs, seq_len)
        tgt_embed = self.tgt_embeddings(tgt)
        tgt_pos_embed = self.tgt_pos_embeddings(tgt_embed)
        return self.decoder(tgt_pos_embed,encoder_output, tgt_mask, src_mask)
        

    def project(self, x):
        return self.projection(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6,
                      h: int = 8, d_ff: int=2048, dropout: float= 0.01) -> Transformer:
    

    src_embeddings = InputEmbedding(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embeddings = InputEmbedding(d_model=d_model, vocab_size=tgt_vocab_size)

    src_pos_embeddings = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos_embeddings = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)

    encoder = Encoder(num_layer=N, num_heads=h, d_model=d_model,d_ff=d_ff, dropout=dropout)
    decoder = Decoder(num_layers=N, num_heads=h, d_ff=d_ff, d_model=d_model, dropout=dropout)

    projection_layer = Projection(d_model=d_model, vocab_size=tgt_vocab_size)

    return Transformer(encoder=encoder,
                decoder=decoder,
                src_embedding=src_embeddings,
                tgt_embedding=tgt_embeddings,
                src_pos_embedding=src_pos_embeddings,
                tgt_pos_embedding=tgt_pos_embeddings,
                projection=projection_layer
                )


