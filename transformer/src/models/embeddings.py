import torch.nn as nn 
import torch
import math

class InputEmbedding(nn.Module):

    def __init__(self,d_model : int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model 
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        '''
            x -> (bs, seq_len)
            output -> (bs, seq_len, embedding_dim)
        '''
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(seq_len, d_model)
        # generate position indices 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # create a frequency divider 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / 8))
        # Apply sin to even indices 
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices 
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       x =  x + self.pe[:, :x.shape[1], :]
       return self.dropout(x)