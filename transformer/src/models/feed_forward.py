import torch.nn as nn
import torch


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):

        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    