import torch.nn as nn 

class ProjectionLayer(nn.Module):

    def __init__(self,d_model,vocab_size):
        self.projection_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (bs, seq_len, d_model) -> (bs, seq_len, vocab_size)
        return self.projection_layer(x)

