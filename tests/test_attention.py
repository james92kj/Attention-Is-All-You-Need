import pytest
from ..src.models.mha_attention import MultiHeadAttention
import torch 

def test_multi_head_attention_shape():

    num_heads = 8
    d_model = 512
    seq_len = 10
    bs = 32

    mha_attention = MultiHeadAttention(
        num_heads=num_heads,
        d_model=d_model
        )

    q = torch.randn(bs,seq_len,d_model)
    k = torch.randn(bs,seq_len,d_model)
    v = torch.randn(bs,seq_len,d_model)

    output, attention_weights = mha_attention(q,k,v)
    assert output.shape == (bs,seq_len,d_model),'Output shape mismatch'
    assert attention_weights.shape == (bs,num_heads,seq_len,seq_len),'attention shape mismatch'

def test_multi_head_compatibility():

    num_heads = 9
    d_model = 512

    # Let's print the remainder first
    print(f"Remainder: {d_model % num_heads}")
   

    with pytest.raises(ValueError):
        MultiHeadAttention(num_heads=num_heads,d_model=d_model)

