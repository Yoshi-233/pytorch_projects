import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from d2l import torch as d2l
import random
import numpy as np


def attention_mask(valid_lens, max_len, num_heads, type="bool"):
    assert type in ["bool", "float"]
    masks = torch.arange(max_len)[None, :] < valid_lens.repeat_interleave(max_len)[:, None]

    if type == "bool":
        return (~masks).repeat_interleave(num_heads, dim=0).reshape(-1, max_len, max_len)
    else:
        masks = torch.zeros_like(masks).masked_fill(~masks, float('-inf'))
        return (~masks).repeat_interleave(num_heads, dim=0).reshape(-1, max_len, max_len)


def key_mask(valid_lens, max_len):
    masks = torch.ones(valid_lens.shape[0], max_len, dtype=torch.bool)

    for i, len in enumerate(valid_lens):
        masks[i, :len] = False

    return masks

def square_subsequent_mask(batch_size, max_len, num_heads):
        return nn.Transformer.generate_square_subsequent_mask(max_len).repeat(batch_size * num_heads, 1, 1)

if __name__ == '__main__':
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self_atten_mask = attention_mask(torch.tensor([2, 3]), 5, 2, type="bool")
        print(self_atten_mask)
        key_padding_mask = key_mask(torch.tensor([2, 3]), 5)
        print(key_padding_mask)
        square_mask = square_subsequent_mask(2, 5, 2)
        print(square_mask)
