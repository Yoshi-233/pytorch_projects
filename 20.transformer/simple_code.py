import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from d2l import torch as d2l


def attention_mask(src_lens, tgt_lens, max_len, num_heads, type="bool"):
        assert type in ["bool", "float"]
        assert len(src_lens) == len(tgt_lens)
        masks = []
        for src_len, tgt_len in zip(src_lens, tgt_lens):
                src_mask = F.pad(torch.ones(src_len, device=src_lens.device), (0, max_len - src_len))
                tgt_mask = F.pad(torch.ones(tgt_len, device=tgt_lens.device), (0, max_len - tgt_len))
                masks.append(src_mask[:, None] @ tgt_mask[None, :])

        masks = torch.cat(masks, dim=0).reshape(len(src_lens), max_len, max_len)
        bool_masks = (1 - masks).bool()

        if type == "bool":
                return bool_masks.repeat_interleave(num_heads, dim=0)
        else:
                masks = torch.zeros_like(masks).masked_fill(bool_masks, float('-inf'))
                return masks.repeat_interleave(num_heads, dim=0)


if __name__ == '__main__':
        model = nn.MultiheadAttention(embed_dim=8, num_heads=2)
        X = torch.rand(4, 2, 8)
        query = torch.rand(4, 2, 8)
        key = value = torch.rand(4, 2, 8)

        max_len = X.size(0)
        src_lens = torch.tensor([2, 3])
        tgt_lens = torch.tensor([1, 4])
        attn_mask = attention_mask(src_lens, tgt_lens, max_len, 2, type = "float")
        # print(attn_mask)
        key_padding_mask = ~torch.tensor([
                [True, True, True, False],
                [True, True, False, False]
        ])

        output, attn_weights = model(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                     average_attn_weights=False)
        print(output.shape)
        print(attn_weights.shape)
        print(attn_weights)
