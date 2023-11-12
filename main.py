import torch
import torch.nn as nn


class self_attention(nn.Module):
    # embed_size => 輸入維度
    # heads => 幾層head
    def __init__(self, embed_size, heads):
        # define heads and input 維度
        super(self_attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size  # heads

        assert (self.head_dim * heads == embed_size), "不能被整除 !!"

        # define q k v

        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forword(self, value, key, query, mask):
        num = query.shape[0]
        values_len = value.shape[1]
        key_len = key[1]
        query_len = query[1]

        # 轉置

        values = value.reshape(num, values_len, self.heads, self.head_dim)
        keys = key.reshape(num, key_len, self.heads, self.head_dim)
        queries = query.reshape(num, query_len, self.heads, self.head_dim)

        # q(1-n) *k(1-n)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("1e20"))

        # softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # (q*k)(1-n) * v(1-n)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            num, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out
