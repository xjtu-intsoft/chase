import abc
from typing import Optional

import torch
import numpy as np
import math

from duorat.utils import registry


def maybe_mask(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> None:
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
            for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])
        ), "Attention mask shape {} should be broadcastable with attention shape {}".format(
            attn_mask.shape, attn.shape
        )

        attn.data.masked_fill_(attn_mask, -float("inf"))


class Pointer(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


@registry.register("pointer", "Bahdanau")
class BahdanauPointer(Pointer):
    def __init__(self, query_size: int, key_size: int, proj_size: int) -> None:
        super().__init__()
        self.compute_scores = torch.nn.Sequential(
            torch.nn.Linear(query_size + key_size, proj_size),
            torch.nn.Tanh(),
            torch.nn.Linear(proj_size, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query shape: batch x seq_len x query_size
        # keys shape: batch x mem_len x key_size

        # query_expanded shape: batch x num keys x query_size
        query_expanded = query.unsqueeze(2).expand(-1, -1, keys.shape[1], -1)
        keys_expanded = keys.unsqueeze(1).expand(-1, query.shape[1], -1, -1)

        # scores shape: batch x num keys x 1
        attn_logits = self.compute_scores(
            # shape: batch x num keys x query_size + key_size
            torch.cat((query_expanded, keys_expanded), dim=3)
        )
        # scores shape: batch x num keys
        attn_logits = attn_logits.squeeze(3)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


@registry.register("pointer", "BahdanauMemEfficient")
class BahdanauPointerMemEfficient(Pointer):
    def __init__(self, query_size: int, key_size: int, proj_size: int) -> None:
        super().__init__()
        self.query_linear = torch.nn.Linear(query_size, proj_size, bias=False)
        self.key_linear = torch.nn.Linear(key_size, proj_size)

        # Correct weight initialization assuming query_size ~= key_size
        with torch.no_grad():
            self.query_linear.weight /= math.sqrt(2)
            self.key_linear.weight /= math.sqrt(2)
            self.key_linear.bias /= math.sqrt(2)

        self.tanh = torch.nn.Tanh()
        self.proj_linear = torch.nn.Linear(proj_size, 1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query shape: batch x seq_len x query_size
        # keys shape: batch x mem_len x key_size

        h_query = self.query_linear(query)  # batch_size x seq_len x proj_size
        h_keys = self.key_linear(keys)  # batch_size x mem_len x proj_size

        h = h_keys.unsqueeze(1) + h_query.unsqueeze(
            2
        )  # batch_size x seq_len x mem_len x proj_size
        h = self.tanh(h)
        attn_logits = self.proj_linear(h)  # batch_size x seq_len x mem_len x 1

        # scores shape: batch x seq_len x mem_len
        attn_logits = attn_logits.squeeze(3)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


@registry.register("pointer", "ScaledDotProduct")
class ScaledDotProductPointer(Pointer):
    def __init__(self, query_size: int, key_size: int) -> None:
        super().__init__()
        self.query_proj = torch.nn.Linear(query_size, key_size)
        self.temp = np.power(key_size, 0.5)
        self.key_size = key_size

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query shape: batch x seq_len x query_size
        # keys shape: batch x mem_len x key_size

        _, mem_len, key_size = keys.shape
        batch_size, seq_len, query_size = query.shape

        # proj_query shape: batch x seq_len x key_size x 1
        proj_query = self.query_proj(query).unsqueeze(-1)
        # reshaped_query shape: (batch*seq_len) x key_size x 1
        reshaped_query = proj_query.reshape(-1, key_size, 1)

        # expanded_keys shape: batch x seq_len x mem_len x key_size
        expanded_keys = keys.unsqueeze(1).repeat((1, seq_len, 1, 1))
        # reshaped_keys shape: (batch*seq_len) x mem_len x key_size
        reshaped_keys = expanded_keys.reshape(-1, mem_len, key_size)

        # attn_logits shape: batch x num keys
        attn_logits = torch.bmm(reshaped_keys, reshaped_query).squeeze(2) / self.temp
        reshaped_attn_logits = attn_logits.reshape(batch_size, seq_len, mem_len)
        maybe_mask(reshaped_attn_logits, attn_mask)
        return reshaped_attn_logits
