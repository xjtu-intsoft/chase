from typing import Tuple, Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F


class RelationAwareMultiheadAttention(nn.Module):
    """Relation-Aware Multi-Headed Attention (RAMHA)."""

    def __init__(
        self,
        embed_dim: int,
        k_embed_dim: int,
        v_embed_dim: int,
        num_heads: int,
        attention_dropout: float,
    ) -> None:
        super(RelationAwareMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.k_embed_dim = k_embed_dim
        self.v_embed_dim = v_embed_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5

        self.q_in_proj = nn.Linear(embed_dim, embed_dim)
        self.k_in_proj = nn.Linear(k_embed_dim, embed_dim)
        self.v_in_proj = nn.Linear(v_embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _mask_attention(
        self, attn_weights: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply additive attention mask, broadcasting over attention heads."""

        if attention_mask is not None:
            batch_size, num_heads, seq_b_len, seq_a_len = attn_weights.shape
            assert num_heads == self.num_heads

            assert attention_mask.shape == (batch_size, seq_b_len, seq_a_len)
            # assert attention_mask.dtype == attn_weights.dtype

            return attn_weights + attention_mask.unsqueeze(1)
        else:
            return attn_weights

    def _mask_key_paddings(
        self, attn_weights: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mask attention weights corresponding to <pad> tokens."""

        if key_padding_mask is not None:
            batch_size, num_heads, seq_b_len, seq_a_len = attn_weights.shape
            assert num_heads == self.num_heads

            assert key_padding_mask.shape == (batch_size, seq_a_len)
            # assert key_padding_mask.dtype == torch.bool

            return torch.masked_fill(
                attn_weights,
                mask=key_padding_mask.unsqueeze(1).unsqueeze(2),
                value=-float("inf"),
            )
        else:
            return attn_weights

    def _attn_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        relations_k: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_b_len, _ = query.shape
        _batch_size, seq_a_len, _ = key.shape
        assert _batch_size == batch_size

        q = self._reshape(self.q_in_proj(query) * self.scaling)
        assert q.shape == (batch_size, self.num_heads, seq_b_len, self.head_dim)

        k = self._reshape(self.k_in_proj(key))
        assert k.shape == (batch_size, self.num_heads, seq_a_len, self.head_dim)

        k_t = k.transpose(2, 3)
        assert k_t.shape == (batch_size, self.num_heads, self.head_dim, seq_a_len)

        attn_weights = torch.matmul(q, k_t)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        if relations_k is not None:
            q_t = q.transpose(1, 2)
            assert q_t.shape == (batch_size, seq_b_len, self.num_heads, self.head_dim)

            relations_k_t = relations_k.transpose(2, 3)
            assert relations_k_t.shape == (
                batch_size,
                seq_b_len,
                self.head_dim,
                seq_a_len,
            )

            attn_weights += torch.matmul(q_t, relations_k_t).transpose(1, 2)
            assert attn_weights.shape == (
                batch_size,
                self.num_heads,
                seq_b_len,
                seq_a_len,
            )

        return attn_weights

    def _attn(
        self,
        attn_weights: torch.Tensor,
        value: torch.Tensor,
        relations_v: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Calculate attention output."""

        batch_size, num_heads, seq_b_len, seq_a_len = attn_weights.shape
        assert num_heads == self.num_heads

        v = self._reshape(self.v_in_proj(value))
        assert v.shape == (batch_size, self.num_heads, seq_a_len, self.head_dim)

        attn = torch.matmul(attn_weights, v).transpose(1, 2)
        assert attn.shape == (batch_size, seq_b_len, self.num_heads, self.head_dim)

        if relations_v is not None:
            attn_weights_t = attn_weights.transpose(1, 2)
            assert attn_weights_t.shape == (
                batch_size,
                seq_b_len,
                self.num_heads,
                seq_a_len,
            )

            assert relations_v.shape == (
                batch_size,
                seq_b_len,
                seq_a_len,
                self.head_dim,
            )
            attn += torch.matmul(attn_weights_t, relations_v)
            assert attn.shape == (batch_size, seq_b_len, self.num_heads, self.head_dim)

        attn = attn.reshape(batch_size, seq_b_len, self.embed_dim)

        attn = self.out_proj(attn)
        assert attn.shape == (batch_size, seq_b_len, self.embed_dim)

        return attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for relation-aware multi-headed attention.

        Input shapes:
            - query:            (batch_size, seq_b_len, embed_dim)
            - key:              (batch_size, seq_a_len, k_embed_dim)
            - value:            (batch_size, seq_a_len, v_embed_dim)
            - relations_k:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - relations_v:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - attention_mask:   (batch_size, seq_b_len, seq_a_len), optional
            - key_padding_mask: (batch_size, seq_a_len), optional

        Output shapes:
            - attn:             (batch_size, seq_b_len, embed_dim)
            - attn_weights:     (batch_size, seq_b_len, seq_a_len)
        """

        batch_size, seq_b_len, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        _batch_size, seq_a_len, k_embed_dim = key.shape
        assert _batch_size == batch_size
        assert k_embed_dim == self.k_embed_dim

        _batch_size, _seq_a_len, v_embed_dim = value.shape
        assert _batch_size == batch_size
        assert _seq_a_len == seq_a_len
        assert v_embed_dim == self.v_embed_dim

        attn_weights = self._attn_weights(query, key, relations_k)

        attn_weights = self._mask_attention(attn_weights, attention_mask)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        attn_weights = self._mask_key_paddings(attn_weights, key_padding_mask)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        attn_weights = F.softmax(attn_weights, dim=3)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        attn = self._attn(attn_weights, value, relations_v)
        assert attn.shape == (batch_size, seq_b_len, self.embed_dim)

        # average attention weights over heads
        attn_weights = attn_weights.sum(dim=1, keepdim=False) / self.num_heads
        assert attn_weights.shape == (batch_size, seq_b_len, seq_a_len)

        return attn, attn_weights


def _residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    dropout: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    shape = x.shape
    dtype = x.dtype
    device = x.device
    assert residual.shape == shape
    # assert residual.dtype == dtype
    assert residual.device == device
    x = residual + dropout(x)
    x = norm(x)
    assert x.shape == shape
    # assert x.dtype == dtype
    assert x.device == device
    return x


class DressedRelationAwareMultiheadAttention(nn.Module):
    """Relation-Aware Multi-Headed Attention (RAMHA) with residual connection and layer norm."""

    def __init__(
        self,
        embed_dim: int,
        k_embed_dim: int,
        v_embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super(DressedRelationAwareMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.self_attn = RelationAwareMultiheadAttention(
            embed_dim=embed_dim,
            k_embed_dim=k_embed_dim,
            v_embed_dim=v_embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for relation-aware multi-headed attention
        with residual connection and layer norms.

        Input shapes:
            - query:            (batch_size, seq_b_len, embed_dim)
            - key:              (batch_size, seq_a_len, k_embed_dim)
            - value:            (batch_size, seq_a_len, v_embed_dim)
            - relations_k:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - relations_v:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - attention_mask:   (batch_size, seq_b_len, seq_a_len), optional
            - key_padding_mask: (batch_size, seq_a_len), optional

        Output shapes:
            - y:                (batch_size, seq_b_len, embed_dim)
        """

        batch_size, seq_len, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        z, _ = self.self_attn(
            query=query,
            key=key,
            value=value,
            relations_k=relations_k,
            relations_v=relations_v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        y = _residual(
            z,
            query,
            lambda z_: F.dropout(z_, p=self.dropout, training=self.training),
            self.norm,
        )

        return y


class TransformerMLP(nn.Module):
    """Transformer MLP Layer."""

    def __init__(
        self, embed_dim: int, ffn_dim: int, dropout: float, relu_dropout: float
    ) -> None:
        super(TransformerMLP, self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transformer MLP layer.

        Input shapes:
            - y:                (batch_size, seq_len, embed_dim)

        Output shapes:
            - y:                (batch_size, seq_len, embed_dim)
        """

        batch_size, seq_len, embed_dim = y.shape
        assert embed_dim == self.embed_dim

        residual = y
        y = F.relu(self.fc1(y))
        assert y.shape == (batch_size, seq_len, self.ffn_dim)
        y = F.dropout(y, p=self.relu_dropout, training=self.training)
        y = self.fc2(y)
        y = _residual(
            y,
            residual,
            lambda y_: F.dropout(y_, p=self.dropout, training=self.training),
            self.norm,
        )

        return y


class RATLayer(nn.Module):
    """Relation-Aware Transformer Layer Block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_dropout: float,
        relu_dropout: float,
    ) -> None:
        super(RATLayer, self).__init__()
        self.self_attn = DressedRelationAwareMultiheadAttention(
            embed_dim=embed_dim,
            k_embed_dim=embed_dim,
            v_embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.mlp = TransformerMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            relu_dropout=relu_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the transformer layer.

        Input shapes:
            - x:                (batch_size, seq_len, embed_dim)
            - relations_k:      (batch_size, seq_len, seq_len, head_dim), optional
            - relations_v:      (batch_size, seq_len, seq_len, head_dim), optional
            - attention_mask:   (batch_size, seq_len, seq_len), optional
            - key_padding_mask: (batch_size, seq_len), optional

        Output shapes:
            - x:                (batch_size, seq_len, embed_dim)
        """
        return self.mlp(
            self.self_attn(
                query=x,
                key=x,
                value=x,
                relations_k=relations_k,
                relations_v=relations_v,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
        )


class RATLayerWithMemory(nn.Module):
    """Relation-Aware Transformer Layer Block with Memory."""

    def __init__(
        self,
        embed_dim: int,
        mem_embed_dim: int,
        dropout: float,
        num_heads: int,
        attention_dropout: float,
        relu_dropout: float,
        ffn_dim: int,
    ) -> None:
        super(RATLayerWithMemory, self).__init__()
        self.self_attn = DressedRelationAwareMultiheadAttention(
            embed_dim=embed_dim,
            k_embed_dim=embed_dim,
            v_embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.memory_attn = DressedRelationAwareMultiheadAttention(
            embed_dim=embed_dim,
            k_embed_dim=mem_embed_dim,
            v_embed_dim=mem_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.mlp = TransformerMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            relu_dropout=relu_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        memory_relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        memory_relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        memory_attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the transformer layer with memory.

        Input shapes:
            - x:                       (batch_size, seq_len, embed_dim)
            - memory:                  (batch_size, mem_len, mem_embed_dim)
            - relations_k:             (batch_size, seq_len, seq_len, head_dim), optional
            - memory_relations_k:      (batch_size, seq_len, mem_len, head_dim), optional
            - relations_v:             (batch_size, seq_len, seq_len, head_dim), optional
            - memory_relations_v:      (batch_size, seq_len, mem_len, head_dim), optional
            - attention_mask:          (batch_size, seq_len, seq_len), optional
            - memory_attention_mask:   (batch_size, seq_len, mem_len), optional
            - key_padding_mask:        (batch_size, seq_len), optional
            - memory_key_padding_mask: (batch_size, mem_len), optional

        Output shapes:
            - x:                       (batch_size, seq_len, embed_dim)
        """

        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            relations_k=relations_k,
            relations_v=relations_v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.memory_attn(
            query=x,
            key=memory,
            value=memory,
            relations_k=memory_relations_k,
            relations_v=memory_relations_v,
            attention_mask=memory_attention_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        x = self.mlp(x)

        return x


class RAT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_embed_x: int,
        pad_x_index: int,
        num_embed_r_k: int,
        pad_r_k_index: int,
        num_embed_r_v: int,
        pad_r_v_index: int,
        dropout: float,
        attention_dropout: float,
        relu_dropout: float,
    ):
        super(RAT, self).__init__()

        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim
        self.head_dim = head_dim

        self.num_embed_x = num_embed_x
        self.pad_x_index = pad_x_index
        self.embed_x = nn.Embedding(num_embed_x, embed_dim, padding_idx=pad_x_index)
        self.embed_x_scale = embed_dim ** 0.5

        self.num_embed_r_k = num_embed_r_k
        self.pad_r_k_index = pad_r_k_index
        self.embed_r_k = nn.Embedding(
            num_embed_r_k, head_dim, padding_idx=pad_r_k_index
        )
        self.num_embed_r_v = num_embed_r_v
        self.pad_r_v_index = pad_r_v_index
        self.embed_r_v = nn.Embedding(
            num_embed_r_v, head_dim, padding_idx=pad_r_v_index
        )
        self.embed_r_scale = head_dim ** 0.5

        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                RATLayer(
                    embed_dim=embed_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = nn.Linear(embed_dim, num_embed_x)

    def _forward(
        self,
        x_tokens: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (batch_size, seq_len) = x_tokens.shape
        assert x_tokens.dtype == torch.long
        assert relation_tokens_k.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_k.dtype == torch.long
        assert relation_tokens_v.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_v.dtype == torch.long

        key_padding_mask = x_tokens.eq(self.pad_x_index)
        assert key_padding_mask.shape == (batch_size, seq_len)
        # assert key_padding_mask.dtype == torch.bool

        x = self.embed_x_scale * self.embed_x(x_tokens)
        assert x.shape == (batch_size, seq_len, self.embed_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len, seq_len)
            # assert attention_mask.dtype == x.dtype

        relations_k = self.embed_r_scale * self.embed_r_k(relation_tokens_k)
        assert relations_k.shape == (batch_size, seq_len, seq_len, self.head_dim)

        relations_v = self.embed_r_scale * self.embed_r_v(relation_tokens_v)
        assert relations_v.shape == (batch_size, seq_len, seq_len, self.head_dim)

        for layer in self.layers:
            x = layer(
                x=x,
                relations_k=relations_k,
                relations_v=relations_v,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )

        assert x.shape == (batch_size, seq_len, self.embed_dim)

        return x

    def forward(
        self,
        x_tokens: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the relation-aware transformer.

        Input shapes:
            - x_tokens:          (batch_size, seq_len)
            - relation_tokens_k: (batch_size, seq_len, seq_len)
            - relation_tokens_v: (batch_size, seq_len, seq_len)
            - attention_mask:    (batch_size, seq_len, seq_len), optional

        Output shapes:
            - x_token_logits:    (batch_size, seq_len, num_embed_x)
        """

        (batch_size, seq_len) = x_tokens.shape
        x = self._forward(
            x_tokens=x_tokens,
            relation_tokens_k=relation_tokens_k,
            relation_tokens_v=relation_tokens_v,
            attention_mask=attention_mask,
        )
        x_token_logits = self.proj(x)
        assert x_token_logits.shape == (batch_size, seq_len, self.num_embed_x)

        return x_token_logits


class RATWithMemory(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mem_embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_embed_x: int,
        pad_x_index: int,
        num_embed_r_k: int,
        pad_r_k_index: int,
        num_mem_embed_r_k: int,
        pad_mem_r_k_index: int,
        num_embed_r_v: int,
        pad_r_v_index: int,
        num_mem_embed_r_v: int,
        pad_mem_r_v_index: int,
        dropout: float,
        attention_dropout: float,
        relu_dropout: float,
    ):
        super(RATWithMemory, self).__init__()

        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim
        self.head_dim = head_dim
        self.mem_embed_dim = mem_embed_dim

        self.num_embed_x = num_embed_x
        self.pad_x_index = pad_x_index
        self.embed_x = nn.Embedding(num_embed_x, embed_dim, padding_idx=pad_x_index)
        self.embed_x_scale = embed_dim ** 0.5

        self.num_embed_r_k = num_embed_r_k
        self.pad_r_k_index = pad_r_k_index
        self.embed_r_k = nn.Embedding(
            num_embed_r_k, head_dim, padding_idx=pad_r_k_index
        )
        self.num_mem_embed_r_k = num_mem_embed_r_k
        self.pad_mem_r_k_index = pad_mem_r_k_index
        self.mem_embed_r_k = nn.Embedding(
            num_mem_embed_r_k, head_dim, padding_idx=pad_mem_r_k_index
        )
        self.num_embed_r_v = num_embed_r_v
        self.pad_r_v_index = pad_r_v_index
        self.embed_r_v = nn.Embedding(
            num_embed_r_v, head_dim, padding_idx=pad_r_v_index
        )
        self.num_mem_embed_r_v = num_mem_embed_r_v
        self.pad_mem_r_v_index = pad_mem_r_v_index
        self.mem_embed_r_v = nn.Embedding(
            num_mem_embed_r_v, head_dim, padding_idx=pad_mem_r_v_index
        )
        self.embed_r_scale = head_dim ** 0.5

        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                RATLayerWithMemory(
                    embed_dim=embed_dim,
                    mem_embed_dim=mem_embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = nn.Linear(embed_dim, num_embed_x)

    def _forward(
        self,
        x_tokens: torch.Tensor,
        memory: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        memory_relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        memory_relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        memory_attention_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = x_tokens.shape
        # assert x_tokens.dtype == torch.long
        _batch_size, mem_len, mem_embed_dim = memory.shape
        assert _batch_size == batch_size
        assert mem_embed_dim == self.mem_embed_dim
        assert relation_tokens_k.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_k.dtype == torch.long
        assert memory_relation_tokens_k.shape == (batch_size, seq_len, mem_len)
        # assert memory_relation_tokens_k.dtype == torch.long
        assert relation_tokens_v.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_v.dtype == torch.long
        assert memory_relation_tokens_v.shape == (batch_size, seq_len, mem_len)
        # assert memory_relation_tokens_v.dtype == torch.long

        key_padding_mask = x_tokens.eq(self.pad_x_index)
        assert key_padding_mask.shape == (batch_size, seq_len)
        # assert key_padding_mask.dtype == torch.bool

        if memory_key_padding_mask is not None:
            assert memory_key_padding_mask.shape == (batch_size, mem_len)
            # assert memory_key_padding_mask.dtype == torch.bool

        x = self.embed_x_scale * self.embed_x(x_tokens)
        assert x.shape == (batch_size, seq_len, self.embed_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len, seq_len)
            # assert attention_mask.dtype == x.dtype

        if memory_key_padding_mask is not None:
            assert memory_attention_mask.shape == (batch_size, seq_len, mem_len)
            # assert memory_attention_mask.dtype == x.dtype

        relations_k = self.embed_r_scale * self.embed_r_k(relation_tokens_k)
        assert relations_k.shape == (batch_size, seq_len, seq_len, self.head_dim)

        memory_relations_k = self.embed_r_scale * self.mem_embed_r_k(
            memory_relation_tokens_k
        )
        assert memory_relations_k.shape == (batch_size, seq_len, mem_len, self.head_dim)

        relations_v = self.embed_r_scale * self.embed_r_v(relation_tokens_v)
        assert relations_v.shape == (batch_size, seq_len, seq_len, self.head_dim)

        memory_relations_v = self.embed_r_scale * self.mem_embed_r_v(
            memory_relation_tokens_v
        )
        assert memory_relations_v.shape == (batch_size, seq_len, mem_len, self.head_dim)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                relations_k=relations_k,
                memory_relations_k=memory_relations_k,
                relations_v=relations_v,
                memory_relations_v=memory_relations_v,
                attention_mask=attention_mask,
                memory_attention_mask=memory_attention_mask,
                key_padding_mask=key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        assert x.shape == (batch_size, seq_len, self.embed_dim)

        return x

    def forward(
        self,
        x_tokens: torch.Tensor,
        memory: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        memory_relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        memory_relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        memory_attention_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the relation-aware transformer with memory.

        Input shapes:
            - x_tokens:                 (batch_size, seq_len)
            - memory:                   (batch_size, mem_len, mem_embed_dim)
            - relation_tokens_k:        (batch_size, seq_len, seq_len)
            - memory_relation_tokens_k: (batch_size, seq_len, mem_len)
            - relation_tokens_v:        (batch_size, seq_len, seq_len)
            - memory_relation_tokens_v: (batch_size, seq_len, mem_len)
            - attention_mask:           (batch_size, seq_len, seq_len), optional
            - memory_attention_mask:    (batch_size, seq_len, mem_len), optional
            - memory_key_padding_mask:  (batch_size, mem_len), optional

        Output shapes:
            - x_token_logits:           (batch_size, seq_len, num_embed_x)
        """

        (batch_size, seq_len) = x_tokens.shape
        x = self._forward(
            x_tokens=x_tokens,
            memory=memory,
            relation_tokens_k=relation_tokens_k,
            memory_relation_tokens_k=memory_relation_tokens_k,
            relation_tokens_v=relation_tokens_v,
            memory_relation_tokens_v=memory_relation_tokens_v,
            attention_mask=attention_mask,
            memory_attention_mask=memory_attention_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        x_token_logits = self.proj(x)
        assert x_token_logits.shape == (batch_size, seq_len, self.num_embed_x)

        return x_token_logits
