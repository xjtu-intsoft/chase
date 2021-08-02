import pytest
import torch

from duorat.models.rat import (
    RelationAwareMultiheadAttention,
    DressedRelationAwareMultiheadAttention,
    TransformerMLP,
    RATLayer,
    RATLayerWithMemory,
    RAT,
    RATWithMemory,
)


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request) -> int:
    return request.param


@pytest.fixture(params=[1, 3])
def num_layers(request) -> int:
    return request.param


@pytest.fixture(params=[1, 43])
def num_embed_x(request) -> int:
    return request.param


@pytest.fixture(params=[0])
def pad_x_index(request) -> int:
    return request.param


@pytest.fixture(params=[1, 5])
def num_embed_r_k(request) -> int:
    return request.param


@pytest.fixture(params=[0])
def pad_r_k_index(request) -> int:
    return request.param


@pytest.fixture(params=[1, 5])
def num_mem_embed_r_k(request) -> int:
    return request.param


@pytest.fixture(params=[0])
def pad_mem_r_k_index(request) -> int:
    return request.param


@pytest.fixture(params=[1, 7])
def num_embed_r_v(request) -> int:
    return request.param


@pytest.fixture(params=[0])
def pad_r_v_index(request) -> int:
    return request.param


@pytest.fixture(params=[1, 7])
def num_mem_embed_r_v(request) -> int:
    return request.param


@pytest.fixture(params=[0])
def pad_mem_r_v_index(request) -> int:
    return request.param


@pytest.fixture(params=[1, 13])
def batch_size(request) -> int:
    return request.param


@pytest.fixture(params=[1, 7])
def num_heads(request) -> int:
    return request.param


@pytest.fixture(params=[1, 11])
def head_dim(request) -> int:
    return request.param


@pytest.fixture(params=[1, 63])
def seq_len(request) -> int:
    return request.param


@pytest.fixture(params=[1, 17])
def seq_a_len(request) -> int:
    return request.param


@pytest.fixture(params=[1, 11])
def seq_b_len(request) -> int:
    return request.param


@pytest.fixture(params=[1, 13])
def mem_len(request) -> int:
    return request.param


@pytest.fixture
def embed_dim(num_heads: int, head_dim: int):
    return num_heads * head_dim


@pytest.fixture(params=[1, 13])
def k_embed_dim(request) -> int:
    return request.param


@pytest.fixture(params=[1, 7])
def v_embed_dim(request) -> int:
    return request.param


@pytest.fixture(params=[1, 11])
def mem_embed_dim(request) -> int:
    return request.param


@pytest.fixture(params=[1, 23])
def ffn_dim(request) -> int:
    return request.param


@pytest.fixture(params=[0.25])
def dropout(request) -> float:
    return request.param


@pytest.fixture(params=[0.25])
def attention_dropout(request) -> float:
    return request.param


@pytest.fixture(params=[0.25])
def relu_dropout(request) -> float:
    return request.param


@pytest.fixture
def query(
    batch_size: int, seq_b_len: int, embed_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_b_len, embed_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def key(
    batch_size: int, seq_a_len: int, k_embed_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_a_len, k_embed_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def value(
    batch_size: int, seq_a_len: int, v_embed_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_a_len, v_embed_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def x_tokens(
    batch_size: int, seq_len: int, num_embed_x: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=num_embed_x,
        size=[batch_size, seq_len],
        dtype=torch.long,
        device=device,
    )


@pytest.fixture
def x(
    batch_size: int, seq_len: int, embed_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_len, embed_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def y(
    batch_size: int, seq_len: int, embed_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_len, embed_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def memory(
    batch_size: int, mem_len: int, mem_embed_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, mem_len, mem_embed_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def relation_tokens_k(
    batch_size: int, seq_len: int, num_embed_r_k: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=num_embed_r_k,
        size=[batch_size, seq_len, seq_len],
        dtype=torch.long,
        device=device,
    )


@pytest.fixture
def relations_k(
    batch_size: int, seq_b_len: int, seq_a_len: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_b_len, seq_a_len, head_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def relations_k_(
    batch_size: int, seq_len: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_len, seq_len, head_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def memory_relation_tokens_k(
    batch_size: int,
    seq_len: int,
    mem_len: int,
    num_mem_embed_r_k: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=num_mem_embed_r_k,
        size=[batch_size, seq_len, mem_len],
        dtype=torch.long,
        device=device,
    )


@pytest.fixture
def memory_relations_k(
    batch_size: int, seq_len: int, mem_len: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_len, mem_len, head_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def relation_tokens_v(
    batch_size: int, seq_len: int, num_embed_r_v: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=num_embed_r_v,
        size=[batch_size, seq_len, seq_len],
        dtype=torch.long,
        device=device,
    )


@pytest.fixture
def relations_v(
    batch_size: int, seq_b_len: int, seq_a_len: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_b_len, seq_a_len, head_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def relations_v_(
    batch_size: int, seq_len: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_len, seq_len, head_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def memory_relation_tokens_v(
    batch_size: int,
    seq_len: int,
    mem_len: int,
    num_mem_embed_r_v: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=num_mem_embed_r_v,
        size=[batch_size, seq_len, mem_len],
        dtype=torch.long,
        device=device,
    )


@pytest.fixture
def memory_relations_v(
    batch_size: int, seq_len: int, mem_len: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        [batch_size, seq_len, mem_len, head_dim], dtype=torch.float, device=device
    )


@pytest.fixture
def attention_mask(
    batch_size: int, seq_b_len: int, seq_a_len: int, device: torch.device
) -> torch.Tensor:
    mask = torch.randint(
        low=0,
        high=2,
        size=[batch_size, seq_b_len, seq_a_len],
        dtype=torch.bool,
        device=device,
    )
    return torch.masked_fill(
        torch.zeros(
            [batch_size, seq_b_len, seq_a_len], dtype=torch.float, device=device
        ),
        mask=~mask,
        value=-float("inf"),
    )


@pytest.fixture
def attention_mask_(
    batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    mask = torch.randint(
        low=0,
        high=2,
        size=[batch_size, seq_len, seq_len],
        dtype=torch.bool,
        device=device,
    )
    return torch.masked_fill(
        torch.zeros([batch_size, seq_len, seq_len], dtype=torch.float, device=device),
        mask=~mask,
        value=-float("inf"),
    )


@pytest.fixture
def memory_attention_mask(
    batch_size: int, seq_len: int, mem_len: int, device: torch.device
) -> torch.Tensor:
    mask = torch.randint(
        low=0,
        high=2,
        size=[batch_size, seq_len, mem_len],
        dtype=torch.bool,
        device=device,
    )
    return torch.masked_fill(
        torch.zeros([batch_size, seq_len, mem_len], dtype=torch.float, device=device),
        mask=~mask,
        value=-float("inf"),
    )


@pytest.fixture
def key_padding_mask(
    batch_size: int, seq_a_len: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0, high=2, size=[batch_size, seq_a_len], dtype=torch.bool, device=device
    )


@pytest.fixture
def key_padding_mask_(
    batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0, high=2, size=[batch_size, seq_len], dtype=torch.bool, device=device
    )


@pytest.fixture
def memory_key_padding_mask(
    batch_size: int, mem_len: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0, high=2, size=[batch_size, mem_len], dtype=torch.bool, device=device
    )


def test_ramha_minimal(
    embed_dim: int,
    k_embed_dim: int,
    v_embed_dim: int,
    num_heads: int,
    attention_dropout: float,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relations_k: torch.Tensor,
    relations_v: torch.Tensor,
    attention_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    mha = RelationAwareMultiheadAttention(
        embed_dim=embed_dim,
        k_embed_dim=k_embed_dim,
        v_embed_dim=v_embed_dim,
        num_heads=num_heads,
        attention_dropout=attention_dropout,
    ).to(device=device)
    _ = mha(
        query=query,
        key=key,
        value=value,
        relations_k=relations_k,
        relations_v=relations_v,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )


def test_ramha_batch_consistency(
    embed_dim: int,
    k_embed_dim: int,
    v_embed_dim: int,
    num_heads: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relations_k: torch.Tensor,
    relations_v: torch.Tensor,
    attention_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    mha = RelationAwareMultiheadAttention(
        embed_dim=embed_dim,
        k_embed_dim=k_embed_dim,
        v_embed_dim=v_embed_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
    ).to(device=device)
    attn_batched, attn_weights_batched = mha(
        query=query,
        key=key,
        value=value,
        relations_k=relations_k,
        relations_v=relations_v,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )
    attn_split, attn_weights_split = map(
        lambda t: torch.cat(t, dim=0),
        zip(
            *list(
                mha(
                    query=_q.unsqueeze(0),
                    key=_k.unsqueeze(0),
                    value=_v.unsqueeze(0),
                    relations_k=_relations_k.unsqueeze(0),
                    relations_v=_relations_v.unsqueeze(0),
                    attention_mask=_attention_mask.unsqueeze(0),
                    key_padding_mask=_key_padding_mask.unsqueeze(0),
                )
                for _q, _k, _v, _relations_k, _relations_v, _attention_mask, _key_padding_mask in zip(
                    query,
                    key,
                    value,
                    relations_k,
                    relations_v,
                    attention_mask,
                    key_padding_mask,
                )
            )
        ),
    )
    assert attn_batched.shape == attn_split.shape
    assert attn_weights_batched.shape == attn_weights_split.shape
    assert (
        torch.allclose(attn_batched, attn_split, rtol=1e-3, atol=1e-4, equal_nan=True)
        is True
    )
    assert (
        torch.allclose(
            attn_weights_batched,
            attn_weights_split,
            rtol=1e-3,
            atol=1e-4,
            equal_nan=True,
        )
        is True
    )


def test_dressed_ramha_minimal(
    embed_dim: int,
    k_embed_dim: int,
    v_embed_dim: int,
    num_heads: int,
    dropout: float,
    attention_dropout: float,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relations_k: torch.Tensor,
    relations_v: torch.Tensor,
    attention_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    dressed_mha = DressedRelationAwareMultiheadAttention(
        embed_dim=embed_dim,
        k_embed_dim=k_embed_dim,
        v_embed_dim=v_embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
    ).to(device=device)
    _ = dressed_mha(
        query=query,
        key=key,
        value=value,
        relations_k=relations_k,
        relations_v=relations_v,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )


def test_dressed_ramha_batch_consistency(
    embed_dim: int,
    k_embed_dim: int,
    v_embed_dim: int,
    num_heads: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relations_k: torch.Tensor,
    relations_v: torch.Tensor,
    attention_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    dressed_mha = DressedRelationAwareMultiheadAttention(
        embed_dim=embed_dim,
        k_embed_dim=k_embed_dim,
        v_embed_dim=v_embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        attention_dropout=0.0,
    ).to(device=device)
    y_batched = dressed_mha(
        query=query,
        key=key,
        value=value,
        relations_k=relations_k,
        relations_v=relations_v,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )
    y_split = torch.cat(
        list(
            dressed_mha(
                query=_q.unsqueeze(0),
                key=_k.unsqueeze(0),
                value=_v.unsqueeze(0),
                relations_k=_relations_k.unsqueeze(0),
                relations_v=_relations_v.unsqueeze(0),
                attention_mask=_attention_mask.unsqueeze(0),
                key_padding_mask=_key_padding_mask.unsqueeze(0),
            )
            for _q, _k, _v, _relations_k, _relations_v, _attention_mask, _key_padding_mask in zip(
                query,
                key,
                value,
                relations_k,
                relations_v,
                attention_mask,
                key_padding_mask,
            )
        ),
        dim=0,
    )
    assert y_batched.shape == y_split.shape
    assert (
        torch.allclose(y_batched, y_split, rtol=1e-3, atol=1e-4, equal_nan=True) is True
    )


def test_transformer_mlp_minimal(
    embed_dim: int,
    ffn_dim: int,
    dropout: float,
    relu_dropout: float,
    y: torch.Tensor,
    device: torch.device,
) -> None:
    mlp = TransformerMLP(
        embed_dim=embed_dim, ffn_dim=ffn_dim, dropout=dropout, relu_dropout=relu_dropout
    ).to(device=device)
    _ = mlp(y)


def test_rat_layer_minimal(
    embed_dim: int,
    dropout: float,
    num_heads: int,
    attention_dropout: float,
    relu_dropout: float,
    ffn_dim: int,
    x: torch.Tensor,
    relations_k_: torch.Tensor,
    relations_v_: torch.Tensor,
    attention_mask_: torch.Tensor,
    key_padding_mask_: torch.Tensor,
    device: torch.device,
) -> None:
    layer = RATLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=relu_dropout,
    ).to(device=device)
    _ = layer(
        x=x,
        relations_k=relations_k_,
        relations_v=relations_v_,
        attention_mask=attention_mask_,
        key_padding_mask=key_padding_mask_,
    )


def test_rat_layer_batch_consistency(
    embed_dim: int,
    num_heads: int,
    ffn_dim: int,
    x: torch.Tensor,
    relations_k_: torch.Tensor,
    relations_v_: torch.Tensor,
    attention_mask_: torch.Tensor,
    key_padding_mask_: torch.Tensor,
    device: torch.device,
) -> None:
    layer = RATLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ).to(device=device)
    x_batched = layer(
        x=x,
        relations_k=relations_k_,
        relations_v=relations_v_,
        attention_mask=attention_mask_,
        key_padding_mask=key_padding_mask_,
    )
    x_split = torch.cat(
        list(
            layer(
                x=_x.unsqueeze(0),
                relations_k=_relations_k.unsqueeze(0),
                relations_v=_relations_v.unsqueeze(0),
                attention_mask=_attention_mask.unsqueeze(0),
                key_padding_mask=_key_padding_mask.unsqueeze(0),
            )
            for _x, _relations_k, _relations_v, _attention_mask, _key_padding_mask in zip(
                x, relations_k_, relations_v_, attention_mask_, key_padding_mask_
            )
        ),
        dim=0,
    )
    assert x_batched.shape == x_split.shape
    assert (
        torch.allclose(x_batched, x_split, rtol=1e-3, atol=1e-4, equal_nan=True) is True
    )


def test_rat_layer_with_memory_minimal(
    embed_dim: int,
    mem_embed_dim: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    attention_dropout: float,
    relu_dropout: float,
    x: torch.Tensor,
    memory: torch.Tensor,
    relations_k_: torch.Tensor,
    memory_relations_k: torch.Tensor,
    relations_v_: torch.Tensor,
    memory_relations_v: torch.Tensor,
    attention_mask_: torch.Tensor,
    memory_attention_mask: torch.Tensor,
    key_padding_mask_: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    layer = RATLayerWithMemory(
        embed_dim=embed_dim,
        mem_embed_dim=mem_embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=relu_dropout,
    ).to(device=device)
    _ = layer(
        x=x,
        memory=memory,
        relations_k=relations_k_,
        memory_relations_k=memory_relations_k,
        relations_v=relations_v_,
        memory_relations_v=memory_relations_v,
        attention_mask=attention_mask_,
        memory_attention_mask=memory_attention_mask,
        key_padding_mask=key_padding_mask_,
        memory_key_padding_mask=memory_key_padding_mask,
    )


def test_rat_layer_with_memory_batch_consistency(
    embed_dim: int,
    mem_embed_dim: int,
    num_heads: int,
    ffn_dim: int,
    x: torch.Tensor,
    memory: torch.Tensor,
    relations_k_: torch.Tensor,
    memory_relations_k: torch.Tensor,
    relations_v_: torch.Tensor,
    memory_relations_v: torch.Tensor,
    attention_mask_: torch.Tensor,
    memory_attention_mask: torch.Tensor,
    key_padding_mask_: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    layer = RATLayerWithMemory(
        embed_dim=embed_dim,
        mem_embed_dim=mem_embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ).to(device=device)
    x_batched = layer(
        x=x,
        memory=memory,
        relations_k=relations_k_,
        memory_relations_k=memory_relations_k,
        relations_v=relations_v_,
        memory_relations_v=memory_relations_v,
        attention_mask=attention_mask_,
        memory_attention_mask=memory_attention_mask,
        key_padding_mask=key_padding_mask_,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    x_split = torch.cat(
        list(
            layer(
                x=_x.unsqueeze(0),
                memory=_memory.unsqueeze(0),
                relations_k=_relations_k.unsqueeze(0),
                memory_relations_k=_memory_relations_k.unsqueeze(0),
                relations_v=_relations_v.unsqueeze(0),
                memory_relations_v=_memory_relations_v.unsqueeze(0),
                attention_mask=_attention_mask.unsqueeze(0),
                memory_attention_mask=_memory_attention_mask.unsqueeze(0),
                key_padding_mask=_key_padding_mask.unsqueeze(0),
                memory_key_padding_mask=_memory_key_padding_mask.unsqueeze(0),
            )
            for (
                _x,
                _memory,
                _relations_k,
                _memory_relations_k,
                _relations_v,
                _memory_relations_v,
                _attention_mask,
                _memory_attention_mask,
                _key_padding_mask,
                _memory_key_padding_mask,
            ) in zip(
                x,
                memory,
                relations_k_,
                memory_relations_k,
                relations_v_,
                memory_relations_v,
                attention_mask_,
                memory_attention_mask,
                key_padding_mask_,
                memory_key_padding_mask,
            )
        ),
        dim=0,
    )
    assert x_batched.shape == x_split.shape
    assert (
        torch.allclose(x_batched, x_split, rtol=1e-3, atol=1e-4, equal_nan=True) is True
    )


def test_unirat_minimal(
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
    x_tokens: torch.Tensor,
    relation_tokens_k: torch.Tensor,
    relation_tokens_v: torch.Tensor,
    attention_mask_: torch.Tensor,
    device: torch.device,
) -> None:
    unirat = RAT(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_embed_x=num_embed_x,
        pad_x_index=pad_x_index,
        num_embed_r_k=num_embed_r_k,
        pad_r_k_index=pad_r_k_index,
        num_embed_r_v=num_embed_r_v,
        pad_r_v_index=pad_r_v_index,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=relu_dropout,
    ).to(device=device)
    _ = unirat(
        x_tokens=x_tokens,
        relation_tokens_k=relation_tokens_k,
        relation_tokens_v=relation_tokens_v,
        attention_mask=attention_mask_,
    )


def test_unirat_batch_consistency(
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
    x_tokens: torch.Tensor,
    relation_tokens_k: torch.Tensor,
    relation_tokens_v: torch.Tensor,
    attention_mask_: torch.Tensor,
    device: torch.device,
) -> None:
    unirat = RAT(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_embed_x=num_embed_x,
        pad_x_index=pad_x_index,
        num_embed_r_k=num_embed_r_k,
        pad_r_k_index=pad_r_k_index,
        num_embed_r_v=num_embed_r_v,
        pad_r_v_index=pad_r_v_index,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ).to(device=device)
    x_token_logits_batched = unirat(
        x_tokens=x_tokens,
        relation_tokens_k=relation_tokens_k,
        relation_tokens_v=relation_tokens_v,
        attention_mask=attention_mask_,
    )
    x_token_logits_split = torch.cat(
        list(
            unirat(
                x_tokens=_x_tokens.unsqueeze(0),
                relation_tokens_k=_relation_tokens_k.unsqueeze(0),
                relation_tokens_v=_relation_tokens_v.unsqueeze(0),
                attention_mask=_attention_mask.unsqueeze(0),
            )
            for _x_tokens, _relation_tokens_k, _relation_tokens_v, _attention_mask in zip(
                x_tokens, relation_tokens_k, relation_tokens_v, attention_mask_
            )
        ),
        dim=0,
    )
    assert x_token_logits_batched.shape == x_token_logits_split.shape
    assert (
        torch.allclose(
            x_token_logits_batched,
            x_token_logits_split,
            rtol=1e-3,
            atol=1e-4,
            equal_nan=True,
        )
        is True
    )


def test_unirat_with_memory_minimal(
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
    x_tokens: torch.Tensor,
    memory: torch.Tensor,
    relation_tokens_k: torch.Tensor,
    memory_relation_tokens_k: torch.Tensor,
    relation_tokens_v: torch.Tensor,
    memory_relation_tokens_v: torch.Tensor,
    attention_mask_: torch.Tensor,
    memory_attention_mask: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    unirat = RATWithMemory(
        num_layers=num_layers,
        embed_dim=embed_dim,
        mem_embed_dim=mem_embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_embed_x=num_embed_x,
        pad_x_index=pad_x_index,
        num_embed_r_k=num_embed_r_k,
        pad_r_k_index=pad_r_k_index,
        num_mem_embed_r_k=num_mem_embed_r_k,
        pad_mem_r_k_index=pad_mem_r_k_index,
        num_embed_r_v=num_embed_r_v,
        pad_r_v_index=pad_r_v_index,
        num_mem_embed_r_v=num_mem_embed_r_v,
        pad_mem_r_v_index=pad_mem_r_v_index,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=relu_dropout,
    ).to(device=device)
    _ = unirat(
        x_tokens=x_tokens,
        memory=memory,
        relation_tokens_k=relation_tokens_k,
        memory_relation_tokens_k=memory_relation_tokens_k,
        relation_tokens_v=relation_tokens_v,
        memory_relation_tokens_v=memory_relation_tokens_v,
        attention_mask=attention_mask_,
        memory_attention_mask=memory_attention_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )


def test_unirat_with_memory_batch_consistency(
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
    x_tokens: torch.Tensor,
    memory: torch.Tensor,
    relation_tokens_k: torch.Tensor,
    memory_relation_tokens_k: torch.Tensor,
    relation_tokens_v: torch.Tensor,
    memory_relation_tokens_v: torch.Tensor,
    attention_mask_: torch.Tensor,
    memory_attention_mask: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
    device: torch.device,
) -> None:
    unirat = RATWithMemory(
        num_layers=num_layers,
        embed_dim=embed_dim,
        mem_embed_dim=mem_embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_embed_x=num_embed_x,
        pad_x_index=pad_x_index,
        num_embed_r_k=num_embed_r_k,
        pad_r_k_index=pad_r_k_index,
        num_mem_embed_r_k=num_mem_embed_r_k,
        pad_mem_r_k_index=pad_mem_r_k_index,
        num_embed_r_v=num_embed_r_v,
        pad_r_v_index=pad_r_v_index,
        num_mem_embed_r_v=num_mem_embed_r_v,
        pad_mem_r_v_index=pad_mem_r_v_index,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ).to(device=device)
    x_token_logits_batched = unirat(
        x_tokens=x_tokens,
        memory=memory,
        relation_tokens_k=relation_tokens_k,
        memory_relation_tokens_k=memory_relation_tokens_k,
        relation_tokens_v=relation_tokens_v,
        memory_relation_tokens_v=memory_relation_tokens_v,
        attention_mask=attention_mask_,
        memory_attention_mask=memory_attention_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    x_token_logits_split = torch.cat(
        list(
            unirat(
                x_tokens=_x_tokens.unsqueeze(0),
                memory=_memory.unsqueeze(0),
                relation_tokens_k=_relation_tokens_k.unsqueeze(0),
                memory_relation_tokens_k=_memory_relation_tokens_k.unsqueeze(0),
                relation_tokens_v=_relation_tokens_v.unsqueeze(0),
                memory_relation_tokens_v=_memory_relation_tokens_v.unsqueeze(0),
                attention_mask=_attention_mask.unsqueeze(0),
                memory_attention_mask=_memory_attention_mask.unsqueeze(0),
                memory_key_padding_mask=_memory_key_padding_mask.unsqueeze(0),
            )
            for (
                _x_tokens,
                _memory,
                _relation_tokens_k,
                _memory_relation_tokens_k,
                _relation_tokens_v,
                _memory_relation_tokens_v,
                _attention_mask,
                _memory_attention_mask,
                _memory_key_padding_mask,
            ) in zip(
                x_tokens,
                memory,
                relation_tokens_k,
                memory_relation_tokens_k,
                relation_tokens_v,
                memory_relation_tokens_v,
                attention_mask_,
                memory_attention_mask,
                memory_key_padding_mask,
            )
        ),
        dim=0,
    )
    assert x_token_logits_batched.shape == x_token_logits_split.shape
    assert (
        torch.allclose(
            x_token_logits_batched,
            x_token_logits_split,
            rtol=1e-3,
            atol=1e-4,
            equal_nan=True,
        )
        is True
    )
