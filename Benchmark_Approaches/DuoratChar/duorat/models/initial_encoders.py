import abc
import math
from typing import Optional, Tuple

import torch
from torch import nn


from transformers import BertModel

from duorat.models.utils import _flip_attention_mask
from duorat.models.rat import RATLayer
from duorat.preproc.offline import (
    TransformerDuoRATPreproc,
    BertDuoRATPreproc,
    SingletonGloVe,
)
from duorat.preproc.utils import pad_nd_tensor
from duorat.types import DuoRATInputSegmentBatch
from duorat.utils import registry


# not yet in pytorch main tree, therefore temporarily taken from https://github.com/pytorch/examples/blob/632d385444ae16afe3e4003c94864f9f97dc8541/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        assert pe.shape == (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            position_ids: the positions associated with the sequence (optional).
        Shape:
            x: [batch size, sequence length, embed dim]
            position_ids: [batch size, sequence length]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        batch_size, seq_len, embed_dim = x.shape
        if position_ids is not None:
            assert position_ids.shape == (batch_size, seq_len)
            x = x + torch.gather(
                self.pe.expand(batch_size, self.max_len, embed_dim),
                dim=1,
                index=position_ids.unsqueeze(dim=2).expand(
                    batch_size, seq_len, embed_dim
                ),
            )
        else:
            x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)
        assert x.shape == (batch_size, seq_len, embed_dim)
        return x


class InitialEncoder(abc.ABC, nn.Module):
    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def max_supported_input_length(self) -> Optional[int]:
        pass

    @abc.abstractmethod
    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        input_attention_mask: torch.Tensor,
        input_key_padding_mask: torch.Tensor,
        input_token_type_ids: torch.Tensor,
        input_position_ids: torch.Tensor,
        input_source_gather_index: torch.Tensor,
        input_segments: Tuple[DuoRATInputSegmentBatch, ...],
    ) -> torch.Tensor:
        pass


class SingletonGloVeEmbedding(nn.Module):

    _embed = None

    def __init__(self):
        SingletonGloVeEmbedding._construct_if_needed()
        super().__init__()

    @property
    def embedding_dim(self):
        return SingletonGloVeEmbedding._embed.embedding_dim

    def forward(self, x):
        return SingletonGloVeEmbedding._embed(x.cpu()).to(x.device)

    @staticmethod
    def _construct_if_needed():
        if SingletonGloVeEmbedding._embed is None:
            glove = SingletonGloVe()
            # two hacks here:
            # - add a zero vector for <unk>
            # - remove the first vector that corresponds to ","
            SingletonGloVeEmbedding._embed = nn.Embedding.from_pretrained(
                torch.cat([torch.zeros_like(glove.vectors[[0]]), glove.vectors[1:]]),
                freeze=True,
            )

    def __setstate__(self, state):
        self._construct_if_needed()
        super().__setstate__(state)


@registry.register("initial_encoder", "Transformer")
class TransformerEncoder(InitialEncoder):
    def __init__(
        self,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_layers: int,
        use_attention_mask: bool,
        use_position_ids: bool,
        use_positional_embedding: bool,
        preproc: TransformerDuoRATPreproc,
    ) -> None:
        super(TransformerEncoder, self).__init__()

        assert isinstance(preproc, TransformerDuoRATPreproc)

        # embedding table for GloVe tokens that appear in the training data
        self.input_embed_a = SingletonGloVeEmbedding()
        self._embed_dim = self.input_embed_a.embedding_dim

        assert self._embed_dim % num_heads == 0

        # embedding table for tokens that appear in the training data and are not in GloVe
        self.input_embed_b = nn.Embedding(
            num_embeddings=len(preproc.input_vocab_b), embedding_dim=self._embed_dim,
        )

        # positional embedding table
        self.positional_embed = PositionalEncoding(
            d_model=self._embed_dim, dropout=dropout,
        )

        self.layers = nn.ModuleList(
            [
                RATLayer(
                    embed_dim=self._embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=dropout,
                    relu_dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.use_attention_mask = use_attention_mask
        self.use_position_ids = use_position_ids
        self.use_positional_embedding = use_positional_embedding

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def max_supported_input_length(self) -> Optional[int]:
        return None

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        input_attention_mask: torch.Tensor,
        input_key_padding_mask: torch.Tensor,
        input_token_type_ids: torch.Tensor,
        input_position_ids: torch.Tensor,
        input_source_gather_index: torch.Tensor,
        input_segments: Tuple[DuoRATInputSegmentBatch, ...],
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        # assert input_a.dtype == torch.long
        (batch_size, max_input_length) = input_a.shape

        # assert input_b.dtype == torch.long
        assert input_b.shape == (batch_size, max_input_length)

        emb_a = self.input_embed_a(input_a.to(device=device))
        assert emb_a.shape == (batch_size, max_input_length, self._embed_dim)
        assert not torch.isnan(emb_a).any()

        emb_b = self.input_embed_b(input_b.to(device=device))
        assert emb_b.shape == (batch_size, max_input_length, self._embed_dim)
        assert not torch.isnan(emb_b).any()

        input = emb_a + emb_b

        del emb_a
        del emb_b

        if self.use_position_ids:
            _input_position_ids = input_position_ids.to(device=device)
            assert _input_position_ids.shape == (batch_size, max_input_length)
        else:
            _input_position_ids = None

        if self.use_positional_embedding:
            input = self.positional_embed(input, position_ids=_input_position_ids)
            assert input.shape == (batch_size, max_input_length, self._embed_dim)

        if self.use_position_ids:
            del _input_position_ids

        if self.use_attention_mask:
            # attend according to attention mask
            # assert input_attention_mask.dtype == torch.bool
            assert input_attention_mask.shape == (
                batch_size,
                max_input_length,
                max_input_length,
            )
            _input_attention_mask = _flip_attention_mask(
                mask=input_attention_mask.to(device=device)
            )
            _input_key_padding_mask = None
        else:
            # attend everywhere except padding
            _input_attention_mask = None
            # assert input_key_padding_mask.dtype == torch.bool
            assert input_key_padding_mask.shape == (batch_size, max_input_length)
            _input_key_padding_mask = ~input_key_padding_mask.to(device=device)

        for layer in self.layers:
            assert not torch.isnan(input).any()
            input = layer(
                x=input,
                relations_k=None,
                relations_v=None,
                attention_mask=_input_attention_mask,
                key_padding_mask=_input_key_padding_mask,
            )
        assert input.shape == (batch_size, max_input_length, self._embed_dim)
        assert input.device == device

        (_batch_size, max_src_length) = input_source_gather_index.shape
        assert _batch_size == batch_size
        _input_source_gather_index = (
            input_source_gather_index.to(device=device)
            .unsqueeze(dim=2)
            .expand(batch_size, max_src_length, self.embed_dim)
        )
        source = torch.gather(input, dim=1, index=_input_source_gather_index)
        assert source.shape == (batch_size, max_src_length, self.embed_dim)
        assert source.device == device

        return source


@registry.register("initial_encoder", "Bert")
class BertEncoder(InitialEncoder):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        trainable: bool,
        num_return_layers: int,
        embed_dim: int,
        use_dedicated_gpu: bool,
        use_affine_transformation: bool,
        use_attention_mask: bool,
        use_token_type_ids: bool,
        use_position_ids: bool,
        use_segments: bool,
        preproc: BertDuoRATPreproc,
    ) -> None:
        super(BertEncoder, self).__init__()

        assert isinstance(preproc, BertDuoRATPreproc)

        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
        )
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.trainable = trainable
        assert 0 <= num_return_layers <= self.bert.config.num_hidden_layers + 1
        self.num_return_layers = num_return_layers

        self._bert_embed_dim = self.num_return_layers * self.bert.config.hidden_size
        if use_affine_transformation:
            self._embed_dim = embed_dim
            self.linear = nn.Linear(self._bert_embed_dim, self.embed_dim)
        else:
            self._embed_dim = self._bert_embed_dim

        if use_dedicated_gpu:
            self.cuda(1)

        self.use_dedicated_gpu = use_dedicated_gpu
        self.use_affine_transformation = use_affine_transformation
        self.use_attention_mask = use_attention_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_position_ids = use_position_ids
        self.use_segments = use_segments

    @property
    def max_supported_input_length(self) -> Optional[int]:
        if self.use_position_ids or self.use_segments:
            return None
        else:
            return self.bert.config.max_position_embeddings

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _forward_segment(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        input_attention_mask: torch.Tensor,
        input_key_padding_mask: torch.Tensor,
        input_token_type_ids: torch.Tensor,
        input_position_ids: torch.Tensor,
    ):
        device = next(self.parameters()).device

        # assert input_a.dtype == torch.long
        (batch_size, max_input_length) = input_a.shape

        # assert input_b.dtype == torch.long
        assert input_b.shape == (batch_size, max_input_length)
        # TODO: assert that input_b does not contain anything but UNK or padding

        if self.use_attention_mask:
            # attend according to attention mask
            # assert input_attention_mask.dtype == torch.bool
            assert input_attention_mask.shape == (
                batch_size,
                max_input_length,
                max_input_length,
            )
            _input_attention_mask = input_attention_mask.to(device=device)
        else:
            # use the key padding mask as attention mask
            # assert input_key_padding_mask.dtype == torch.bool
            assert input_key_padding_mask.shape == (batch_size, max_input_length)
            _input_attention_mask = input_key_padding_mask.to(device=device)

        if self.use_token_type_ids:
            # assert input_token_type_ids.dtype == torch.long
            assert input_key_padding_mask.shape == (batch_size, max_input_length)
            _input_token_type_ids = input_token_type_ids.to(device=device)
        else:
            _input_token_type_ids = None

        if self.use_position_ids:
            # assert input_position_ids.dtype == torch.long
            assert input_position_ids.shape == (batch_size, max_input_length)
            _input_position_ids = input_position_ids.to(device=device)
        else:
            _input_position_ids = None

        last_layer_hidden_state, _pooled_output, all_hidden_states = self.bert(
            input_a.to(device=device),
            attention_mask=_input_attention_mask,
            token_type_ids=_input_token_type_ids,
            position_ids=_input_position_ids,
        )

        assert len(all_hidden_states) == self.bert.config.num_hidden_layers + 1
        # assert all(
        #     hidden_state.dtype == torch.float for hidden_state in all_hidden_states
        # )
        assert all(
            hidden_state.shape
            == (batch_size, max_input_length, self.bert.config.hidden_size)
            for hidden_state in all_hidden_states
        )
        assert all(hidden_state.device == device for hidden_state in all_hidden_states)
        assert all_hidden_states[-1].data_ptr() == last_layer_hidden_state.data_ptr()

        output = torch.cat(all_hidden_states[-self.num_return_layers :], 2)
        assert output.shape == (batch_size, max_input_length, self._bert_embed_dim)
        assert output.device == device

        if self.use_affine_transformation:
            output = self.linear(output)
        assert output.shape == (batch_size, max_input_length, self.embed_dim)
        assert output.device == device

        return output

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        input_attention_mask: torch.Tensor,
        input_key_padding_mask: torch.Tensor,
        input_token_type_ids: torch.Tensor,
        input_position_ids: torch.Tensor,
        input_source_gather_index: torch.Tensor,
        input_segments: Tuple[DuoRATInputSegmentBatch, ...],
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        if self.use_segments:
            source_tensors = []
            batch_size = len(input_segments)
            for segment in input_segments:
                output = self._forward_segment(
                    input_a=segment.input_a,
                    input_b=segment.input_b,
                    input_attention_mask=segment.input_attention_mask,
                    input_key_padding_mask=segment.input_key_padding_mask,
                    input_token_type_ids=segment.input_token_type_ids,
                    input_position_ids=segment.input_position_ids,
                )
                (segment_size, max_input_segment_length, _embed_dim) = output.shape
                assert _embed_dim == self.embed_dim
                (_segment_size, src_length) = segment.input_source_gather_index.shape
                assert _segment_size == segment_size
                _input_source_gather_index = (
                    segment.input_source_gather_index.to(device=device)
                    .unsqueeze(dim=2)
                    .expand(segment_size, src_length, self.embed_dim)
                )
                source_tensor = torch.gather(
                    output, dim=1, index=_input_source_gather_index,
                )
                del _input_source_gather_index
                assert segment.input_source_gather_index_mask.shape == (
                    segment_size,
                    src_length,
                )
                _input_source_gather_index_mask = ~(
                    segment.input_source_gather_index_mask.to(device=device)
                    .unsqueeze(dim=2)
                    .expand(segment_size, src_length, self.embed_dim)
                )
                source_tensor = torch.masked_fill(
                    source_tensor, mask=_input_source_gather_index_mask, value=0,
                )
                del _input_source_gather_index_mask
                source_tensor = torch.sum(source_tensor, dim=0, keepdim=False,)
                assert source_tensor.shape == (src_length, self.embed_dim)
                source_tensors.append(source_tensor)
            source = pad_nd_tensor(
                tensors=source_tensors,
                num_padding_dimensions=2,
                batch_first=True,
                padding_value=0,
            )
            (_batch_size, _max_src_length, _embed_dim) = source.shape
            assert _batch_size == batch_size
            assert _embed_dim == self.embed_dim
            assert source.device == device
        else:
            output = self._forward_segment(
                input_a=input_a,
                input_b=input_b,
                input_attention_mask=input_attention_mask,
                input_key_padding_mask=input_key_padding_mask,
                input_token_type_ids=input_token_type_ids,
                input_position_ids=input_position_ids,
            )
            (batch_size, max_input_length, _embed_dim) = output.shape
            assert _embed_dim == self.embed_dim
            (_batch_size, max_src_length) = input_source_gather_index.shape
            assert _batch_size == batch_size
            source = torch.gather(
                output,
                dim=1,
                index=(
                    input_source_gather_index.to(device=device)
                    .unsqueeze(dim=2)
                    .expand(batch_size, max_src_length, self.embed_dim)
                ),
            )
            assert source.shape == (batch_size, max_src_length, self.embed_dim)
            assert source.device == device

        if self.use_dedicated_gpu:
            return source.cuda(0)
        else:
            return source
