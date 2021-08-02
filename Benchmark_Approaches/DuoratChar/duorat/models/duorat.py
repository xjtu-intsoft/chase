import logging
from collections import defaultdict, deque
from typing import List, Tuple, Optional, Dict, Any, Deque, Sequence
from functools import partial, lru_cache

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

from duorat.models.initial_encoders import InitialEncoder
from duorat.models.pointer import Pointer
from duorat.asdl.asdl import ASDLPrimitiveType
from duorat.asdl.transition_system import Partial, MaskAction

from duorat.models.utils import _flip_attention_mask
from duorat.preproc.duorat import (
    duo_rat_decoder_batch,
    duo_rat_encoder_batch,
    DuoRATEncoderItemBuilder,
    DuoRATDecoderItemBuilder,
    DuoRATHypothesis,
    duo_rat_encoder_item,
    duo_rat_decoder_item,
    duo_rat_item,
)
from duorat.preproc.masking import (
    mask_duo_rat_decoder_batch,
    BernoulliMaskConfig,
)
from duorat.preproc.offline import DuoRATPreproc
from duorat.preproc.relations import (
    freeze_order,
    source_relation_types,
    target_relation_types,
    memory_relation_types,
    DefaultSourceRelation,
    DefaultTargetRelation,
    DefaultMemoryRelation,
)
from duorat.types import (
    RATPreprocItem,
    DuoRATEncoderItem,
    DuoRATBatch,
    DuoRATEncoderBatch,
    DuoRATDecoderBatch,
    QuestionToken,
    ColumnToken,
    TableToken,
    ActionToken,
    Token,
    InputId,
    DuoRATDecoderItem,
    DuoRATItem,
    NoScoping,
    AttentionScope,
    AttentionScopeName,
    FineScoping,
    Scoping,
)
from duorat.models.rat import RATLayerWithMemory, RATLayer
from duorat.utils import registry
from duorat.utils.beam_search import beam_search, Candidate, FinishedBeam

logger = logging.getLogger(__name__)


@registry.register("model", "DuoRAT")
class DuoRATModel(torch.nn.Module):
    """
    A seq2tree model based on relation-aware transformers (RAT)
    Both the encoder and the decoder are RAT.
    The decoder is trained with masked-token prediction.
    """

    def __init__(self, preproc: DuoRATPreproc, encoder: dict, decoder: dict) -> None:
        super(DuoRATModel, self).__init__()

        self.preproc = preproc

        # First-stage encoder
        self.initial_encoder: InitialEncoder = registry.construct(
            "initial_encoder", encoder["initial_encoder"], preproc=preproc
        )

        self.predict_relations = encoder.get("predict_relations", False)
        self.relation_aware_values = encoder.get("relation_aware_values", True)
        self.max_source_length: Optional[int] = encoder.get("max_source_length", None)

        self.schema_input_token_ordering = encoder.get(
            "schema_input_token_ordering", "[column][table]"
        )
        self.schema_source_token_ordering = encoder.get(
            "schema_source_token_ordering", "[column][table]"
        )

        self.grammar_constrained_training = decoder.get(
            "grammar_constrained_training", True
        )
        self.grammar_constrained_inference = decoder.get(
            "grammar_constrained_inference", True
        )

        self.encoder_rat_embed_dim = self.initial_encoder.embed_dim
        self.encoder_rat_head_dim = (
            self.encoder_rat_embed_dim // encoder["rat_num_heads"]
        )
        assert (
            self.encoder_rat_head_dim * encoder["rat_num_heads"]
            == self.encoder_rat_embed_dim
        )
        assert self.encoder_rat_embed_dim % encoder["rat_num_heads"] == 0
        self.mem_embed_dim = self.encoder_rat_embed_dim
        self.decoder_rat_embed_dim = (
            decoder["action_embed_dim"]
            + decoder["field_embed_dim"]
            + decoder["type_embed_dim"]
        )
        self.decoder_rat_head_dim = (
            self.decoder_rat_embed_dim // decoder["rat_num_heads"]
        )
        assert (
            self.decoder_rat_head_dim * decoder["rat_num_heads"]
            == self.decoder_rat_embed_dim
        )
        assert self.decoder_rat_embed_dim % decoder["rat_num_heads"] == 0
        self.decoder_loss = decoder.get("loss", "log_likelihood")

        # embedding table for all ASDL production rules + Reduce + MASK + decoder vocab
        self.target_embed = nn.Embedding(
            num_embeddings=len(preproc.target_vocab),
            embedding_dim=decoder["action_embed_dim"],
        )
        # embedding table for ASDL fields in constructors + NO_FIELD (field of root node)
        self.frontier_field_embed = nn.Embedding(
            num_embeddings=len(preproc.transition_system.grammar.fields) + 1,
            embedding_dim=decoder["field_embed_dim"],
        )
        # embedding table for ASDL types
        self.frontier_field_type_embed = nn.Embedding(
            num_embeddings=len(preproc.transition_system.grammar.types),
            embedding_dim=decoder["type_embed_dim"],
        )

        # attention scoping
        self.input_attention_scoping: Scoping = registry.construct(
            "attention_scoping",
            encoder.get(
                "input_attention_scoping",
                {
                    "name": "FineScoping",
                    "question_sees_columns": False,
                    "question_sees_tables": False,
                    "columns_see_question": False,
                    "columns_see_each_other": False,
                    "columns_see_tables": False,
                    "tables_see_question": False,
                    "tables_see_columns": False,
                    "tables_see_each_other": False,
                    "target_sees_question": False,
                    "target_sees_columns": False,
                    "target_sees_tables": False,
                },
            ),
        )
        self.source_attention_scoping: Scoping = registry.construct(
            "attention_scoping",
            encoder.get("source_attention_scoping", {"name": "NoScoping"}),
        )
        self.target_attention_scoping: Scoping = registry.construct(
            "attention_scoping",
            encoder.get("target_attention_scoping", {"name": "NoScoping"}),
        )

        # relation types
        self.source_relation_types = freeze_order(
            (source_relation_types(**(encoder.get("source_relation_types", {}))),)
        )
        assert self.source_relation_types[DefaultSourceRelation()] == 0
        self.target_relation_types = freeze_order(
            (target_relation_types(**(decoder.get("target_relation_types", {}))),)
        )
        assert self.target_relation_types[DefaultTargetRelation()] == 0
        self.memory_relation_types = freeze_order(
            (memory_relation_types(**(decoder.get("memory_relation_types", {}))),)
        )
        assert self.memory_relation_types[DefaultMemoryRelation()] == 0

        # relation embedding tables
        self.source_relation_embed = nn.Embedding(
            num_embeddings=len(self.source_relation_types),
            embedding_dim=self.encoder_rat_head_dim,
            padding_idx=self.source_relation_types[DefaultSourceRelation()],
        )
        self.target_relation_embed = nn.Embedding(
            num_embeddings=len(self.target_relation_types),
            embedding_dim=self.decoder_rat_head_dim,
            padding_idx=self.target_relation_types[DefaultTargetRelation()],
        )
        self.memory_relation_embed = nn.Embedding(
            num_embeddings=len(self.memory_relation_types),
            embedding_dim=self.decoder_rat_head_dim,
            padding_idx=self.memory_relation_types[DefaultMemoryRelation()],
        )

        self.encoder_rat_layers = nn.ModuleList(
            [
                RATLayer(
                    embed_dim=self.encoder_rat_embed_dim,
                    num_heads=encoder["rat_num_heads"],
                    ffn_dim=encoder["rat_ffn_dim"],
                    dropout=encoder["rat_dropout"],
                    attention_dropout=encoder["rat_attention_dropout"],
                    relu_dropout=encoder["rat_relu_dropout"],
                )
                for _ in range(encoder["rat_num_layers"])
            ]
        )

        # decoder transformer layers with memory
        self.decoder_rat_layers = nn.ModuleList(
            [
                RATLayerWithMemory(
                    embed_dim=self.decoder_rat_embed_dim,
                    mem_embed_dim=self.mem_embed_dim,
                    num_heads=decoder["rat_num_heads"],
                    ffn_dim=decoder["rat_ffn_dim"],
                    dropout=decoder["rat_dropout"],
                    attention_dropout=decoder["rat_attention_dropout"],
                    relu_dropout=decoder["rat_relu_dropout"],
                )
                for _ in range(decoder["rat_num_layers"])
            ]
        )

        # Copy decision layer
        self.copy_logprob = nn.Sequential(
            nn.Linear(in_features=self.decoder_rat_embed_dim, out_features=2),
            nn.LogSoftmax(dim=-1),
        )

        # Copy pointer network
        self.pointer_network: Pointer = registry.construct(
            "pointer",
            decoder.get("pointer", {"name": "Bahdanau", "proj_size": 50}),
            query_size=self.decoder_rat_embed_dim,
            key_size=self.mem_embed_dim,
        )

        # Generation projection layer
        self.out_proj = nn.Linear(
            in_features=self.decoder_rat_embed_dim,
            out_features=len(preproc.target_vocab),
        )

        self.mask_sampling_config = BernoulliMaskConfig(p_mask=decoder["p_mask"])

    def compute_loss(
        self, preproc_items: List[RATPreprocItem], debug=False
    ) -> torch.Tensor:
        duo_rat_batch = self.items_to_duo_rat_batch(preproc_items)
        decoder_batch = duo_rat_batch.decoder_batch
        memory, output = self.forward(batch=duo_rat_batch)
        assert not torch.isnan(memory).any()
        assert not torch.isnan(output).any()
        loss = self._compute_loss(
            memory=memory,
            output=output,
            target_key_padding_mask=decoder_batch.target_key_padding_mask,
            valid_copy_mask=decoder_batch.valid_copy_mask,
            copy_target_mask=decoder_batch.copy_target_mask,
            valid_actions_mask=decoder_batch.valid_actions_mask,
            target=decoder_batch.target,
        ).mean()
        return loss

    def items_to_duo_rat_batch(
        self, preproc_items: List[RATPreprocItem]
    ) -> DuoRATBatch:
        items = self.preproc_items_to_duorat_items(preproc_items)
        decoder_batch = mask_duo_rat_decoder_batch(
            batch=duo_rat_decoder_batch(
                items=tuple(item.decoder_item for item in items)
            ),
            action_relation_types=self.target_relation_types,
            memory_relation_types=self.memory_relation_types,
            mask_sampling_config=self.mask_sampling_config,
            mask_value=self.preproc.target_vocab[MaskAction()],
        )
        duo_rat_batch = DuoRATBatch(
            encoder_batch=duo_rat_encoder_batch(
                items=tuple(item.encoder_item for item in items)
            ),
            decoder_batch=decoder_batch,
        )
        return duo_rat_batch

    def preproc_items_to_duorat_items(
        self, preproc_items: List[RATPreprocItem]
    ) -> Tuple[DuoRATItem]:
        def _skip_filter(encoder_item: DuoRATEncoderItem) -> bool:
            source_length = encoder_item.input_source_gather_index.shape[0]
            if (
                self.max_source_length is not None
                and source_length > self.max_source_length
            ):
                logger.warning(
                    "source length exceeds maximum source length, {} > {}, skipping".format(
                        source_length, self.max_source_length
                    )
                )
                return False
            else:
                return True

        items = tuple(
            item
            for item in (
                self._get_item_cached(preproc_item=preproc_item)
                for preproc_item in preproc_items
            )
            if _skip_filter(encoder_item=item.encoder_item)
            # Skip examples with 0 actions (should not happen during training)
            and len(item.decoder_item.valid_actions_mask) > 0
        )
        return items

    def _compute_loss(
        self,
        memory: torch.Tensor,
        output: torch.Tensor,
        target_key_padding_mask: torch.Tensor,
        valid_copy_mask: torch.Tensor,
        copy_target_mask: torch.Tensor,
        valid_actions_mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        p_copy_gen_logprobs = self.copy_logprob(output)
        (batch_size, seq_len) = target.shape
        assert p_copy_gen_logprobs.shape == (batch_size, seq_len, 2)
        assert not torch.isnan(p_copy_gen_logprobs).any()

        copy_logits = self.pointer_network(query=output, keys=memory)
        # Grammar-based constraining
        if self.grammar_constrained_training:
            masked_copy_logits = copy_logits.masked_fill(
                mask=~valid_copy_mask.to(device=device), value=float("-inf"),
            )
        else:
            masked_copy_logits = copy_logits
        copy_log_probs = F.log_softmax(masked_copy_logits, dim=2)
        copy_log_likelihood = torch.logsumexp(
            copy_log_probs.masked_fill(
                mask=~copy_target_mask.to(device=device), value=float("-inf"),
            ),
            dim=2,
        )
        assert copy_log_likelihood.shape == (batch_size, seq_len)
        assert not torch.isnan(copy_log_likelihood).any()

        gen_logits = self.out_proj(output)
        assert not torch.isnan(gen_logits).any()
        # Grammar-based constraining
        if self.grammar_constrained_training:
            masked_gen_logits = gen_logits.masked_fill(
                mask=~valid_actions_mask.to(device=device), value=float("-inf"),
            )
        else:
            masked_gen_logits = gen_logits
        gen_log_probs = F.log_softmax(masked_gen_logits, dim=2)
        gen_log_likelihood = gen_log_probs.gather(
            dim=2, index=target.to(device=device).unsqueeze(2)
        ).squeeze(2)
        assert gen_log_likelihood.shape == (batch_size, seq_len)

        flipped_target_key_padding_mask = ~target_key_padding_mask.to(device=device)
        overall_log_likelihood = torch.logsumexp(
            torch.stack(
                [
                    copy_log_likelihood.masked_fill(
                        mask=flipped_target_key_padding_mask, value=0
                    )
                    + p_copy_gen_logprobs[:, :, 0],
                    gen_log_likelihood.masked_fill(
                        mask=flipped_target_key_padding_mask, value=0
                    )
                    + p_copy_gen_logprobs[:, :, 1],
                ],
                dim=0,
            ),
            dim=0,
            keepdim=False,
        )

        if self.decoder_loss == "log_likelihood":
            losses = overall_log_likelihood
        elif self.decoder_loss == "expected_per_token_acc":
            losses = torch.exp(overall_log_likelihood)
        else:
            raise ValueError("unknown decoder loss")

        losses = -losses.masked_select(target_key_padding_mask.to(device=device))
        assert not torch.isnan(losses).any()
        return losses

    def eval_on_batch(self, batch):
        mean_loss = self.compute_loss(batch)
        batch_size = len(batch)
        result = {
            "loss": mean_loss.item() * batch_size,
            "total": batch_size,
        }
        return result

    def forward(self, batch: DuoRATBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self._encode(batch=batch.encoder_batch)
        target = self._decode(memory=source, batch=batch.decoder_batch)
        return source, target

    def _encode(self, batch: DuoRATEncoderBatch) -> torch.Tensor:
        (batch_size, _max_input_length) = batch.input_a.shape
        source = self.initial_encoder(
            input_a=batch.input_a,
            input_b=batch.input_b,
            input_attention_mask=batch.input_attention_mask,
            input_key_padding_mask=batch.input_key_padding_mask,
            input_token_type_ids=batch.input_token_type_ids,
            input_position_ids=batch.input_position_ids,
            input_source_gather_index=batch.input_source_gather_index,
            input_segments=batch.input_segments,
        )
        (_batch_size, max_src_length, _encoder_rat_embed_dim) = source.shape
        assert _batch_size == batch_size
        assert _encoder_rat_embed_dim == self.encoder_rat_embed_dim
        source = self._encode_source(
            source=source,
            source_relations=batch.source_relations,
            source_attention_mask=batch.source_attention_mask,
            source_key_padding_mask=batch.source_key_padding_mask,
        )
        assert source.shape == (batch_size, max_src_length, self.encoder_rat_embed_dim)
        return source

    def _encode_source(
        self,
        source: torch.Tensor,
        source_relations: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        source_relations = source_relations.to(device)
        (batch_size, max_src_length, _encoder_rat_embed_dim) = source.shape

        _source_relations = self.source_relation_embed(source_relations)
        assert _source_relations.shape == (
            batch_size,
            max_src_length,
            max_src_length,
            self.encoder_rat_head_dim,
        )

        _source_attention_mask = _flip_attention_mask(source_attention_mask).to(
            device=device
        )
        _source_key_padding_mask = ~source_key_padding_mask.to(device=device)

        for layer in self.encoder_rat_layers:
            source = layer(
                x=source,
                relations_k=_source_relations,
                relations_v=_source_relations
                if self.relation_aware_values
                else torch.zeros_like(_source_relations),
                attention_mask=_source_attention_mask,
                key_padding_mask=_source_key_padding_mask,
            )
        assert source.shape == (batch_size, max_src_length, self.encoder_rat_embed_dim)
        return source

    def _decode(self, memory: torch.Tensor, batch: DuoRATDecoderBatch) -> torch.Tensor:
        device = next(self.parameters()).device

        (batch_size, max_src_length, _encoder_rat_embed_dim) = memory.shape
        target = torch.cat(
            (
                self.target_embed(self._get_targets_as_input(batch).to(device=device)),
                self.frontier_field_embed(batch.frontier_fields.to(device=device)),
                self.frontier_field_type_embed(
                    batch.frontier_field_types.to(device=device)
                ),
            ),
            dim=2,
        )
        _batch_size, max_tgt_length, _decoder_rat_embed_dim = target.shape
        assert _batch_size == batch_size
        assert _decoder_rat_embed_dim == self.decoder_rat_embed_dim
        target_relations = self.target_relation_embed(
            batch.target_relations.to(device=device)
        )
        assert target_relations.shape == (
            batch_size,
            max_tgt_length,
            max_tgt_length,
            self.decoder_rat_head_dim,
        )
        memory_relations = self.memory_relation_embed(
            self._get_memory_relations(batch).to(device=device)
        )
        assert memory_relations.shape == (
            batch_size,
            max_tgt_length,
            max_src_length,
            self.decoder_rat_head_dim,
        )
        target_attention_mask = _flip_attention_mask(
            mask=batch.target_attention_mask.to(device=device)
        )
        memory_attention_mask = _flip_attention_mask(
            mask=batch.memory_attention_mask.to(device=device)
        )
        target_key_padding_mask = ~batch.target_key_padding_mask.to(device=device)
        memory_key_padding_mask = ~batch.memory_key_padding_mask.to(device=device)
        for layer in self.decoder_rat_layers:
            target = layer(
                x=target,
                memory=memory,
                relations_k=target_relations,
                memory_relations_k=memory_relations,
                relations_v=target_relations,
                memory_relations_v=memory_relations,
                attention_mask=target_attention_mask,
                memory_attention_mask=memory_attention_mask,
                key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        assert target.shape == (batch_size, max_tgt_length, self.decoder_rat_embed_dim)
        return target

    @staticmethod
    def _get_targets_as_input(batch: DuoRATDecoderBatch) -> torch.Tensor:
        return batch.masked_target

    @staticmethod
    def _get_memory_relations(batch: DuoRATDecoderBatch) -> torch.Tensor:
        return batch.memory_relations

    def parse(
        self,
        preproc_items: List[RATPreprocItem],
        decode_max_time_step: int,
        beam_size: int,
    ) -> List[FinishedBeam]:
        assert len(preproc_items) == 1
        if not self.grammar_constrained_inference:
            assert beam_size == 1
        preproc_item = preproc_items[0]
        encoder_item, encoder_item_builder = self._get_encoder_item(
            preproc_item=preproc_item
        )
        memory = self._encode(batch=duo_rat_encoder_batch(items=[encoder_item]))
        return self.parse_decode(
            encoder_item_builder=encoder_item_builder,
            memory=memory,
            beam_size=beam_size,
            decode_max_time_step=decode_max_time_step,
            grammar_constrained_inference=self.grammar_constrained_inference,
        )

    def _get_encoder_item(
        self, preproc_item: RATPreprocItem, device: Optional[torch.device] = None
    ) -> Tuple[DuoRATEncoderItem, DuoRATEncoderItemBuilder]:
        if device is None:
            device = next(self.parameters()).device
        return duo_rat_encoder_item(
            preproc_item=preproc_item,
            input_a_str_to_id=self.preproc.input_a_str_to_id,
            input_b_str_to_id=self.preproc.input_b_str_to_id,
            max_supported_input_length=self.initial_encoder.max_supported_input_length,
            input_attention_scoping=self.input_attention_scoping,
            source_attention_scoping=self.source_attention_scoping,
            source_relation_types=self.source_relation_types,
            schema_input_token_ordering=self.schema_input_token_ordering,
            schema_source_token_ordering=self.schema_source_token_ordering,
            device=device,
        )

    def _get_decoder_item(
        self,
        preproc_item: RATPreprocItem,
        positioned_source_tokens: Sequence[Token[InputId, str]],
        device: Optional[torch.device] = None,
    ) -> Tuple[DuoRATDecoderItem, DuoRATDecoderItemBuilder]:
        if device is None:
            device = next(self.parameters()).device
        return duo_rat_decoder_item(
            preproc_item=preproc_item,
            positioned_source_tokens=positioned_source_tokens,
            target_vocab=self.preproc.target_vocab,
            transition_system=self.preproc.transition_system,
            allow_unk=True,
            source_attention_scoping=self.source_attention_scoping,
            target_attention_scoping=self.target_attention_scoping,
            target_relation_types=self.target_relation_types,
            memory_relation_types=self.memory_relation_types,
            device=device,
        )

    @lru_cache(maxsize=None)
    def _get_item_cached(self, preproc_item: RATPreprocItem) -> DuoRATItem:
        device = torch.device("cpu")
        return duo_rat_item(
            preproc_item=preproc_item,
            get_encoder_item=partial(self._get_encoder_item, device=device),
            get_decoder_item=partial(self._get_decoder_item, device=device),
        )

    def parse_decode(
        self,
        encoder_item_builder: DuoRATEncoderItemBuilder,
        memory: torch.Tensor,
        beam_size: int,
        decode_max_time_step: int,
        grammar_constrained_inference: bool,
    ) -> List[FinishedBeam]:
        question_position_map: Dict[Any, Deque[Any]] = defaultdict(deque)
        columns_position_map: Dict[Any, Deque[Any]] = defaultdict(deque)
        tables_position_map: Dict[Any, Deque[Any]] = defaultdict(deque)
        for positioned_source_token in encoder_item_builder.positioned_source_tokens:
            if isinstance(positioned_source_token, QuestionToken):
                question_position_map[positioned_source_token.raw_value].append(
                    positioned_source_token.position
                )
            elif isinstance(positioned_source_token, ColumnToken):
                columns_position_map[positioned_source_token.value].append(
                    positioned_source_token.position
                )
            elif isinstance(positioned_source_token, TableToken):
                tables_position_map[positioned_source_token.value].append(
                    positioned_source_token.position
                )
            else:
                raise ValueError(
                    "Unsupported token type: {}".format(
                        positioned_source_token.__repr__()
                    )
                )

        initial_hypothesis = DuoRATHypothesis(
            beam_builder=DuoRATDecoderItemBuilder(
                positioned_source_tokens=encoder_item_builder.positioned_source_tokens,
                target_vocab=self.preproc.target_vocab,
                transition_system=self.preproc.transition_system,
                allow_unk=False,
                source_attention_scoping=self.source_attention_scoping,
                target_attention_scoping=self.target_attention_scoping,
                target_relation_types=self.target_relation_types,
                memory_relation_types=self.memory_relation_types,
            ),
            scores=[],
            tokens=[],
        )
        res = beam_search(
            initial_hypothesis,
            beam_size,
            decode_max_time_step,
            get_new_hypothesis=lambda candidate: DuoRATHypothesis(
                beam_builder=candidate.prev_hypothesis.beam_builder.add_action_token(
                    action_token=ActionToken(
                        key=candidate.token,
                        value=candidate.token,
                        scope=AttentionScope(scope_name=AttentionScopeName.TARGET),
                    ),
                    copy=True,
                ),
                score=candidate.score,
                tokens=candidate.prev_hypothesis.tokens + [candidate.token],
                scores=candidate.prev_hypothesis.scores + [candidate.score],
            ),
            get_continuations=partial(
                self.get_continuations,
                memory=memory,
                question_position_map=question_position_map,
                columns_position_map=columns_position_map,
                tables_position_map=tables_position_map,
                grammar_constrained_inference=grammar_constrained_inference,
            ),
        )
        return [
            FinishedBeam(
                ast=hypothesis.beam_builder.parsing_result.res, score=hypothesis.score
            )
            for hypothesis in res
        ]

    def get_continuations(
        self,
        beam_hypotheses: List[DuoRATHypothesis],
        step: int,
        memory: torch.Tensor,
        question_position_map: Dict[Any, Deque[Any]],
        columns_position_map: Dict[Any, Deque[Any]],
        tables_position_map: Dict[Any, Deque[Any]],
        grammar_constrained_inference: bool,
    ):
        device = next(self.parameters()).device
        # we have to make copies of the builders here so that the additions of the mask actions
        # are confined to the for loop:
        decoder_batch = duo_rat_decoder_batch(
            items=[
                hypothesis.beam_builder.add_action_token(
                    action_token=ActionToken(
                        key=MaskAction(),
                        value=MaskAction(),
                        scope=AttentionScope(scope_name=AttentionScopeName.TARGET),
                    ),
                    copy=True,
                ).build(device=device)
                for hypothesis in beam_hypotheses
            ]
        )
        expanded_memory = memory.expand(len(beam_hypotheses), -1, -1)
        output = self._decode(memory=expanded_memory, batch=decoder_batch)
        p_copy_gen_logprobs = self.copy_logprob(output)
        (batch_size, seq_len) = decoder_batch.target.shape
        assert p_copy_gen_logprobs.shape == (batch_size, seq_len, 2)
        assert not torch.isnan(p_copy_gen_logprobs).any()
        copy_logits = self.pointer_network(query=output, keys=expanded_memory)
        gen_logits = self.out_proj(output)
        # For each hypothesis, record all possible continuations
        continuations = []
        for hypothesis_id, hypothesis in enumerate(beam_hypotheses):
            assert isinstance(hypothesis.beam_builder.parsing_result, Partial)
            continuations += self.get_hyp_continuations(
                decoder_batch=decoder_batch,
                copy_logits=copy_logits,
                gen_logits=gen_logits,
                p_copy_gen_logprobs=p_copy_gen_logprobs,
                hypothesis_id=hypothesis_id,
                hypothesis=hypothesis,
                step=step,
                question_position_map=question_position_map,
                columns_position_map=columns_position_map,
                tables_position_map=tables_position_map,
                grammar_constrained_inference=grammar_constrained_inference,
            )
        return continuations

    def get_hyp_continuations(
        self,
        decoder_batch: DuoRATDecoderBatch,
        copy_logits: torch.Tensor,
        gen_logits: torch.Tensor,
        p_copy_gen_logprobs: torch.Tensor,
        hypothesis_id: int,
        hypothesis: DuoRATHypothesis,
        step: int,
        question_position_map: Dict[Any, Deque[Any]],
        columns_position_map: Dict[Any, Deque[Any]],
        tables_position_map: Dict[Any, Deque[Any]],
        grammar_constrained_inference: bool,
    ):
        device = next(self.parameters()).device
        assert isinstance(hypothesis.beam_builder.parsing_result, Partial)
        continuations = []
        # Copy continuations
        if (
            hypothesis.beam_builder.parsing_result.frontier_field is None
            or not isinstance(
                hypothesis.beam_builder.parsing_result.frontier_field.type,
                ASDLPrimitiveType,
            )
        ):
            pass
        else:
            if grammar_constrained_inference:
                masked_copy_logits = copy_logits.masked_fill(
                    mask=~decoder_batch.valid_copy_mask.to(device=device),
                    value=float("-inf"),
                )
            else:
                masked_copy_logits = copy_logits
            copy_log_probs = F.log_softmax(masked_copy_logits, dim=2)
            for position_map in [
                question_position_map,
                columns_position_map,
                tables_position_map,
            ]:
                for token_value, positions in position_map.items():
                    if (
                        any(decoder_batch.valid_copy_mask[hypothesis_id, -1, positions])
                        or not grammar_constrained_inference
                    ):
                        score = (
                            torch.logsumexp(
                                copy_log_probs[hypothesis_id, step, positions], dim=0,
                            )
                            + p_copy_gen_logprobs[hypothesis_id, step, 0]
                        )
                        continuations.append(
                            Candidate(
                                token=self.preproc.transition_system.get_gen_token_action(
                                    primitive_type=hypothesis.beam_builder.parsing_result.frontier_field.type
                                )(
                                    token=token_value
                                ),
                                score=hypothesis.score + score.item(),
                                prev_hypothesis=hypothesis,
                            )
                        )
        # Vocab continuations
        if grammar_constrained_inference:
            masked_gen_logits = gen_logits.masked_fill(
                mask=~decoder_batch.valid_actions_mask.to(device=device),
                value=float("-inf"),
            )
        else:
            masked_gen_logits = gen_logits
        gen_log_probs = F.log_softmax(masked_gen_logits, dim=2)
        action_ids = (
            decoder_batch.valid_actions_mask[hypothesis_id, -1].nonzero(as_tuple=False)
            if grammar_constrained_inference
            else range(decoder_batch.valid_actions_mask.shape[2])
        )
        for valid_action_id in action_ids:
            # Never continue with a MaskAction.
            if self.preproc.target_vocab.itos[valid_action_id] == MaskAction():
                continue
            score = (
                gen_log_probs[hypothesis_id, step, valid_action_id]
                + p_copy_gen_logprobs[hypothesis_id, step, 1]
            )
            continuations.append(
                Candidate(
                    token=self.preproc.target_vocab.itos[valid_action_id],
                    score=hypothesis.score + score.item(),
                    prev_hypothesis=hypothesis,
                )
            )
        return continuations
