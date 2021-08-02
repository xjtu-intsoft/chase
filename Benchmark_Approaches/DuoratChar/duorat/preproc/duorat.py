import itertools
import logging
from collections import defaultdict, deque
from copy import deepcopy, copy
from dataclasses import dataclass, replace, field
from typing import (
    Tuple,
    Callable,
    Optional,
    Sequence,
    Iterable,
    Dict,
    Type,
    Deque,
)

import torch
from torchtext.vocab import Vocab

from duorat.asdl.action_info import ActionInfo
from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.asdl.transition_system import (
    Pos,
    Partial,
    Result,
    Done,
    MaskAction,
    Action,
    TransitionSystem,
)
from duorat.preproc.memory import (
    MemoryAttentionMaskBuilder,
    MemoryKeyPaddingMaskBuilder,
)
from duorat.preproc.relations import (
    SourceRelation,
    TargetRelation,
    MemoryRelation,
    TargetRelationsBuilder,
    MemoryRelationsBuilder,
    SourceRelationsBuilder,
)
from duorat.preproc.target import (
    ValidActionsMaskBuilder,
    ValidCopyMaskBuilder,
    CopyTargetMaskBuilder,
    index_frontier_field,
    index_frontier_field_type,
)
from duorat.preproc.tokens import (
    question_input_tokens,
    TokenTensorBuilder,
    KeyPaddingMaskBuilder,
    AttentionMaskBuilder,
    TokenTypeIdTensorBuilder,
    PositionIdTensorBuilder,
    question_source_tokens,
    action_tokens,
    schema_input_tokens,
    schema_source_tokens,
)
from duorat.preproc.utils import pad_nd_tensor
from duorat.types import (
    Token,
    VT,
    VT_P,
    FrozenDict,
    DuoRATItem,
    RATPreprocItem,
    InputId,
    SQLSchema,
    DuoRATEncoderItem,
    ActionInfoToken,
    QuestionToken,
    DuoRATDecoderItem,
    DuoRATEncoderBatch,
    DuoRATDecoderBatch,
    Sparse1DTensorBuilder,
    DuoRATInputSegmentBatch,
    DuoRATInputSegment,
    ActionToken,
    Sparse1DMaskTensorBuilder,
    FineScoping,
    AttentionScope,
    Scoping,
)
from duorat.utils.beam_search import Hypothesis

logger = logging.getLogger(__name__)


@dataclass
class InputToSourceGatherIndexBuilder(object):
    input_token_map: Dict[Tuple[Type[Token], InputId], Token[InputId, VT]] = field(
        default_factory=dict
    )
    source_token_map: Dict[Tuple[Type[Token], InputId], Token[InputId, VT_P]] = field(
        default_factory=dict
    )
    sparse_1d_tensor_builder: Sparse1DTensorBuilder = field(
        default_factory=lambda: Sparse1DTensorBuilder()
    )
    sparse_1d_mask_tensor_builder: Sparse1DMaskTensorBuilder = field(
        default_factory=lambda: Sparse1DMaskTensorBuilder()
    )

    def add_input_token(
        self, input_token: Token[InputId, VT], copy: bool = False
    ) -> "InputToSourceGatherIndexBuilder":
        builder = deepcopy(self) if copy is True else self
        if (type(input_token), input_token.key) not in builder.input_token_map:
            builder.input_token_map[(type(input_token), input_token.key)] = input_token
            if (type(input_token), input_token.key) in builder.source_token_map:
                builder.sparse_1d_tensor_builder.append(
                    index=builder.source_token_map[
                        (type(input_token), input_token.key)
                    ].position,
                    value=input_token.position,
                )
                builder.sparse_1d_mask_tensor_builder.append(
                    index=builder.source_token_map[
                        (type(input_token), input_token.key)
                    ].position
                )
        return builder

    def add_input_tokens(
        self, input_tokens: Iterable[Token[InputId, VT]], copy: bool = False
    ) -> "InputToSourceGatherIndexBuilder":
        builder = deepcopy(self) if copy is True else self
        for input_token in input_tokens:
            builder.add_input_token(input_token=input_token)
        return builder

    def add_source_token(
        self, source_token: Token[InputId, VT_P], copy: bool = False
    ) -> "InputToSourceGatherIndexBuilder":
        builder = deepcopy(self) if copy is True else self
        if (type(source_token), source_token.key) not in builder.source_token_map:
            builder.source_token_map[
                (type(source_token), source_token.key)
            ] = source_token
            if (type(source_token), source_token.key) in builder.input_token_map:
                builder.sparse_1d_tensor_builder.append(
                    index=source_token.position,
                    value=builder.input_token_map[
                        (type(source_token), source_token.key)
                    ].position,
                )
                builder.sparse_1d_mask_tensor_builder.append(
                    index=source_token.position,
                )
        self.sparse_1d_tensor_builder.resize(size=1 + source_token.position)
        self.sparse_1d_mask_tensor_builder.resize(size=1 + source_token.position)
        return builder

    def add_source_tokens(
        self, source_tokens: Iterable[Token[InputId, VT_P]], copy: bool = False
    ) -> "InputToSourceGatherIndexBuilder":
        builder = deepcopy(self) if copy is True else self
        for source_token in source_tokens:
            builder.add_source_token(source_token=source_token)
        return builder

    def build(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.sparse_1d_tensor_builder.build(device=device),
            self.sparse_1d_mask_tensor_builder.build(device=device),
        )


@dataclass
class DuoRATInputSegmentBuilder(object):
    input_a_str_to_id: Callable[[str], int]
    input_b_str_to_id: Callable[[str], int]
    input_attention_scoping: Scoping
    input_token_max_position_pointer: int = 0
    input_a_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    input_b_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    input_key_padding_mask_builder: KeyPaddingMaskBuilder = field(
        default_factory=lambda: KeyPaddingMaskBuilder()
    )
    input_attention_mask_builder: AttentionMaskBuilder = field(init=False)
    input_token_type_ids_builder: TokenTypeIdTensorBuilder = field(
        default_factory=lambda: TokenTypeIdTensorBuilder()
    )
    input_position_ids_builder: PositionIdTensorBuilder = field(
        default_factory=lambda: PositionIdTensorBuilder()
    )
    input_to_source_gather_index_builder: InputToSourceGatherIndexBuilder = field(
        default_factory=lambda: InputToSourceGatherIndexBuilder()
    )

    def __post_init__(self):
        self.input_attention_mask_builder = AttentionMaskBuilder(
            scoping=self.input_attention_scoping
        )

    def add_input_token(
        self, input_token: Token[InputId, str], copy: bool = False
    ) -> "DuoRATInputSegmentBuilder":
        builder = deepcopy(self) if copy is True else self
        positioned_input_token = replace(
            input_token,
            position=Pos(
                builder.input_token_max_position_pointer + input_token.position
            ),
        )
        builder.input_token_max_position_pointer = (
            builder.input_token_max_position_pointer + input_token.position + 1
        )
        builder.input_a_builder.add_token(
            token=replace(
                positioned_input_token,
                value=self.input_a_str_to_id(positioned_input_token.value),
            )
        )
        builder.input_b_builder.add_token(
            token=replace(
                positioned_input_token,
                value=self.input_b_str_to_id(positioned_input_token.value),
            )
        )
        builder.input_key_padding_mask_builder.add_token(token=positioned_input_token)
        builder.input_attention_mask_builder.add_token(token=positioned_input_token)
        builder.input_token_type_ids_builder.add_token(token=positioned_input_token)
        builder.input_position_ids_builder.add_token(token=positioned_input_token)
        builder.input_to_source_gather_index_builder.add_input_token(
            input_token=positioned_input_token
        )
        return builder

    def add_input_tokens(
        self, input_tokens: Iterable[Token[InputId, str]], copy: bool = False
    ) -> "DuoRATInputSegmentBuilder":
        builder = deepcopy(self) if copy is True else self
        for input_token in input_tokens:
            builder.add_input_token(input_token=input_token)
        return builder

    def add_positioned_source_token(
        self, positioned_source_token: Token[InputId, str], copy: bool = False
    ) -> "DuoRATInputSegmentBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.input_to_source_gather_index_builder.add_source_token(
            source_token=positioned_source_token
        )
        return builder

    def add_positioned_source_tokens(
        self,
        positioned_source_tokens: Iterable[Token[InputId, str]],
        copy: bool = False,
    ) -> "DuoRATInputSegmentBuilder":
        builder = deepcopy(self) if copy is True else self
        for positioned_source_token in positioned_source_tokens:
            builder.add_positioned_source_token(
                positioned_source_token=positioned_source_token
            )
        return builder

    def build(self, device: torch.device) -> DuoRATInputSegment:
        (
            input_source_gather_index,
            input_source_gather_index_mask,
        ) = self.input_to_source_gather_index_builder.build(device)
        return DuoRATInputSegment(
            input_a=self.input_a_builder.build(device=device),
            input_b=self.input_b_builder.build(device=device),
            input_attention_mask=self.input_attention_mask_builder.build(device=device),
            input_key_padding_mask=self.input_key_padding_mask_builder.build(
                device=device
            ),
            input_token_type_ids=self.input_token_type_ids_builder.build(device=device),
            input_position_ids=self.input_position_ids_builder.build(device=device),
            input_source_gather_index=input_source_gather_index,
            input_source_gather_index_mask=input_source_gather_index_mask,
        )


@dataclass
class DuoRATEncoderItemBuilder(object):
    input_a_str_to_id: Callable[[str], int]
    input_b_str_to_id: Callable[[str], int]
    sql_schema: SQLSchema
    input_attention_scoping: Scoping
    source_attention_scoping: Scoping
    source_relation_types: FrozenDict[SourceRelation, int]
    max_supported_input_length: Optional[int] = None
    input_token_max_position_pointer: int = 0
    input_a_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    input_b_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    input_key_padding_mask_builder: KeyPaddingMaskBuilder = field(
        default_factory=lambda: KeyPaddingMaskBuilder()
    )
    input_attention_mask_builder: AttentionMaskBuilder = field(init=False)
    input_token_type_ids_builder: TokenTypeIdTensorBuilder = field(
        default_factory=lambda: TokenTypeIdTensorBuilder()
    )
    input_position_ids_builder: PositionIdTensorBuilder = field(
        default_factory=lambda: PositionIdTensorBuilder()
    )
    input_segment_builders: Dict[AttentionScope, DuoRATInputSegmentBuilder] = field(
        init=False
    )
    positioned_source_tokens: Deque[Token[InputId, str]] = field(default_factory=deque)
    source_token_max_position_pointer: int = 0
    input_to_source_gather_index_builder: InputToSourceGatherIndexBuilder = field(
        default_factory=lambda: InputToSourceGatherIndexBuilder()
    )
    source_key_padding_mask_builder: KeyPaddingMaskBuilder = field(
        default_factory=lambda: KeyPaddingMaskBuilder()
    )
    source_attention_mask_builder: AttentionMaskBuilder = field(init=False)
    source_relations_builder: SourceRelationsBuilder = field(init=False)

    def __post_init__(self):
        self.input_segment_builders = defaultdict(
            lambda: DuoRATInputSegmentBuilder(
                input_a_str_to_id=self.input_a_str_to_id,
                input_b_str_to_id=self.input_b_str_to_id,
                input_attention_scoping=self.input_attention_scoping,
            )
        )
        self.input_attention_mask_builder = AttentionMaskBuilder(
            scoping=self.input_attention_scoping
        )
        self.source_attention_mask_builder = AttentionMaskBuilder(
            scoping=self.source_attention_scoping
        )
        self.source_relations_builder = SourceRelationsBuilder(
            sql_schema=self.sql_schema, relation_types=self.source_relation_types,
        )

    def add_input_token(
        self, input_token: Token[InputId, str], copy: bool = False
    ) -> "DuoRATEncoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.input_segment_builders[input_token.scope].add_input_token(
            input_token=input_token
        )
        if (
            builder.max_supported_input_length is not None
            and builder.input_token_max_position_pointer + input_token.position + 1
            > builder.max_supported_input_length
        ):
            logger.warning(
                "input token tensor has been truncated to {} tokens, "
                "original length was {} tokens".format(
                    builder.max_supported_input_length,
                    builder.input_token_max_position_pointer + input_token.position + 1,
                )
            )
            return builder
        else:
            positioned_input_token = replace(
                input_token,
                position=Pos(
                    builder.input_token_max_position_pointer + input_token.position
                ),
            )
            builder.input_token_max_position_pointer = (
                builder.input_token_max_position_pointer + input_token.position + 1
            )
            builder.input_a_builder.add_token(
                token=replace(
                    positioned_input_token,
                    value=self.input_a_str_to_id(positioned_input_token.value),
                )
            )
            builder.input_b_builder.add_token(
                token=replace(
                    positioned_input_token,
                    value=self.input_b_str_to_id(positioned_input_token.value),
                )
            )
            builder.input_key_padding_mask_builder.add_token(
                token=positioned_input_token
            )
            builder.input_attention_mask_builder.add_token(token=positioned_input_token)
            builder.input_token_type_ids_builder.add_token(token=positioned_input_token)
            builder.input_position_ids_builder.add_token(token=positioned_input_token)
            builder.input_to_source_gather_index_builder.add_input_token(
                input_token=positioned_input_token
            )
            builder.source_relations_builder.add_input_token(
                input_token=positioned_input_token
            )
        return builder

    def add_input_tokens(
        self, input_tokens: Iterable[Token[InputId, str]], copy: bool = False
    ) -> "DuoRATEncoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        input_tokens = list(input_tokens)

        if (
            builder.max_supported_input_length is not None
            and builder.input_token_max_position_pointer + len(input_tokens)
            > builder.max_supported_input_length
        ):
            logger.warning(
                "input token tensor has been truncated to {} tokens, "
                "original length was {} tokens".format(
                    builder.max_supported_input_length,
                    builder.input_token_max_position_pointer + len(input_tokens),
                )
            )
            input_tokens = input_tokens[:(builder.max_supported_input_length - builder.input_token_max_position_pointer)]

        for input_token in input_tokens:
            builder.add_input_token(input_token=input_token)
        return builder

    def add_source_token(
        self, source_token: Token[InputId, str], copy: bool = False
    ) -> "DuoRATEncoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        positioned_source_token = replace(
            source_token,
            position=Pos(
                builder.source_token_max_position_pointer + source_token.position
            ),
        )
        builder.positioned_source_tokens.append(positioned_source_token)
        builder.source_token_max_position_pointer = (
            builder.source_token_max_position_pointer + source_token.position + 1
        )
        for _, input_segment_builder in builder.input_segment_builders.items():
            input_segment_builder.add_positioned_source_token(
                positioned_source_token=positioned_source_token
            )
        builder.input_to_source_gather_index_builder.add_source_token(
            source_token=positioned_source_token
        )
        builder.source_key_padding_mask_builder.add_token(token=positioned_source_token)
        builder.source_attention_mask_builder.add_token(token=positioned_source_token)
        builder.source_relations_builder.add_source_token(
            source_token=positioned_source_token
        )
        return builder

    def add_source_tokens(
        self, source_tokens: Iterable[Token[InputId, str]], copy: bool = False
    ) -> "DuoRATEncoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        for source_token in source_tokens:
            builder.add_source_token(source_token=source_token)
        return builder

    def build(self, device: torch.device) -> DuoRATEncoderItem:
        input_source_gather_index, _ = self.input_to_source_gather_index_builder.build(
            device=device
        )
        return DuoRATEncoderItem(
            input_a=self.input_a_builder.build(device=device),
            input_b=self.input_b_builder.build(device=device),
            input_attention_mask=self.input_attention_mask_builder.build(device=device),
            input_key_padding_mask=self.input_key_padding_mask_builder.build(
                device=device
            ),
            input_token_type_ids=self.input_token_type_ids_builder.build(device=device),
            input_position_ids=self.input_position_ids_builder.build(device=device),
            input_segments=tuple(
                input_segment_builder.build(device=device)
                for _, input_segment_builder in self.input_segment_builders.items()
            ),
            input_source_gather_index=input_source_gather_index,
            source_relations=self.source_relations_builder.build(device=device),
            source_attention_mask=self.source_attention_mask_builder.build(
                device=device
            ),
            source_key_padding_mask=self.source_key_padding_mask_builder.build(
                device=device
            ),
        )


@dataclass
class DuoRATDecoderItemBuilder(object):
    positioned_source_tokens: Sequence[Token[InputId, str]]
    target_vocab: Vocab
    transition_system: TransitionSystem
    allow_unk: bool
    source_attention_scoping: Scoping
    target_attention_scoping: Scoping
    target_relation_types: FrozenDict[TargetRelation, int]
    memory_relation_types: FrozenDict[MemoryRelation, int]
    action_token_max_position_pointer: int = 0
    parsing_result: Result[AbstractSyntaxTree] = field(init=False)
    target_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    frontier_fields_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    frontier_field_types_builder: TokenTensorBuilder = field(
        default_factory=lambda: TokenTensorBuilder()
    )
    target_attention_mask_builder: AttentionMaskBuilder = field(init=False)
    target_key_padding_mask_builder: KeyPaddingMaskBuilder = field(
        default_factory=lambda: KeyPaddingMaskBuilder()
    )
    memory_attention_mask_builder: MemoryAttentionMaskBuilder = field(init=False)
    memory_key_padding_mask_builder: MemoryKeyPaddingMaskBuilder = field(init=False)
    valid_copy_mask_builder: ValidCopyMaskBuilder = field(init=False)
    copy_target_mask_builder: CopyTargetMaskBuilder = field(init=False)
    valid_actions_mask_builder: ValidActionsMaskBuilder = field(init=False)
    target_relations_builder: TargetRelationsBuilder = field(init=False)
    memory_relations_builder: MemoryRelationsBuilder = field(init=False)

    def __deepcopy__(self, memo) -> "DuoRATDecoderItemBuilder":
        builder = copy(self)
        builder.target_builder = deepcopy(self.target_builder)
        builder.frontier_fields_builder = deepcopy(self.frontier_fields_builder)
        builder.frontier_field_types_builder = deepcopy(
            self.frontier_field_types_builder
        )
        builder.target_attention_mask_builder = deepcopy(
            self.target_attention_mask_builder
        )
        builder.target_key_padding_mask_builder = deepcopy(
            self.target_key_padding_mask_builder
        )
        builder.memory_attention_mask_builder = deepcopy(
            self.memory_attention_mask_builder
        )
        builder.memory_key_padding_mask_builder = deepcopy(
            self.memory_key_padding_mask_builder
        )
        builder.valid_copy_mask_builder = deepcopy(self.valid_copy_mask_builder)
        builder.copy_target_mask_builder = deepcopy(self.copy_target_mask_builder)
        builder.valid_actions_mask_builder = deepcopy(self.valid_actions_mask_builder)
        builder.target_relations_builder = deepcopy(self.target_relations_builder)
        builder.memory_relations_builder = deepcopy(self.memory_relations_builder)
        return builder

    def __post_init__(self):
        self.parsing_result = self.transition_system.parse()
        self.target_attention_mask_builder = AttentionMaskBuilder(
            scoping=self.target_attention_scoping
        )
        self.memory_attention_mask_builder = MemoryAttentionMaskBuilder(
            source_scoping=self.source_attention_scoping,
            target_scoping=self.target_attention_scoping,
        )
        self.memory_attention_mask_builder.add_source_tokens(
            source_tokens=self.positioned_source_tokens
        )
        self.memory_key_padding_mask_builder = MemoryKeyPaddingMaskBuilder(
            source_scoping=self.source_attention_scoping
        )
        self.memory_key_padding_mask_builder.add_source_tokens(
            source_tokens=self.positioned_source_tokens
        )
        self.valid_copy_mask_builder = ValidCopyMaskBuilder(
            source_tokens=self.positioned_source_tokens
        )
        self.copy_target_mask_builder = CopyTargetMaskBuilder(
            source_tokens=self.positioned_source_tokens
        )
        self.valid_actions_mask_builder = ValidActionsMaskBuilder(
            question_tokens=[
                token
                for token in self.positioned_source_tokens
                if isinstance(token, QuestionToken)
            ],
            target_vocab=self.target_vocab,
            transition_system=self.transition_system,
            allow_unk=self.allow_unk,
        )
        self.target_relations_builder = TargetRelationsBuilder(
            relation_types=self.target_relation_types
        )
        self.memory_relations_builder = MemoryRelationsBuilder(
            source_tokens=self.positioned_source_tokens,
            relation_types=self.memory_relation_types,
        )

    def add_action_token(
        self, action_token: ActionToken[Action], copy: bool = False,
    ) -> "DuoRATDecoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        if isinstance(builder.parsing_result, Done):
            raise ValueError("A complete action sequence cannot be continued")
        elif isinstance(builder.parsing_result, Partial):
            positioned_action_token = replace(
                action_token,
                position=Pos(
                    builder.action_token_max_position_pointer + action_token.position
                ),
            )
            builder.action_token_max_position_pointer = (
                builder.action_token_max_position_pointer + action_token.position + 1
            )
            action_info = ActionInfo(
                action=positioned_action_token.value,
                parent_pos=builder.parsing_result.parent_pos,
                frontier_field=builder.parsing_result.frontier_field,
            )
            # Don't try to parse a mask action
            if positioned_action_token.value != MaskAction():
                builder.parsing_result = builder.parsing_result.cont(
                    positioned_action_token.position, positioned_action_token.value
                )
            positioned_action_info_token = ActionInfoToken(
                key=action_info,
                value=action_info,
                position=positioned_action_token.position,
                scope=positioned_action_token.scope,
            )
            builder._add_positioned_action_info_token(
                positioned_action_info_token=positioned_action_info_token
            )
            return builder
        else:
            raise ValueError("Invalid parsing state: {}".format(builder.parsing_result))

    def add_action_tokens(
        self, action_tokens: Iterable[ActionToken[Action]], copy: bool = False
    ) -> "DuoRATDecoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        for action_token in action_tokens:
            builder.add_action_token(action_token=action_token)
        return builder

    def _add_positioned_action_info_token(
        self,
        positioned_action_info_token: ActionInfoToken[ActionInfo],
        copy: bool = False,
    ) -> "DuoRATDecoderItemBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.target_builder.add_token(
            token=replace(
                positioned_action_info_token,
                value=builder.target_vocab[positioned_action_info_token.value.action],
            )
        )
        builder.frontier_fields_builder.add_token(
            token=replace(
                positioned_action_info_token,
                value=index_frontier_field(
                    action_info=positioned_action_info_token.value,
                    grammar=builder.transition_system.grammar,
                ),
            )
        )
        builder.frontier_field_types_builder.add_token(
            token=replace(
                positioned_action_info_token,
                value=index_frontier_field_type(
                    action_info=positioned_action_info_token.value,
                    grammar=builder.transition_system.grammar,
                ),
            )
        )
        builder.target_attention_mask_builder.add_token(
            token=positioned_action_info_token
        )
        builder.target_key_padding_mask_builder.add_token(
            token=positioned_action_info_token
        )
        builder.memory_attention_mask_builder.add_target_token(
            target_token=positioned_action_info_token
        )
        builder.valid_copy_mask_builder.add_token(token=positioned_action_info_token)
        builder.copy_target_mask_builder.add_token(token=positioned_action_info_token)
        builder.valid_actions_mask_builder.add_token(token=positioned_action_info_token)
        builder.target_relations_builder.add_token(token=positioned_action_info_token)
        builder.memory_relations_builder.add_token(token=positioned_action_info_token)
        return builder

    def build(self, device: torch.device) -> DuoRATDecoderItem:
        target = self.target_builder.build(device=device)
        shifted_target = torch.cat(
            (
                target.new_full(size=(1,), fill_value=self.target_vocab[MaskAction()]),
                target[:-1],
            ),
            dim=0,
        )
        memory_relations = self.memory_relations_builder.build(device=device)
        shifted_memory_relations = torch.cat(
            (
                memory_relations.new_full(
                    size=(1, memory_relations.shape[1]), fill_value=0
                ),
                memory_relations[:-1],
            ),
            dim=0,
        )
        return DuoRATDecoderItem(
            masked_target=target,
            shifted_target=shifted_target,
            frontier_fields=self.frontier_fields_builder.build(device=device),
            frontier_field_types=self.frontier_field_types_builder.build(device=device),
            target_relations=self.target_relations_builder.build(device=device),
            target_attention_mask=self.target_attention_mask_builder.build(
                device=device
            ),
            target_key_padding_mask=self.target_key_padding_mask_builder.build(
                device=device
            ),
            memory_relations=memory_relations,
            shifted_memory_relations=shifted_memory_relations,
            memory_attention_mask=self.memory_attention_mask_builder.build(
                device=device
            ),
            memory_key_padding_mask=self.memory_key_padding_mask_builder.build(
                device=device
            ),
            valid_copy_mask=self.valid_copy_mask_builder.build(device=device),
            copy_target_mask=self.copy_target_mask_builder.build(device=device),
            valid_actions_mask=self.valid_actions_mask_builder.build(device=device),
            target=target,
        )


class DuoRATHypothesis(Hypothesis[DuoRATDecoderItemBuilder]):
    def is_finished(self) -> bool:
        return isinstance(self.beam_builder.parsing_result, Done)


def duo_rat_encoder_item(
    preproc_item: RATPreprocItem,
    input_a_str_to_id: Callable[[str], int],
    input_b_str_to_id: Callable[[str], int],
    max_supported_input_length: Optional[int],
    input_attention_scoping: Scoping,
    source_attention_scoping: Scoping,
    source_relation_types: FrozenDict[SourceRelation, int],
    schema_input_token_ordering: str,
    schema_source_token_ordering: str,
    device: torch.device,
) -> Tuple[DuoRATEncoderItem, DuoRATEncoderItemBuilder]:
    encoder_item_builder = DuoRATEncoderItemBuilder(
        input_a_str_to_id=input_a_str_to_id,
        input_b_str_to_id=input_b_str_to_id,
        input_attention_scoping=input_attention_scoping,
        source_attention_scoping=source_attention_scoping,
        sql_schema=preproc_item.sql_schema,
        source_relation_types=source_relation_types,
        max_supported_input_length=max_supported_input_length,
    )
    encoder_item_builder.add_input_tokens(
        input_tokens=itertools.chain(
            question_input_tokens(
                question=preproc_item.question, scoping=input_attention_scoping
            ),
            schema_input_tokens(
                sql_schema=preproc_item.sql_schema,
                schema_token_ordering=schema_input_token_ordering,
                scoping=input_attention_scoping,
            ),
        )
    )
    encoder_item_builder.add_source_tokens(
        source_tokens=itertools.chain(
            question_source_tokens(
                question=preproc_item.question, scoping=source_attention_scoping
            ),
            schema_source_tokens(
                sql_schema=preproc_item.sql_schema,
                schema_token_ordering=schema_source_token_ordering,
                scoping=source_attention_scoping,
            ),
        )
    )

    encoder_item = encoder_item_builder.build(device=device)
    return encoder_item, encoder_item_builder


def duo_rat_decoder_item(
    preproc_item: RATPreprocItem,
    positioned_source_tokens: Sequence[Token[InputId, str]],
    target_vocab: Vocab,
    transition_system: TransitionSystem,
    allow_unk: bool,
    source_attention_scoping: Scoping,
    target_attention_scoping: Scoping,
    target_relation_types: FrozenDict[TargetRelation, int],
    memory_relation_types: FrozenDict[MemoryRelation, int],
    device: torch.device,
) -> Tuple[DuoRATDecoderItem, DuoRATDecoderItemBuilder]:
    decoder_item_builder = DuoRATDecoderItemBuilder(
        positioned_source_tokens=positioned_source_tokens,
        target_vocab=target_vocab,
        transition_system=transition_system,
        allow_unk=allow_unk,
        source_attention_scoping=source_attention_scoping,
        target_attention_scoping=target_attention_scoping,
        target_relation_types=target_relation_types,
        memory_relation_types=memory_relation_types,
    )
    decoder_item_builder.add_action_tokens(
        action_tokens=action_tokens(
            actions=preproc_item.actions, scoping=target_attention_scoping
        )
    )
    decoder_item = decoder_item_builder.build(device=device)
    return decoder_item, decoder_item_builder


def duo_rat_item(
    preproc_item: RATPreprocItem,
    get_encoder_item: Callable[
        [RATPreprocItem], Tuple[DuoRATEncoderItem, DuoRATEncoderItemBuilder]
    ],
    get_decoder_item: Callable[
        [RATPreprocItem, Sequence[Token[InputId, str]]],
        Tuple[DuoRATDecoderItem, DuoRATDecoderItemBuilder],
    ],
) -> DuoRATItem:
    encoder_item, encoder_item_builder = get_encoder_item(preproc_item)
    decoder_item, _ = get_decoder_item(
        preproc_item, encoder_item_builder.positioned_source_tokens
    )
    return DuoRATItem(encoder_item=encoder_item, decoder_item=decoder_item)


def duo_rat_encoder_batch(items: Iterable[DuoRATEncoderItem]) -> DuoRATEncoderBatch:
    def _pad(
        tensors: Sequence[torch.Tensor],
        num_padding_dimensions: int = 1,
        padding_value: int = 0,
    ) -> torch.Tensor:
        return pad_nd_tensor(
            tensors=tensors,
            num_padding_dimensions=num_padding_dimensions,
            batch_first=True,
            padding_value=padding_value,
        )

    def _input_segment(
        input_segments: Iterable[DuoRATInputSegment],
    ) -> DuoRATInputSegmentBatch:
        input_key_padding_mask = _pad(
            tensors=[segment.input_key_padding_mask for segment in input_segments],
        )
        input_attention_mask = torch.masked_fill(
            _pad(
                tensors=[segment.input_attention_mask for segment in input_segments],
                num_padding_dimensions=2,
                padding_value=1,
            ),
            mask=torch.unsqueeze(~input_key_padding_mask, dim=1),
            value=0,
        )

        return DuoRATInputSegmentBatch(
            input_a=_pad(tensors=[segment.input_a for segment in input_segments]),
            input_b=_pad(tensors=[segment.input_b for segment in input_segments]),
            input_attention_mask=input_attention_mask,
            input_key_padding_mask=input_key_padding_mask,
            input_token_type_ids=_pad(
                tensors=[segment.input_token_type_ids for segment in input_segments],
            ),
            input_position_ids=_pad(
                tensors=[segment.input_position_ids for segment in input_segments],
            ),
            input_source_gather_index=_pad(
                tensors=[
                    segment.input_source_gather_index for segment in input_segments
                ],
            ),
            input_source_gather_index_mask=_pad(
                tensors=[
                    segment.input_source_gather_index_mask for segment in input_segments
                ],
            ),
        )

    input_key_padding_mask = _pad(
        tensors=[item.input_key_padding_mask for item in items]
    )
    input_attention_mask = torch.masked_fill(
        _pad(
            tensors=[item.input_attention_mask for item in items],
            num_padding_dimensions=2,
            padding_value=1,
        ),
        mask=torch.unsqueeze(~input_key_padding_mask, dim=1),
        value=0,
    )
    source_key_padding_mask = _pad(
        tensors=[item.source_key_padding_mask for item in items]
    )
    source_attention_mask = torch.masked_fill(
        _pad(
            tensors=[item.source_attention_mask for item in items],
            num_padding_dimensions=2,
            padding_value=1,
        ),
        mask=torch.unsqueeze(~source_key_padding_mask, dim=1),
        value=0,
    )

    return DuoRATEncoderBatch(
        input_a=_pad(tensors=[item.input_a for item in items]),
        input_b=_pad(tensors=[item.input_b for item in items]),
        input_attention_mask=input_attention_mask,
        input_key_padding_mask=input_key_padding_mask,
        input_token_type_ids=_pad(
            tensors=[item.input_token_type_ids for item in items]
        ),
        input_position_ids=_pad(tensors=[item.input_position_ids for item in items]),
        input_segments=tuple(
            _input_segment(input_segments=item.input_segments) for item in items
        ),
        input_source_gather_index=_pad(
            tensors=[item.input_source_gather_index for item in items]
        ),
        source_relations=_pad(
            tensors=[item.source_relations for item in items], num_padding_dimensions=2
        ),
        source_attention_mask=source_attention_mask,
        source_key_padding_mask=source_key_padding_mask,
    )


def duo_rat_decoder_batch(items: Iterable[DuoRATDecoderItem]) -> DuoRATDecoderBatch:
    def _pad(
        getter: Callable[[DuoRATDecoderItem], torch.Tensor],
        num_padding_dimensions: int = 1,
        padding_value: int = 0,
    ) -> torch.Tensor:
        return pad_nd_tensor(
            tensors=[getter(item) for item in items],
            num_padding_dimensions=num_padding_dimensions,
            batch_first=True,
            padding_value=padding_value,
        )

    target_key_padding_mask = _pad(lambda item: item.target_key_padding_mask)
    target_attention_mask = torch.masked_fill(
        _pad(
            lambda item: item.target_attention_mask,
            num_padding_dimensions=2,
            padding_value=1,
        ),
        mask=torch.unsqueeze(~target_key_padding_mask, dim=1),
        value=0,
    )
    memory_key_padding_mask = _pad(lambda item: item.memory_key_padding_mask)
    memory_attention_mask = torch.masked_fill(
        _pad(
            lambda item: item.memory_attention_mask,
            num_padding_dimensions=2,
            padding_value=1,
        ),
        mask=torch.unsqueeze(~memory_key_padding_mask, dim=1),
        value=0,
    )

    return DuoRATDecoderBatch(
        masked_target=_pad(lambda item: item.masked_target),
        shifted_target=_pad(lambda item: item.shifted_target),
        frontier_fields=_pad(lambda item: item.frontier_fields),
        frontier_field_types=_pad(lambda item: item.frontier_field_types),
        target_relations=_pad(
            lambda item: item.target_relations, num_padding_dimensions=2
        ),
        target_attention_mask=target_attention_mask,
        target_key_padding_mask=target_key_padding_mask,
        memory_relations=_pad(
            lambda item: item.memory_relations, num_padding_dimensions=2
        ),
        shifted_memory_relations=_pad(
            lambda item: item.shifted_memory_relations, num_padding_dimensions=2
        ),
        memory_attention_mask=memory_attention_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        valid_copy_mask=_pad(
            lambda item: item.valid_copy_mask, num_padding_dimensions=2
        ),
        copy_target_mask=_pad(
            lambda item: item.copy_target_mask, num_padding_dimensions=2
        ),
        valid_actions_mask=_pad(lambda item: item.valid_actions_mask),
        target=_pad(lambda item: item.target),
    )
