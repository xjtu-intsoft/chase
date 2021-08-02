from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Iterable

import torch
from torchtext.vocab import Vocab

from duorat.asdl.action_info import ActionInfo
from duorat.asdl.asdl import ASDLGrammar, ASDLPrimitiveType
from duorat.asdl.transition_system import (
    ReduceAction,
    GenTokenAction,
    TransitionSystem,
)
from duorat.types import (
    KT,
    VT,
    KT_P,
    Token,
    QuestionToken,
    ColumnToken,
    Sparse2DMaskTensorBuilder,
    TableToken,
)


@dataclass
class ValidActionsMaskBuilder(object):
    question_tokens: Sequence[Token[KT, str]]
    target_vocab: Vocab
    transition_system: TransitionSystem
    allow_unk: bool
    previous_action_info: Optional[ActionInfo] = None
    sparse_2d_mask_tensor_builder: Sparse2DMaskTensorBuilder = field(
        default_factory=lambda: Sparse2DMaskTensorBuilder()
    )

    def __deepcopy__(self, memo):
        builder = copy(self)
        builder.previous_action_info = copy(self.previous_action_info)
        builder.sparse_2d_mask_tensor_builder = deepcopy(
            self.sparse_2d_mask_tensor_builder
        )
        return builder

    def add_token(
        self, token: Token[KT, ActionInfo], copy: bool = False
    ) -> "ValidActionsMaskBuilder":
        builder = deepcopy(self) if copy is True else self

        can_be_copied = isinstance(
            token.value.action, GenTokenAction
        ) and token.value.action.token in [
            token.value for token in builder.question_tokens
        ]

        for _index in map(
            lambda action_index: (token.position, action_index[1]),
            filter(
                lambda action_index: self.transition_system.valid_action_predicate(
                    action=action_index[0],
                    previous_action=(
                        builder.previous_action_info.action
                        if builder.previous_action_info
                        else None
                    ),
                    frontier_field=token.value.frontier_field,
                    allow_unk=(
                        False if can_be_copied else builder.allow_unk
                    ),  # UNK not allowed if the target can be copied from the question
                ),
                builder.target_vocab.stoi.items(),
            ),
        ):
            builder.sparse_2d_mask_tensor_builder.append(index=_index)

        builder.sparse_2d_mask_tensor_builder.resize(
            size=(1 + token.position, len(builder.target_vocab))
        )

        builder.previous_action_info = token.value

        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, ActionInfo]], copy: bool = False
    ) -> "ValidActionsMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_mask_tensor_builder.build(device=device)


class _ValidMaskBuilder(object):
    source_tokens: Sequence[Token[KT, str]]
    sparse_2d_mask_tensor_builder: Sparse2DMaskTensorBuilder
    source_group_token_max_position: int = field(init=False)

    def __deepcopy__(self, memo) -> "_ValidMaskBuilder":
        builder = copy(self)
        builder.sparse_2d_mask_tensor_builder = deepcopy(
            self.sparse_2d_mask_tensor_builder
        )
        return builder

    def __post_init__(self):
        self.source_group_token_max_position = max(
            map(lambda _token: _token.position, self.source_tokens)
        )

    @staticmethod
    def _predicate(
        source_token: Token[KT, str], action_info_token: Token[KT_P, ActionInfo]
    ) -> bool:
        raise NotImplementedError

    def _valid_copy_source_tokens(
        self, action_info: ActionInfo
    ) -> Iterator[Token[KT, VT]]:
        if (
            action_info.frontier_field is None
            or isinstance(action_info.action, ReduceAction)
            or not isinstance(action_info.frontier_field.type, ASDLPrimitiveType)
            or action_info.action is None
        ):
            return iter(tuple())
        else:
            assert isinstance(action_info.frontier_field.type, ASDLPrimitiveType)
            if action_info.frontier_field.type.name == "column":
                # position of all the columns in the memory
                return (
                    token
                    for token in self.source_tokens
                    if isinstance(token, ColumnToken)
                )
            elif action_info.frontier_field.type.name == "table":
                # position of all the tables in the memory
                return (
                    token
                    for token in self.source_tokens
                    if isinstance(token, TableToken)
                )
            else:
                # position of question tokens in the memory
                return (
                    token
                    for token in self.source_tokens
                    if isinstance(token, QuestionToken)
                )

    def add_token(
        self, token: Token[KT_P, ActionInfo], copy: bool = False
    ) -> "_ValidMaskBuilder":
        builder = deepcopy(self) if copy is True else self

        for _index in (
            (token.position, source_token.position)
            for source_token in builder._valid_copy_source_tokens(
                action_info=token.value
            )
            if builder._predicate(source_token=source_token, action_info_token=token)
        ):
            builder.sparse_2d_mask_tensor_builder.append(index=_index,)

        builder.sparse_2d_mask_tensor_builder.resize(
            size=(1 + token.position, 1 + builder.source_group_token_max_position)
        )

        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT_P, ActionInfo]], copy: bool = False
    ) -> "_ValidMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_mask_tensor_builder.build(device=device)


@dataclass
class ValidCopyMaskBuilder(_ValidMaskBuilder):
    source_tokens: Sequence[Token[KT, str]]
    sparse_2d_mask_tensor_builder: Sparse2DMaskTensorBuilder = field(
        default_factory=lambda: Sparse2DMaskTensorBuilder()
    )
    _source_group_token_max_position: Optional[int] = None

    @staticmethod
    def _predicate(
        source_token: Token[KT, str], action_info_token: Token[KT_P, ActionInfo]
    ) -> bool:
        return True


@dataclass
class CopyTargetMaskBuilder(_ValidMaskBuilder):
    source_tokens: Sequence[Token[KT, str]]
    sparse_2d_mask_tensor_builder: Sparse2DMaskTensorBuilder = field(
        default_factory=lambda: Sparse2DMaskTensorBuilder()
    )
    _source_group_token_max_position: Optional[int] = None

    @staticmethod
    def _predicate(
        source_token: Token[KT, str], action_info_token: Token[KT_P, ActionInfo]
    ) -> bool:
        return (
            isinstance(action_info_token.value.action, GenTokenAction)
            and action_info_token.value.action.token == source_token.value
        )


def index_frontier_field(action_info: ActionInfo, grammar: ASDLGrammar) -> int:
    return (
        grammar.field2id[action_info.frontier_field]
        if action_info.frontier_field is not None
        else len(grammar.fields)
    )


def index_frontier_field_type(action_info: ActionInfo, grammar: ASDLGrammar) -> int:
    return (
        grammar.type2id[action_info.frontier_field.type]
        if action_info.frontier_field is not None
        else grammar.type2id[grammar.root_type]  # type for root node
    )
