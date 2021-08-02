import itertools
from collections import deque, defaultdict
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import Iterable, Deque, Dict, Set, Tuple, Generator

import torch

from duorat.asdl.transition_system import Pos
from duorat.types import (
    Token,
    KT,
    VT,
    Sparse2DMaskTensorBuilder,
    Sparse1DMaskTensorBuilder,
    Scoping,
    AttentionScopeName,
    NoScoping,
    AttentionScope,
    CoarseScoping,
    FineScoping,
)


@dataclass
class MemoryAttentionMaskBuilder(object):
    source_scoping: Scoping
    target_scoping: Scoping
    source_scope_positions: Dict[AttentionScope, Set[Pos]] = field(init=False)
    target_scope_positions: Dict[AttentionScope, Set[Pos]] = field(init=False)
    sparse_2d_mask_tensor_builder: Sparse2DMaskTensorBuilder = field(
        default_factory=lambda: Sparse2DMaskTensorBuilder()
    )

    def __post_init__(self):
        self.source_scope_positions = defaultdict(set)
        self.target_scope_positions = defaultdict(set)

    def __deepcopy__(self, memo) -> "MemoryAttentionMaskBuilder":
        builder = copy(self)
        builder.source_scope_positions = defaultdict(set)
        for scope, tokens in self.source_scope_positions.items():
            builder.source_scope_positions[scope] = copy(tokens)
        builder.target_scope_positions = defaultdict(set)
        for scope, tokens in self.target_scope_positions.items():
            builder.target_scope_positions[scope] = copy(tokens)
        builder.sparse_2d_mask_tensor_builder = deepcopy(
            self.sparse_2d_mask_tensor_builder
        )
        return builder

    def _get_source_scopes_by_name(
        self, scope_name: AttentionScopeName
    ) -> Set[AttentionScope]:
        return set(
            _scope
            for _scope in self.source_scope_positions.keys()
            if _scope.scope_name == scope_name
        )

    def _get_target_scopes_by_name(
        self, scope_name: AttentionScopeName
    ) -> Set[AttentionScope]:
        return set(
            _scope
            for _scope in self.target_scope_positions.keys()
            if _scope.scope_name == scope_name
        )

    def _scope_connections(self, target_scope: AttentionScope) -> Set[AttentionScope]:
        if not isinstance(self.target_scoping, NoScoping):
            raise NotImplementedError
        if isinstance(self.source_scoping, NoScoping):
            if target_scope.scope_name == AttentionScopeName.TARGET:
                source_scopes = self._get_source_scopes_by_name(
                    scope_name=AttentionScopeName.SOURCE
                )
                return source_scopes
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(target_scope.scope_name)
                )
        elif isinstance(self.source_scoping, CoarseScoping):
            if target_scope.scope_name == AttentionScopeName.TARGET:
                if self.source_scoping.target_sees_question:
                    question_scopes = self._get_source_scopes_by_name(
                        scope_name=AttentionScopeName.QUESTION
                    )
                else:
                    question_scopes = set()
                if self.source_scoping.target_sees_schema:
                    schema_scopes = self._get_source_scopes_by_name(
                        scope_name=AttentionScopeName.SCHEMA
                    )
                else:
                    schema_scopes = set()
                return question_scopes | schema_scopes
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(target_scope.scope_name)
                )
        elif isinstance(self.source_scoping, FineScoping):
            if target_scope.scope_name == AttentionScopeName.TARGET:
                if self.source_scoping.target_sees_question:
                    question_scopes = self._get_source_scopes_by_name(
                        scope_name=AttentionScopeName.QUESTION
                    )
                else:
                    question_scopes = set()
                if self.source_scoping.target_sees_columns:
                    column_scopes = self._get_source_scopes_by_name(
                        scope_name=AttentionScopeName.COLUMN
                    )
                else:
                    column_scopes = set()
                if self.source_scoping.target_sees_tables:
                    table_scopes = self._get_source_scopes_by_name(
                        scope_name=AttentionScopeName.TABLE
                    )
                else:
                    table_scopes = set()
                return question_scopes | column_scopes | table_scopes
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(target_scope.scope_name)
                )
        else:
            raise NotImplementedError

    @staticmethod
    def _make_mask(
        attend_from: Iterable[Pos], attend_to: Iterable[Pos],
    ) -> Generator[Tuple[Pos, Pos], None, None]:
        attend_from_to = (
            (from_pos, to_pos) for from_pos in attend_from for to_pos in attend_to
        )
        return attend_from_to

    def add_source_token(
        self, source_token: Token[KT, VT], copy: bool = False
    ) -> "MemoryAttentionMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.source_scope_positions[source_token.scope].add(source_token.position)
        this_scope = source_token.scope
        attend_self: Set[Pos] = {source_token.position}
        mask: Set[Tuple[Pos, Pos]] = set().union(
            *(
                builder._make_mask(
                    attend_from=builder.target_scope_positions[that_scope],
                    attend_to=attend_self,
                )
                for that_scope in builder.target_scope_positions.keys()
                if this_scope in builder._scope_connections(target_scope=that_scope)
            )
        )
        for _index in mask:
            builder.sparse_2d_mask_tensor_builder.append(index=_index)
        builder.sparse_2d_mask_tensor_builder.resize(
            size=(0, 1 + source_token.position)
        )
        return builder

    def add_source_tokens(
        self, source_tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "MemoryAttentionMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for source_token in source_tokens:
            builder.add_source_token(source_token=source_token)
        return builder

    def add_target_token(
        self, target_token: Token[KT, VT], copy: bool = False
    ) -> "MemoryAttentionMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.target_scope_positions[target_token.scope].add(target_token.position)
        this_scope = target_token.scope
        attend_self: Set[Pos] = {target_token.position}
        mask: Set[Tuple[Pos, Pos]] = set().union(
            *(
                builder._make_mask(
                    attend_from=attend_self,
                    attend_to=builder.source_scope_positions[that_scope],
                )
                for that_scope in builder._scope_connections(target_scope=this_scope)
            )
        )
        for _index in mask:
            builder.sparse_2d_mask_tensor_builder.append(index=_index)
        builder.sparse_2d_mask_tensor_builder.resize(
            size=(1 + target_token.position, 0)
        )
        return builder

    def add_target_tokens(
        self, target_tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "MemoryAttentionMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for target_token in target_tokens:
            builder.add_target_token(target_token=target_token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_mask_tensor_builder.build(device=device)


@dataclass
class MemoryKeyPaddingMaskBuilder(object):
    source_scoping: Scoping
    sparse_1d_mask_tensor_builder: Sparse1DMaskTensorBuilder = field(
        default_factory=lambda: Sparse1DMaskTensorBuilder()
    )

    def add_source_token(
        self, source_token: Token[KT, VT], copy: bool = False
    ) -> "MemoryKeyPaddingMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        if isinstance(self.source_scoping, NoScoping):
            if source_token.scope.scope_name == AttentionScopeName.SOURCE:
                builder.sparse_1d_mask_tensor_builder.append(
                    index=source_token.position
                )
            else:
                builder.sparse_1d_mask_tensor_builder.resize(
                    size=1 + source_token.position
                )
        elif isinstance(self.source_scoping, CoarseScoping):
            if source_token.scope.scope_name in (
                AttentionScopeName.QUESTION,
                AttentionScopeName.SCHEMA,
            ):
                builder.sparse_1d_mask_tensor_builder.append(
                    index=source_token.position
                )
            else:
                builder.sparse_1d_mask_tensor_builder.resize(
                    size=1 + source_token.position
                )
        elif isinstance(self.source_scoping, FineScoping):
            if source_token.scope.scope_name in (
                AttentionScopeName.QUESTION,
                AttentionScopeName.COLUMN,
                AttentionScopeName.TABLE,
            ):
                builder.sparse_1d_mask_tensor_builder.append(
                    index=source_token.position
                )
            else:
                builder.sparse_1d_mask_tensor_builder.resize(
                    size=1 + source_token.position
                )
        else:
            raise NotImplementedError
        return builder

    def add_source_tokens(
        self, source_tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "MemoryKeyPaddingMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for source_token in source_tokens:
            builder.add_source_token(source_token=source_token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_1d_mask_tensor_builder.build(device=device)
