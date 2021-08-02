import itertools
from collections import defaultdict
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import (
    Generator,
    Optional,
    Dict,
    Iterable,
    Callable,
    Set,
    Tuple,
)

import torch

from duorat.asdl.transition_system import Action, Pos
from duorat.types import (
    Token,
    KT,
    VT,
    QuestionToken,
    SQLSchema,
    ColumnToken,
    TableToken,
    ActionToken,
    BidirectionalAttention,
    BackwardAttention,
    ForwardAttention,
    Sparse1DTensorBuilder,
    Sparse1DMaskTensorBuilder,
    Sparse2DMaskTensorBuilder,
    PreprocQuestionToken,
    TableId,
    ColumnId,
    Scoping,
    AttentionKind,
    AttentionScope,
    NoScoping,
    CoarseScoping,
    FineScoping,
    AttentionScopeName,
)


def question_input_tokens(
    question: Iterable[PreprocQuestionToken], scoping: Scoping
) -> Generator[QuestionToken[str], None, None]:
    if isinstance(scoping, NoScoping):
        scope = AttentionScope(scope_name=AttentionScopeName.INPUT)
    elif isinstance(scoping, CoarseScoping) or isinstance(scoping, FineScoping):
        scope = AttentionScope(scope_name=AttentionScopeName.QUESTION)
    else:
        raise NotImplementedError
    return (
        QuestionToken(
            key=token.key,
            value=token.value,
            scope=scope,
            raw_value=token.value,
            match_tags=token.match_tags,
        )
        for token in question
    )


def column_input_tokens(
    sql_schema: SQLSchema, column_id: ColumnId, scoping: Scoping
) -> Generator[ColumnToken[str], None, None]:
    if isinstance(scoping, NoScoping):
        make_scope = lambda _column_id: AttentionScope(
            scope_name=AttentionScopeName.INPUT
        )
    elif isinstance(scoping, CoarseScoping):
        make_scope = lambda _column_id: AttentionScope(
            scope_name=AttentionScopeName.SCHEMA
        )
    elif isinstance(scoping, FineScoping):
        make_scope = lambda column_id: AttentionScope(
            scope_name=AttentionScopeName.COLUMN, scope_extension=column_id
        )
    else:
        raise NotImplementedError
    return (
        ColumnToken(key=column_id, value=s, scope=make_scope(column_id))
        for s in sql_schema.tokenized_column_names[column_id]
    )


def table_input_tokens(
    sql_schema: SQLSchema, table_id: TableId, scoping: Scoping
) -> Generator[TableToken[str], None, None]:
    if isinstance(scoping, NoScoping):
        make_scope = lambda _table_id: AttentionScope(
            scope_name=AttentionScopeName.INPUT
        )
    elif isinstance(scoping, CoarseScoping):
        make_scope = lambda _table_id: AttentionScope(
            scope_name=AttentionScopeName.SCHEMA
        )
    elif isinstance(scoping, FineScoping):
        make_scope = lambda table_id: AttentionScope(
            scope_name=AttentionScopeName.TABLE, scope_extension=table_id
        )
    else:
        raise NotImplementedError
    return (
        TableToken(key=table_id, value=s, scope=make_scope(table_id))
        for s in sql_schema.tokenized_table_names[table_id]
    )


def column_source_tokens(
    sql_schema: SQLSchema, column_id: ColumnId, scoping: Scoping
) -> Tuple[ColumnToken[ColumnId]]:
    if isinstance(scoping, NoScoping):
        make_scope = lambda _column_id: AttentionScope(
            scope_name=AttentionScopeName.SOURCE
        )
    elif isinstance(scoping, CoarseScoping):
        make_scope = lambda _column_id: AttentionScope(
            scope_name=AttentionScopeName.SCHEMA
        )
    elif isinstance(scoping, FineScoping):
        make_scope = lambda column_id: AttentionScope(
            scope_name=AttentionScopeName.COLUMN, scope_extension=column_id
        )
    else:
        raise NotImplementedError
    return (ColumnToken(key=column_id, value=column_id, scope=make_scope(column_id)),)


def table_source_tokens(
    sql_schema: SQLSchema, table_id: TableId, scoping: Scoping
) -> Tuple[TableToken[TableId]]:
    if isinstance(scoping, NoScoping):
        make_scope = lambda _table_id: AttentionScope(
            scope_name=AttentionScopeName.SOURCE
        )
    elif isinstance(scoping, CoarseScoping):
        make_scope = lambda _table_id: AttentionScope(
            scope_name=AttentionScopeName.SCHEMA
        )
    elif isinstance(scoping, FineScoping):
        make_scope = lambda table_id: AttentionScope(
            scope_name=AttentionScopeName.TABLE, scope_extension=table_id
        )
    else:
        raise NotImplementedError
    return (TableToken(key=table_id, value=table_id, scope=make_scope(table_id)),)


def table_and_columns_tokens(
    sql_schema: SQLSchema,
    table_id: TableId,
    scoping: Scoping,
    get_column_tokens: Callable[[SQLSchema, ColumnId, Scoping], Iterable[ColumnToken]],
    get_table_tokens: Callable[[SQLSchema, TableId, Scoping], Iterable[TableToken]],
) -> Iterable[Token]:
    columns = itertools.chain(
        *(
            get_column_tokens(sql_schema, column_id, scoping)
            for column_id in sql_schema.table_to_columns[table_id]
        )
    )
    return itertools.chain(get_table_tokens(sql_schema, table_id, scoping), columns)


def schema_source_tokens(
    sql_schema: SQLSchema, schema_token_ordering: str, scoping: Scoping
) -> Iterable[Token]:
    return schema_tokens(
        sql_schema,
        schema_token_ordering,
        scoping,
        get_column_tokens=column_source_tokens,
        get_table_tokens=table_source_tokens,
    )


def schema_input_tokens(
    sql_schema: SQLSchema, schema_token_ordering: str, scoping: Scoping
) -> Iterable[Token]:
    return schema_tokens(
        sql_schema,
        schema_token_ordering,
        scoping,
        get_column_tokens=column_input_tokens,
        get_table_tokens=table_input_tokens,
    )


def schema_tokens(
    sql_schema: SQLSchema,
    schema_token_ordering: str,
    scoping: Scoping,
    get_column_tokens: Callable[[SQLSchema, ColumnId, Scoping], Iterable[ColumnToken]],
    get_table_tokens: Callable[[SQLSchema, TableId, Scoping], Iterable[TableToken]],
) -> Iterable[Token]:
    if schema_token_ordering == "[table][column]":
        # First all the tables, then all the columns
        tables = itertools.chain(
            *(
                get_table_tokens(sql_schema, table_id, scoping)
                for table_id in sql_schema.tokenized_table_names.keys()
            )
        )
        columns = itertools.chain(
            *(
                get_column_tokens(sql_schema, column_id, scoping)
                for column_id in sql_schema.tokenized_column_names.keys()
            )
        )
        return itertools.chain(tables, columns)
    elif schema_token_ordering == "[column][table]":
        # First all the columns, then all the tables
        tables = itertools.chain(
            *(
                get_table_tokens(sql_schema, table_id, scoping)
                for table_id in sql_schema.tokenized_table_names.keys()
            )
        )
        columns = itertools.chain(
            *(
                get_column_tokens(sql_schema, column_id, scoping)
                for column_id in sql_schema.tokenized_column_names.keys()
            )
        )
        return itertools.chain(columns, tables)
    elif schema_token_ordering == "[table[column]]":
        # Table Column Column Table Column Column

        # Columns that have no table (`*`, essentially)
        columns_without_table = itertools.chain(
            *(
                get_column_tokens(sql_schema, column_id, scoping)
                for column_id in sql_schema.tokenized_column_names.keys()
                if sql_schema.column_to_table[column_id] is None
            )
        )

        return itertools.chain(
            columns_without_table,
            *(
                table_and_columns_tokens(
                    sql_schema, table_id, scoping, get_column_tokens, get_table_tokens
                )
                for table_id, table_name in sql_schema.tokenized_table_names.items()
            ),
        )
    else:
        raise ValueError(
            f"Invalid value for schema_token_ordering: {schema_token_ordering}"
        )


def question_source_tokens(
    question: Iterable[PreprocQuestionToken], scoping: Scoping
) -> Generator[QuestionToken[str], None, None]:
    if isinstance(scoping, NoScoping):
        scope = AttentionScope(scope_name=AttentionScopeName.SOURCE)
    elif isinstance(scoping, CoarseScoping) or isinstance(scoping, FineScoping):
        scope = AttentionScope(scope_name=AttentionScopeName.QUESTION)
    else:
        raise NotImplementedError
    return (
        QuestionToken(
            key=token.key,
            value=token.value,
            raw_value=token.raw_value,
            scope=scope,
            match_tags=token.match_tags,
        )
        for token in question
    )


def action_tokens(
    actions: Iterable[Action], scoping: Scoping
) -> Generator[ActionToken[Action], None, None]:
    if isinstance(scoping, NoScoping):
        scope = AttentionScope(scope_name=AttentionScopeName.TARGET)
    else:
        raise NotImplementedError
    return (ActionToken(key=action, value=action, scope=scope) for action in actions)


@dataclass
class TokenTensorBuilder(object):
    sparse_1d_tensor_builder: Sparse1DTensorBuilder = field(
        default_factory=lambda: Sparse1DTensorBuilder()
    )

    def add_token(
        self, token: Token[KT, int], copy: bool = False
    ) -> "TokenTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.sparse_1d_tensor_builder.append(index=token.position, value=token.value)
        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, int]], copy: bool = False
    ) -> "TokenTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_1d_tensor_builder.build(device=device)


@dataclass
class TokenTypeIdTensorBuilder(object):
    type_id: int = 1
    scope: Optional[AttentionScope] = None
    sparse_1d_tensor_builder: Sparse1DTensorBuilder = field(
        default_factory=lambda: Sparse1DTensorBuilder()
    )

    def add_token(
        self, token: Token[KT, VT], copy: bool = False
    ) -> "TokenTypeIdTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        if token.scope != builder.scope:
            builder.type_id = 1 - builder.type_id
            builder.scope = token.scope
        builder.sparse_1d_tensor_builder.append(
            index=token.position, value=builder.type_id
        )
        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "TokenTypeIdTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_1d_tensor_builder.build(device=device)


@dataclass
class PositionIdTensorBuilder(object):
    position_id: int = 0
    scope: Optional[AttentionScope] = None
    sparse_1d_tensor_builder: Sparse1DTensorBuilder = field(
        default_factory=lambda: Sparse1DTensorBuilder()
    )

    def add_token(
        self, token: Token[KT, VT], copy: bool = False
    ) -> "PositionIdTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        if token.scope != builder.scope:
            builder.position_id = 0
            builder.scope = token.scope
        else:
            builder.position_id += 1
        builder.sparse_1d_tensor_builder.append(
            index=token.position, value=builder.position_id
        )
        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "PositionIdTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_1d_tensor_builder.build(device=device)


@dataclass
class KeyPaddingMaskBuilder(object):
    r"""
Builds a boolean mask that is `True`/`1` for all token positions and `False`/`0` for all padding positions.

Thus, if it were to be used in a transformer model, it would need to be inverted first.
"""
    sparse_1d_mask_tensor_builder: Sparse1DMaskTensorBuilder = field(
        default_factory=lambda: Sparse1DMaskTensorBuilder()
    )

    def add_token(
        self, token: Token[KT, VT], copy: bool = False
    ) -> "KeyPaddingMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.sparse_1d_mask_tensor_builder.append(index=token.position)
        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "KeyPaddingMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_1d_mask_tensor_builder.build(device=device)


@dataclass
class AttentionMaskBuilder(object):
    r"""
Builds a boolean mask that is `True`/`1` for all positions that can be attended to and `False`/`0` for all others.

Thus, if it were to be used in a transformer model where attention masks are additive masks, first `True` would need to
be mapped to `0` and `False` would need to be mapped to `float("-inf")`.
"""
    scoping: Scoping
    scope_positions: Dict[AttentionScope, Set[Pos]] = field(init=False)
    sparse_2d_mask_tensor_builder: Sparse2DMaskTensorBuilder = field(
        default_factory=lambda: Sparse2DMaskTensorBuilder()
    )

    def __post_init__(self):
        self.scope_positions = defaultdict(set)

    def __deepcopy__(self, memo) -> "AttentionMaskBuilder":
        builder = copy(self)
        builder.scope_positions = defaultdict(set)
        for scope, tokens in self.scope_positions.items():
            builder.scope_positions[scope] = copy(tokens)
        builder.sparse_2d_mask_tensor_builder = deepcopy(
            self.sparse_2d_mask_tensor_builder
        )
        return builder

    def _get_scopes_by_name(
        self, scope_name: AttentionScopeName
    ) -> Set[AttentionScope]:
        return set(
            _scope
            for _scope in self.scope_positions.keys()
            if _scope.scope_name == scope_name
        )

    def _scope_connections(self, scope: AttentionScope) -> Set[AttentionScope]:
        if isinstance(self.scoping, NoScoping):
            if scope.scope_name in (
                AttentionScopeName.INPUT,
                AttentionScopeName.SOURCE,
                AttentionScopeName.TARGET,
            ):
                self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                return self_scopes
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(scope.scope_name)
                )
        elif isinstance(self.scoping, CoarseScoping):
            if scope.scope_name == AttentionScopeName.QUESTION:
                self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                if self.scoping.question_sees_schema:
                    schema_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.SCHEMA
                    )
                else:
                    schema_scopes = set()
                return self_scopes | schema_scopes
            elif scope.scope_name == AttentionScopeName.SCHEMA:
                self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                if self.scoping.schema_sees_question:
                    question_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.QUESTION
                    )
                else:
                    question_scopes = set()
                return self_scopes | question_scopes
            elif scope.scope_name == AttentionScopeName.TARGET:
                self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                return self_scopes
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(scope.scope_name)
                )
        elif isinstance(self.scoping, FineScoping):
            if scope.scope_name == AttentionScopeName.QUESTION:
                self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                if self.scoping.question_sees_columns:
                    column_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.COLUMN
                    )
                else:
                    column_scopes = set()
                if self.scoping.question_sees_tables:
                    tables_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.TABLE
                    )
                else:
                    tables_scopes = set()
                return self_scopes | column_scopes | tables_scopes
            elif scope.scope_name == AttentionScopeName.COLUMN:
                if self.scoping.columns_see_each_other:
                    self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                else:
                    self_scopes = {scope}
                if self.scoping.columns_see_question:
                    question_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.QUESTION
                    )
                else:
                    question_scopes = set()
                if self.scoping.columns_see_tables:
                    table_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.TABLE
                    )
                else:
                    table_scopes = set()
                return self_scopes | question_scopes | table_scopes
            elif scope.scope_name == AttentionScopeName.TABLE:
                if self.scoping.tables_see_each_other:
                    self_scopes = self._get_scopes_by_name(scope_name=scope.scope_name)
                else:
                    self_scopes = {scope}
                if self.scoping.tables_see_question:
                    question_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.QUESTION
                    )
                else:
                    question_scopes = set()
                if self.scoping.tables_see_columns:
                    column_scopes = self._get_scopes_by_name(
                        scope_name=AttentionScopeName.COLUMN
                    )
                else:
                    column_scopes = set()
                return self_scopes | question_scopes | column_scopes
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(scope.scope_name)
                )
        else:
            raise NotImplementedError

    def _scope_attention_kind(self, scope: AttentionScope) -> AttentionKind:
        if isinstance(self.scoping, NoScoping):
            if scope.scope_name in (
                AttentionScopeName.INPUT,
                AttentionScopeName.SOURCE,
            ):
                return BidirectionalAttention()
            elif scope.scope_name == AttentionScopeName.TARGET:
                return BackwardAttention()
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(scope.scope_name)
                )
        elif isinstance(self.scoping, CoarseScoping):
            if scope.scope_name in (
                AttentionScopeName.QUESTION,
                AttentionScopeName.SCHEMA,
            ):
                return BidirectionalAttention()
            elif scope.scope_name == AttentionScopeName.TARGET:
                return BackwardAttention()
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(scope.scope_name)
                )
        elif isinstance(self.scoping, FineScoping):
            if scope.scope_name in (
                AttentionScopeName.QUESTION,
                AttentionScopeName.TABLE,
                AttentionScopeName.COLUMN,
            ):
                return BidirectionalAttention()
            elif scope.scope_name == AttentionScopeName.TARGET:
                return BackwardAttention()
            else:
                raise ValueError(
                    "Unexpected attention scope: {}".format(scope.scope_name)
                )
        else:
            raise NotImplementedError

    @staticmethod
    def _constrain_attention(
        attention_kind: AttentionKind,
        attend_from_to: Generator[Tuple[Pos, Pos], None, None],
    ) -> Generator[Tuple[Pos, Pos], None, None]:
        if attention_kind == BidirectionalAttention():
            return attend_from_to
        elif attention_kind == BackwardAttention():
            return (
                (from_pos, to_pos)
                for (from_pos, to_pos) in attend_from_to
                if from_pos >= to_pos
            )
        elif attention_kind == ForwardAttention():
            return (
                (from_pos, to_pos)
                for (from_pos, to_pos) in attend_from_to
                if from_pos <= to_pos
            )
        else:
            raise ValueError("Unexpected attention kind: {}".format(attention_kind))

    @classmethod
    def _make_mask(
        cls,
        attention_kind: AttentionKind,
        attend_from: Iterable[Pos],
        attend_to: Iterable[Pos],
    ) -> Generator[Tuple[Pos, Pos], None, None]:
        attend_from_to = (
            (from_pos, to_pos) for from_pos in attend_from for to_pos in attend_to
        )
        return cls._constrain_attention(
            attention_kind=attention_kind, attend_from_to=attend_from_to
        )

    def add_token(
        self, token: Token[KT, VT], copy: bool = False
    ) -> "AttentionMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.scope_positions[token.scope].add(token.position)
        this_scope = token.scope
        attend_self: Set[Pos] = {token.position}
        mask: Set[Tuple[Pos, Pos]] = set().union(
            *(
                builder._make_mask(
                    attention_kind=builder._scope_attention_kind(scope=this_scope),
                    attend_from=attend_self,
                    attend_to=set().union(
                        *(
                            builder.scope_positions[that_scope]
                            for that_scope in builder._scope_connections(
                                scope=this_scope
                            )
                        )
                    ),
                ),
                set().union(
                    *(
                        builder._make_mask(
                            attention_kind=builder._scope_attention_kind(
                                scope=that_scope
                            ),
                            attend_from=builder.scope_positions[that_scope],
                            attend_to=attend_self,
                        )
                        for that_scope in builder.scope_positions.keys()
                        if this_scope in builder._scope_connections(scope=that_scope)
                    )
                ),
            )
        )
        for _index in mask:
            builder.sparse_2d_mask_tensor_builder.append(
                index=_index, size=(1 + token.position, 1 + token.position)
            )
        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, VT]], copy: bool = False
    ) -> "AttentionMaskBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_mask_tensor_builder.build(device=device)
