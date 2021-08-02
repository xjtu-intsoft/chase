from collections import deque, OrderedDict
from copy import deepcopy, copy
from enum import Enum, auto
from uuid import UUID

from dataclasses import dataclass, field
from typing import (
    Tuple,
    Mapping,
    Optional,
    Generic,
    TypeVar,
    NewType,
    Union,
    Deque,
    Dict,
    List,
    Set,
    FrozenSet,
    ClassVar,
)

import torch

from duorat.asdl.action_info import ActionInfo
from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.asdl.transition_system import Pos, Action, Done
from duorat.utils import registry

T = TypeVar("T")  # Any type.
KT = TypeVar("KT")  # Key type.
KT_P = TypeVar("KT_P")  # Other key type.
VT = TypeVar("VT")  # Value type.
VT_P = TypeVar("VT_P")  # Other value type.

X = TypeVar("X")
A = TypeVar("A")
B = TypeVar("B")


class FrozenDict(Mapping[KT, VT], Generic[KT, VT]):
    """
    An immutable wrapper around dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability is desired.
    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash


class OrderedFrozenDict(FrozenDict):
    """Ordered version of FrozenDict"""

    dict_cls = OrderedDict

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            # Take into account the order of the keys
            h ^= hash(tuple(self._dict.keys()))
            self._hash = h
        return self._hash


frozendict = FrozenDict


QuestionTokenId = NewType("QuestionTokenId", UUID)
TableId = NewType("TableId", str)
ColumnId = NewType("ColumnId", str)
InputId = Union[QuestionTokenId, ColumnId, TableId]


@dataclass(order=True, frozen=True)
class SQLSchema(object):
    column_names: FrozenDict[ColumnId, str]
    tokenized_column_names: FrozenDict[ColumnId, Tuple[str, ...]]
    original_column_names: FrozenDict[ColumnId, str]
    table_names: FrozenDict[TableId, str]
    tokenized_table_names: FrozenDict[TableId, Tuple[str, ...]]
    original_table_names: FrozenDict[TableId, str]
    column_to_table: FrozenDict[ColumnId, Optional[TableId]]
    table_to_columns: FrozenDict[TableId, Tuple[ColumnId, ...]]
    foreign_keys: FrozenDict[ColumnId, ColumnId]
    foreign_keys_tables: FrozenDict[TableId, Tuple[TableId, ...]]
    primary_keys: Tuple[ColumnId, ...]
    db_id: str
    db_path: Optional[str]


PositionMap = Dict[T, Deque[Pos]]


class AttentionKind(object):
    pass


@dataclass(order=True, frozen=True)
class BidirectionalAttention(AttentionKind):
    pass


@dataclass(order=True, frozen=True)
class BackwardAttention(AttentionKind):
    pass


@dataclass(order=True, frozen=True)
class ForwardAttention(AttentionKind):
    pass


class Scoping(object):
    pass


@registry.register("attention_scoping", "NoScoping")
@dataclass(order=True, frozen=True)
class NoScoping(Scoping):
    pass


@registry.register("attention_scoping", "CoarseScoping")
@dataclass(order=True, frozen=True)
class CoarseScoping(Scoping):
    question_sees_schema: bool
    schema_sees_question: bool
    target_sees_question: bool = True
    target_sees_schema: bool = True


@registry.register("attention_scoping", "FineScoping")
@dataclass(order=True, frozen=True)
class FineScoping(Scoping):
    question_sees_columns: bool
    question_sees_tables: bool
    columns_see_question: bool
    columns_see_each_other: bool
    columns_see_tables: bool
    tables_see_question: bool
    tables_see_columns: bool
    tables_see_each_other: bool
    target_sees_question: bool = True
    target_sees_columns: bool = True
    target_sees_tables: bool = True


class AttentionScopeName(Enum):
    INPUT = auto()
    SOURCE = auto()
    QUESTION = auto()
    SCHEMA = auto()
    COLUMN = auto()
    TABLE = auto()
    TARGET = auto()


@dataclass(order=True, frozen=True)
class AttentionScope(object):
    scope_name: AttentionScopeName
    scope_extension: Optional[str] = None


class Token(Generic[KT, VT]):
    key: KT
    value: VT
    scope: AttentionScope
    position: Pos


class MatchConfidence(object):
    pass


@dataclass(order=False, frozen=True)
class HighConfidenceMatch(MatchConfidence):
    def __le__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return True
        elif isinstance(other, LowConfidenceMatch):
            return False
        else:
            return NotImplemented

    def __lt__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return False
        elif isinstance(other, LowConfidenceMatch):
            return False
        else:
            return NotImplemented

    def __ge__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return True
        elif isinstance(other, LowConfidenceMatch):
            return True
        else:
            return NotImplemented

    def __gt__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return False
        elif isinstance(other, LowConfidenceMatch):
            return True
        else:
            return NotImplemented


@dataclass(order=False, frozen=True)
class LowConfidenceMatch(MatchConfidence):
    def __le__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return True
        elif isinstance(other, LowConfidenceMatch):
            return True
        else:
            return NotImplemented

    def __lt__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return True
        elif isinstance(other, LowConfidenceMatch):
            return False
        else:
            return NotImplemented

    def __ge__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return False
        elif isinstance(other, LowConfidenceMatch):
            return True
        else:
            return NotImplemented

    def __gt__(self, other: "MatchConfidence") -> bool:
        if isinstance(other, HighConfidenceMatch):
            return False
        elif isinstance(other, LowConfidenceMatch):
            return False
        else:
            return NotImplemented


class MatchTag(object):
    confidence: MatchConfidence


# Table < Column < Value


@dataclass(order=False, frozen=True)
class TableMatchTag(MatchTag):
    confidence: MatchConfidence
    table_id: TableId

    def __le__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return (self.confidence, self.table_id) <= (
                other.confidence,
                other.table_id,
            )
        elif isinstance(other, ColumnMatchTag):
            return True
        elif isinstance(other, ValueMatchTag):
            return True
        else:
            return NotImplemented

    def __lt__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return (self.confidence, self.table_id) < (other.confidence, other.table_id)
        elif isinstance(other, ColumnMatchTag):
            return True
        elif isinstance(other, ValueMatchTag):
            return True
        else:
            return NotImplemented

    def __ge__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return (self.confidence, self.table_id) >= (
                other.confidence,
                other.table_id,
            )
        elif isinstance(other, ColumnMatchTag):
            return False
        elif isinstance(other, ValueMatchTag):
            return False
        else:
            return NotImplemented

    def __gt__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return (self.confidence, self.table_id) > (other.confidence, other.table_id)
        elif isinstance(other, ColumnMatchTag):
            return False
        elif isinstance(other, ValueMatchTag):
            return False
        else:
            return NotImplemented


@dataclass(order=False, frozen=True)
class ColumnMatchTag(MatchTag):
    confidence: MatchConfidence
    column_id: ColumnId
    table_id: TableId

    def __le__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return False
        elif isinstance(other, ColumnMatchTag):
            return (self.confidence, self.column_id, self.table_id) <= (
                other.confidence,
                other.column_id,
                other.table_id,
            )
        elif isinstance(other, ValueMatchTag):
            return True
        else:
            return NotImplemented

    def __lt__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return False
        elif isinstance(other, ColumnMatchTag):
            return (self.confidence, self.column_id, self.table_id) < (
                other.confidence,
                other.column_id,
                other.table_id,
            )
        elif isinstance(other, ValueMatchTag):
            return True
        else:
            return NotImplemented

    def __ge__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return True
        elif isinstance(other, ColumnMatchTag):
            return (self.confidence, self.column_id, self.table_id) >= (
                other.confidence,
                other.column_id,
                other.table_id,
            )
        elif isinstance(other, ValueMatchTag):
            return False
        else:
            return NotImplemented

    def __gt__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return True
        elif isinstance(other, ColumnMatchTag):
            return (self.confidence, self.column_id, self.table_id) > (
                other.confidence,
                other.column_id,
                other.table_id,
            )
        elif isinstance(other, ValueMatchTag):
            return False
        else:
            return NotImplemented


@dataclass(order=False, frozen=True)
class ValueMatchTag(MatchTag):
    confidence: MatchConfidence
    column_id: ColumnId
    table_id: TableId
    value: str

    def __le__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return False
        elif isinstance(other, ColumnMatchTag):
            return False
        elif isinstance(other, ValueMatchTag):
            return (self.confidence, self.column_id, self.table_id, self.value) <= (
                other.confidence,
                other.column_id,
                other.table_id,
                other.value,
            )
        else:
            return NotImplemented

    def __lt__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return False
        elif isinstance(other, ColumnMatchTag):
            return False
        elif isinstance(other, ValueMatchTag):
            return (self.confidence, self.column_id, self.table_id, self.value) < (
                other.confidence,
                other.column_id,
                other.table_id,
                other.value,
            )
        else:
            return NotImplemented

    def __ge__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return True
        elif isinstance(other, ColumnMatchTag):
            return True
        elif isinstance(other, ValueMatchTag):
            return (self.confidence, self.column_id, self.table_id, self.value) >= (
                other.confidence,
                other.column_id,
                other.table_id,
                other.value,
            )
        else:
            return NotImplemented

    def __gt__(self, other: "MatchTag") -> bool:
        if isinstance(other, TableMatchTag):
            return True
        elif isinstance(other, ColumnMatchTag):
            return True
        elif isinstance(other, ValueMatchTag):
            return (self.confidence, self.column_id, self.table_id, self.value) > (
                other.confidence,
                other.column_id,
                other.table_id,
                other.value,
            )
        else:
            return NotImplemented


@dataclass
class TaggedToken(object):
    value: str
    tag: str
    match_tags: Deque[MatchTag]
    raw_value: str = ""


TaggedSequence = List[TaggedToken]


@dataclass(order=True, frozen=True)
class PreprocQuestionToken(object):
    key: QuestionTokenId
    value: str
    raw_value: str = ""
    match_tags: Tuple[MatchTag, ...] = tuple()


@dataclass(order=True, frozen=True)
class QuestionToken(Token[QuestionTokenId, VT]):
    key: QuestionTokenId
    value: VT
    raw_value: VT
    scope: AttentionScope
    position: Pos = Pos(0)
    match_tags: Tuple[MatchTag, ...] = tuple()


@dataclass(order=True, frozen=True)
class ColumnToken(Token[ColumnId, VT]):
    key: ColumnId
    value: VT
    scope: AttentionScope
    position: Pos = Pos(0)


@dataclass(order=True, frozen=True)
class TableToken(Token[TableId, VT]):
    key: TableId
    value: VT
    scope: AttentionScope
    position: Pos = Pos(0)


@dataclass(order=True, frozen=True)
class ActionToken(Token[Action, VT]):
    key: Action
    value: VT
    scope: AttentionScope
    position: Pos = Pos(0)


@dataclass(order=True, frozen=True)
class ActionInfoToken(Token[ActionInfo, VT]):
    key: ActionInfo
    value: VT
    scope: AttentionScope
    position: Pos = Pos(0)


@dataclass(order=True, frozen=True)
class RATPreprocItem(object):
    question: Tuple[PreprocQuestionToken, ...]
    sql_schema: SQLSchema
    actions: Tuple[Action, ...]


@dataclass
class DuoRATInputSegment(object):
    input_a: torch.Tensor
    input_b: torch.Tensor
    input_attention_mask: torch.Tensor
    input_key_padding_mask: torch.Tensor
    input_token_type_ids: torch.Tensor
    input_position_ids: torch.Tensor
    input_source_gather_index: torch.Tensor
    input_source_gather_index_mask: torch.Tensor


@dataclass
class DuoRATEncoderItem(object):
    input_a: torch.Tensor
    input_b: torch.Tensor
    input_attention_mask: torch.Tensor
    input_key_padding_mask: torch.Tensor
    input_token_type_ids: torch.Tensor
    input_position_ids: torch.Tensor
    input_segments: Tuple[DuoRATInputSegment, ...]
    input_source_gather_index: torch.Tensor
    source_relations: torch.Tensor
    source_attention_mask: torch.Tensor
    source_key_padding_mask: torch.Tensor


@dataclass
class DuoRATDecoderItem(object):
    masked_target: torch.Tensor
    shifted_target: torch.Tensor
    frontier_fields: torch.Tensor
    frontier_field_types: torch.Tensor
    target_relations: torch.Tensor
    target_attention_mask: torch.Tensor
    target_key_padding_mask: torch.Tensor
    memory_relations: torch.Tensor
    shifted_memory_relations: torch.Tensor
    memory_attention_mask: torch.Tensor
    memory_key_padding_mask: torch.Tensor
    valid_copy_mask: torch.Tensor
    copy_target_mask: torch.Tensor
    valid_actions_mask: torch.Tensor
    target: torch.Tensor


@dataclass
class DuoRATItem(object):
    encoder_item: DuoRATEncoderItem
    decoder_item: DuoRATDecoderItem


@dataclass
class DuoRATInputSegmentBatch(object):
    input_a: torch.Tensor
    input_b: torch.Tensor
    input_attention_mask: torch.Tensor
    input_key_padding_mask: torch.Tensor
    input_token_type_ids: torch.Tensor
    input_position_ids: torch.Tensor
    input_source_gather_index: torch.Tensor
    input_source_gather_index_mask: torch.Tensor


@dataclass
class DuoRATEncoderBatch(object):
    input_a: torch.Tensor
    input_b: torch.Tensor
    input_attention_mask: torch.Tensor
    input_key_padding_mask: torch.Tensor
    input_token_type_ids: torch.Tensor
    input_position_ids: torch.Tensor
    input_segments: Tuple[DuoRATInputSegmentBatch, ...]
    input_source_gather_index: torch.Tensor
    source_relations: torch.Tensor
    source_attention_mask: torch.Tensor
    source_key_padding_mask: torch.Tensor


@dataclass
class DuoRATDecoderBatch(object):
    masked_target: torch.Tensor
    shifted_target: torch.Tensor
    frontier_fields: torch.Tensor
    frontier_field_types: torch.Tensor
    target_relations: torch.Tensor
    target_attention_mask: torch.Tensor
    target_key_padding_mask: torch.Tensor
    memory_relations: torch.Tensor
    shifted_memory_relations: torch.Tensor
    memory_attention_mask: torch.Tensor
    memory_key_padding_mask: torch.Tensor
    valid_copy_mask: torch.Tensor
    copy_target_mask: torch.Tensor
    valid_actions_mask: torch.Tensor
    target: torch.Tensor


@dataclass
class DuoRATBatch(object):
    encoder_batch: DuoRATEncoderBatch
    decoder_batch: DuoRATDecoderBatch


@dataclass
class Sparse1DTensorBuilder(object):
    index: Deque[int] = field(default_factory=deque)
    value: Deque[int] = field(default_factory=deque)
    size: int = 0

    def __deepcopy__(self, memo) -> "Sparse1DTensorBuilder":
        builder = copy(self)
        builder.index = copy(self.index)
        builder.value = copy(self.value)
        return builder

    def empty(self, copy: bool = False) -> "Sparse1DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index = deque()
        builder.value = deque()
        return builder

    def resize(self, size: int, copy: bool = False) -> "Sparse1DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.size = max(builder.size, size)
        return builder

    def append(
        self, index: int, value: int, size: Optional[int] = None, copy: bool = False
    ) -> "Sparse1DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index.append(index)
        builder.value.append(value)
        builder.size = max(builder.size, 1 + index)
        if size is not None:
            builder.size = max(builder.size, size)
        return builder

    def extend(
        self, other: "Sparse1DTensorBuilder", copy: bool = False
    ) -> "Sparse1DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index.extend(other.index)
        builder.value.extend(other.value)
        builder.size = max(builder.size, other.size)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        assert len(set(self.index)) == len(self.index)
        return torch.sparse_coo_tensor(
            torch.tensor((self.index,), dtype=torch.long, device=device),
            torch.tensor(self.value, dtype=torch.long, device=device),
            torch.Size((self.size,)),
            dtype=torch.long,
            device=device,
        ).to_dense()


@dataclass
class Sparse2DTensorBuilder(object):
    index_0: Deque[int] = field(default_factory=deque)
    index_1: Deque[int] = field(default_factory=deque)
    value: Deque[int] = field(default_factory=deque)
    size_0: int = 0
    size_1: int = 0

    def __deepcopy__(self, memo) -> "Sparse2DTensorBuilder":
        builder = copy(self)
        builder.index_0 = copy(self.index_0)
        builder.index_1 = copy(self.index_1)
        builder.value = copy(self.value)
        return builder

    def empty(self, copy: bool = False) -> "Sparse2DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index_0 = deque()
        builder.index_1 = deque()
        builder.value = deque()
        return builder

    def resize(self, size: Tuple[int, int], copy=False) -> "Sparse2DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.size_0 = max(builder.size_0, size[0])
        builder.size_1 = max(builder.size_1, size[1])
        return builder

    def append(
        self,
        index: Tuple[int, int],
        value: int,
        size: Optional[Tuple[int, int]] = None,
        copy: bool = False,
    ) -> "Sparse2DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index_0.append(index[0])
        builder.index_1.append(index[1])
        builder.value.append(value)
        builder.size_0 = max(builder.size_0, 1 + index[0])
        builder.size_1 = max(builder.size_1, 1 + index[1])
        if size is not None:
            builder.size_0 = max(builder.size_0, size[0])
            builder.size_1 = max(builder.size_1, size[1])
        return builder

    def extend(
        self, other: "Sparse2DTensorBuilder", copy: bool = False
    ) -> "Sparse2DTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index_0.extend(other.index_0)
        builder.index_1.extend(other.index_1)
        builder.value.extend(other.value)
        builder.size_0 = max(builder.size_0, other.size_0)
        builder.size_1 = max(builder.size_1, other.size_1)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        assert len(set(zip(self.index_0, self.index_1))) == len(
            list(zip(self.index_0, self.index_1))
        )
        return torch.sparse_coo_tensor(
            torch.tensor((self.index_0, self.index_1), dtype=torch.long, device=device),
            torch.tensor(self.value, dtype=torch.long, device=device),
            torch.Size((self.size_0, self.size_1)),
            dtype=torch.long,
            device=device,
        ).to_dense()


@dataclass
class Sparse1DMaskTensorBuilder(object):
    index: Deque[int] = field(default_factory=deque)
    size: int = 0

    def __deepcopy__(self, memo) -> "Sparse1DMaskTensorBuilder":
        builder = copy(self)
        builder.index = copy(self.index)
        return builder

    def empty(self, copy: bool = False) -> "Sparse1DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index = deque()
        return builder

    def resize(self, size: int, copy: bool = False) -> "Sparse1DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.size = max(builder.size, size)
        return builder

    def append(
        self, index: int, size: Optional[int] = None, copy: bool = False
    ) -> "Sparse1DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index.append(index)
        builder.size = max(builder.size, 1 + index)
        if size is not None:
            builder.size = max(builder.size, size)
        return builder

    def extend(
        self, other: "Sparse1DMaskTensorBuilder", copy: bool = False
    ) -> "Sparse1DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index.extend(other.index)
        builder.size = max(builder.size, other.size)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        assert len(set(self.index)) == len(self.index)
        return (
            torch.sparse_coo_tensor(
                torch.tensor((self.index,), dtype=torch.long, device=device),
                torch.ones(
                    size=torch.Size((len(self.index),)),
                    dtype=torch.long,
                    device=device,
                ),
                torch.Size((self.size,)),
                dtype=torch.long,
                device=device,
            )
            .to_dense()
            .bool()
        )


@dataclass
class Sparse2DMaskTensorBuilder(object):
    index_0: Deque[int] = field(default_factory=deque)
    index_1: Deque[int] = field(default_factory=deque)
    size_0: int = 0
    size_1: int = 0

    def __deepcopy__(self, memo) -> "Sparse2DMaskTensorBuilder":
        builder = copy(self)
        builder.index_0 = copy(self.index_0)
        builder.index_1 = copy(self.index_1)
        return builder

    def empty(self, copy: bool = False) -> "Sparse2DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index_0 = deque()
        builder.index_1 = deque()
        return builder

    def resize(
        self, size: Tuple[int, int], copy: bool = False
    ) -> "Sparse2DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.size_0 = max(builder.size_0, size[0])
        builder.size_1 = max(builder.size_1, size[1])
        return builder

    def append(
        self,
        index: Tuple[int, int],
        size: Optional[Tuple[int, int]] = None,
        copy: bool = False,
    ) -> "Sparse2DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index_0.append(index[0])
        builder.index_1.append(index[1])
        builder.size_0 = max(builder.size_0, 1 + index[0])
        builder.size_1 = max(builder.size_1, 1 + index[1])
        if size is not None:
            builder.size_0 = max(builder.size_0, size[0])
            builder.size_1 = max(builder.size_1, size[1])
        return builder

    def extend(
        self, other: "Sparse2DMaskTensorBuilder", copy: bool = False
    ) -> "Sparse2DMaskTensorBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.index_0.extend(other.index_0)
        builder.index_1.extend(other.index_1)
        builder.size_0 = max(builder.size_0, other.size_0)
        builder.size_1 = max(builder.size_1, other.size_1)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        assert len(set(zip(self.index_0, self.index_1))) == len(
            list(zip(self.index_0, self.index_1))
        )
        return (
            torch.sparse_coo_tensor(
                torch.tensor(
                    (self.index_0, self.index_1), dtype=torch.long, device=device
                ),
                torch.ones(
                    size=torch.Size((len(self.index_0),)),
                    dtype=torch.long,
                    device=device,
                ),
                torch.Size((self.size_0, self.size_1)),
                dtype=torch.long,
                device=device,
            )
            .to_dense()
            .bool()
        )
