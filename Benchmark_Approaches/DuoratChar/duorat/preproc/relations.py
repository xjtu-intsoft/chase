import itertools
import logging
from collections import deque, defaultdict
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import Generator, Tuple, Union, Iterable, Dict, Sequence, Deque

import torch

from duorat.asdl.action_info import ActionInfo
from duorat.asdl.asdl import ASDLPrimitiveType
from duorat.asdl.transition_system import Pos, GenTokenAction, ReduceAction, MaskAction
from duorat.types import (
    FrozenDict,
    frozendict,
    T,
    PositionMap,
    SQLSchema,
    ColumnId,
    TableId,
    Token,
    QuestionToken,
    ColumnToken,
    TableToken,
    KT,
    InputId,
    Sparse2DTensorBuilder,
    TableMatchTag,
    HighConfidenceMatch,
    ColumnMatchTag,
    ValueMatchTag,
    LowConfidenceMatch,
    MatchTag,
    MatchConfidence,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Relation(object):
    pass


class SourceRelation(Relation):
    pass


@dataclass(order=True, frozen=True)
class DefaultSourceRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class QQDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class QQDistRelation(SourceRelation):
    dist: int


@dataclass(order=True, frozen=True)
class QCDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class QTDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TQDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CQDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CCDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CTDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TCDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TTDefaultRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CCForeignKeyForwardRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CCForeignKeyBackwardRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CCTableMatchRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CCDistRelation(SourceRelation):
    dist: int


@dataclass(order=True, frozen=True)
class CTForeignKeyRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CTPrimaryKeyRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CTTableMatchRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class CTAnyTableRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TCForeignKeyRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TCPrimaryKeyRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TCTableMatchRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TCAnyTableRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TTForeignKeyForwardRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TTForeignKeyBackwardRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TTForeignKeyBidirectionalRelation(SourceRelation):
    pass


@dataclass(order=True, frozen=True)
class TTDistRelation(SourceRelation):
    dist: int


@dataclass(order=True, frozen=True)
class QCMatchRelation(SourceRelation):
    confidence: MatchConfidence
    value_match: bool


@dataclass(order=True, frozen=True)
class QTMatchRelation(SourceRelation):
    confidence: MatchConfidence
    value_match: bool


@dataclass(order=True, frozen=True)
class CQMatchRelation(SourceRelation):
    confidence: MatchConfidence
    value_match: bool


@dataclass(order=True, frozen=True)
class TQMatchRelation(SourceRelation):
    confidence: MatchConfidence
    value_match: bool


class MemoryRelation(Relation):
    pass


@dataclass(order=True, frozen=True)
class DefaultMemoryRelation(MemoryRelation):
    pass


@dataclass(order=True, frozen=True)
class CopiedFromRelation(MemoryRelation):
    pass


class TargetRelation(Relation):
    pass


@dataclass(order=True, frozen=True)
class DefaultTargetRelation(TargetRelation):
    pass


@dataclass(order=True, frozen=True)
class ParentChildRelation(TargetRelation):
    pass


@dataclass(order=True, frozen=True)
class ChildParentRelation(TargetRelation):
    pass


@dataclass(order=True, frozen=True)
class IdentityActionRelation(TargetRelation):
    pass


@dataclass(order=True, frozen=True)
class SiblingDistRelation(TargetRelation):
    dist: int


def source_relation_types(
    qq_identity: bool = True,
    qq_max_dist: int = 2,
    cc_foreign_key: bool = True,
    cc_table_match: bool = True,
    cc_identity: bool = True,
    cc_max_dist: int = 2,
    ct_foreign_key: bool = True,
    ct_table_match: bool = True,
    tc_table_match: bool = True,
    tc_foreign_key: bool = True,
    tt_identity: bool = True,
    tt_max_dist: int = 2,
    tt_foreign_key: bool = True,
    use_schema_linking: bool = True,
    only_exact_matches: bool = False,  # If True, turns off the partial matches
    high_confidence_db_content_schema_linking: bool = True,
    low_confidence_db_content_schema_linking: bool = False,
    db_content_schema_linking=None,  # Deprecated
) -> Iterable[SourceRelation]:
    return itertools.chain(
        (
            DefaultSourceRelation(),
            QQDefaultRelation(),
            QCDefaultRelation(),
            CQDefaultRelation(),
            QTDefaultRelation(),
            TQDefaultRelation(),
            CCDefaultRelation(),
            CTDefaultRelation(),
            TCDefaultRelation(),
            TTDefaultRelation(),
        ),
        (
            QQDistRelation(dist)
            for dist in range(-qq_max_dist, qq_max_dist + 1)
            if qq_identity or dist != 0
        ),
        (
            (CCForeignKeyForwardRelation(), CCForeignKeyBackwardRelation())
            if cc_foreign_key
            else ()
        ),
        ((CCTableMatchRelation(),) if cc_table_match else ()),
        (
            CCDistRelation(dist)
            for dist in range(-cc_max_dist, cc_max_dist + 1)
            if cc_identity or dist != 0
        ),
        ((CTForeignKeyRelation(),) if ct_foreign_key else ()),
        (
            (CTPrimaryKeyRelation(), CTTableMatchRelation(), CTAnyTableRelation(),)
            if ct_table_match
            else ()
        ),
        ((TCForeignKeyRelation(),) if tc_foreign_key else ()),
        (
            (TCPrimaryKeyRelation(), TCTableMatchRelation(), TCAnyTableRelation(),)
            if tc_table_match
            else ()
        ),
        (
            (
                TTForeignKeyForwardRelation(),
                TTForeignKeyBackwardRelation(),
                TTForeignKeyBidirectionalRelation(),
            )
            if tt_foreign_key
            else ()
        ),
        (
            TTDistRelation(dist)
            for dist in range(-tt_max_dist, tt_max_dist + 1)
            if tt_identity or dist != 0
        ),
        (
            (
                QCMatchRelation(confidence=HighConfidenceMatch(), value_match=False),
                QTMatchRelation(confidence=HighConfidenceMatch(), value_match=False),
                CQMatchRelation(confidence=HighConfidenceMatch(), value_match=False),
                TQMatchRelation(confidence=HighConfidenceMatch(), value_match=False),
            )
            if use_schema_linking
            else ()
        ),
        (
            (
                QCMatchRelation(confidence=LowConfidenceMatch(), value_match=False),
                QTMatchRelation(confidence=LowConfidenceMatch(), value_match=False),
                CQMatchRelation(confidence=LowConfidenceMatch(), value_match=False),
                TQMatchRelation(confidence=LowConfidenceMatch(), value_match=False),
            )
            if use_schema_linking and not only_exact_matches
            else ()
        ),
        (
            (
                QCMatchRelation(confidence=HighConfidenceMatch(), value_match=True),
                # QTMatchRelation(confidence=HighConfidenceMatch(), value_match=True),
                CQMatchRelation(confidence=HighConfidenceMatch(), value_match=True),
                # TQMatchRelation(confidence=HighConfidenceMatch(), value_match=True),
            )
            if high_confidence_db_content_schema_linking
            else ()
        ),
        (
            (
                QCMatchRelation(confidence=LowConfidenceMatch(), value_match=True),
                # QTMatchRelation(confidence=LowConfidenceMatch(), value_match=True),
                CQMatchRelation(confidence=LowConfidenceMatch(), value_match=True),
                # TQMatchRelation(confidence=LowConfidenceMatch(), value_match=True),
            )
            if low_confidence_db_content_schema_linking
            else ()
        ),
    )


def memory_relation_types(
    copied_from_relation: bool = True,
) -> Iterable[MemoryRelation]:
    return itertools.chain(
        (DefaultMemoryRelation(),),
        ((CopiedFromRelation(),) if copied_from_relation else ()),
    )


def target_relation_types(
    identity_relation: bool = True,
    sibling_relation_clipping_distance: int = 2,
    parent_child_relation: bool = True,
) -> Iterable[TargetRelation]:
    return itertools.chain(
        (DefaultTargetRelation(),),
        (ParentChildRelation(), ChildParentRelation(),)
        if parent_child_relation
        else (),
        (IdentityActionRelation(),) if identity_relation else (),
        (
            SiblingDistRelation(dist)
            for dist in range(
                -sibling_relation_clipping_distance,
                sibling_relation_clipping_distance + 1,
            )
            if dist != 0
        ),
    )


def freeze_order(xxs: Iterable[Iterable[T]]) -> FrozenDict[T, int]:
    return frozendict({rel: i for i, rel in enumerate(itertools.chain(*xxs))})


@dataclass
class SourceRelationsBuilder(object):
    sql_schema: SQLSchema
    relation_types: FrozenDict[SourceRelation, int]
    input_tokens: Deque[Token[InputId, str]] = field(default_factory=deque)
    source_question_tokens: Deque[QuestionToken[str]] = field(default_factory=deque)
    columns_position_map: PositionMap[ColumnId] = field(
        default_factory=lambda: defaultdict(deque)
    )
    tables_position_map: PositionMap[TableId] = field(
        default_factory=lambda: defaultdict(deque)
    )
    sparse_2d_tensor_builder: Sparse2DTensorBuilder = field(
        default_factory=lambda: Sparse2DTensorBuilder()
    )

    def __deepcopy__(self, memo) -> "SourceRelationsBuilder":
        builder = copy(self)
        builder.input_tokens = copy(self.input_tokens)
        builder.source_question_tokens = copy(self.source_question_tokens)
        builder.columns_position_map = defaultdict(deque)
        for s, positions in self.columns_position_map.items():
            builder.columns_position_map[s] = copy(positions)
        builder.tables_position_map = defaultdict(deque)
        for s, positions in self.tables_position_map.items():
            builder.tables_position_map[s] = copy(positions)
        builder.sparse_2d_tensor_builder = deepcopy(self.sparse_2d_tensor_builder)
        return builder

    def add_input_token(
        self, input_token: Token[InputId, str], copy: bool = False
    ) -> "SourceRelationsBuilder":
        builder = deepcopy(self) if copy is True else self
        builder.input_tokens.append(input_token)
        return builder

    def add_input_tokens(
        self, input_tokens: Iterable[Token[InputId, str]], copy: bool = False
    ) -> "SourceRelationsBuilder":
        builder = deepcopy(self) if copy is True else self
        for input_token in input_tokens:
            builder.add_input_token(input_token=input_token)
        return builder

    def _qq_relations(
        self, position: Pos, other_position: Pos
    ) -> Generator[
        Tuple[Pos, Pos, Union[QQDefaultRelation, QQDistRelation]], None, None
    ]:
        relation = QQDistRelation(position - other_position)
        if relation in self.relation_types:
            yield position, other_position, relation
        elif QQDefaultRelation() in self.relation_types:
            yield position, other_position, QQDefaultRelation()

    def _cc_relations(
        self,
        column_id: ColumnId,
        position: Pos,
        other_column_id: ColumnId,
        other_position: Pos,
    ) -> Generator[
        Tuple[
            Pos,
            Pos,
            Union[
                CCDefaultRelation,
                CCDistRelation,
                CCForeignKeyForwardRelation,
                CCForeignKeyBackwardRelation,
                CCTableMatchRelation,
            ],
        ],
        None,
        None,
    ]:
        if column_id == other_column_id:
            # sibling positions share the same column
            dist_relation = CCDistRelation(position - other_position)
            if dist_relation in self.relation_types:
                yield position, other_position, dist_relation
            elif CCDefaultRelation() in self.relation_types:
                yield position, other_position, CCDefaultRelation()
        elif (
            CCForeignKeyForwardRelation() in self.relation_types
            and column_id in self.sql_schema.foreign_keys
            and self.sql_schema.foreign_keys[column_id] == other_column_id
        ):
            yield position, other_position, CCForeignKeyForwardRelation()
        elif (
            CCForeignKeyBackwardRelation() in self.relation_types
            and other_column_id in self.sql_schema.foreign_keys
            and self.sql_schema.foreign_keys[other_column_id] == column_id
        ):
            yield position, other_position, CCForeignKeyBackwardRelation()
        elif (
            CCTableMatchRelation() in self.relation_types
            and self.sql_schema.column_to_table[column_id]
            == self.sql_schema.column_to_table[other_column_id]
        ):
            yield position, other_position, CCTableMatchRelation()
        elif CCDefaultRelation() in self.relation_types:
            yield position, other_position, CCDefaultRelation()

    def _tt_relations(
        self,
        table_id: TableId,
        position: Pos,
        other_table_id: TableId,
        other_position: Pos,
    ) -> Generator[
        Tuple[
            Pos,
            Pos,
            Union[
                TTDefaultRelation,
                TTDistRelation,
                TTForeignKeyForwardRelation,
                TTForeignKeyBackwardRelation,
                TTForeignKeyBidirectionalRelation,
            ],
        ],
        None,
        None,
    ]:
        if table_id == other_table_id:
            # sibling positions share the same table
            relation = TTDistRelation(position - other_position)
            if relation in self.relation_types:
                yield position, other_position, relation
            elif TTDefaultRelation() in self.relation_types:
                yield position, other_position, TTDefaultRelation()
        elif (
            TTForeignKeyBidirectionalRelation() in self.relation_types
            and other_table_id
            in self.sql_schema.foreign_keys_tables.get(table_id, tuple())
            and table_id
            in self.sql_schema.foreign_keys_tables.get(other_table_id, tuple())
        ):
            yield position, other_position, TTForeignKeyBidirectionalRelation()
        elif TTForeignKeyForwardRelation() in self.relation_types and (
            other_table_id in self.sql_schema.foreign_keys_tables.get(table_id, tuple())
        ):
            yield position, other_position, TTForeignKeyForwardRelation()
        elif TTForeignKeyBackwardRelation() in self.relation_types and (
            table_id in self.sql_schema.foreign_keys_tables.get(other_table_id, tuple())
        ):
            yield position, other_position, TTForeignKeyBackwardRelation()
        elif TTDefaultRelation() in self.relation_types:
            yield position, other_position, TTDefaultRelation()

    def _ct_relations(
        self,
        column_id: ColumnId,
        column_position: Pos,
        table_id: TableId,
        table_position: Pos,
    ) -> Generator[
        Tuple[
            Pos,
            Pos,
            Union[
                CTDefaultRelation,
                CTAnyTableRelation,
                CTForeignKeyRelation,
                CTPrimaryKeyRelation,
                CTTableMatchRelation,
                TCDefaultRelation,
                TCAnyTableRelation,
                TCForeignKeyRelation,
                TCPrimaryKeyRelation,
                TCTableMatchRelation,
            ],
        ],
        None,
        None,
    ]:
        if (
            column_id in self.sql_schema.foreign_keys
            and self.sql_schema.column_to_table[column_id]
            == self.sql_schema.column_to_table[self.sql_schema.foreign_keys[column_id]]
        ):
            if CTForeignKeyRelation() in self.relation_types:
                yield column_position, table_position, CTForeignKeyRelation()
            elif CTDefaultRelation() in self.relation_types:
                yield column_position, table_position, CTDefaultRelation()
            if TCForeignKeyRelation() in self.relation_types:
                yield table_position, column_position, TCForeignKeyRelation()
            elif TCDefaultRelation() in self.relation_types:
                yield table_position, column_position, TCDefaultRelation()
        elif column_id not in self.sql_schema.column_to_table:
            # for the * wild card column
            if CTAnyTableRelation() in self.relation_types:
                yield column_position, table_position, CTAnyTableRelation()
            elif CTDefaultRelation() in self.relation_types:
                yield column_position, table_position, CTDefaultRelation()
            if TCAnyTableRelation() in self.relation_types:
                yield table_position, column_position, TCAnyTableRelation()
            elif TCDefaultRelation() in self.relation_types:
                yield table_position, column_position, TCDefaultRelation()
        elif (
            table_id == self.sql_schema.column_to_table[column_id]
            and column_id in self.sql_schema.primary_keys
        ):
            if CTPrimaryKeyRelation() in self.relation_types:
                yield column_position, table_position, CTPrimaryKeyRelation()
            elif CTDefaultRelation() in self.relation_types:
                yield column_position, table_position, CTDefaultRelation()
            if TCPrimaryKeyRelation() in self.relation_types:
                yield table_position, column_position, TCPrimaryKeyRelation()
            elif TCDefaultRelation() in self.relation_types:
                yield table_position, column_position, TCDefaultRelation()
        elif (
            table_id == self.sql_schema.column_to_table[column_id]
            and column_id not in self.sql_schema.primary_keys
        ):
            if CTTableMatchRelation() in self.relation_types:
                yield column_position, table_position, TCTableMatchRelation()
            elif CTDefaultRelation() in self.relation_types:
                yield column_position, table_position, CTDefaultRelation()
            if TCTableMatchRelation() in self.relation_types:
                yield table_position, column_position, TCTableMatchRelation()
            elif TCDefaultRelation() in self.relation_types:
                yield table_position, column_position, TCDefaultRelation()
        else:
            if CTDefaultRelation() in self.relation_types:
                yield column_position, table_position, CTDefaultRelation()
            if TCDefaultRelation() in self.relation_types:
                yield table_position, column_position, TCDefaultRelation()

    def _qc_relations(
        self,
        question_match_tags: Tuple[MatchTag, ...],
        question_position: Pos,
        column_id: ColumnId,
        column_position: Pos,
    ) -> Generator[
        Tuple[
            Pos,
            Pos,
            Union[
                QCDefaultRelation, QCMatchRelation, CQDefaultRelation, CQMatchRelation,
            ],
        ],
        None,
        None,
    ]:
        question_match_tags = list(
            match_tag
            for match_tag in question_match_tags
            if (
                isinstance(match_tag, ColumnMatchTag)
                or isinstance(match_tag, ValueMatchTag)
            )
            and match_tag.column_id == column_id
        )
        if len(question_match_tags) == 1:
            question_match_tag = question_match_tags[0]
        elif len(question_match_tags) == 0:
            question_match_tag = None
        else:
            # raise ValueError(f"Found more than 1 q-c matches: {question_match_tags}")
            logger.warning(f"Found more than 1 q-c matches: {question_match_tags}")
            question_match_tag = question_match_tags[0]
        if question_match_tag is not None:
            qc_match_relation = QCMatchRelation(
                confidence=question_match_tag.confidence,
                value_match=isinstance(question_match_tag, ValueMatchTag),
            )
            if qc_match_relation in self.relation_types:
                yield question_position, column_position, qc_match_relation
            elif QCDefaultRelation() in self.relation_types:
                yield question_position, column_position, QCDefaultRelation()
            cq_match_relation = CQMatchRelation(
                confidence=question_match_tag.confidence,
                value_match=isinstance(question_match_tag, ValueMatchTag),
            )
            if cq_match_relation in self.relation_types:
                yield column_position, question_position, cq_match_relation
            elif CQDefaultRelation() in self.relation_types:
                yield column_position, question_position, CQDefaultRelation()
        else:
            if QCDefaultRelation() in self.relation_types:
                yield question_position, column_position, QCDefaultRelation()
            if CQDefaultRelation() in self.relation_types:
                yield column_position, question_position, CQDefaultRelation()

    def _qt_relations(
        self,
        question_match_tags: Tuple[MatchTag, ...],
        question_position: Pos,
        table_id: TableId,
        table_position: Pos,
    ) -> Generator[
        Tuple[
            Pos,
            Pos,
            Union[
                QTDefaultRelation, QTMatchRelation, TQDefaultRelation, TQMatchRelation,
            ],
        ],
        None,
        None,
    ]:
        question_match_tags = list(
            match_tag
            for match_tag in question_match_tags
            if (
                isinstance(match_tag, TableMatchTag)
                # or isinstance(match_tag, ColumnMatchTag)
                # or isinstance(match_tag, ValueMatchTag)
            )
            and match_tag.table_id == table_id
        )
        if len(question_match_tags) == 1:
            question_match_tag = question_match_tags[0]
        elif len(question_match_tags) == 0:
            question_match_tag = None
        else:
            raise ValueError(f"Found more than 1 q-t matches: {question_match_tags}")

        if question_match_tag is not None:
            qt_match_relation = QTMatchRelation(
                confidence=question_match_tag.confidence,
                value_match=isinstance(question_match_tag, ValueMatchTag),
            )
            if qt_match_relation in self.relation_types:
                yield question_position, table_position, qt_match_relation
            elif QTDefaultRelation() in self.relation_types:
                yield question_position, table_position, QTDefaultRelation()
            tq_match_relation = TQMatchRelation(
                confidence=question_match_tag.confidence,
                value_match=isinstance(question_match_tag, ValueMatchTag),
            )
            if tq_match_relation in self.relation_types:
                yield table_position, question_position, tq_match_relation
            elif TQDefaultRelation() in self.relation_types:
                yield table_position, question_position, TQDefaultRelation()
        else:
            if QTDefaultRelation() in self.relation_types:
                yield question_position, table_position, QTDefaultRelation()
            if TQDefaultRelation() in self.relation_types:
                yield table_position, question_position, TQDefaultRelation()

    def _source_relations(
        self, source_token: Token[InputId, str]
    ) -> Generator[
        Tuple[Pos, Pos, SourceRelation], None, None,
    ]:
        if isinstance(source_token, QuestionToken):
            self.source_question_tokens.append(source_token)
            for other_token in self.source_question_tokens:
                yield from self._qq_relations(
                    position=source_token.position, other_position=other_token.position
                )
                if source_token.position != other_token.position:
                    yield from self._qq_relations(
                        position=other_token.position,
                        other_position=source_token.position,
                    )
            for column_id, column_positions in self.columns_position_map.items():
                for column_position in column_positions:
                    yield from self._qc_relations(
                        question_match_tags=source_token.match_tags,
                        question_position=source_token.position,
                        column_id=column_id,
                        column_position=column_position,
                    )
            for table_id, table_positions in self.tables_position_map.items():
                for table_position in table_positions:
                    yield from self._qt_relations(
                        question_match_tags=source_token.match_tags,
                        question_position=source_token.position,
                        table_id=table_id,
                        table_position=table_position,
                    )
        elif isinstance(source_token, ColumnToken):
            self.columns_position_map[source_token.key].append(source_token.position)
            assert source_token.key in self.columns_position_map
            for other_column_id, other_positions in self.columns_position_map.items():
                for other_position in other_positions:
                    yield from self._cc_relations(
                        column_id=source_token.key,
                        position=source_token.position,
                        other_column_id=other_column_id,
                        other_position=other_position,
                    )
                    if source_token.position != other_position:
                        yield from self._cc_relations(
                            column_id=other_column_id,
                            position=other_position,
                            other_column_id=source_token.key,
                            other_position=source_token.position,
                        )
            for table_id, table_positions in self.tables_position_map.items():
                for table_position in table_positions:
                    yield from self._ct_relations(
                        column_id=source_token.key,
                        column_position=source_token.position,
                        table_id=table_id,
                        table_position=table_position,
                    )
            for question_token in self.source_question_tokens:
                yield from self._qc_relations(
                    question_match_tags=question_token.match_tags,
                    question_position=question_token.position,
                    column_id=source_token.key,
                    column_position=source_token.position,
                )
        elif isinstance(source_token, TableToken):
            for column_id, column_positions in self.columns_position_map.items():
                for column_position in column_positions:
                    yield from self._ct_relations(
                        column_id=column_id,
                        column_position=column_position,
                        table_id=source_token.key,
                        table_position=source_token.position,
                    )
            self.tables_position_map[source_token.key].append(source_token.position)
            assert source_token.key in self.tables_position_map
            for other_table_id, other_positions in self.tables_position_map.items():
                for other_position in other_positions:
                    yield from self._tt_relations(
                        table_id=source_token.key,
                        position=source_token.position,
                        other_table_id=other_table_id,
                        other_position=other_position,
                    )
                    if source_token.position != other_position:
                        yield from self._tt_relations(
                            table_id=other_table_id,
                            position=other_position,
                            other_table_id=source_token.key,
                            other_position=source_token.position,
                        )
            for question_token in self.source_question_tokens:
                yield from self._qt_relations(
                    question_match_tags=question_token.match_tags,
                    question_position=question_token.position,
                    table_id=source_token.key,
                    table_position=source_token.position,
                )
        else:
            raise ValueError(
                "Unsupported token type: {}".format(source_token.__repr__())
            )

    def add_source_token(
        self, source_token: Token[InputId, str], copy: bool = False
    ) -> "SourceRelationsBuilder":
        builder = deepcopy(self) if copy is True else self

        for position, other_position, relation in builder._source_relations(
            source_token=source_token
        ):
            builder.sparse_2d_tensor_builder.append(
                index=(position, other_position),
                value=builder.relation_types[relation],
            )

        builder.sparse_2d_tensor_builder.resize(
            size=(1 + source_token.position, 1 + source_token.position)
        )

        return builder

    def add_source_tokens(
        self, source_tokens: Iterable[Token[InputId, str]], copy: bool = False
    ) -> "SourceRelationsBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in source_tokens:
            builder.add_source_token(source_token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_tensor_builder.build(device=device)


def mask_source_relation_tensor(
    _relation_tensor: torch.Tensor,
    relation_types: FrozenDict[SourceRelation, int],
    mask_token_mask: torch.Tensor,
) -> torch.Tensor:
    # No masking (yet) to perform on source relations
    return _relation_tensor


@dataclass
class MemoryRelationsBuilder(object):
    source_tokens: Sequence[Token[InputId, str]]
    relation_types: FrozenDict[MemoryRelation, int]
    question_position_map: PositionMap[str] = field(
        default_factory=lambda: defaultdict(deque)
    )
    columns_position_map: PositionMap[str] = field(
        default_factory=lambda: defaultdict(deque)
    )
    tables_position_map: PositionMap[str] = field(
        default_factory=lambda: defaultdict(deque)
    )
    source_token_max_position: int = field(init=False)
    sparse_2d_tensor_builder: Sparse2DTensorBuilder = field(
        default_factory=lambda: Sparse2DTensorBuilder()
    )

    def __deepcopy__(self, memo) -> "MemoryRelationsBuilder":
        builder = copy(self)
        builder.question_position_map = defaultdict(deque)
        for s, positions in self.question_position_map.items():
            builder.question_position_map[s] = copy(positions)
        builder.columns_position_map = defaultdict(deque)
        for s, positions in self.columns_position_map.items():
            builder.columns_position_map[s] = copy(positions)
        builder.tables_position_map = defaultdict(deque)
        for s, positions in self.tables_position_map.items():
            builder.tables_position_map[s] = copy(positions)
        builder.sparse_2d_tensor_builder = deepcopy(self.sparse_2d_tensor_builder)
        return builder

    def __post_init__(self):
        for token in self.source_tokens:
            if isinstance(token, QuestionToken):
                self.question_position_map[token.value].append(token.position)
            elif isinstance(token, ColumnToken):
                self.columns_position_map[token.key].append(token.position)
            elif isinstance(token, TableToken):
                self.tables_position_map[token.key].append(token.position)
            else:
                raise ValueError("Unsupported token type: {}".format(token.__repr__()))
        self.source_token_max_position = max(
            map(lambda _token: _token.position, self.source_tokens)
        )

    def _copied_from_relations(
        self, token: Token[KT, ActionInfo]
    ) -> Generator[Tuple[Pos, Pos, TargetRelation], None, None]:
        if CopiedFromRelation() in self.relation_types:
            # Root, Productions, </primitive> token, inference MASK token: cannot copy
            if (
                token.value.frontier_field is None
                or isinstance(token.value.action, ReduceAction)
                or isinstance(token.value.action, MaskAction)
                or not isinstance(token.value.frontier_field.type, ASDLPrimitiveType)
                or token.value.action is None
                # or action_info.action.token == END_STRING_TOKEN
            ):
                position_map: Dict[str, Deque[int]] = dict()
            # Primitives: can possibly copy
            else:
                assert isinstance(token.value.frontier_field.type, ASDLPrimitiveType)
                if token.value.frontier_field.type.name == "column":
                    # position of all the columns in the memory
                    position_map = self.columns_position_map
                elif token.value.frontier_field.type.name == "table":
                    # position of all the tables in the memory
                    position_map = self.tables_position_map
                else:
                    # position of question tokens in the memory
                    position_map = self.question_position_map

            if (
                position_map
                and isinstance(token.value.action, GenTokenAction)
                and token.value.action.token in position_map
            ):
                for position in position_map[token.value.action.token]:
                    yield token.position, position, CopiedFromRelation()

    def add_token(
        self, token: Token[KT, ActionInfo], copy: bool = False
    ) -> "MemoryRelationsBuilder":
        builder = deepcopy(self) if copy is True else self
        for position, other_position, relation in itertools.chain(
            builder._copied_from_relations(token=token),
        ):
            builder.sparse_2d_tensor_builder.append(
                index=(position, other_position),
                value=builder.relation_types[relation],
            )

        builder.sparse_2d_tensor_builder.resize(
            size=(1 + token.position, 1 + builder.source_token_max_position)
        )

        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, ActionInfo]], copy: bool = False
    ) -> "MemoryRelationsBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_tensor_builder.build(device=device)


def mask_memory_relation_tensor(
    _relation_tensor: torch.Tensor,
    relation_types: FrozenDict[MemoryRelation, int],
    mask_token_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_tgt_length, _max_src_length = _relation_tensor.shape
    assert (batch_size, max_tgt_length) == mask_token_mask.shape

    if CopiedFromRelation() in relation_types:
        # Remove copied_from relation if input is MASK token (having the copied_from relation would be cheating)
        return _relation_tensor.masked_fill(
            mask=mask_token_mask.unsqueeze(2),
            value=relation_types[DefaultMemoryRelation()],
        )


@dataclass
class TargetRelationsBuilder(object):
    relation_types: FrozenDict[TargetRelation, int]
    parents: Dict[Pos, Deque[Token[KT, ActionInfo]]] = field(default_factory=dict)
    sparse_2d_tensor_builder: Sparse2DTensorBuilder = field(
        default_factory=lambda: Sparse2DTensorBuilder()
    )

    def __deepcopy__(self, memo) -> "TargetRelationsBuilder":
        builder = copy(self)
        builder.parents = dict(
            (pos, copy(tokens)) for pos, tokens in self.parents.items()
        )
        builder.sparse_2d_tensor_builder = deepcopy(self.sparse_2d_tensor_builder)
        return builder

    def _identity_action_relations(
        self, token: Token[KT, ActionInfo]
    ) -> Generator[Tuple[Pos, Pos, TargetRelation], None, None]:
        if IdentityActionRelation() in self.relation_types:
            yield token.position, token.position, IdentityActionRelation()

    def _parent_child_relations(
        self, token: Token[KT, ActionInfo]
    ) -> Generator[Tuple[Pos, Pos, TargetRelation], None, None]:
        if token.value.parent_pos is not None:
            if ChildParentRelation() in self.relation_types:
                yield (
                    token.position,
                    token.value.parent_pos,
                    ChildParentRelation(),
                )
            if ParentChildRelation() in self.relation_types:
                yield (
                    token.value.parent_pos,
                    token.position,
                    ParentChildRelation(),
                )

    def _sibling_relations(
        self, token: Token[KT, ActionInfo]
    ) -> Generator[Tuple[Pos, Pos, TargetRelation], None, None]:
        if any(
            isinstance(relation, SiblingDistRelation)
            for relation in self.relation_types.keys()
        ):
            if token.value.parent_pos is not None:
                sibling_tokens: Deque[Token[KT, ActionInfo]] = self.parents.get(
                    token.value.parent_pos, deque()
                )
                sibling_tokens.append(token)

                token_sib_idx = sibling_tokens.index(token)

                for sib_idx, sibling_token in enumerate(sibling_tokens):
                    relation = SiblingDistRelation(token_sib_idx - sib_idx)
                    if relation in self.relation_types:
                        yield token.position, sibling_token.position, relation
                    if token != sibling_token:
                        relation = SiblingDistRelation(sib_idx - token_sib_idx)
                        if relation in self.relation_types:
                            yield sibling_token.position, token.position, relation

                self.parents.update({token.value.parent_pos: sibling_tokens})

    def add_token(
        self, token: Token[KT, ActionInfo], copy: bool = False
    ) -> "TargetRelationsBuilder":
        builder = deepcopy(self) if copy is True else self

        for position, other_position, relation in itertools.chain(
            builder._identity_action_relations(token=token),
            builder._parent_child_relations(token=token),
            builder._sibling_relations(token=token),
        ):
            builder.sparse_2d_tensor_builder.append(
                index=(position, other_position),
                value=builder.relation_types[relation],
            )

        builder.sparse_2d_tensor_builder.resize(
            size=(1 + token.position, 1 + token.position)
        )

        return builder

    def add_tokens(
        self, tokens: Iterable[Token[KT, ActionInfo]], copy: bool = False
    ) -> "TargetRelationsBuilder":
        builder = deepcopy(self) if copy is True else self
        for token in tokens:
            builder.add_token(token=token)
        return builder

    def build(self, device: torch.device) -> torch.Tensor:
        return self.sparse_2d_tensor_builder.build(device=device)


def mask_target_relation_tensor(
    _relation_tensor: torch.Tensor,
    relation_types: FrozenDict[TargetRelation, int],
    mask_token_mask: torch.Tensor,
) -> torch.Tensor:
    # No masking (yet) to perform on target relations
    return _relation_tensor
