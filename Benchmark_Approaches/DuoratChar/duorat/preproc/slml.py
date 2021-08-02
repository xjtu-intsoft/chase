import itertools
import re
from collections import deque, defaultdict
from copy import deepcopy
from html import unescape, escape
from html.parser import HTMLParser
from random import Random
from typing import (
    Deque,
    List,
    Callable,
    Tuple,
    Dict,
    Optional,
    Sequence,
    Iterable,
    Union,
)
from uuid import UUID

from duorat.types import (
    PreprocQuestionToken,
    MatchTag,
    TableMatchTag,
    SQLSchema,
    HighConfidenceMatch,
    LowConfidenceMatch,
    TableId,
    MatchConfidence,
    ColumnId,
    ColumnMatchTag,
    ValueMatchTag,
    QuestionTokenId,
    QuestionToken,
    TaggedToken,
)
from duorat.utils.tokenization import AbstractTokenizer


class SLMLParser(HTMLParser):
    """
    Schema Linking Markup Language Parser (SLML).
    a human readable and writable representation of tags that is using the familiar HTML syntax and tooling
    (syntax highlighting, auto formatting, parsing, etc.).
    SLML provides an easy and convenient way to annotate and read schema linking information in the user’s question.
    """

    _question_tokens: Deque[PreprocQuestionToken]
    _match_stack: Deque[MatchTag]
    _tokenizer: AbstractTokenizer
    _rd: Random
    _initial_seed: int

    def __init__(
        self, sql_schema: SQLSchema, tokenizer: AbstractTokenizer, seed: int = 0,
    ):
        self._initial_seed = seed
        super(SLMLParser, self).__init__()
        self.sql_schema = sql_schema
        self._tokenizer = tokenizer

    def reset(self) -> None:
        super(SLMLParser, self).reset()
        self._question_tokens = deque()
        self._match_stack = deque()
        self._rd = Random()
        self._rd.seed(self._initial_seed)

    @property
    def question_tokens(self) -> Tuple[PreprocQuestionToken, ...]:
        return tuple(self._question_tokens)

    @staticmethod
    def _handle_confidence(tag: str, _attrs: Dict[str, str]) -> MatchConfidence:
        """
        Try to match the match confidence level specified in the tag to a valid confidence level.
        Currently, we support only high or low confidence matches.
        """
        if "confidence" in _attrs:
            if _attrs["confidence"].lower() == "high":
                confidence = HighConfidenceMatch()
            elif _attrs["confidence"].lower() == "low":
                confidence = LowConfidenceMatch()
            else:
                raise ValueError(
                    "Invalid confidence value `{}` in tag `{}`".format(
                        _attrs["confidence"], tag
                    )
                )
        else:
            # If confidence value is missing, assume high confidence by default
            confidence = HighConfidenceMatch()
        return confidence

    def _handle_table(self, tag: str, _attrs: Dict[str, str]) -> TableId:
        """
        Try to find the table id corresponding to the table name annotation in the tag.
        """
        if "table" in _attrs:
            table_id: Optional[TableId] = None
            for _table_id, _table_name in self.sql_schema.original_table_names.items():
                if _table_name.lower() == unescape(_attrs["table"]).lower():
                    table_id = _table_id
            if table_id is None:
                raise ValueError(
                    "Invalid table name `{}` in tag `{}`".format(_attrs["table"], tag)
                )
        else:
            raise ValueError("Table name missing from tag `{}`".format(tag))
        return table_id

    def _handle_column(
        self, tag: str, _attrs: Dict[str, str]
    ) -> Tuple[TableId, ColumnId]:
        """
        Try to find the column id and table id corresponding to the column and table name
        annotations in the tag.

        We have to do a joint comparison of column and table names because column names may not be
        unique in the schema.
        """
        if "table" in _attrs:
            table_id: Optional[TableId] = None
            if "column" in _attrs:
                column_id: Optional[ColumnId] = None
                for (
                    _column_id,
                    _column_name,
                ) in self.sql_schema.original_column_names.items():
                    if _column_id in self.sql_schema.column_to_table:
                        _table_id = self.sql_schema.column_to_table[_column_id]
                        if _table_id is not None:
                            _table_name = self.sql_schema.original_table_names[
                                _table_id
                            ]
                            if (
                                _table_name.lower() == unescape(_attrs["table"]).lower()
                                and _column_name.lower()
                                == unescape(_attrs["column"]).lower()
                            ):
                                table_id = _table_id
                                column_id = _column_id
                if table_id is None or column_id is None:
                    raise ValueError(
                        "Invalid table and/or column names `{}`, `{}`, "
                        "respectively, in tag `{}`".format(
                            _attrs["table"], _attrs["column"], tag
                        )
                    )
            else:
                raise ValueError("Column name missing from tag `{}`".format(tag))
        else:
            raise ValueError("Table name missing from tag `{}`".format(tag))
        return table_id, column_id

    @staticmethod
    def _handle_value(tag: str, _attrs: Dict[str, str]) -> str:
        """
        Try to extract the value annotation in the tag.
        """
        if "value" in _attrs:
            return unescape(_attrs["value"])
        else:
            raise ValueError("Value missing from tag `{}`".format(tag))

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        """
        Handle the opening of new tags. If the tag is valid, a `MatchTag` object is pushed to the
        parser's internal stack.
        """
        # lower the case of the attribute keys
        _attrs = {k.lower(): v for k, v in attrs}

        # identify the tag
        if tag.lower() == "tm":
            # table match tag
            confidence = self._handle_confidence(tag=tag, _attrs=_attrs)
            table_id = self._handle_table(tag=tag, _attrs=_attrs)
            self._match_stack.append(
                TableMatchTag(confidence=confidence, table_id=table_id)
            )
        elif tag.lower() == "cm":
            # column match tab
            confidence = self._handle_confidence(tag=tag, _attrs=_attrs)
            table_id, column_id = self._handle_column(tag=tag, _attrs=_attrs)
            self._match_stack.append(
                ColumnMatchTag(
                    confidence=confidence, table_id=table_id, column_id=column_id
                )
            )
        elif tag.lower() == "vm":
            # value match tag
            confidence = self._handle_confidence(tag=tag, _attrs=_attrs)
            table_id, column_id = self._handle_column(tag=tag, _attrs=_attrs)
            value = self._handle_value(tag=tag, _attrs=_attrs)
            self._match_stack.append(
                ValueMatchTag(
                    confidence=confidence,
                    table_id=table_id,
                    column_id=column_id,
                    value=value,
                )
            )
        else:
            # invalid tag
            raise ValueError("Invalid tag `{}`", tag)

    def handle_endtag(self, tag: str) -> None:
        """
        Handle the end of a tag. Tags only be closed if the last element in the parser's internal
        `MatchTag` stack matches the encountered tag. Otherwise, an exception is thrown.
        """
        if len(self._match_stack) > 0:
            match_tag = self._match_stack[-1]
            if isinstance(match_tag, TableMatchTag):
                if tag.lower() == "tm":
                    self._match_stack.pop()
                else:
                    raise ValueError(
                        "Unexpected attempt to close tag `{}`, expected `{}`".format(
                            tag, "tm"
                        )
                    )
            elif isinstance(match_tag, ColumnMatchTag):
                if tag.lower() == "cm":
                    self._match_stack.pop()
                else:
                    raise ValueError(
                        "Unexpected attempt to close tag `{}`, expected `{}`".format(
                            tag, "cm"
                        )
                    )
            elif isinstance(match_tag, ValueMatchTag):
                if tag.lower() == "vm":
                    self._match_stack.pop()
                else:
                    raise ValueError(
                        "Unexpected attempt to close tag `{}`, expected `{}`".format(
                            tag, "vm"
                        )
                    )
            else:
                raise RuntimeError("Invalid match tag state `{}`".format(match_tag))
        else:
            raise ValueError(
                "Trying to close tag `{}` although no tag has been opened yet".format(
                    tag
                )
            )

    def handle_data(self, data: str) -> None:
        """
        Handle every piece of input string data that is not tags. We HTML unescape the string, tokenize it, look up
        the opened tags, and emit tokens attached with all the matching data that is in the current context.
        """
        match_tags: Tuple[MatchTag, ...] = tuple(self._match_stack)
        for token, raw_token in self._tokenizer.tokenize_with_raw(unescape(data)):
            self._question_tokens.append(
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID(int=self._rd.getrandbits(128))),
                    value=token,
                    raw_value=raw_token,
                    match_tags=match_tags,
                )
            )


class SLMLBuilder(object):
    """
    Schema Linking Markup Language Builder
    """

    _raw_question_tokens: Deque[str]
    _match_tag_map: Dict[MatchTag, Deque[int]]
    _detokenize: Callable[[Sequence[str]], str]

    def __init__(
        self, sql_schema: SQLSchema, detokenize: Callable[[Sequence[str]], str],
    ):
        super(SLMLBuilder, self).__init__()
        self.sql_schema = sql_schema
        self._raw_question_tokens = deque()
        self._match_tag_map = defaultdict(deque)
        self._detokenize = detokenize

    def add_question_token(
        self,
        question_token: Union[TaggedToken, PreprocQuestionToken, QuestionToken[str]],
        copy: bool = False,
    ) -> "SLMLBuilder":
        builder = deepcopy(self) if copy is True else self
        for match_tag in set(question_token.match_tags):
            builder._match_tag_map[match_tag].append(len(builder._raw_question_tokens))
        builder._raw_question_tokens.append(question_token.raw_value)
        return builder

    def add_question_tokens(
        self,
        question_tokens: Iterable[
            Union[TaggedToken, PreprocQuestionToken, QuestionToken[str]]
        ],
        copy: bool = False,
    ) -> "SLMLBuilder":
        builder = deepcopy(self) if copy is True else self
        for question_token in question_tokens:
            builder.add_question_token(question_token=question_token)
        return builder

    def build(self) -> str:
        match_tag_spans: Dict[int, Deque[Tuple[int, int, MatchTag]]] = defaultdict(
            deque
        )

        for match_tag, positions in self._match_tag_map.items():
            spans: Deque[Tuple[int, int]] = deque()
            for position in sorted(positions):
                if len(spans) > 0 and spans[-1][1] + 1 == position:
                    spans[-1] = (spans[-1][0], position)
                else:
                    spans.append((position, position))

            for span in spans:
                span_len = span[1] - span[0] + 1
                match_tag_spans[span_len].append((span[0], span[1], match_tag))

        for span_len, _spans in sorted(
            match_tag_spans.items(), key=lambda t: t[0], reverse=True
        ):
            new_spans: Deque[Tuple[int, int, MatchTag]] = deque()
            for _span in _spans:
                span_split = False
                for other_span in itertools.chain(
                    *(
                        other_spans
                        for other_span_len, other_spans in match_tag_spans.items()
                        if other_span_len > span_len
                    ),
                    new_spans
                ):
                    if _span[0] < other_span[0] <= _span[1]:
                        # other_span: _ _ x x x _ _
                        # _span:      _ x x x _ _ _
                        # 1st split:  _ x _ _ _ _ _
                        # 2nd split:  _ _ x x _ _ _
                        match_tag_spans[_span[1] - other_span[0] + 1].appendleft(
                            (other_span[0], _span[1], _span[2])
                        )
                        match_tag_spans[other_span[0] - _span[0]].appendleft(
                            (_span[0], other_span[0] - 1, _span[2])
                        )
                        span_split = True
                        break
                    elif _span[0] <= other_span[1] < _span[1]:
                        # other_span: _ _ x x x _ _
                        # _span:      _ _ _ x x x _
                        # 1st split:  _ _ _ x x _ _
                        # 2nd split:  _ _ _ _ _ x _
                        match_tag_spans[_span[1] - other_span[1]].appendleft(
                            (other_span[1] + 1, _span[1], _span[2])
                        )
                        match_tag_spans[other_span[1] - _span[0] + 1].appendleft(
                            (_span[0], other_span[1], _span[2])
                        )
                        span_split = True
                        break
                    else:
                        pass
                if not span_split:
                    new_spans.append(_span)
            match_tag_spans[span_len] = new_spans

        slml_question: Deque[str] = deque()
        for idx, s in enumerate(self._raw_question_tokens):
            open_tags: Deque[str] = deque()
            close_tags: Deque[str] = deque()
            for span_len, _spans in sorted(
                match_tag_spans.items(), key=lambda t: t[0], reverse=True
            ):
                for (start, end, match_tag) in sorted(_spans, key=lambda t: t[2]):
                    if idx == start:
                        if isinstance(match_tag, TableMatchTag):
                            table_name = escape(
                                self.sql_schema.original_table_names[match_tag.table_id]
                            )
                            if match_tag.confidence == HighConfidenceMatch():
                                confidence = "high"
                            elif match_tag.confidence == LowConfidenceMatch():
                                confidence = "low"
                            else:
                                raise ValueError(
                                    "Invalid confidence {}".format(match_tag.confidence)
                                )
                            open_tags.append(
                                '<tm table="{}" confidence="{}">'.format(
                                    table_name, confidence
                                )
                            )
                        elif isinstance(match_tag, ColumnMatchTag):
                            table_name = escape(
                                self.sql_schema.original_table_names[match_tag.table_id]
                            )
                            column_name = escape(
                                self.sql_schema.original_column_names[
                                    match_tag.column_id
                                ]
                            )
                            if match_tag.confidence == HighConfidenceMatch():
                                confidence = "high"
                            elif match_tag.confidence == LowConfidenceMatch():
                                confidence = "low"
                            else:
                                raise ValueError(
                                    "Invalid confidence {}".format(match_tag.confidence)
                                )
                            open_tags.append(
                                '<cm table="{}" column="{}" confidence="{}">'.format(
                                    table_name, column_name, confidence
                                )
                            )
                        elif isinstance(match_tag, ValueMatchTag):
                            table_name = escape(
                                self.sql_schema.original_table_names[match_tag.table_id]
                            )
                            column_name = escape(
                                self.sql_schema.original_column_names[
                                    match_tag.column_id
                                ]
                            )
                            value = escape(match_tag.value)
                            if match_tag.confidence == HighConfidenceMatch():
                                confidence = "high"
                            elif match_tag.confidence == LowConfidenceMatch():
                                confidence = "low"
                            else:
                                raise ValueError(
                                    "Invalid confidence {}".format(match_tag.confidence)
                                )
                            open_tags.append(
                                '<vm table="{}" column="{}" value="{}" confidence="{}">'.format(
                                    table_name, column_name, value, confidence
                                )
                            )
                        else:
                            raise ValueError("Invalid match tag {}".format(match_tag))
                    if idx == end:
                        if isinstance(match_tag, TableMatchTag):
                            close_tags.appendleft("</tm>")
                        elif isinstance(match_tag, ColumnMatchTag):
                            close_tags.appendleft("</cm>")
                        elif isinstance(match_tag, ValueMatchTag):
                            close_tags.appendleft("</vm>")
                        else:
                            raise ValueError("Invalid match tag {}".format(match_tag))
            slml_question.append("".join(open_tags) + escape(s) + "".join(close_tags))

        return self._detokenize(slml_question)


def pretty_format_slml(slml: str):
    """Make every tag and content word start from a new line."""
    slml = re.sub("\n", "", slml)
    slml = re.sub("([^^])<", "\g<1>\n<", slml)
    slml = re.sub(">([^ \n])", ">\n\g<1>", slml)
    return slml
