import abc
import json
from collections import deque

import _jsonnet
from dataclasses import replace
from typing import (
    Tuple,
    Generator,
    Optional,
    Callable,
    Iterable,
    Sequence,
)

from duorat.preproc.slml import SLMLBuilder
from duorat.preproc.utils import has_subsequence
from duorat.types import (
    SQLSchema,
    ValueMatchTag,
    HighConfidenceMatch,
    TableMatchTag,
    ColumnMatchTag,
    TaggedToken,
    TaggedSequence,
    LowConfidenceMatch,
    MatchConfidence,
)
from duorat.utils import registry
from duorat.utils.db_content import (
    pre_process_words,
    match_db_content,
    EntryType,
)
from duorat.utils.tokenization import AbstractTokenizer


class AbstractSchemaLinker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def question_to_slml(self, question: str, sql_schema: SQLSchema) -> str:
        pass


MATCH_CONFIDENCE = {
    "high": HighConfidenceMatch(),
    "low": LowConfidenceMatch(),
    "none": None,
}


@registry.register("schema_linker", "SpiderSchemaLinker")
class SpiderSchemaLinker(AbstractSchemaLinker):
    def __init__(
        self,
        tokenizer: dict,
        max_n_gram: int = 5,
        with_stemming: bool = False,
        blocking_match: bool = True,
        whole_entry_db_content_confidence: str = "high",
        partial_entry_db_content_confidence: str = "low",
    ):
        super(SpiderSchemaLinker, self).__init__()
        self.max_n_gram = max_n_gram
        self.with_stemming = with_stemming
        self.blocking_match = blocking_match
        self.whole_entry_db_content_confidence = MATCH_CONFIDENCE[
            whole_entry_db_content_confidence
        ]
        self.partial_entry_db_content_confidence = MATCH_CONFIDENCE[
            partial_entry_db_content_confidence
        ]
        self.tokenizer: AbstractTokenizer = registry.construct("tokenizer", tokenizer)

    def question_to_slml(self, question: str, sql_schema: SQLSchema,) -> str:
        def is_chinese(tok):
            if '\u4e00' <= tok <= '\u9fff' or tok == '>' or tok == '\"':
                return True
            else:
                return False

        def chinese_tokenize(question):
            tokenized_question = []
            en_str = ''
            for token in question:
                if is_chinese(token):
                    if en_str != '':
                        tokenized_question.append((en_str, en_str))
                        en_str = ''
                    tokenized_question.append((token, token))
                else:
                    en_str += token
            if en_str != '':
                tokenized_question.append((en_str, en_str))
            return tokenized_question
        tagged_question_tokens = tag_question_with_schema_links(
            tokenized_question=chinese_tokenize(question),
            sql_schema=sql_schema,
            tokenize=self.tokenizer.tokenize,
            max_n_gram=self.max_n_gram,
            with_stemming=self.with_stemming,
            blocking_match=self.blocking_match,
            whole_entry_db_content_confidence=self.whole_entry_db_content_confidence,
            partial_entry_db_content_confidence=self.partial_entry_db_content_confidence,
        )
        slml_builder = SLMLBuilder(
            sql_schema=sql_schema, detokenize=self.tokenizer.detokenize
        )
        slml_builder.add_question_tokens(question_tokens=tagged_question_tokens)
        slml_question = slml_builder.build()
        return slml_question


def tag_question_with_schema_links(
    tokenized_question: Iterable[Tuple[str, str]],
    sql_schema: SQLSchema,
    tokenize: Callable[[str], Sequence[str]],
    max_n_gram: int,
    with_stemming: bool,
    blocking_match: bool,
    whole_entry_db_content_confidence: Optional[MatchConfidence],
    partial_entry_db_content_confidence: Optional[MatchConfidence],
) -> TaggedSequence:
    """

    :param tokenized_question:
    :param sql_schema:
    :param tokenize:
    :param max_n_gram:
    :param with_stemming:
    :param blocking_match:
    :param whole_entry_db_content_confidence:
    :param partial_entry_db_content_confidence:
    :return:
    """

    def _entry_type_to_confidence(entry_type):
        if entry_type is EntryType.WHOLE_ENTRY:
            return whole_entry_db_content_confidence
        else:
            return partial_entry_db_content_confidence

    # init BIO tags
    tagged_question_tokens = [
        TaggedToken(value=token, raw_value=raw_token, tag=OUTSIDE, match_tags=deque())
        for token, raw_token in tokenized_question
    ]

    for n in range(max_n_gram, 0, -1):
        for (start, end), question_n_gram in get_spans(
            tagged_sequence=tagged_question_tokens, n=n
        ):
            # Try to match column names
            for column_id, column_name in sql_schema.column_names.items():
                if (
                    column_id
                    not in sql_schema.column_to_table
                    # or sql_schema.column_to_table[column_id] is None
                ):
                    continue
                table_id = sql_schema.column_to_table[column_id]
                match = span_matches_entity(
                    tagged_span=question_n_gram,
                    entity_name=column_name,
                    tokenize=tokenize,
                    with_stemming=with_stemming,
                )

                # Try to match using db-content only if a column match did not succeed.
                # That means column matches have precedence!
                db_content_matches = []
                if match is NO_MATCH:
                    if table_id is not None:
                        db_content_matches = match_db_content(
                            [token.value for token in question_n_gram],
                            sql_schema.original_column_names[column_id],
                            sql_schema.original_table_names[table_id],
                            sql_schema.db_id,
                            sql_schema.db_path,
                            with_stemming=with_stemming,
                        )
                        # Filter non-None confidence.
                        db_content_matches = [
                            entry
                            for entry in db_content_matches
                            if _entry_type_to_confidence(entry[0]) is not None
                        ]
                        # Tag the sequence if a match was found
                        if db_content_matches:
                            match = VALUE_MATCH

                # Tag the sequence if a match was found
                if match is not NO_MATCH:
                    # Block the sequence
                    if blocking_match:
                        set_tags(
                            tagged_sequence=tagged_question_tokens, start=start, end=end
                        )
                    for idx in range(start, end):
                        if match is VALUE_MATCH:
                            # Only keep the match with the highest confidence
                            entry_type, match_value = max(
                                db_content_matches,
                                key=lambda match: _entry_type_to_confidence(match[0])
                            )
                            match_tag = ValueMatchTag(
                                confidence=_entry_type_to_confidence(entry_type),
                                column_id=column_id,
                                table_id=table_id,
                                value=match_value,
                            )
                            tagged_question_tokens[idx].match_tags.append(match_tag)
                        else:
                            match_tag = ColumnMatchTag(
                                confidence=(
                                    HighConfidenceMatch()
                                    if match == EXACT_MATCH
                                    else LowConfidenceMatch()
                                ),
                                column_id=column_id,
                                table_id=table_id,
                            )
                            tagged_question_tokens[idx].match_tags.append(match_tag)

    # reset BIO tags
    tagged_question_tokens: TaggedSequence = [
        replace(t, tag=OUTSIDE) for t in tagged_question_tokens
    ]

    for n in range(max_n_gram, 0, -1):
        for (start, end), question_n_gram in get_spans(
            tagged_sequence=tagged_question_tokens, n=n
        ):
            # Try to match table names
            for table_id, table_name in sql_schema.table_names.items():
                match = span_matches_entity(
                    tagged_span=question_n_gram,
                    entity_name=table_name,
                    tokenize=tokenize,
                    with_stemming=with_stemming,
                )

                # Tag the sequence if a match was found
                if match is not NO_MATCH:
                    # Block the sequence
                    if blocking_match:
                        set_tags(
                            tagged_sequence=tagged_question_tokens, start=start, end=end
                        )
                    for idx in range(start, end):
                        tagged_question_tokens[idx].match_tags.append(
                            TableMatchTag(
                                confidence=(
                                    HighConfidenceMatch()
                                    if match == EXACT_MATCH
                                    else LowConfidenceMatch()
                                ),
                                table_id=table_id,
                            )
                        )

    return tagged_question_tokens


BEGIN = "B"
INSIDE = "I"
OUTSIDE = "O"

EXACT_MATCH = "exact_match"
PARTIAL_MATCH = "partial_match"
VALUE_MATCH = "value_match"
NO_MATCH = None


def set_tags(tagged_sequence: TaggedSequence, start: int, end: int) -> None:
    if start < end:
        tagged_sequence[start].tag = BEGIN
    for idx in range(start + 1, end):
        tagged_sequence[idx].tag = INSIDE


def get_spans(
    tagged_sequence: TaggedSequence, n: int
) -> Generator[Tuple[Tuple[int, int], TaggedSequence], None, None]:
    """Generate untagged spans from sequence"""
    start = 0
    end = n
    while end <= len(tagged_sequence):
        # yield span only if not yet tagged
        if all(
            [tagged_token.tag == OUTSIDE for tagged_token in tagged_sequence[start:end]]
        ):
            yield (start, end), tagged_sequence[start:end]
        start += 1
        end += 1


def span_matches_entity(
    tagged_span: TaggedSequence,
    entity_name: str,
    tokenize: Callable[[str], Sequence[str]],
    with_stemming: bool,
) -> Optional[str]:
    """
    Check if span and entity match (modulo stemming if desired)
    """
    if with_stemming:
        span_seq = pre_process_words(
            [tagged_token.value for tagged_token in tagged_span],
            with_stemming=with_stemming,
        )
        entity_name_seq = pre_process_words(
            tokenize(entity_name), with_stemming=with_stemming
        )
        if span_seq == entity_name_seq:
            return EXACT_MATCH
        elif has_subsequence(seq=entity_name_seq, subseq=span_seq):
            return PARTIAL_MATCH
        else:
            return NO_MATCH
    else:
        span_str = " ".join([tagged_token.value for tagged_token in tagged_span])
        entity_name_str = entity_name
        if span_str == entity_name_str:
            return EXACT_MATCH
        elif span_str in entity_name_str:
            return PARTIAL_MATCH
        elif len(tagged_span) == 1 and entity_name_str in span_str:
            # When span length is 1, also test the other inclusion
            return PARTIAL_MATCH
        else:
            return NO_MATCH
