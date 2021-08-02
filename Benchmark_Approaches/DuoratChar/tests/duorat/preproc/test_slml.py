from html import escape
from typing import List, Tuple, Optional
from uuid import UUID

import pytest

from duorat.preproc.slml import SLMLParser, SLMLBuilder
from duorat.types import (
    SQLSchema,
    RATPreprocItem,
    ColumnMatchTag,
    HighConfidenceMatch,
    ColumnId,
    TableId,
    TableMatchTag,
    MatchTag,
    ValueMatchTag,
    LowConfidenceMatch,
    PreprocQuestionToken,
    QuestionTokenId,
)


@pytest.fixture()
def sql_schema(preproc_data: List[RATPreprocItem]) -> SQLSchema:
    return preproc_data[0].sql_schema


@pytest.mark.parametrize(
    "question,expected",
    [
        ("", []),
        (
            """
            how many
            <tm table="singer" confidence="high">
                singers
            </tm>
            do we have?
            """,
            [
                ("how", ()),
                ("many", ()),
                (
                    "singers",
                    (
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("1")
                        ),
                    ),
                ),
                ("do", ()),
                ("we", ()),
                ("have?", ()),
            ],
        ),
        (
            """
            how many
            <tm table="singer">
                singers
            </tm>
            do we have?
            """,
            [
                ("how", ()),
                ("many", ()),
                (
                    "singers",
                    (
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("1")
                        ),
                    ),
                ),
                ("do", ()),
                ("we", ()),
                ("have?", ()),
            ],
        ),
        (
            escape(
                """
                how many
                <tm table="singer">
                    singers
                </tm>
                do we have?
                """
            ),
            [
                ("how", ()),
                ("many", ()),
                ("<tm", ()),
                ('table="singer">', ()),
                ("singers", ()),
                ("</tm>", ()),
                ("do", ()),
                ("we", ()),
                ("have?", ()),
            ],
        ),
        (
            """
            show
            <cm table="singer" column="name" confidence="high">
                name
            </cm>
            ,
            <cm table="singer" column="country" confidence="high">
                country
            </cm>
            ,
            <cm table="singer" column="age" confidence="high">
                age
            </cm>
            for all
            <tm table="singer" confidence="high">
                singers
            </tm>
            ordered by
            <cm table="singer" column="age" confidence="high">
                age
            </cm>
            from the oldest to the youngest.
            """,
            [
                ("show", ()),
                (
                    "name",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("9"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                (",", ()),
                (
                    "country",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("10"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                (",", ()),
                (
                    "age",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                ("for", ()),
                ("all", ()),
                (
                    "singers",
                    (
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("1")
                        ),
                    ),
                ),
                ("ordered", ()),
                ("by", ()),
                (
                    "age",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                ("from", ()),
                ("the", ()),
                ("oldest", ()),
                ("to", ()),
                ("the", ()),
                ("youngest.", ()),
            ],
        ),
        (
            """
            What is the average, minimum, and maximum
            <cm table="singer" column="age" confidence="high">
                age
            </cm>
            of all
            <tm table="singer" confidence="high">
                singers
            </tm>
            from
            <vm table="singer" column="country" value="France" confidence="high">
                France
            </vm>
            ?
            """,
            [
                ("What", ()),
                ("is", ()),
                ("the", ()),
                ("average,", ()),
                ("minimum,", ()),
                ("and", ()),
                ("maximum", ()),
                (
                    "age",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                ("of", ()),
                ("all", ()),
                (
                    "singers",
                    (
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("1")
                        ),
                    ),
                ),
                ("from", ()),
                (
                    "France",
                    (
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("10"),
                            table_id=TableId("1"),
                            value="France",
                        ),
                    ),
                ),
                ("?", ()),
            ],
        ),
        (
            """
            What are the
            <cm table="concert" column="concert_name" confidence="high">
                <tm table="concert" confidence="high">
                    <cm table="concert" column="concert_id" confidence="low">
                        <cm table="singer_in_concert" column="concert_id" confidence="low">
                            concert
                        </cm>
                    </cm>
                </tm>
                <cm table="stadium" column="name" confidence="high">
                    <cm table="singer" column="name" confidence="high">
                        <cm table="singer" column="song_name" confidence="low">
                            names
                        </cm>
                    </cm>
                </cm>
            </cm>
            ?
            """,
            [
                ("What", ()),
                ("are", ()),
                ("the", ()),
                (
                    "concert",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("16"),
                            table_id=TableId("2"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("2")
                        ),
                        ColumnMatchTag(
                            confidence=LowConfidenceMatch(),
                            column_id=ColumnId("15"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=LowConfidenceMatch(),
                            column_id=ColumnId("20"),
                            table_id=TableId("3"),
                        ),
                    ),
                ),
                (
                    "names",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("16"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("3"),
                            table_id=TableId("0"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("9"),
                            table_id=TableId("1"),
                        ),
                        ColumnMatchTag(
                            confidence=LowConfidenceMatch(),
                            column_id=ColumnId("11"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                ("?", ()),
            ],
        ),
        (
            """
            <cm table="concert" column="year">
                a
                <tm table="stadium">
                    b
                    <cm table="singer" column="age">
                        c
                    </cm>
                </tm>
                <cm table="singer" column="age">
                    <tm table="stadium">
                        d
                    </tm>
                    e
                    <vm table="concert" column="theme" value="f g h i j">
                        f g
                    </vm>
                </cm>
                <vm table="concert" column="theme" value="f g h i j">
                    h i j
                </vm>
                k l
            </cm>
            """,
            [
                (
                    "a",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                    ),
                ),
                (
                    "b",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("0")
                        ),
                    ),
                ),
                (
                    "c",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("0")
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                (
                    "d",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("0")
                        ),
                    ),
                ),
                (
                    "e",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                (
                    "f",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                (
                    "g",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                (
                    "h",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                (
                    "i",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                (
                    "j",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                (
                    "k",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                    ),
                ),
                (
                    "l",
                    (
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                    ),
                ),
            ],
        ),
    ],
)
def test_slml_parser(
    sql_schema: SQLSchema,
    question: str,
    expected: List[Tuple[str, Tuple[MatchTag, ...]]],
) -> None:
    parser = SLMLParser(sql_schema=sql_schema, tokenize=lambda s: s.split())
    parser.feed(data=question)
    parser.close()
    tokens = [(token.value, token.match_tags) for token in parser.question_tokens]
    assert tokens == expected


@pytest.mark.parametrize(
    "question,seed",
    [
        (
            """
            how many
            <tm table="singer">
                singers
            </tm>
            do we have?
            """,
            None,
        ),
        (
            """
            how many
            <tm table="singer">
                singers
            </tm>
            do we have?
            """,
            1,
        ),
    ],
)
def test_slml_parser_uuid_determinism(
    sql_schema: SQLSchema, question: str, seed: Optional[int],
):
    tokens: List[Tuple[PreprocQuestionToken, ...]] = []
    if seed is not None:
        parser_1 = SLMLParser(
            sql_schema=sql_schema, tokenize=lambda s: s.split(), seed=seed
        )
        parser_2 = SLMLParser(
            sql_schema=sql_schema, tokenize=lambda s: s.split(), seed=seed
        )
    else:
        parser_1 = SLMLParser(sql_schema=sql_schema, tokenize=lambda s: s.split())
        parser_2 = SLMLParser(sql_schema=sql_schema, tokenize=lambda s: s.split())
    for parser in [parser_1, parser_2]:
        parser.feed(data=question)
        parser.close()
        tokens.append(parser.question_tokens)
        parser.reset()
        parser.feed(data=question)
        parser.close()
        tokens.append(parser.question_tokens)
    for t in tokens:
        for tp in tokens:
            assert t == tp


def test_slml_parser_uuid_uniqueness(sql_schema: SQLSchema):
    question = " ".join(["a"] * 1000)
    for seed in range(0, 100, 1):
        parser = SLMLParser(
            sql_schema=sql_schema, tokenize=lambda s: s.split(), seed=seed
        )
        parser.feed(data=question)
        parser.close()
        uuids = [token.key for token in parser.question_tokens]
        assert len(uuids) == len(set(uuids))


@pytest.mark.parametrize(
    "question_tokens,expected",
    [
        ([], ""),
        (
            [
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("28509f71-b14d-4213-9537-044d30167091")),
                    value="how",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("2fd16bc4-db00-4aca-8cee-645ddf5ca754")),
                    value="many",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("e81f836d-5959-46ff-a97a-fdfc52d140de")),
                    value="singers",
                    match_tags=(
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("1")
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("a866f273-eff0-4da2-bbf6-d93bc5e491c0")),
                    value="do",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("0fe5573e-508f-4844-8ade-61e27072fe13")),
                    value="we",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("ec642fda-74f2-4f72-bd78-d503f89d8821")),
                    value="have?",
                    match_tags=(),
                ),
            ],
            'how many <tm table="singer" confidence="high">singers</tm> do we have?',
        ),
        (
            [
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("f7dd9d43-86f3-4221-abcf-b25b61c8ead4")),
                    value="What",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("66f5bf25-2448-4703-ac67-21aa0a4aff6d")),
                    value="are",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("e4106705-1e63-45ac-8a4a-6b60597a8735")),
                    value="the",
                    match_tags=(),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("0695bb3a-bd8c-4569-bd69-eb148508c47e")),
                    value="concert",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("16"),
                            table_id=TableId("2"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("2")
                        ),
                        ColumnMatchTag(
                            confidence=LowConfidenceMatch(),
                            column_id=ColumnId("15"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=LowConfidenceMatch(),
                            column_id=ColumnId("20"),
                            table_id=TableId("3"),
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("61601714-7866-416c-a47e-5703beddc338")),
                    value="names",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("16"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("3"),
                            table_id=TableId("0"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("9"),
                            table_id=TableId("1"),
                        ),
                        ColumnMatchTag(
                            confidence=LowConfidenceMatch(),
                            column_id=ColumnId("11"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("cbe45efb-17d5-4635-a139-94786452e280")),
                    value="?",
                    match_tags=(),
                ),
            ],
            "What are the "
            '<cm table="concert" column="concert_Name" confidence="high">'
            '<tm table="concert" confidence="high">'
            '<cm table="concert" column="concert_ID" confidence="low">'
            '<cm table="singer_in_concert" column="concert_ID" confidence="low">'
            "concert"
            "</cm>"
            "</cm>"
            "</tm>"
            " "
            '<cm table="singer" column="Song_Name" confidence="low">'
            '<cm table="stadium" column="Name" confidence="high">'
            '<cm table="singer" column="Name" confidence="high">'
            "names"
            "</cm>"
            "</cm>"
            "</cm>"
            "</cm>"
            " ?",
        ),
        (
            [
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("1e143270-4841-44e4-b72f-9bd4beec8ac8")),
                    value="a",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("948235a5-2a9d-41b8-80b9-aeb6ad9a4406")),
                    value="b",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("0")
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("55793c81-3a3d-4d59-81f3-2e884f31f932")),
                    value="c",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("0")
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("e4f00749-3213-427a-a859-c4f16f25af1f")),
                    value="d",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                        TableMatchTag(
                            confidence=HighConfidenceMatch(), table_id=TableId("0")
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("cd2d73d8-4dda-47b2-b409-9caba69bc374")),
                    value="e",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("80767053-9b44-4689-9bb6-a182feecdbe3")),
                    value="f",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("b2989407-da8f-411b-a309-4bc4ad82fcf5")),
                    value="g",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("13"),
                            table_id=TableId("1"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("d651c044-9a74-4ee8-83a9-2f431e34dc49")),
                    value="h",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("fb0ed745-dac6-4d76-b406-c59884de6955"),),
                    value="i",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("166725f3-3a3d-48be-bed4-7dc7bb6fb9f3")),
                    value="j",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                        ValueMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("17"),
                            table_id=TableId("2"),
                            value="f g h i j",
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("d4fe1f38-671d-4f80-b9b5-3fe680c34498")),
                    value="k",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                    ),
                ),
                PreprocQuestionToken(
                    key=QuestionTokenId(UUID("c8beb545-954c-49ac-a586-e957d8d120c7")),
                    value="l",
                    match_tags=(
                        ColumnMatchTag(
                            confidence=HighConfidenceMatch(),
                            column_id=ColumnId("19"),
                            table_id=TableId("2"),
                        ),
                    ),
                ),
            ],
            '<cm table="concert" column="Year" confidence="high">'
            "a "
            '<tm table="stadium" confidence="high">'
            "b"
            "</tm>"
            " "
            '<cm table="singer" column="Age" confidence="high">'
            '<tm table="stadium" confidence="high">'
            "c d"
            "</tm>"
            " e "
            '<vm table="concert" column="Theme" value="f g h i j" confidence="high">'
            "f g"
            "</vm>"
            "</cm>"
            " "
            '<vm table="concert" column="Theme" value="f g h i j" confidence="high">'
            "h i j"
            "</vm>"
            " k l"
            "</cm>",
        ),
    ],
)
def test_slml_builder(
    sql_schema: SQLSchema, question_tokens: List[PreprocQuestionToken], expected: str
) -> None:
    builder = SLMLBuilder(sql_schema=sql_schema, detokenize=lambda ls: " ".join(ls))
    builder.add_question_tokens(question_tokens=question_tokens)
    slml_question = builder.build()
    assert slml_question == expected
