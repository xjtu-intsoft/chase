import pytest

from duorat.types import (
    HighConfidenceMatch,
    LowConfidenceMatch,
    ValueMatchTag,
    ColumnMatchTag,
    TableMatchTag,
    TableId,
    ColumnId,
)


def test_order_match_confidence() -> None:
    assert HighConfidenceMatch() <= HighConfidenceMatch()
    assert LowConfidenceMatch() <= HighConfidenceMatch()
    assert not HighConfidenceMatch() <= LowConfidenceMatch()
    assert LowConfidenceMatch() <= LowConfidenceMatch()

    assert not HighConfidenceMatch() < HighConfidenceMatch()
    assert LowConfidenceMatch() < HighConfidenceMatch()
    assert not HighConfidenceMatch() < LowConfidenceMatch()
    assert not LowConfidenceMatch() < LowConfidenceMatch()

    assert HighConfidenceMatch() >= HighConfidenceMatch()
    assert not LowConfidenceMatch() >= HighConfidenceMatch()
    assert HighConfidenceMatch() >= LowConfidenceMatch()
    assert LowConfidenceMatch() >= LowConfidenceMatch()

    assert not HighConfidenceMatch() > HighConfidenceMatch()
    assert not LowConfidenceMatch() > HighConfidenceMatch()
    assert HighConfidenceMatch() > LowConfidenceMatch()
    assert not LowConfidenceMatch() > LowConfidenceMatch()


@pytest.fixture()
def table_match_tag() -> TableMatchTag:
    return TableMatchTag(confidence=HighConfidenceMatch(), table_id=TableId(""))


@pytest.fixture()
def column_match_tag() -> ColumnMatchTag:
    return ColumnMatchTag(
        confidence=HighConfidenceMatch(), table_id=TableId(""), column_id=ColumnId("")
    )


@pytest.fixture()
def value_match_tag() -> ValueMatchTag:
    return ValueMatchTag(
        confidence=HighConfidenceMatch(),
        table_id=TableId(""),
        column_id=ColumnId(""),
        value="",
    )


def test_order_match_tag(
    table_match_tag: TableMatchTag,
    column_match_tag: ColumnMatchTag,
    value_match_tag: ValueMatchTag,
) -> None:
    assert table_match_tag <= table_match_tag
    assert table_match_tag <= column_match_tag
    assert table_match_tag <= value_match_tag
    assert not column_match_tag <= table_match_tag
    assert column_match_tag <= column_match_tag
    assert column_match_tag <= value_match_tag
    assert not value_match_tag <= table_match_tag
    assert not value_match_tag <= column_match_tag
    assert value_match_tag <= value_match_tag

    assert not table_match_tag < table_match_tag
    assert table_match_tag < column_match_tag
    assert table_match_tag < value_match_tag
    assert not column_match_tag < table_match_tag
    assert not column_match_tag < column_match_tag
    assert column_match_tag < value_match_tag
    assert not value_match_tag < table_match_tag
    assert not value_match_tag < column_match_tag
    assert not value_match_tag < value_match_tag

    assert table_match_tag >= table_match_tag
    assert not table_match_tag >= column_match_tag
    assert not table_match_tag >= value_match_tag
    assert column_match_tag >= table_match_tag
    assert column_match_tag >= column_match_tag
    assert not column_match_tag >= value_match_tag
    assert value_match_tag >= table_match_tag
    assert value_match_tag >= column_match_tag
    assert value_match_tag >= value_match_tag

    assert not table_match_tag > table_match_tag
    assert not table_match_tag > column_match_tag
    assert not table_match_tag > value_match_tag
    assert column_match_tag > table_match_tag
    assert not column_match_tag > column_match_tag
    assert not column_match_tag > value_match_tag
    assert value_match_tag > table_match_tag
    assert value_match_tag > column_match_tag
    assert not value_match_tag > value_match_tag
