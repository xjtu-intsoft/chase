import json
import os

import pytest
from tqdm import tqdm

# avoid import errors
import duorat.preproc
from duorat.utils.db_content import match_db_content


def column_identifiers(config):
    tables = json.load(open(config["data"]["train"]["tables_paths"][0], "r"))

    # Iterate through all columns in all the databases
    for db in tables:
        for table_id, column_name in db["column_names_original"]:
            if table_id >= 0:
                table_name = db["table_names_original"][table_id]
                yield db["db_id"], table_name, column_name


@pytest.mark.parametrize(
    "span, expected",
    [
        [["tiger"], True],
        [["woods"], True],
        [["tige"], False],
        [["tiger", "woods"], False],
    ],
)
def test_db_content_match(span, expected):
    database_dir = "/home/bgs/Dev/data/spider/database"
    db_id = "department_management"
    column_name = "name"
    table_name = "head"
    db_path = os.path.join(database_dir, db_id, f"{db_id}.sqlite")
    match = match_db_content(span, column_name, table_name, db_id, db_path, with_stemming=False)
    assert (len(match) > 0) == expected
