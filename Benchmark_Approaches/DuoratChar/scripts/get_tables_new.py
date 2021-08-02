# get_tables_new.py
# Raymond Li, 2020-04-27
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import json
import os
import re
import sys
from typing import Dict

from duorat.preproc.utils import refine_schema_names
from third_party.spider.preprocess.get_tables import dump_db_json_schema


if __name__ == "__main__":
    """
    Extract tables.json for a single sqlite DB
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python get_tables.py [sqlite file] [output file name e.g. output.json]"
        )
        sys.exit()
    sqlite_file = sys.argv[1]
    output_file = sys.argv[2]

    assert sqlite_file.endswith('.sqlite')
    db_id = os.path.basename(sqlite_file)[:-7]
    schema = dump_db_json_schema(sqlite_file, db_id)
    schema = refine_schema_names(schema)

    with open(output_file, "wt") as out:
        json.dump([schema], out, sort_keys=True, indent=2, separators=(",", ": "))
