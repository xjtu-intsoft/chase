import json
import re
import uuid

from third_party.spider.preprocess.schema import _get_schemas_from_json, Schema
from third_party.spider.process_sql import get_sql
from typing import Optional

def go_spider_sql(db_id: str, sql: str) -> Optional[dict]:
    tables_fpath = "data/database/{}/tables.json".format(db_id)

    with open(tables_fpath, "r") as f:
        tables_data = json.load(f)

    schemas, db_names, tables = _get_schemas_from_json(tables_data)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)

    sql = re.sub(
        r"COUNT\(DISTINCT \"(.*)\"\)", r'COUNT(DISTINCT "\1".id)', sql
    )
    sql = (
        sql.replace('"', "")
        # .replace("(", " ")
        # .replace(")", " ")
        .replace("ILIKE", "=")
    )
    sql = re.sub(
        r"WHERE \((.*)\) AND \((.*)\) AND \((.*)\) AND \((.*)\) AND \((.*)\)", r'WHERE \1 AND \2 AND \3 AND \4 AND \5', sql
    )
    sql = re.sub(
        r"WHERE \((.*)\) AND \((.*)\) AND \((.*)\) AND \((.*)\)", r'WHERE \1 AND \2 AND \3 AND \4', sql
    )
    sql = re.sub(
        r"WHERE \((.*)\) AND \((.*)\) AND \((.*)\)", r'WHERE \1 AND \2 AND \3', sql
    )
    sql = re.sub(
        r"WHERE \((.*)\) AND \((.*)\)", r'WHERE \1 AND \2', sql
    )
    sql = re.sub(
        r"WHERE \((.*)\)", r'WHERE \1', sql
    )
    # print(sql)
    pattern: Pattern[str] = re.compile(r'(?P<term>(\w|\.|\"|\(|\)|(COUNT\s*)|(\s*DISTINCT\s*))+) AS (?P<alias>(\w|\.)+)')
    aliases: List[dict] = []
    for match in pattern.finditer(sql):
        if match.group('term') is not None and match.group('alias') is not None:
            aliases.append({
                "alias": match.group('alias'),
                "term": match.group('term')
            })
    # print(aliases)
    sql = re.sub(
        r" AS (\w|\.)+", r"", sql
    )
    for d in aliases:
        # replace term with a unique string to prevent part of it to be replaced by itself
        unique_string = str(uuid.uuid4())
        while unique_string in sql:
            unique_string = str(uuid.uuid4())
        sql = sql.replace(d["term"], unique_string)
        sql = sql.replace(" " + d["alias"], " " + d["term"])
        sql = sql.replace(unique_string, d["term"])

    try:
        spider_sql = get_sql(schema, sql)
    except Exception as e:
        print(sql, e)
        spider_sql = None

    return spider_sql


def main():
    examples_in = "data/database/fda/examples.json"
    examples_out = "data/database/fda/examples_with_spider_sql.json"
    db_id = "fda"
    table_file = "data/database/fda/tables.json"

    with open(examples_in, "r") as f:
        examples = json.load(f)

    res = []
    n_success = 0
    for i, ex in enumerate(examples):
        if ex.get('query'):
            try:
                sql = go_spider_sql(db_id, ex['query'])
                if sql:
                    ex["sql"] = sql
                    n_success += 1
            except (
                KeyError,
                AssertionError,
                RuntimeError,
                json.decoder.JSONDecodeError,
            ) as e:
                print(i, e)
        res.append(ex)

    print(f"Successfully converted {n_success} queries to Spider SQL")
    with open(examples_out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
