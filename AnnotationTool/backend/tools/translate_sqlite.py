# coding=utf8

import os
import sqlite3
import shutil
from pymongo import MongoClient

"""
Translate the database schema of SparC to Chinese
"""


client = MongoClient()
db = client.contextual_semparse


def prepare_sqlite_file(database_id: str) -> str:
    copy_path = os.path.join("..", "data", "database", database_id, "%s.sqlite" % database_id)
    basepath = os.path.join("..", "data", "translated_database", database_id)
    if not os.path.exists(basepath):
        os.mkdir(basepath)

    new_sqlite_path = os.path.join(basepath, "%s.sqlite" % database_id)
    if os.path.exists(new_sqlite_path):
        # rm
        os.remove(new_sqlite_path)

    # copy sqlite
    shutil.copy(copy_path, new_sqlite_path)
    return new_sqlite_path


def get_column_id(column_names_original: str, table_id: int, column_name: str) -> int:
    for cid, (tid, cn) in enumerate(column_names_original):
        if tid == table_id and cn.lower() == column_name.lower():
            return cid
    raise Exception("Column Table id: %d int, Name: %s not Found." % (table_id, column_name,))


def translate(database_id: str):
    query_key = {"database_id": database_id}
    original_schema = db.databases.find_one(query_key)
    schema = db.translated_databases.find_one(query_key)

    db_path = prepare_sqlite_file(database_id)
    conn = sqlite3.connect(db_path)

    print("Database Id: ", database_id)
    for tid, tn in enumerate(original_schema['table_names_original']):
        print("%s: " % tn)
        if tn.lower() == "sqlite_sequence":
            continue
        cursor = conn.execute("SELECT name FROM PRAGMA_TABLE_INFO('%s');" % tn)
        for cn in cursor.fetchall():
            # find column index
            print(cn)
            cid = get_column_id(original_schema['column_names_original'], tid, cn[0])
            # Rename
            new_name = schema['column_names_original'][cid][1]
            assert " " not in new_name
            rename_sql = "ALTER Table %s RENAME COLUMN '%s' TO '%s'" % (tn, cn[0], new_name,)
            print(rename_sql)
            conn.execute(rename_sql)
        # Rename table
        new_table_name = schema['table_names_original'][tid]
        rename_sql = "ALTER Table %s RENAME TO %s" % (tn, new_table_name,)
        print(rename_sql)
        conn.execute(rename_sql)
        print("==\n")


def main():
    path = os.path.join("..", "data", "translated_database")
    if not os.path.exists(path):
        os.mkdir(path)
    for doc in db.translated_databases.find({}):
        db_id = doc['database_id']
        translate(db_id)


if __name__ == "__main__":
    main()
