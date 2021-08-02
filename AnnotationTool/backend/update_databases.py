# coding=utf8

"""
Purpose of this script:
1. Add all DuSQL database to database collection if they are not exists
2. Add a label to distinguish DuSQL database and SparC database
3. Add a label to indicate whether or not we can label conversations for this database
"""

import os
import json
import codecs
from typing import List, Dict
from pymongo import MongoClient


client = MongoClient()
db = client.contextual_semparse
database_collection = db.databases

SPARC_DATABASE_PATH = os.path.join("data", "CSparC", "tables.json")
DUSQL_DATABASE_PATH = os.path.join("data", "DuSQL", "tables.json")
ALLOW_CONVERSATION_DB_PATH = os.path.join("data", "conversation_databases.txt")


def read_data(path) -> List[Dict]:
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def read_raw_text_data(path) -> List[str]:
    with open(path, 'r', encoding='utf8') as f:
        results = list()
        for line in f:
            line = line.strip()
            results.append(line)
    return results


def main():
    # Sparc
    sparc_databases = read_data(SPARC_DATABASE_PATH)
    for database in sparc_databases:
        db_id = database['db_id']
        database['database_id'] = database['db_id']
        del database['db_id']
        database['source'] = 'sparc'
        database['labelConversation'] = False
        doc = database_collection.find_one({"database_id": db_id})
        assert doc is not None
        database_collection.update({"database_id": db_id}, database)

    # DuSQL
    dusql_databases = read_data(DUSQL_DATABASE_PATH)
    for database in dusql_databases:
        db_id = database['db_id']
        database['database_id'] = database['db_id']
        del database['db_id']
        database['source'] = 'dusql'
        database['labelConversation'] = False
        doc = database_collection.find_one({"database_id": db_id})
        if doc is None:
            # insert
            database_collection.insert_one(database)
        else:
            # update
            database_collection.update({"database_id": db_id}, database)

    # Allow conversations
    valid_database_ids = read_raw_text_data(ALLOW_CONVERSATION_DB_PATH)
    for db_id in valid_database_ids:
        doc = database_collection.find_one({"database_id": db_id})
        assert doc is not None
        doc['labelConversation'] = True
        database_collection.save(doc)


if __name__ == '__main__':
    main()
