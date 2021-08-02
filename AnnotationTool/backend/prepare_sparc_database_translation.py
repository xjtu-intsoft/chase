# coding=utf8

"""
1. Traverse all the dusql database
2. Get potential translations of each column and tables
"""

import os
import json
import codecs
from typing import List, Dict
from pymongo import MongoClient

client = MongoClient()
db = client.contextual_semparse
database_collection = db.databases
annotation_collection = db.annotations


def find_translation(database):
    db_id = database['database_id']
    db_annotations = annotation_collection.find({"database_id": db_id})
    table_map, column_map = dict(), dict()
    for doc in db_annotations:
        interaction = doc['interaction']
        for turn in interaction:
            tokens = turn['tokenized_annotated_chinese_question']
            linkings = turn['schema_linking']
            for link in linkings:
                beg, end = link['beg'], link['end']
                link_type = link['type']
                if link_type == 'column':
                    column_id = link['entity']['columnId']
                    if column_id not in column_map:
                        column_map[column_id] = set()
                    column_map[column_id].add("".join(tokens[beg:end+1]))
                elif link_type == 'table':
                    table_id = link['entity']['tableId']
                    if table_id not in table_map:
                        table_map[table_id] = set()
                    table_map[table_id].add("".join(tokens[beg:end+1]))
    print(table_map)
    print(column_map)


def main():
    databases = database_collection.find({"source": "sparc"})
    for database in databases:
        print(database['database_id'])
        find_translation(database)


if __name__ == '__main__':
    main()
