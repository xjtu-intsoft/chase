# coding=utf8

import os
import json
import codecs
from pymongo import MongoClient

client = MongoClient()
db = client.contextual_semparse
collection = db.raw_data
database_collection = db.databases


def read_data(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def main():
    basepath = os.path.join('data', 'CSparC')
    dev_path = os.path.join(basepath, 'dev_zh-CN.json')
    train_path = os.path.join(basepath, 'train_zh-CN.json')
    example_index = 1
    collection.delete_many({})
    database_collection.delete_many({})
    for (split, path) in [('dev', dev_path,), ('train', train_path,)]:
        processed_examples = list()
        examples = read_data(path)
        for example in examples:
            database_id = example['database_id']
            example_id = "ex_%d" % (example_index)
            is_annotated = False
            interactions = example['interaction']
            processed_interactions = list()
            for tid, turn in enumerate(interactions):
                question_id = "ex_%d_q_%d" % (example_index, tid + 1)
                sql = turn['query']
                sql_dict = turn['sql']
                english_question = turn['utterance']
                google_chinese_question = turn['utterance_zh-CN']
                baidu_chinese_question = turn['utterance_zh-CN_du']
                processed_interactions.append({
                    "question_id": question_id,
                    "sql": sql,
                    "sql_dict": sql_dict,
                    "english_question": english_question,
                    "google_chinese_question": google_chinese_question,
                    "baidu_chinese_question": baidu_chinese_question
                })
            processed_examples.append({
                "split": split,
                "database_id": database_id,
                "example_id": example_id,
                "is_annotated": is_annotated,
                "interaction": processed_interactions
            })
            example_index += 1
        insert_results = collection.insert_many(processed_examples)
        assert len(insert_results.inserted_ids) == len(processed_examples)

    # Databases
    database_path = os.path.join(basepath, 'tables.json')
    databases = read_data(database_path)
    processed_databases = list()
    for database in databases:
        database['database_id'] = database['db_id']
        del database['db_id']
        processed_databases.append(database)
    insert_results = database_collection.insert_many(processed_databases)
    assert len(insert_results.inserted_ids) == len(processed_databases)


if __name__ == '__main__':
    main()
