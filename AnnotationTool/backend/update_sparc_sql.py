# coding=utf8

import os
import json
import codecs
from pymongo import MongoClient

client = MongoClient()
db = client.contextual_semparse
collection = db.raw_data
annotation_collection = db.annotations


def read_data(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def main():
    basepath = os.path.join('data', 'CSparC')
    dev_path = os.path.join(basepath, 'dev_zh-CN.json')
    train_path = os.path.join(basepath, 'train_zh-CN.json')
    example_index = 1
    for (split, path) in [('dev', dev_path,), ('train', train_path,)]:
        examples = read_data(path)
        for example in examples:
            database_id = example['database_id']
            example_id = "ex_%d" % (example_index)
            query_key = {"database_id": database_id, "example_id": example_id}
            target_annotation = annotation_collection.find_one(query_key)
            target_raw_data = collection.find_one(query_key)
            assert target_annotation is not None and target_raw_data is not None

            interactions = example['interaction']
            for tid, turn in enumerate(interactions):
                question_id = "ex_%d_q_%d" % (example_index, tid + 1)
                sql = turn['query']
                sql_dict = turn['sql']
                english_question = turn['utterance']
                google_chinese_question = turn['utterance_zh-CN']
                baidu_chinese_question = turn['utterance_zh-CN_du']

                target_raw_data_turn = target_raw_data['interaction'][tid]
                target_annotation_turn = target_annotation['interaction'][tid]
                assert target_raw_data_turn['question_id'] == question_id and target_raw_data_turn['english_question'] == english_question
                assert target_annotation_turn['question_id'] == question_id and target_annotation_turn['english_question'] == english_question
                target_raw_data_turn['sql'] = sql
                target_raw_data_turn['sql_dict'] = sql_dict
                target_annotation_turn['sql'] = sql
                target_annotation_turn['sql_dict'] = sql_dict

                annotation_collection.save(target_annotation)
                collection.save(target_raw_data)

            example_index += 1


if __name__ == '__main__':
    main()
