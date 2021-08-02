# coding=utf8

from pprint import pprint
from pymongo import MongoClient

client = MongoClient()
db = client.contextual_semparse
annotation_collection = db.annotations


def main():
    annotations = annotation_collection.find()
    user_question_count = dict()
    user_annotation_count = dict()
    for a in annotations:
        if 'created_by' not in a:
            continue
        user = a['created_by']
        if user not in user_question_count:
            assert user not in user_annotation_count
            user_question_count[user] = 0
            user_annotation_count[user] = 0
        user_annotation_count[user] += 1
        user_question_count[user] += len(a['interaction'])
    pprint(user_question_count)
    pprint(user_annotation_count)


if __name__ == '__main__':
    main()