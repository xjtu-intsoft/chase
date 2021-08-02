# coding=utf8

import json
import requests
from pprint import pprint
from pymongo import MongoClient


BASEURL = 'http://127.0.0.1:5000/api/'


def test_get_annotation():
    url = BASEURL + 'annotate'
    response = requests.get(url)
    result = response.json()
    pprint(result)


def get_db_inst():
    client = MongoClient()
    db = client.contextual_semparse
    return db


def test_count():
    db = get_db_inst()
    print(db.raw_data.count_documents({"is_annotated": False}))


def test_db_count():
    db = get_db_inst()
    results = db.conversations.aggregate([{"$group":{"_id":"$database_id", "count":{"$sum":1}}}])
    results = list(results)
    print(results)


def test_get_sql_entity():
    url = BASEURL + 'sql/'
    payload = {
        "sql": "SELECT name ,  country ,  age FROM singer ORDER BY age DESC",
        "databaseId": "concert_singer"
    }
    response = requests.post(url, json=payload)
    result = response.json()
    pprint(result)


def test_get_database_list():
    url = BASEURL + 'db_list/'
    response = requests.get(url)
    result = response.json()
    pprint(result)


if __name__ == '__main__':
    test_get_database_list()
