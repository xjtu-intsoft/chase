# coding=utf8

import copy
from typing import Dict, List
from pprint import pprint
from pymongo import MongoClient
from translate_sql import translate as translate_sql


def translate_linking(database: Dict, linking: List[Dict]) -> List[Dict]:
    translated_linking = list()
    for link in linking:
        tl = copy.deepcopy(link)
        if tl['type'] == 'table':
            value = database['table_names_original'][tl['entity']['tableId']]
            tl['entity']['value'] = value
        elif tl['type'] == 'column':
            value = database['column_names_original'][tl['entity']['columnId']][1]
            tl['entity']['value'] = value
        else:
            assert tl['type'] == 'value'
            column = database['column_names_original'][tl['entity']['columnId']][1]
            tl['entity']['column'] = column
        translated_linking.append(tl)
    return translated_linking


def translate(database: Dict, original_database: Dict, annotation: Dict) -> Dict:
    dataset = "sparc"
    db_id = annotation['database_id']
    example_id = annotation['example_id']
    split = annotation['split']
    created_by = annotation['created_by']
    create_date = annotation['create_date']
    is_problematic = annotation['is_problematic']
    problematic_note = annotation['problematic_note'] if "problematic_note" in annotation else ""

    turns = annotation['interaction']
    processed_interactions = list()
    for tid, turn in enumerate(turns):
        question_id = turn['question_id']
        question = turn['annoated_chinese_question']
        tokenized_question = turn['tokenized_annotated_chinese_question']
        schema_linking = turn['schema_linking']
        contextual_phenomena = turn['contextual_phenomena']
        sql = turn['sql']
        translated_sql = translate_sql(database, original_database, sql)
        processed_interactions.append({
            "question_id": question_id,
            "question": question,
            "sql": translated_sql,
            "tokenized_question": tokenized_question,
            "schema_linking": translate_linking(database, schema_linking),
            "contextual_phenomena": contextual_phenomena
        })

    return {
        "database_id": db_id,
        "example_id": example_id,
        "dataset": "sparc",
        "split": split,
        "created_by": created_by,
        "create_date": create_date,
        "is_problematic": is_problematic,
        "problematic_note": problematic_note,
        "interaction": processed_interactions,
        "is_delete": False,
    }


def main():
    client = MongoClient()
    db = client.contextual_semparse
    annotations_collection = db.annotations
    original_databases_collection = db.databases
    translated_databases_collection = db.translated_databases
    refine_sparc_collection = db.refine_sparc

    translations = list()
    for database in translated_databases_collection.find():
        print("DB id: %s" % database['database_id'])
        original_database = original_databases_collection.find_one({"database_id": database['database_id']})
        for annotation in annotations_collection.find({"database_id": database['database_id']}):
            translation = translate(database, original_database, annotation)
            translations.append(translation)

    # Before insertion
    print(refine_sparc_collection.count())

    # Insert
    updated_exampled_ids = set()
    for t in translations:
        example_id = t['example_id']
        if example_id in updated_exampled_ids:
            continue
        updated_exampled_ids.add(example_id)
        doc = refine_sparc_collection.find_one({"example_id": example_id})
        if doc is None:
            refine_sparc_collection.insert_one(t)
        else:
            database_id = doc['database_id']
            database = translated_databases_collection.find_one({"database_id": database_id})
            original_database = original_databases_collection.find_one({"database_id": database_id})
            for turn in doc['interaction']:
                sql = turn['sql']
                translated_sql = translate_sql(database, original_database, sql)
                turn['sql'] = translated_sql
            # update
            refine_sparc_collection.update({"example_id": example_id}, doc)

    # After insertion
    print(refine_sparc_collection.count())


if __name__ == '__main__':
    main()
