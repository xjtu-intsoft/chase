# coding=utf8

from pymongo import MongoClient

client = MongoClient()
db = client.contextual_semparse
collection = db.databases


def main(db_id: str, column: str):
    database = collection.find_one({"database_id": db_id})
    column_names_original = database['column_names_original']
    table_names_original = database['table_names_original']
    for idx, (tid, cn) in enumerate(column_names_original):
        if cn.lower() == column.lower():
            print(table_names_original[tid], idx)



if __name__ == '__main__':
    main("assets_maintenance", "engineer_id")