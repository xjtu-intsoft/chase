# coding=utf8

from typing import Dict
from pymongo import MongoClient
from .utils import get_sql, has_Chinese


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                if (table_names_original[tab_id]):
                    key = table_names_original[tab_id]
                    val = col
                else:
                    key = table_names_original[tab_id].lower()
                    val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            if has_Chinese(tab):
                key = tab
            else:
                key = tab.lower()
            idMap[key] = i

        return idMap


def parse_sql(sql: str, database: Dict) -> Dict:
    column_names_original = database['column_names_original']
    table_names_original = database['table_names_original']
    schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
    table = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
    for i, tabn in enumerate(table_names_original):
        if has_Chinese(str(tabn)):
            tn = str(tabn)
            cols = [str(col) for td, col in column_names_original if td == i]
            schema[tn] = cols
        else:
            tn = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[tn] = cols

    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)
    return sql_label


def test():
    sql = 'select T1.创始人, T1.首席执行官 from 公司 AS T1 JOIN 公司 AS T2 where T1.首席执行官 = T2.创始人 and T1.公司名 != "百度集团"'
    db_id = "互联网企业"

    client = MongoClient()
    db_collection = client.contextual_semparse.databases
    translated_database = db_collection.find_one({"database_id": db_id})

    sql_dict = parse_sql(sql, translated_database)
    print(sql_dict)


if __name__ == '__main__':
    test()
