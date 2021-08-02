# coding=utf8

import json
from typing import Dict, List
from pprint import pprint
from pymongo import MongoClient

"""
Parse SQL dict according the following documentation
https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql
"""

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


def get_sql_entity(sql_dict: Dict) -> Dict:
    table_set, column_set, value_set = set(), set(), set()

    # From
    from_clause = sql_dict['from']
    for (table_type, table_id) in from_clause['table_units']:
        if table_type == 'sql':
            subquery_results = get_sql_entity(table_id)
            table_set |= subquery_results['table_set']
            column_set |= subquery_results['column_set']
            value_set |= subquery_results['value_set']
        else:
            assert table_type == 'table_unit'
            table_set.add(table_id)

    # Select
    select_clause = sql_dict['select']
    for (agg_id, val_unit) in select_clause[1]:
        unit_op, col_unit1, col_unit2 = val_unit
        _, col_id, _ = col_unit1
        column_set.add(col_id)
        if col_unit2 is not None:
            _, col_id, _ = col_unit2
            column_set.add(col_id)

    # Group By
    group_by_clause = sql_dict['groupBy']
    if len(group_by_clause) > 0:
        for col_unit in group_by_clause:
            _, col_id, _ = col_unit
            column_set.add(col_id)

    # Order By
    order_by_clause = sql_dict['orderBy']
    if len(order_by_clause) > 0:
        for val_unit in order_by_clause[1]:
            unit_op, col_unit1, col_unit2 = val_unit
            _, col_id, _ = col_unit1
            column_set.add(col_id)
            if col_unit2 is not None:
                _, col_id, _ = col_unit2
                column_set.add(col_id)

    # Limit
    limit_clause = sql_dict['limit']
    if limit_clause is not None:
        assert isinstance(limit_clause, int)
        value_set.add((0, limit_clause))

    # where
    where_clause = sql_dict['where']
    for cond_unit in where_clause:
        if isinstance(cond_unit, str):
            continue
        not_op, op_id, val_unit, val1, val2 = cond_unit
        unit_op, col_unit1, col_unit2 = val_unit
        if col_unit1 is not None:
            _, col_id1, _ = col_unit1
            column_set.add(col_id1)
        if col_unit2 is not None:
            _, col_id2, _ = col_unit2
            column_set.add(col_id2)
        for col_unit, val in [(col_unit1, val1,), (col_unit1, val2,)]:
            if val is None:
                continue
            if isinstance(val, dict):
                subquery_results = get_sql_entity(val)
                table_set |= subquery_results['table_set']
                column_set |= subquery_results['column_set']
                value_set |= subquery_results['value_set']
            elif isinstance(val, list) or isinstance(val, tuple):
                # Value is another column
                _, val_col_id, _ = val
                column_set.add(val_col_id)
            else:
                _, col_id, _ = col_unit
                value_set.add((col_id, val))

    # Having
    have_clause = sql_dict['having']
    for cond_unit in have_clause:
        if isinstance(cond_unit, str):
            continue
        not_op, op_id, val_unit, val1, val2 = cond_unit
        unit_op, col_unit1, col_unit2 = val_unit
        if col_unit1 is not None:
            _, col_id1, _ = col_unit1
            column_set.add(col_id1)
        if col_unit2 is not None:
            _, col_id2, _ = col_unit2
            column_set.add(col_id2)

        for col_unit, val in [(col_unit1, val1,), (col_unit1, val2,)]:
            if val is None:
                continue
            if isinstance(val, dict):
                subquery_results = get_sql_entity(val)
                table_set |= subquery_results['table_set']
                column_set |= subquery_results['column_set']
                value_set |= subquery_results['value_set']
            else:
                _, col_id, _ = col_unit
                value_set.add((col_id, val))

    for op in ['intersect', 'union', 'except']:
        if sql_dict[op] is not None:
            op_result = get_sql_entity(sql_dict[op])
            table_set |= op_result['table_set']
            column_set |= op_result['column_set']
            value_set |= op_result['value_set']

    return {
        "table_set": table_set,
        "column_set": column_set,
        "value_set": value_set
    }


def get_db_entity(database: Dict) -> List[Dict]:
    table_names_original = database['table_names_original']
    entities = [{"tableName": tn, "tableId": tid, "columns": list()} for tid, tn in enumerate(table_names_original)]
    column_types = database['column_types']
    column_names_original = database['column_names_original']
    assert len(column_types) == len(column_names_original)
    for col_id, ((table_id, column_name), column_type) in enumerate(zip(column_names_original, column_types)):
        if col_id == 0:
            continue
        entities[table_id]['columns'].append({
            "name": column_name,
            "type": column_type,
            "id": col_id
        })
    return entities


def get_db_entity_graph(database: Dict) -> Dict[str, List]:
    table_names_original = database['table_names_original']
    nodes = [{"tableName": tn, "tableId": tid} for tid, tn in enumerate(table_names_original)]
    edges = list()
    column_names_original = database['column_names_original']
    foreign_keys = database['foreign_keys']
    for pk, fk in foreign_keys:
        pk_col = column_names_original[pk]
        fk_col = column_names_original[fk]
        edges.append({
            "node1": nodes[pk_col[0]]["tableName"],
            "node2": nodes[fk_col[0]]["tableName"],
            "col1": pk_col[1],
            "col2": fk_col[1]
        })
    return {
        "nodes": nodes, "edges": edges
    }



def get_db_entity_by_id(database: Dict, entity_ids: Dict) -> Dict:
    table_ids, column_ids = entity_ids['table_set'], entity_ids['column_set']
    tables = [{"tableId": tid, "name": database['table_names_original'][tid], "inSQL": True} for tid in table_ids]
    for tid, tname in enumerate(database['table_names_original']):
        if tid not in table_ids:
            tables.append({"tableId": tid, "name": tname, "inSQL": False})

    columns = [{"columnId": cid, "type": database['column_types'][cid],
                "tableName": "*" if cid == 0 else database['table_names_original'][database['column_names_original'][cid][0]],
                "name": database['column_names_original'][cid][-1]} for cid in column_ids]
    values = list()
    for (cid, value) in entity_ids['value_set']:
        values.append({
            "column": database['column_names_original'][cid][-1],
            "columnId": cid,
            "value": value
        })

    # Add primary key; Primary keys are organized by column
    primary_keys = database['primary_keys']
    primary_key_columns = list()
    for pk_col_id in primary_keys:
        if pk_col_id in column_ids:
            continue
        pk_tid = database['column_names_original'][pk_col_id][0]
        pk_table_name = database['table_names_original'][pk_tid]
        if pk_tid not in table_ids or pk_table_name in [c['tableName'] for c in columns]:
            continue
        columns.append({"columnId": pk_col_id, "type": database['column_types'][pk_col_id],
                        "tableName": pk_table_name,
                        "name": database['column_names_original'][pk_col_id][-1]})

    return {
        "tables": tables, "columns": columns, "values": values
    }


def test_with_specific_sql():
    sql_dict = {
        'except': None,
        'from': {'conds': [],
                'table_units': [['table_unit', 0]]},
        'groupBy': [],
        'having': [],
        'intersect': None,
        'limit': None,
        'orderBy': [],
        'select': [False,
                    [[0, [0, [0, 4, False], None]]]],
        'union': None,
        'where': [[False,
                    2,
                    [0, [0, 2, False], None],
                    '"JetBlue Airways"',
                    None]]
    }
    results = get_sql_entity(sql_dict)
    pprint(results)


def test_with_all_sql():
    client = MongoClient()
    db = client.contextual_semparse
    collection = db.raw_data
    db_collection = db.databases
    for doc in collection.find({}):
        db_id = doc['database_id']
        curr_database = db_collection.find_one({"database_id": db_id})
        interaction = doc['interaction']
        print(db_id)
        for turn in interaction:
            sql_dict = turn['sql_dict']
            print(turn['sql'])
            print(sql_dict)
            entity_ids = get_sql_entity(sql_dict)
            pprint(entity_ids)
            try:
                get_db_entity_by_id(curr_database, entity_ids)
            except:
                print(curr_database)
                raise Exception("List index out of range")


if __name__ == '__main__':
    test_with_all_sql()