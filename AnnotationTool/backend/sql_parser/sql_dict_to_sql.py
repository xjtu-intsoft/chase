# coding=utf8

"""
Parse an SQL dict to a SQL string
"""

from typing import List, Dict
from pymongo import MongoClient


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


def get_column_name(col_id: int, column_names_original: List, table_names_original: List, table_alias: Dict) -> str:
    if col_id == 0:
        # "*"
        return "*"
    tid, cn = column_names_original[col_id]
    tn = table_names_original[tid]
    if len(table_alias) > 0:
        target_alias = None
        for alias, name in table_alias.items():
            if tn.lower() == name.lower():
                target_alias = alias
                break
        if target_alias is None:
            return cn
        return "T%d.%s" % (target_alias, cn)
    return cn


def parse_select_clause(select_clause: Dict, column_names_original: List,
                        table_names_original: List, table_alias: Dict) -> str:
    select_clause_str = "SELECT "
    is_distinct = select_clause[0]
    if is_distinct:
        select_clause_str += "DISTINCT "

    selections = list()
    for (agg_id, val_unit) in select_clause[1]:
        agg = AGG_OPS[agg_id]
        unit_op, col_unit1, col_unit2 = val_unit
        col_id_1 = col_unit1[1]
        col_name_1 = get_column_name(col_id_1, column_names_original, table_names_original, table_alias)
        if col_unit2 is not None:
            col_op = UNIT_OPS[unit_op]
            col_id_2 = col_unit2[1]
            col_name_2 = get_column_name(col_id_2, column_names_original, table_names_original, table_alias)
        else:
            col_op, col_id_2, col_name_2 = None, None, None
        if col_op is not None:
            assert col_id_2 is not None
            if agg == "none":
                selections.append("%s %s %s" % (col_name_1, col_op, col_name_2,))
            else:
                selections.append("%s(%s %s %s)" % (agg, col_name_1, col_op, col_name_2,))
        else:
            if agg == "none":
                selections.append("%s" % col_name_1)
            else:
                selections.append("%s(%s)" % (agg, col_name_1,))
    select_clause_str = select_clause_str + ", ".join(selections)
    return select_clause_str


def parse_from_clause(from_clause: Dict, table_names_original: Dict,
                      column_names_original: Dict, table_alias: Dict, database: Dict) -> str:
    table_units = from_clause['table_units']
    conds = from_clause['conds']
    assert len(table_units) == sum([1 if isinstance(c, list) else 0 for c in conds]) + 1 or len(conds) >= len(table_units)
    from_clause_str = "FROM "

    if len(table_units) == 1:
        assert len(conds) == 0
        tu = table_units[0]
        # No Alias
        if tu[0] == "table_unit":
            table_name = table_names_original[tu[1]]
        else:
            # subquery
            table_name = "(%s)" % parse(database, tu[1], table_alias)
        from_clause_str += table_name
        return from_clause_str

    alias_begin_index = len(table_alias) + 1

    meaningful_conds = [c for c in conds if isinstance(c, list)]
    for idx, tu in enumerate(table_units):
        if tu[0] == "table_unit":
            table_name = table_names_original[tu[1]]
        else:
            # subquery
            table_name = "(%s)" % parse(database, tu[1], table_alias)

        # Add Alias
        alias = alias_begin_index
        assert alias not in table_alias
        table_alias[alias] = table_name
        alias_begin_index += 1

        if idx == 0:
            # Add alias
            from_clause_str += "%s AS T%d" % (table_name, alias)
            continue

        cond = meaningful_conds[idx - 1]
        col_id_1, col_id_2 = cond[2][1][1], cond[3][1]
        cond_str = "ON %s = %s" % (
            get_column_name(col_id_1, column_names_original, table_names_original, table_alias),
            get_column_name(col_id_2, column_names_original, table_names_original, table_alias),
        )
        from_clause_str += " JOIN %s AS T%d %s" % (table_name, alias, cond_str)

    if not (len(table_units) == sum([1 if isinstance(c, list) else 0 for c in conds]) + 1):
        last_index = idx - 1
        for cond in meaningful_conds[last_index + 1:]:
            col_id_1, col_id_2 = cond[2][1][1], cond[3][1]
            cond_str = "ON %s = %s" % (
                get_column_name(col_id_1, column_names_original, table_names_original, table_alias),
                get_column_name(col_id_2, column_names_original, table_names_original, table_alias),
            )
            from_clause_str += " AND %s" % cond_str
    return from_clause_str


def parse_where_clause(where_clause: List, table_names_original: Dict,
                       column_names_original: Dict, table_alias: Dict, database: Dict) -> str:
    where_clause_units = list()
    for cond_unit in where_clause:
        if isinstance(cond_unit, str):
            where_clause_units.append(cond_unit)
            continue
        not_op, op_id, val_unit, val1, val2 = cond_unit
        # Column
        op_column = None
        unit_op_id, col_unit1, col_unit2 = val_unit
        unit_op = UNIT_OPS[unit_op_id]
        if unit_op = "none":
            assert col_unit2 is None
            op_column = get_column_name(col_unit1[1], column_names_original,
                                        table_names_original, table_alias)
        else:
            assert col_unit2 is not None
            op_column_1 = get_column_name(col_unit1[1], column_names_original,
                                          table_names_original, table_alias)
            op_column_2 = get_column_name(col_unit2[1], column_names_original,
                                          table_names_original, table_alias)
            op_column = "%s %s %s"(op_column_1, unit_op, op_column_2)

        where_op = WHERE_OPS[op_id]

        # Value
        values = list()
        for val in [val1, val2]:
            if isinstance(val, dict):
                subquery_results = get_sql_entity(val)
                table_set |= subquery_results['table_set']
                column_set |= subquery_results['column_set']
                value_set |= subquery_results['value_set']
            elif isinstance(val, list):
                # Value is another column
                _, val_col_id, _ = val
                column_set.add(val_col_id)
            else:
                _, col_id, _ = col_unit
                value_set.add((col_id, val))


def parse(database: Dict, sql_dict: Dict, table_alias: Dict = None) -> str:
    column_names_original = database['column_names_original']
    table_names_original = database['table_names_original']
    if table_alias is None:
        table_alias = dict()

    from_clause_str = parse_from_clause(sql_dict['from'], table_names_original, column_names_original, table_alias, database)
    print(from_clause_str)

    select_clause_str = parse_select_clause(sql_dict['select'], column_names_original, table_names_original, table_alias)
    print(select_clause_str)

    where_clause_str = parse_where_clause(sql_dict['where'], column_names_original, table_names_original, table_alias, database)


def test_with_all_sql():
    client = MongoClient()
    db = client.contextual_semparse
    collection = db.raw_data
    db_collection = db.databases
    for doc in collection.find({}):
        db_id = doc['database_id']
        curr_database = db_collection.find_one({"database_id": db_id})
        interaction = doc['interaction']
        for turn in interaction:
            sql_dict = turn['sql_dict']
            sql = turn['sql']
            print(db_id)
            print(turn['english_question'])
            print(sql)
            parse(curr_database, sql_dict)
            print("==\n\n")


if __name__ == '__main__':
    test_with_all_sql()
