# coding=utf8

import os
import re
import copy
import json
from typing import List, Dict, Tuple
import sqlite3
from nltk import word_tokenize
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


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap, self._id2table, self._id2column = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    @property
    def id2table(self):
        return self._id2table

    @property
    def id2column(self):
        return self._id2column

    def is_table(self, name: str) -> bool:
        if name.lower() in self._idMap:
            return True
        return False

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        idMap, id2table, id2column = {}, {}, {}
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
                id2column[i] = "*"
            else:
                if is_Chinese(table_names_original[tab_id]):
                    key = table_names_original[tab_id]
                    val = col
                else:
                    key = table_names_original[tab_id].lower()
                    val = col.lower()
                idMap[key + "." + val] = i
                id2column[i] = val

        for i, tab in enumerate(table_names_original):
            if is_Chinese(tab):
                key = tab
            else:
                key = tab.lower()
            idMap[key] = i
            id2table[i] = key

        return idMap, id2table, id2column

    @classmethod
    def get_schema(cls, database: Dict):
        column_names_original = database['column_names_original']
        table_names_original = database['table_names_original']
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        table = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            if is_Chinese(str(tabn)):
                tn = str(tabn)
                cols = [str(col) for td, col in column_names_original if td == i]
                schema[tn] = cols
            else:
                tn = str(tabn.lower())
                cols = [str(col.lower()) for td, col in column_names_original if td == i]
                schema[tn] = cols

        return cls(schema, table)


def update_tokens(all_tokens: List[str], sub_tokens: List[str], beg_idx: int):
    for idx, token in enumerate(sub_tokens):
        all_tokens[beg_idx + idx] = token


def is_Chinese(string: str) -> bool:
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        return False


def tokenize(string: str) -> List[str]:
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() if not is_Chinese(word) else word for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks: List[str]) -> Dict[str, str]:
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_table_unit(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str],
                     schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str]) -> Tuple:
    """
    :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    # translate
    translated_sql_tokens[idx] = translated_schema.id2table[schema.idMap[key]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_col(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str],
              schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str], 
              default_tables: List[str] = None) -> Tuple:
    """
    :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        # Translate
        if schema.is_table(alias):
            _alias = translated_schema.id2table[schema.idMap[alias]]
            translated_sql_tokens[start_idx] = _alias + "." + translated_schema.id2column[schema.idMap[key]]
        else:
            translated_sql_tokens[start_idx] = alias + "." + translated_schema.id2column[schema.idMap[key]]
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            # Translate
            translated_sql_tokens[start_idx] = translated_schema.id2column[schema.idMap[key]]
            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str],
                   schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str], 
                   default_tables: List[str] = None) -> Tuple:
    """
    :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_value(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str],
                schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str], 
                default_tables: List[str] = None) -> Tuple:
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = translate_clause(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] not in ['and', 'or'] and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            translated_sub_tokens = translated_sql_tokens[start_idx: end_idx]
            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, translated_schema, 
                                      translated_sub_tokens, default_tables)
            # update
            update_tokens(translated_sql_tokens, translated_sub_tokens, start_idx)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_val_unit(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str],
                   schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str], 
                   default_tables: List[str] = None) -> Tuple:
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, translated_schema, 
                                    translated_sql_tokens, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, translated_schema,
                                        translated_sql_tokens, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_condition(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str],
                    schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str], 
                    default_tables: List[str] = None) -> Tuple:
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def translate_from(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str], 
                   schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str]) -> Tuple:
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = translate_clause(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens)
            table_units.append((TABLE_TYPE['table_unit'], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, translated_schema, 
                                              translated_sql_tokens, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def translate_select(toks: List[str], start_idx: int, tables_with_alias: Dict, schema: Schema, 
                     translated_schema: Schema, translated_sql_tokens: List[str], default_tables=None) -> Tuple:
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def translate_where(toks: List[str], start_idx: int, tables_with_alias: Dict, schema: Schema, 
                    translated_schema: Schema, translated_sql_tokens: List[str], default_tables=None) -> Tuple:
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, translated_schema, 
                                 translated_sql_tokens, default_tables)
    return idx, conds


def translate_group_by(toks: List[str], start_idx: int, tables_with_alias: Dict, schema: Schema, 
                       translated_schema: Schema, translated_sql_tokens: List[str], default_tables=None) -> Tuple:
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def translate_having(toks: List[str], start_idx: int, tables_with_alias: Dict, schema: Schema, 
                     translated_schema: Schema, translated_sql_tokens: List[str], default_tables=None) -> Tuple:
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
    return idx, conds


def translate_order_by(toks: List[str], start_idx: int, tables_with_alias: Dict, schema: Schema, 
                       translated_schema: Schema, translated_sql_tokens: List[str], default_tables=None) -> Tuple:
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, translated_schema,
                                       translated_sql_tokens, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def translate_limit(toks: List[str], start_idx: int) -> Tuple:
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def skip_semicolon(toks: List[str], start_idx: int) -> int:
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


def translate_clause(toks: List[str], start_idx: int, tables_with_alias: Dict[str, str], 
                     schema: Schema, translated_schema: Schema, translated_sql_tokens: List[str]) -> Tuple[int, Dict]:
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1
    
    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = translate_from(toks, start_idx, tables_with_alias, schema,
                                                                      translated_schema, translated_sql_tokens)
    sql['from'] = {'table_units': table_units, 'conds': conds}

    # select clause
    _, select_col_units = translate_select(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units

    # where clause
    idx, where_conds = translate_where(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
    sql['where'] = where_conds

    # group by clause
    idx, group_col_units = translate_group_by(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
    sql['groupBy'] = group_col_units

    # having clause
    idx, having_conds = translate_having(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
    sql['having'] = having_conds

    # order by clause
    idx, order_col_units = translate_order_by(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens, default_tables)
    sql['orderBy'] = order_col_units

    # limit clause
    idx, limit_val = translate_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = translate_clause(toks, idx, tables_with_alias, schema, translated_schema, translated_sql_tokens)
        sql[sql_op] = IUE_sql

    return idx, sql


def join_sql_tokens(tokens: List[str]) -> str:
    sql = " ".join(tokens)
    sql = re.sub(r"(count|max|min|avg|sum)\s*\(\s*(.*?)\s*\)", r"\1(\2)", sql)
    # Remove redundant space
    sql = re.sub(r" +", " ", sql)
    return sql


def translate(database: Dict, original_database: Dict, sql: str) -> str:
    schema = Schema.get_schema(database)
    original_schema = Schema.get_schema(original_database)

    sql_tokens = tokenize(sql)
    tables_with_alias = get_tables_with_alias(original_schema.schema, sql_tokens)
    print(tables_with_alias)

    translated_sql_tokens = copy.deepcopy(sql_tokens)
    translate_clause(sql_tokens, 0, tables_with_alias, original_schema, schema, translated_sql_tokens)

    return join_sql_tokens(translated_sql_tokens)


def bulk_test(database_id: str):
    client = MongoClient()
    db = client.contextual_semparse
    query_key = {"database_id": database_id}
    original_database = db.databases.find_one(query_key)
    database = db.translated_databases.find_one(query_key)

    for conversation in db.raw_data.find(query_key):
        interaction = conversation['interaction']
        for turn in interaction:
            sql = turn['sql']
            question = turn['english_question']
            print(question)
            print(sql)
            translated_sql = translate(database, original_database, sql)
            print(translated_sql)
            print("==\n")


def compare_execution_results(results1: List[Tuple], results2: List[Tuple]) -> bool:
    if type(results1) != type(results2):
        return False
    if isinstance(results1, list):
        if len(results1) != len(results2):
            return False
        for record1, record2 in zip(results1, results2):
            for idx, value in enumerate(record1):
                if record2[idx] != value:
                    return False
    return True


def compare_sql(database_id: str, sql: str, translated_sql: str) -> bool:
    base_path = os.path.join("..", "data")
    db_path = os.path.join(base_path, "database", database_id, "%s.sqlite" % database_id)
    translated_db_path = os.path.join(base_path, "translated_database", database_id, "%s.sqlite" % database_id)

    sql_results, translated_sql_results = None, None
    try:
        conn = sqlite3.connect(db_path)
        sql_results = conn.execute(sql.replace("! =", "!=")).fetchall()
        # print(sql_results)
    except Exception as e:
        print(e)
        sql_results = None

    try:
        conn = sqlite3.connect(translated_db_path)
        translated_sql_results = conn.execute(translated_sql).fetchall()
        # print(translated_sql_results)
    except Exception as e:
        print(e)
        translated_sql_results = None

    return compare_execution_results(sql_results, translated_sql_results)


def test_with_execution(database_id: str = None):
    client = MongoClient()
    db = client.contextual_semparse
    if database_id is not None:
        query_key = {"database_id": database_id}
        original_database = db.databases.find_one(query_key)
        database = db.translated_databases.find_one(query_key)

        for conversation in db.raw_data.find(query_key):
            interaction = conversation['interaction']
            for turn in interaction:
                sql = turn['sql']
                question = turn['english_question']
                print(question)
                print(sql)
                translated_sql = translate(database, original_database, sql)
                print(translated_sql)
                assert compare_sql(database_id, sql, translated_sql)
                print("==\n")
    else:
        for database in db.translated_databases.find({}):
            db_id = database['database_id']
            query_key = {"database_id": db_id}
            original_database = db.databases.find_one(query_key)

            for conversation in db.raw_data.find(query_key):
                interaction = conversation['interaction']
                for turn in interaction:
                    sql = turn['sql']
                    question = turn['english_question']
                    print(db_id)
                    print(question)
                    print(sql)
                    translated_sql = translate(database, original_database, sql)
                    print(translated_sql)
                    assert compare_sql(db_id, sql, translated_sql)
                    print("==\n")

def test():
    client = MongoClient()
    db = client.contextual_semparse

    query_key = {"database_id": "concert_singer"}
    original_database = db.databases.find_one(query_key)
    database = db.translated_databases.find_one(query_key)

    sql = "SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id JOIN concert AS T3 ON T1.concert_id  =  T3.concert_id WHERE T3.year  =  2014"
    translated_sql = translate(database, original_database, sql)
    print(translated_sql)


if __name__ == '__main__':
    db_id = "flight_2"
    test_with_execution(None)