# coding=utf8

import json
import sqlite3
from typing import Dict


def execute_sql(db_path: str, sql: str) -> Dict:
    conn = sqlite3.connect(f'file:%s?mode=ro' % db_path, uri=True)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        data = cursor.fetchall()
        data_description = cursor.description
    finally:
        conn.close()
    return {
        "data_details": data, "data_description": data_description
    }