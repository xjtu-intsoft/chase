# CHASE: A Large-Scale and Pragmatic Chinese Dataset for Cross-Database Context-Dependent Text-to-SQL

CHASE is a large-scale and pragmatic Chinese dataset for cross-database context-dependent text-to-SQL task (natural language interfaces for relational databases). It is released along with our ACL 2021 paper: CHASE: A Large-Scale and Pragmatic Chinese Dataset for Cross-Database Context-Dependent Text-to-SQL. This repo contains our dataset CHASE.

### Citation



### Data Content and Format

#### Question, SQL, and Parsed SQL

Each file in`train.json` and `dev.json` contains the following fields:
- `database_id`: the database id to which this interaction is addressed.
- `interaction`: the query interaction including multiple DB query questions. For each question in the interaction, it includes:
  - `utterance`: the natural language question
  - `utterance_toks`: the natural language question tokens
  - `query`: the SQL query corresponding to the question. 
  - `sql`: parsed results of this SQL query using `process_sql.py`. Please refer to the [Spider Github page](https://github.com/taoyds/spider) for the detailed documentation.

```
    {
        "database_id": "party_host",
        "interaction": [
            {
                "utterance": "主办方都有谁？",
                "utterance_toks": [
                    "主",
                    "办",
                    "方",
                    ...
                    "？"
                ],
                "query": "select 姓名 from 主办方",
                "sql": {
                    "except": null,
                    "from": {
                        "conds": [],
                        "table_units": [
                            [
                                "table_unit",
                                1
                            ]
                        ]
                    },
                    ...
                    "where": []
                }
            },
            {
                "utterance": "他们来自哪些不同的国家？",
                "utterance_toks": [
                    "他",
                    "们",
                    ...
                    "？"
                ],
                "query": "select distinct 国籍 from 主办方",
                "sql": {
                    "except": null,
                    "from": {
                        "conds": [],
                        "table_units": [
                            [
                                "table_unit",
                                1
                            ]
                        ]
                    },
                    ...
                    "where": []
                }
            },
            {
                "utterance": "每个国家有多少个主办方？",
                "utterance_toks": [
                    "每",
                    "个",
                    "国",
                    "家",
                    ...
                    "？"
                ],
                "query": "select 国籍 , count(*) from 主办方 group by 国籍",
                "sql": {
                    "except": null,
                    "from": {
                        "conds": [],
                        "table_units": [
                            [
                                "table_unit",
                                1
                            ]
                        ]
                    },
                    ...
                    "where": []
                }
            }
        ]
    }
```

#### Tables

`tables.json` contains the following information for each database:
- `db_id`: database id
- `table_names_original`: original table names stored in the database.
- `table_names`: cleaned and normalized table names. We make sure the table names are meaningful. [to be changed]
- `column_names_original`: original column names stored in the database. Each column looks like: `[0, "派对主题"]`. `0` is the index of table names in `table_names`, which is `"派对"` in this case. `"派对主题"` is the column name. 
- `column_names`: cleaned and normalized column names. We make sure the column names are meaningful. [to be changed]
- `column_types`: data type of each column
- `foreign_keys`: foreign keys in the database. `[11, 7]` means column indices in the `column_names`. These two columns are foreign keys of two different tables.
- `primary_keys`: primary keys in the database. Each number is the index of `column_names`.

```
    {
        "db_id": "party_host",
        "table_names_original": [
            "派对",
            "主办方",
            "派对主办方"
        ],
        "table_names": [
            "派对",
            "主办方",
            "派对主办方"
        ],
        "column_names_original": [
            [
                -1,
                "*"
            ],
            [
                0,
                "派对"
            ],
            [
                0,
                "派对主题"
            ],
            [
                0,
                "地点"
            ],
            ...
        ],
        "column_names": [
            [
                -1,
                "*"
            ],
            [
                0,
                "派对"
            ],
            [
                0,
                "派对主题"
            ],
            [
                0,
                "地点"
            ],
            ...
        ],
        "column_types": [
            "text",
            "number",
            "text",
            "text",
            ...
        ],
        "foreign_keys": [
            [
                11,
                1
            ],
            [
                12,
                7
            ]
        ],
        "primary_keys": [
            1,
            7,
            11
        ]
    }
```