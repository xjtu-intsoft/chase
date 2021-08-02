# MIT License
#
# Copyright (c) 2019 seq2struct contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import dataclasses
import json
from typing import Optional, Tuple, List, Iterable

import attr
import networkx as nx
from pydantic.dataclasses import dataclass
from pydantic.main import BaseConfig
from torch.utils.data import Dataset

from duorat.utils import registry
from third_party.spider import evaluation
from third_party.spider.preprocess.schema import get_schemas_from_json, Schema
from third_party.spider.process_sql import get_sql


@dataclass
class SpiderTable:
    id: int
    name: List[str]
    unsplit_name: str
    orig_name: str
    columns: List["SpiderColumn"] = dataclasses.field(default_factory=list)
    primary_keys: List[str] = dataclasses.field(default_factory=list)


@dataclass
class SpiderColumn:
    id: int
    table: Optional[SpiderTable]
    name: List[str]
    unsplit_name: str
    orig_name: str
    type: str
    foreign_key_for: Optional[str] = None


SpiderTable.__pydantic_model__.update_forward_refs()


class SpiderSchemaConfig:
    arbitrary_types_allowed = True


@dataclass(config=SpiderSchemaConfig)
class SpiderSchema(BaseConfig):
    db_id: str
    tables: Tuple[SpiderTable, ...]
    columns: Tuple[SpiderColumn, ...]
    foreign_key_graph: nx.DiGraph
    orig: dict


@dataclass
class SpiderItem:
    question: str
    slml_question: Optional[str]
    query: str
    spider_sql: dict
    spider_schema: SpiderSchema
    db_path: str
    orig: dict


def schema_dict_to_spider_schema(schema_dict):
    tables = tuple(
        SpiderTable(id=i, name=name.split(), unsplit_name=name, orig_name=orig_name,)
        for i, (name, orig_name) in enumerate(
            zip(schema_dict["table_names"], schema_dict["table_names_original"])
        )
    )
    columns = tuple(
        SpiderColumn(
            id=i,
            table=tables[table_id] if table_id >= 0 else None,
            name=col_name.split(),
            unsplit_name=col_name,
            orig_name=orig_col_name,
            type=col_type,
        )
        for i, ((table_id, col_name), (_, orig_col_name), col_type,) in enumerate(
            zip(
                schema_dict["column_names"],
                schema_dict["column_names_original"],
                schema_dict["column_types"],
            )
        )
    )

    # Link columns to tables
    for column in columns:
        if column.table:
            column.table.columns.append(column)

    for column_id in schema_dict["primary_keys"]:
        # Register primary keys
        column = columns[column_id]
        column.table.primary_keys.append(column)

    foreign_key_graph = nx.DiGraph()
    for source_column_id, dest_column_id in schema_dict["foreign_keys"]:
        # Register foreign keys
        source_column = columns[source_column_id]
        dest_column = columns[dest_column_id]
        source_column.foreign_key_for = dest_column
        foreign_key_graph.add_edge(
            source_column.table.id,
            dest_column.table.id,
            columns=(source_column_id, dest_column_id),
        )
        foreign_key_graph.add_edge(
            dest_column.table.id,
            source_column.table.id,
            columns=(dest_column_id, source_column_id),
        )

    db_id = schema_dict["db_id"]
    return SpiderSchema(db_id, tables, columns, foreign_key_graph, schema_dict)


def load_tables(paths):
    schemas = {}
    eval_foreign_key_maps = {}

    for path in paths:
        schema_dicts = json.load(open(path))
        for schema_dict in schema_dicts:
            db_id = schema_dict["db_id"]
            assert db_id not in schemas
            schemas[db_id] = schema_dict_to_spider_schema(schema_dict)
            eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


def load_original_schemas(tables_paths):
    all_schemas = {}
    for path in tables_paths:
        schemas, db_ids, tables = get_schemas_from_json(path)
        for db_id in db_ids:
            all_schemas[db_id] = Schema(schemas[db_id], tables[db_id])
    return all_schemas


@registry.register("dataset", "spider")
class SpiderDataset(Dataset):
    def __init__(self, paths: List[str], tables_paths: List[str], db_path: str):
        self.paths = paths
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)
        original_schemas = load_original_schemas(tables_paths)

        for path in paths:
            try:
                raw_data = json.load(open(path))
            except FileNotFoundError as e:
                print(e)
                continue
            for entry in raw_data:
                if "sql" not in entry:
                    entry["sql"] = get_sql(
                        original_schemas[entry["db_id"]], entry["query"]
                    )
                item = SpiderItem(
                    question=entry["question"],
                    slml_question=entry.get("slml_question", None),
                    query=entry["query"],
                    spider_sql=entry["sql"],
                    spider_schema=self.schemas[entry["db_id"]],
                    db_path=self.get_db_path(entry["db_id"]),
                    orig=entry,
                )
                self.examples.append(item)

    def get_db_path(self, db_id: str):
        return os.path.join(self.db_path, db_id, db_id + ".sqlite")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> SpiderItem:
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path, self.foreign_key_maps, "match"
            )
            self.results = []

        def add(self, item: SpiderItem, inferred_code: str) -> None:
            self.results.append(
                self.evaluator.evaluate_one(
                    db_name=item.spider_schema.db_id,
                    gold=item.query,
                    predicted=inferred_code,
                )
            )

        def evaluate_all(
            self, idx: int, item: SpiderItem, inferred_codes: Iterable[str]
        ) -> Tuple[int, list]:
            beams = [
                self.evaluator.evaluate_one(
                    db_name=item.spider_schema.db_id,
                    gold=item.query,
                    predicted=inferred_code,
                )
                for inferred_code in inferred_codes
            ]
            return idx, beams

        def finalize(self) -> dict:
            self.evaluator.finalize()
            return {"per_item": self.results, "total_scores": self.evaluator.scores}
