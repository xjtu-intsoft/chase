# interactive
# Raymond Li, 2020-04-27
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import json
import os
import sqlite3
import re
import subprocess
from typing import Optional

import _jsonnet
import torch

from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.datasets.spider import (
    SpiderItem,
    load_tables,
    SpiderSchema,
    schema_dict_to_spider_schema,
)
from duorat.preproc.utils import preprocess_schema_uncached, refine_schema_names
from duorat.types import RATPreprocItem, SQLSchema, Dict
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from duorat.utils import registry
import duorat.models
from duorat.utils import saver as saver_mod


class ModelLoader:
    def __init__(self, config, from_heuristic: bool = False):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)
        self.from_heuristic = from_heuristic
        if from_heuristic:
            config["model"]["preproc"]["grammar"]["output_from"] = False

        # 0. Construct preprocessors
        self.model_preproc = registry.construct(
            "preproc", self.config["model"]["preproc"],
        )
        self.model_preproc.load()

    def load_model(self, logdir, step, allow_untrained=False, load_best=True):
        """Load a model (identified by the config used for construction) and return it"""
        # 1. Construct model
        model = registry.construct(
            "model", self.config["model"], preproc=self.model_preproc,
        )
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        # 2. Restore its parameters
        saver = saver_mod.Saver(model, None)
        last_step, best_validation_metric = saver.restore(
            logdir, step=step, map_location=self.device, load_best=load_best
        )
        if not allow_untrained and not last_step:
            raise Exception("Attempting to infer on untrained model")
        return model


class DuoratAPI(object):
    """Adds minimal preprocessing code to the DuoRAT model."""

    def __init__(self, logdir: str, config_path: str):
        self.config = json.loads(_jsonnet.evaluate_file(config_path))
        self.inferer = ModelLoader(self.config)
        self.preproc = self.inferer.model_preproc
        self.model = self.inferer.load_model(logdir, step=None)

    def infer_query(
        self, question: str, spider_schema: SpiderSchema, preprocessed_schema: SQLSchema
    ):
        # TODO: we should only need the preprocessed schema here
        spider_item = SpiderItem(
            question=question,
            slml_question=None,
            query="",
            spider_sql={},
            spider_schema=spider_schema,
            db_path="",
            orig={},
        )
        preproc_item: RATPreprocItem = self.preproc.preprocess_item(
            spider_item,
            preprocessed_schema,
            AbstractSyntaxTree(production=None, fields=(), created_time=None),
        )
        finished_beams = self.model.parse(
            [preproc_item], decode_max_time_step=500, beam_size=1
        )
        if not finished_beams:
            return {
                "slml_question": spider_item.slml_question,
                "query": "",
                "score": -1,
            }
        parsed_query = self.model.preproc.transition_system.ast_to_surface_code(
            asdl_ast=finished_beams[0].ast
        )
        parsed_query = self.model.preproc.transition_system.spider_grammar.unparse(
            parsed_query, spider_schema=spider_schema
        )
        return {
            "slml_question": spider_item.slml_question,
            "query": fix_detokenization(parsed_query),
            "score": finished_beams[0].score,
        }


def fix_detokenization(query: str):
    query = query.replace('" ', '"').replace(' "', '"')
    query = query.replace("% ", "%").replace(" %", "%")
    query = re.sub("(\d) . (\d)", "\g<1>.\g<2>", query)
    return query


def add_collate_nocase(query: str):
    value_regexps = ['"[^"]*"', "'[^']*'"]
    value_strs = []
    for regex in value_regexps:
        value_strs += re.findall(regex, query)
    for str_ in set(value_strs):
        query = query.replace(str_, str_ + " COLLATE NOCASE ")
    return query


def convert_csv_to_sqlite(csv_path: str):
    # TODO: infer types when importing
    db_path = csv_path + ".sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    subprocess.run(["sqlite3", db_path, ".mode csv", f".import {csv_path} Data"])
    return db_path


class DuoratOnDatabase(object):
    """Run DuoRAT model on a given database."""

    def __init__(self, duorat: DuoratAPI, db_path: str, schema_path: Optional[str]):
        self.duorat = duorat
        self.db_path = db_path

        if self.db_path.endswith(".sqlite"):
            pass
        elif self.db_path.endswith(".csv"):
            self.db_path = convert_csv_to_sqlite(self.db_path)
        else:
            raise ValueError("expected either .sqlite or .csv file")

        # Get SQLSchema
        if schema_path:
            schemas, _ = load_tables([schema_path])
            if len(schemas) != 1:
                raise ValueError()
            self.schema: Dict = next(iter(schemas.values()))
        else:
            self.schema: Dict = dump_db_json_schema(self.db_path, "")
            self.schema: SpiderSchema = schema_dict_to_spider_schema(
                refine_schema_names(self.schema)
            )

        self.preprocessed_schema: SQLSchema = preprocess_schema_uncached(
            schema=self.schema,
            db_path=self.db_path,
            tokenize=self.duorat.preproc._schema_tokenize,
        )

    def infer_query(self, question):
        return self.duorat.infer_query(question, self.schema, self.preprocessed_schema)

    def execute(self, query):
        conn = sqlite3.connect(self.db_path)
        # Temporary Hack: makes sure all literals are collated in a case-insensitive way
        query = add_collate_nocase(query)
        results = conn.execute(query).fetchall()
        conn.close()
        return results
