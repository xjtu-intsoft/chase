# interactive
# Raymond Li, 2020-04-27
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import json
import os
import argparse

import _jsonnet
import tqdm

from duorat.api import DuoratAPI, DuoratOnDatabase
from duorat.utils.evaluation import find_any_config
from duorat.preproc.utils import preprocess_schema_uncached
from duorat.datasets.spider import SpiderDataset
from duorat.preproc.slml import pretty_format_slml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Infer queries for questions about a data section. The output format"\
        " is compatible with official Spider eval scripts.")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config",
        help="The configuration file. By default, an arbitrary configuration from the logdir is loaded")
    parser.add_argument("--data-config",
        help="Dataset section configuration",
        required=True)
    parser.add_argument(
        "--questions",
        help="The path to the questions in Spider format."
             "By default, use questions specified by --data-config",
    )
    parser.add_argument(
        "--output-spider",
        help="Path to save outputs in the Spider format")
    parser.add_argument(
        "--output-google",
        help="Path to save output in the Google format")
    args = parser.parse_args()

    if args.output_spider is None and args.output_google is None:
        raise ValueError("specify output destination in either Google or Michigan format")

    config_path = find_any_config(args.logdir) if args.config is None else args.config
    api = DuoratAPI(args.logdir, config_path)

    data_config = json.loads(_jsonnet.evaluate_file(
        args.data_config, tla_codes={'prefix': '"data/"'}))
    if data_config['name'] != 'spider':
        raise ValueError()
    del data_config['name']
    if args.questions:
        data_config['paths'] = [args.questions]
    dataset = SpiderDataset(**data_config)

    sql_schemas = {}
    for db_id in dataset.schemas:
        spider_schema = dataset.schemas[db_id]
        sql_schemas[db_id] = preprocess_schema_uncached(
            schema=spider_schema,
            db_path=dataset.get_db_path(db_id),
            tokenize=api.preproc._schema_tokenize,
        )

    if args.output_spider and os.path.exists(args.output_spider):
        os.remove(args.output_spider)

    output_items = []
    for item in tqdm.tqdm(dataset):
        db_id = item.spider_schema.db_id
        result = api.infer_query(
            item.question, item.spider_schema, sql_schemas[db_id])
        print("QUESTION:", item.question)
        print("SLML:")
        print(pretty_format_slml(result['slml_question']))
        print("PREDICTION:", result['query'])
        print("GOLD:", item.query)
        output_items.append({
            'utterance': item.question,
            'gold': item.query,
            'database_path': item.db_path,
            'empty_database_path': item.db_path,
            'predictions': [result['query']],
            'scores': [result['score']]
        })
        if args.output_spider:
            with open(args.output_spider, 'at') as output:
                print(result['query'], db_id, sep='\t', file=output)
    if args.output_google:
        with open(args.output_google, 'wt') as output:
            json.dump(output_items, output)
