# interactive
# Raymond Li, 2020-04-27
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import argparse
import glob
import re

import _jsonnet

from duorat.api import DuoratAPI, DuoratOnDatabase
from duorat.utils.evaluation import find_any_config
from duorat.preproc.slml import pretty_format_slml


class Interactive(object):

    def __init__(self, duorat: DuoratOnDatabase):
        self.duorat = duorat

    def ask_any_question(self, question):
        results = self.duorat.infer_query(question)

        print(pretty_format_slml(results['slml_question']))
        print(f'{results["query"]}  ({results["score"]})')
        try:
            results = self.duorat.execute(results['query'])
            print(results)
        except Exception as e:
            print(str(e))

    def show_schema(self):
        for table in self.duorat.schema.tables:
            print("Table", f"{table.name} ({table.orig_name})")
            for column in table.columns:
                print("    Column", f"{column.name} ({column.orig_name})")

    def run(self):
        self.show_schema()

        while True:
            question = input("Ask a question: ")
            self.ask_any_question(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config",
        help="The configuration file. By default, an arbitrary configuration from the logdir is loaded")
    parser.add_argument(
        "--db-path", required=True,
        help="The path to the sqlite database or csv file"
    )
    parser.add_argument(
        "--schema-path",
        help="The path to the tables.json file with human-readable database schema."
    )
    args = parser.parse_args()

    config_path = find_any_config(args.logdir) if args.config is None else args.config
    interactive = Interactive(DuoratOnDatabase(DuoratAPI(args.logdir, config_path),
                                               args.db_path, args.schema_path))
    interactive.run()
