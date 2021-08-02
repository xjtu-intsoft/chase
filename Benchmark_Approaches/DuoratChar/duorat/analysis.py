import os
import random
from typing import Iterable, Dict, Any

import pandas
from IPython.display import display

from duorat.datasets.spider import SpiderItem, SpiderDataset
from duorat.asdl.lang.spider.spider import SpiderGrammar
from duorat.utils.evaluation import load_from_lines


def show_question(ex):
    print(ex.question)
    print(ex.query)


def show_question_set(question_set, k=10):
    rng = random.Random(1)
    if k > len(question_set):
        k = len(question_set)
    for idx in rng.sample(list(range(len(question_set))), k=k):
        print(idx)
        show_question(question_set[idx])
        print()


def load_outputs(
    experiment_path: str, output_file: str, trial_ids: Iterable[int] = None
) -> Dict[int, Any]:
    if trial_ids is None:
        trial_ids = [int(trial_id) for trial_id in os.listdir(experiment_path)]

    all_outputs = {}
    for trial_id in trial_ids:
        path = f"{experiment_path}/{trial_id}/{output_file}"
        with open(path) as src:
            all_outputs[trial_id] = list(load_from_lines(list(src)))
    return all_outputs


def evaluate_outputs(
    dataset: SpiderDataset, all_outputs: Dict[int, Any]
) -> pandas.DataFrame:
    columns = ("trial_id", "qid", "exact", "group_acc")
    metrics = SpiderDataset.Metrics(dataset)
    data = []
    for trial_id, outputs in all_outputs.items():
        assert len(outputs) == len(dataset)
        for qid, (example, predicted) in enumerate(zip(dataset, outputs)):
            r = metrics.evaluator.evaluate_one(
                example.spider_schema.db_id, example.orig["query"], predicted[0]
            )
            data.append(
                (trial_id, qid, int(r["exact"]), int(r["partial"]["group"]["acc"]))
            )

    return pandas.DataFrame(data, columns=columns)


def compute_dataset_metadata(
    examples: Iterable[SpiderItem], grammar: SpiderGrammar
) -> pandas.DataFrame:
    all_metadata = []
    for ex in examples:
        parse = grammar.parse_sql(ex.spider_sql)
        metadata = {
            "group_by": "group_by" in parse,
            "order_by": "order_by" in parse,
            "per": "per" in ex.question.split(),
        }
        metadata["num_tables"] = len(parse["from"]["table_units"])
        # the number of where clauses is easier to infer from the original SPIDER SQL format
        metadata["num_where"] = (len(ex.spider_sql["where"]) + 1) // 2
        metadata["db_id"] = ex.spider_schema.db_id

        def recurse(path, node):
            if node["_type"] == "sql":
                if path:
                    metadata["nested"] = True
                if "select" in path:
                    metadata["sql_in_select"] = True
                if "where" in path:
                    metadata["sql_in_where"] = True
                elif "from" in path:
                    metadata["sql_in_from"] = True
                elif "having" in path:
                    metadata["sql_in_having"] = True
                elif "except" in path:
                    metadata["except"] = True
                elif "intersect" in path:
                    metadata["intersect"] = True
                elif "union" in path:
                    metadata["union"] = True
                elif path:
                    metadata["other_nested"] = True
                    print(path)
            if node["_type"] == "Count" and "select" in path:
                metadata["count"] = True
            if node["_type"] == "Count" and "order_by" in path:
                metadata["count_in_order_by"] = True
            for key, child in node.items():
                if isinstance(child, dict):
                    recurse(path + [key], child)
                elif isinstance(child, list):
                    for element in child:
                        recurse(path + [key], element)

        recurse([], parse)
        all_metadata.append(metadata)
    return pandas.DataFrame(all_metadata).fillna(False)


def subset_analysis(df_results, df_metadata, subsets_query=None):
    subset = df_metadata.query(subsets_query) if subsets_query else df_metadata
    num_trials = len(df_results["trial_id"].unique())
    print(f"{len(subset)} examples where {subsets_query}")
    print("Average performance across trials:")
    exact_acc = df_results.join(subset, on="qid", how="inner")["exact"].mean()
    group_by_acc = df_results.join(subset.query("group_by"), on="qid", how="inner")[
        "group_acc"
    ].mean()
    print("Exact accuracy: {}, groupby accuracy: {}".format(exact_acc, group_by_acc))
    print("Flawless performance frequency:")
    display(
        (
            df_results.join(subset, on="qid", how="inner")
            .groupby("qid")[["exact"]]
            .sum()
            >= num_trials
        ).mean()
    )


def show_question_report(dataset, qid, all_outputs, df_results):
    df_results = df_results.query("qid == @qid")

    print(f"QUESTION {qid} ABOUT {dataset[qid].spider_schema.db_id}")
    print(dataset[qid].question)
    print("GOLD")
    print(dataset[qid].orig["query"])
    print("PREDICTED")
    for trial_id, outputs in all_outputs.items():
        print(
            "correct"
            if df_results.query("trial_id == @trial_id")["exact"].item()
            else "wrong",
            outputs[qid][0],
        )
    print()
