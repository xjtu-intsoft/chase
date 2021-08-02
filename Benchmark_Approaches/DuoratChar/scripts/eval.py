import argparse
import json
import os
from typing import List

from duorat.utils import evaluation


def main(args=None, logdir_suffix: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument("--section", required=True)
    parser.add_argument("--inferred", required=True)
    parser.add_argument("--output-eval")
    parser.add_argument("--logdir")
    parser.add_argument("--evaluate-beams-individually", action="store_true")
    args, _ = parser.parse_known_args(args)

    if logdir_suffix:
        args.logdir = os.path.join(args.logdir, *logdir_suffix)

    real_logdir, metrics = evaluation.compute_metrics(
        args.config,
        args.config_args,
        args.section,
        list(evaluation.load_from_lines(open(args.inferred))),
        args.logdir,
        evaluate_beams_individually=args.evaluate_beams_individually,
    )

    if args.output_eval:
        if real_logdir:
            output_path = args.output_eval.replace("__LOGDIR__", real_logdir)
        else:
            output_path = args.output_eval
        with open(output_path, "w") as f:
            json.dump(metrics, f, ensure_ascii=False)
        print("Wrote eval results to {}".format(output_path))
    else:
        print(metrics)


if __name__ == "__main__":
    main()
