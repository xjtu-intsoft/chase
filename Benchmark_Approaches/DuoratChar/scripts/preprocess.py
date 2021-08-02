import argparse
import json
import _jsonnet
import tqdm

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat.preproc import offline, utils

# noinspection PyUnresolvedReferences
from duorat.utils import schema_linker

# noinspection PyUnresolvedReferences
from duorat.asdl.lang import spider

from duorat.utils import registry


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.construct(
            "preproc", self.config["model"]["preproc"],
        )

    def preprocess(self, sections, keep_vocab):
        self.model_preproc.clear_items()
        for section in sections:
            data = registry.construct("dataset", self.config["data"][section])
            for item in tqdm.tqdm(data, desc=section, dynamic_ncols=True):
                to_add, validation_info = self.model_preproc.validate_item(
                    item, section
                )
                if to_add:
                    self.model_preproc.add_item(item, section, validation_info)
        if keep_vocab:
            self.model_preproc.save_examples()
        else:
            self.model_preproc.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument("--sections", nargs='+', default=None,
                        help="Preprocess only the listed sections")
    parser.add_argument("--keep-vocab", action='store_true',
                        help="Keep existing vocabulary files")
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    sections = args.sections if args.sections is not None else config["data"].keys()

    preprocessor = Preprocessor(config)
    preprocessor.preprocess(sections, args.keep_vocab)


if __name__ == "__main__":
    main()
