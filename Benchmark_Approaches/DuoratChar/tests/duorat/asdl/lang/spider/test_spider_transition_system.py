import os
from typing import Dict
from inspect import signature

import pytest
import logging

from duorat.asdl.lang.spider.spider_transition_system import (
    SpiderTransitionSystem,
    SpiderStringAction,
    SpiderTableAction,
    SpiderColumnAction,
    SpiderSingletonAction,
    SpiderObjectAction,
    SpiderIntAction,
)
from duorat.asdl.transition_system import (
    UnkAction,
    MaskAction,
    ApplyRuleAction,
    ReduceAction,
)

from duorat.datasets.spider import SpiderDataset, SpiderItem
from duorat.preproc.offline import DuoRATPreproc
from duorat.utils import registry

# noinspection PyUnresolvedReferences
from tests.duorat.asdl.test_transition_system import (
    mask_action,
    apply_rule_action_a,
    apply_rule_action_b,
    reduce_action,
    unk_action,
)

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
LOGGER = logging.getLogger(__name__)


@pytest.fixture()
def spider_string_action_a() -> SpiderStringAction:
    return SpiderStringAction("a")


@pytest.fixture()
def spider_string_action_b() -> SpiderStringAction:
    return SpiderStringAction("b")


@pytest.fixture()
def spider_table_action_a() -> SpiderTableAction:
    return SpiderTableAction("0")


@pytest.fixture()
def spider_table_action_b() -> SpiderTableAction:
    return SpiderTableAction("1")


@pytest.fixture()
def spider_column_action_a() -> SpiderColumnAction:
    return SpiderColumnAction("0")


@pytest.fixture()
def spider_column_action_b() -> SpiderColumnAction:
    return SpiderColumnAction("1")


@pytest.fixture()
def spider_singleton_action_a() -> SpiderSingletonAction:
    return SpiderSingletonAction("a")


@pytest.fixture()
def spider_singleton_action_b() -> SpiderSingletonAction:
    return SpiderSingletonAction("b")


@pytest.fixture()
def spider_object_action_a() -> SpiderObjectAction:
    return SpiderObjectAction("0.1")


@pytest.fixture()
def spider_object_action_b() -> SpiderObjectAction:
    return SpiderObjectAction("0.2")


@pytest.fixture()
def spider_int_action_a() -> SpiderIntAction:
    return SpiderIntAction("1")


@pytest.fixture()
def spider_int_action_b() -> SpiderIntAction:
    return SpiderIntAction("2")


def test_order_spider_gen_token_action(
    mask_action: MaskAction,
    apply_rule_action_a: ApplyRuleAction,
    apply_rule_action_b: ApplyRuleAction,
    reduce_action: ReduceAction,
    unk_action: UnkAction,
    spider_string_action_a: SpiderStringAction,
    spider_string_action_b: SpiderStringAction,
    spider_table_action_a: SpiderTableAction,
    spider_table_action_b: SpiderTableAction,
    spider_column_action_a: SpiderColumnAction,
    spider_column_action_b: SpiderColumnAction,
    spider_singleton_action_a: SpiderSingletonAction,
    spider_singleton_action_b: SpiderSingletonAction,
    spider_object_action_a: SpiderObjectAction,
    spider_object_action_b: SpiderObjectAction,
    spider_int_action_a: SpiderIntAction,
    spider_int_action_b: SpiderIntAction,
) -> None:
    # we assume that the function parameters are passed in order
    sig = signature(test_order_spider_gen_token_action)
    _locals = locals()
    for i, param_i in enumerate(sig.parameters.keys()):
        for j, param_j in enumerate(sig.parameters.keys()):
            if i <= j:
                assert _locals[param_i] <= _locals[param_j]
                assert not _locals[param_i] > _locals[param_j]
            else:
                assert not _locals[param_i] <= _locals[param_j]
                assert _locals[param_i] > _locals[param_j]
            if i < j:
                assert _locals[param_i] < _locals[param_j]
                assert not _locals[param_i] >= _locals[param_j]
            else:
                assert not _locals[param_i] < _locals[param_j]
                assert _locals[param_i] >= _locals[param_j]


@pytest.fixture()
def duorat_preproc(data_prefix: str) -> DuoRATPreproc:
    preproc = registry.construct(
        "preproc",
        {
            "name": "TransformerDuoRAT",
            "min_freq": 5,
            "max_count": 5000,
            "use_full_glove_vocab": True,
            "save_path": os.path.join(data_prefix, "duorat"),
            "tokenizer": {"name": "CoreNLPTokenizer"},
            "transition_system": {
                "name": "SpiderTransitionSystem",
                "asdl_grammar_path": "duorat/asdl/lang/spider/spider_asdl.txt",
                "tokenizer": {"name": "CoreNLPTokenizer"},
                "output_from": True,
                "use_table_pointer": True,
                "include_literals": True,
                "include_columns": True,
            },
            "schema_linker": {
                "name": "SpiderSchemaLinker",
                "tokenizer": {"name": "CoreNLPTokenizer"},
                "max_n_gram": 5,
                "with_stemming": False,
            },
        },
    )
    assert isinstance(preproc, DuoRATPreproc)
    return preproc


def test_spider_transition_system(
    duorat_preproc: DuoRATPreproc, duorat_data: Dict[str, SpiderDataset]
):
    assert isinstance(duorat_preproc.transition_system, SpiderTransitionSystem)
    duorat_preproc.clear_items()
    for section, dataset in duorat_data.items():
        assert isinstance(dataset, SpiderDataset)
        for item in dataset:
            assert isinstance(item, SpiderItem)
            to_add, asdl_ast = duorat_preproc.validate_item(item=item, section=section)
            # do the tests whenever the RatSQL parser succeeded to parse
            if to_add:
                actions = duorat_preproc.transition_system.get_actions(asdl_ast)
                LOGGER.info(actions)

                asdl_ast = duorat_preproc.transition_system.get_ast(actions)

                assert actions == duorat_preproc.transition_system.get_actions(asdl_ast)

                tree = duorat_preproc.transition_system.ast_to_surface_code(asdl_ast)
                inferred_code = duorat_preproc.transition_system.spider_grammar.unparse(
                    tree=tree, spider_schema=item.spider_schema
                )
                LOGGER.info(inferred_code)
                print(inferred_code)
