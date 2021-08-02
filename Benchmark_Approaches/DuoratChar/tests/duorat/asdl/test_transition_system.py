from inspect import signature

import pytest

from duorat.asdl.asdl import (
    ASDLProduction,
    ASDLPrimitiveType,
    ASDLConstructor,
    ASDLCompositeType,
    Field,
)
from duorat.asdl.transition_system import (
    MaskAction,
    ApplyRuleAction,
    ReduceAction,
    GenTokenAction,
    UnkAction,
)


@pytest.fixture()
def mask_action() -> MaskAction:
    return MaskAction()


@pytest.fixture()
def apply_rule_action_a() -> ApplyRuleAction:
    return ApplyRuleAction(
        production=ASDLProduction(
            type=ASDLCompositeType(name="b"),
            constructor=ASDLConstructor(
                name="b",
                fields=(
                    Field(
                        name="b", type=ASDLCompositeType(name="b"), cardinality="single"
                    ),
                ),
            ),
        )
    )


@pytest.fixture()
def apply_rule_action_b() -> ApplyRuleAction:
    return ApplyRuleAction(
        production=ASDLProduction(
            type=ASDLPrimitiveType(name="a"),
            constructor=ASDLConstructor(name="a", fields=tuple()),
        )
    )


@pytest.fixture()
def reduce_action() -> ReduceAction:
    return ReduceAction()


@pytest.fixture()
def unk_action() -> UnkAction:
    return UnkAction()


@pytest.fixture()
def gen_token_action() -> GenTokenAction:
    res = GenTokenAction()
    setattr(res, "token", "a")
    return res


def test_order_action(
    mask_action: MaskAction,
    apply_rule_action_a: ApplyRuleAction,
    apply_rule_action_b: ApplyRuleAction,
    reduce_action: ReduceAction,
    unk_action: UnkAction,
    gen_token_action: GenTokenAction,
) -> None:
    # we assume that the function parameters are passed in order
    # note that there is no ordering defined on the generic `GenTokenAction`, and therefore we have to exclude it
    sig = signature(test_order_action)
    _locals = locals()
    for i, param_i in enumerate(sig.parameters.keys()):
        for j, param_j in enumerate(sig.parameters.keys()):
            if i <= j:
                if _locals[param_i] != gen_token_action:
                    assert _locals[param_i] <= _locals[param_j]
                if _locals[param_i] != gen_token_action:
                    assert not _locals[param_i] > _locals[param_j]
            else:
                assert not _locals[param_i] <= _locals[param_j]
                assert _locals[param_i] > _locals[param_j]
            if i < j:
                assert _locals[param_i] < _locals[param_j]
                assert not _locals[param_i] >= _locals[param_j]
            else:
                if _locals[param_i] != gen_token_action:
                    assert not _locals[param_i] < _locals[param_j]
                if _locals[param_i] != gen_token_action:
                    assert _locals[param_i] >= _locals[param_j]
