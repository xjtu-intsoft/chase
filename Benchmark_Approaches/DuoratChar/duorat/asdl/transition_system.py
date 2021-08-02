# coding=utf-8
from dataclasses import dataclass
from typing import (
    Generator,
    Tuple,
    TypeVar,
    NewType,
    Generic,
    Callable,
    Sequence,
    List,
    Optional,
    Union,
    Dict,
    Type,
    Iterable,
)
from functools import total_ordering
from duorat.asdl.asdl import (
    ASDLProduction,
    ASDLGrammar,
    Field,
    ASDLPrimitiveType,
    ASDLCompositeType,
)
from duorat.asdl.asdl_ast import AbstractSyntaxTree, RealizedField
from duorat.utils.tokenization import AbstractTokenizer

T = TypeVar("T")  # Any type.
T_P = TypeVar("T_P")  # Any other type.


Pos = NewType("Pos", int)


class Action(object):
    pass

    def __eq__(self, other: "Action"):
        if isinstance(other, self.__class__):
            return True
        elif isinstance(other, Action):
            return False
        else:
            return NotImplemented

    def __lt__(self, other: "Action") -> bool:
        if isinstance(other, Action):
            return ACTION_CLASS_ORDER[self.__class__] < next(
                k for v, k in ACTION_CLASS_ORDER.items() if isinstance(other, v)
            )
        else:
            return NotImplemented


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class MaskAction(Action):
    def __repr__(self):
        return "Mask"


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class ApplyRuleAction(Action):
    production: ASDLProduction

    def __repr__(self):
        return "ApplyRule[%s]" % self.production.__repr__()

    def __eq__(self, other: "Action"):
        if isinstance(other, ApplyRuleAction):
            return self.production == other.production
        elif isinstance(other, Action):
            return False
        else:
            return NotImplemented

    def __lt__(self, other: "Action") -> bool:
        if isinstance(other, ApplyRuleAction):
            return self.production < other.production
        elif isinstance(other, Action):
            return ACTION_CLASS_ORDER[self.__class__] < next(
                k for v, k in ACTION_CLASS_ORDER.items() if isinstance(other, v)
            )
        else:
            return NotImplemented


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class ReduceAction(Action):
    def __repr__(self):
        return "Reduce"


class GenTokenAction(Action):
    token: str


@dataclass(init=False, order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class UnkAction(GenTokenAction):
    token: str = "<unk>"

    def __init__(self) -> None:
        pass

    def __repr__(self):
        return "Unk"


ACTION_CLASS_ORDER: Dict[Type[Action], int] = {
    v: k
    for k, v in enumerate(
        (MaskAction, ApplyRuleAction, ReduceAction, UnkAction, GenTokenAction)
    )
}


class Result(Generic[T]):
    pass


@dataclass
class Done(Result[T]):
    res: T


@dataclass
class Partial(Result[T]):
    cont: Callable[[Pos, Action], Result[T]]
    frontier_field: Optional[Field]
    parent_pos: Optional[Pos]


class ParseError(ValueError):
    pass


class TransitionSystem(object):
    def __init__(self, grammar: ASDLGrammar, tokenizer: AbstractTokenizer):
        self.grammar: ASDLGrammar = grammar
        self.tokenizer: AbstractTokenizer = tokenizer

    def _tokenize(self, s: str) -> Tuple[str, List[str]]:
        tokens = self.tokenizer.tokenize(s)
        # guard against empty token sequences
        if len(tokens) == 0:
            return ("", [])
        else:
            head, *tail = tokens
            return (head, tail)

    def _detokenize(self, xs: Sequence[str]) -> str:
        return self.tokenizer.detokenize(xs)

    def compare_ast(self, hyp_ast: AbstractSyntaxTree, ref_ast: AbstractSyntaxTree):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast: AbstractSyntaxTree):
        raise NotImplementedError

    def surface_code_to_ast(self, code) -> AbstractSyntaxTree:
        raise NotImplementedError

    def get_gen_token_action(self, primitive_type: ASDLPrimitiveType):
        raise NotImplementedError

    def get_ast(self, actions: Iterable[Action]) -> AbstractSyntaxTree:
        r = self.parse()
        for pos, action in enumerate(actions):
            r = r.cont(Pos(pos), action)
            if isinstance(r, Done):
                break
            assert isinstance(r, Partial)
        assert isinstance(r, Done), "Action sequence is incomplete!"
        return r.res

    def get_actions(self, asdl_ast: AbstractSyntaxTree) -> List[Action]:
        """
        generate action sequence given the ASDL Syntax Tree
        """

        def _go_ast(ast: AbstractSyntaxTree) -> Generator[Action, None, None]:
            yield ApplyRuleAction(ast.production)
            for field in ast.fields:
                assert isinstance(field, RealizedField)
                yield from _go_field(field=field)

        def _go_field(field: RealizedField) -> Generator[Action, None, None]:
            if isinstance(field.type, ASDLCompositeType):
                if field.cardinality == "single":
                    assert isinstance(field.value, AbstractSyntaxTree)
                    yield from _go_ast(ast=field.value)
                elif field.cardinality == "optional":
                    if field.value is None:
                        yield ReduceAction()
                    else:
                        assert isinstance(field.value, AbstractSyntaxTree)
                        yield from _go_ast(ast=field.value)
                elif field.cardinality == "multiple":
                    assert isinstance(field.value, tuple)
                    for val in field.value:
                        assert isinstance(val, AbstractSyntaxTree)
                        yield from _go_ast(ast=val)
                    yield ReduceAction()
                else:
                    raise ValueError(
                        "Unexpected field cardinality {!r}".format(field.cardinality)
                    )
            else:
                yield from self.get_primitive_field_actions(field=field)

        return list(_go_ast(ast=asdl_ast))

    def get_primitive_field_actions(
        self, field: RealizedField
    ) -> List[Union[GenTokenAction, ReduceAction]]:
        def _go() -> Generator[Union[GenTokenAction, ReduceAction], None, None]:
            assert isinstance(field.type, ASDLPrimitiveType)
            if field.cardinality is "single":
                assert isinstance(field.value, str)
                head, tail = self._tokenize(s=field.value)
                yield from (
                    self.get_gen_token_action(primitive_type=field.type)(token=token)
                    for token in [head] + tail
                )
                yield ReduceAction()
            elif field.cardinality is "optional":
                if field.value is None:
                    yield ReduceAction()
                else:
                    assert isinstance(field.value, str)
                    head, tail = self._tokenize(s=field.value)
                    yield from (
                        self.get_gen_token_action(primitive_type=field.type)(
                            token=token
                        )
                        for token in [head] + tail
                    )
                    yield ReduceAction()
            elif field.cardinality == "multiple":
                assert isinstance(field.value, tuple)
                for val in field.value:
                    assert isinstance(val, str)
                    head, tail = self._tokenize(s=val)
                    yield from (
                        self.get_gen_token_action(primitive_type=field.type)(
                            token=token
                        )
                        for token in [head] + tail
                    )
                    yield ReduceAction()
                yield ReduceAction()
            else:
                raise ValueError(
                    "Unexpected field cardinality {!r}".format(field.cardinality)
                )

        return list(_go())

    def valid_action_predicate(
        self,
        action: Action,
        previous_action: Optional[Action],
        frontier_field: Optional[Field],
        allow_unk: bool,
    ) -> bool:
        raise NotImplementedError

    def _append(
        self, r: Result[Tuple[T, ...]], r_p: Result[T]
    ) -> Result[Tuple[T, ...]]:
        if isinstance(r, Done):
            if isinstance(r_p, Done):
                return Done(res=r.res + (r_p.res,))
            elif isinstance(r_p, Partial):
                return Partial(
                    cont=lambda pos, action: self._append(
                        r=r, r_p=r_p.cont(pos, action)
                    ),
                    frontier_field=r_p.frontier_field,
                    parent_pos=r_p.parent_pos,
                )
        elif isinstance(r, Partial):
            return Partial(
                cont=lambda pos, action: self._append(r=r.cont(pos, action), r_p=r_p),
                frontier_field=r.frontier_field,
                parent_pos=r.parent_pos,
            )

    def _prepend(
        self, r: Result[Tuple[T, ...]], r_p: Result[T]
    ) -> Result[Tuple[T, ...]]:
        if isinstance(r_p, Done):
            if isinstance(r, Done):
                return Done(res=(r_p.res,) + r.res)
            elif isinstance(r, Partial):
                return Partial(
                    cont=lambda pos, action: self._prepend(
                        r=r.cont(pos, action), r_p=r_p
                    ),
                    frontier_field=r.frontier_field,
                    parent_pos=r.parent_pos,
                )
        elif isinstance(r_p, Partial):
            return Partial(
                cont=lambda pos, action: self._prepend(r=r, r_p=r_p.cont(pos, action)),
                frontier_field=r_p.frontier_field,
                parent_pos=r_p.parent_pos,
            )

    def _parse_fields(
        self, fields: Tuple[Field, ...], parent_pos: Pos
    ) -> Result[Tuple[RealizedField, ...]]:
        def _go(
            fs: Tuple[Field, ...], r: Result[Tuple[RealizedField, ...]]
        ) -> Result[Tuple[RealizedField, ...]]:
            if len(fs) > 0:
                head, *tail = fs
                return _go(
                    fs=tuple(tail),
                    r=self._append(
                        r=r, r_p=self._parse_field(field=head, parent_pos=parent_pos)
                    ),
                )
            else:
                return r

        return _go(fields, Done(res=tuple()))

    def _parse_ast(
        self, frontier_field: Optional[Field], parent_pos: Optional[Pos]
    ) -> Partial[AbstractSyntaxTree]:
        def _cont(pos: Pos, action: Action) -> Result[AbstractSyntaxTree]:
            if isinstance(action, ApplyRuleAction):

                def _go(
                    r: Result[Tuple[RealizedField, ...]]
                ) -> Result[AbstractSyntaxTree]:
                    if isinstance(r, Done):
                        return Done(
                            res=AbstractSyntaxTree(
                                production=action.production, fields=r.res
                            )
                        )
                    elif isinstance(r, Partial):
                        return Partial(
                            cont=lambda pos_p, action_p: _go(r=r.cont(pos_p, action_p)),
                            frontier_field=r.frontier_field,
                            parent_pos=r.parent_pos,
                        )

                return _go(
                    r=self._parse_fields(
                        fields=action.production.fields, parent_pos=pos
                    )
                )
            else:
                raise ParseError(
                    "Expected {!r}, got {!r} at position {!r}".format(
                        ApplyRuleAction, action, pos
                    )
                )

        return Partial(cont=_cont, frontier_field=frontier_field, parent_pos=parent_pos)

    def _parse_string(self, frontier_field: Field, parent_pos: Pos) -> Partial[str]:
        def _go(r: Result[Tuple[str, ...]]) -> Result[str]:
            if isinstance(r, Done):
                res = self._detokenize(r.res)
                return Done(res=res)
            elif isinstance(r, Partial):
                return Partial(
                    cont=lambda pos, action: _go(r=r.cont(pos, action)),
                    frontier_field=r.frontier_field,
                    parent_pos=r.parent_pos,
                )

        return Partial(
            cont=lambda pos, action: _go(
                r=self._many1_till(
                    p=self._parse_gen_token(
                        frontier_field=frontier_field, parent_pos=parent_pos
                    ),
                    end=self._parse_reduce(
                        frontier_field=frontier_field, parent_pos=parent_pos
                    ),
                ).cont(pos, action)
            ),
            frontier_field=frontier_field,
            parent_pos=parent_pos,
        )

    @staticmethod
    def _parse_gen_token(frontier_field: Field, parent_pos: Pos) -> Partial[str]:
        def _cont(pos: Pos, action: Action) -> Done[str]:
            if isinstance(action, GenTokenAction):
                return Done(res=action.token)
            else:
                raise ParseError(
                    "Expected {!r}, got {!r} at position {!r}".format(
                        GenTokenAction, action, pos
                    )
                )

        return Partial(cont=_cont, frontier_field=frontier_field, parent_pos=parent_pos)

    @staticmethod
    def _parse_reduce(frontier_field: Field, parent_pos: Pos) -> Partial[None]:
        def _cont(pos: Pos, action: Action) -> Done[None]:
            if isinstance(action, ReduceAction):
                return Done(res=None)
            else:
                raise ParseError(
                    "Expected {!r}, got {!r} at position {!r}".format(
                        ReduceAction, action, pos
                    )
                )

        return Partial(cont=_cont, frontier_field=frontier_field, parent_pos=parent_pos)

    @staticmethod
    def _or(p: Partial[T], p_p: Partial[T_P],) -> Partial[Union[T, T_P]]:
        assert p.frontier_field == p_p.frontier_field
        assert p.parent_pos == p_p.parent_pos

        def _cont(pos: Pos, action: Action) -> Result[Union[T, T_P]]:
            try:
                return p.cont(pos, action)
            except ParseError:
                return p_p.cont(pos, action)

        return Partial(
            cont=_cont, frontier_field=p.frontier_field, parent_pos=p.parent_pos
        )

    def _many_till(self, p: Partial[T], end: Partial[T_P],) -> Partial[Tuple[T, ...]]:
        def _go(r: Result[Tuple[T, ...]]) -> Partial[Tuple[T, ...]]:
            if isinstance(r, Done):
                # when `r` is done, try to parse `end`
                def _cont(pos: Pos, action: Action) -> Result[Tuple[T, ...]]:
                    try:

                        def _go_p(r_p: Result[T_P]) -> Result[T]:
                            if isinstance(r_p, Done):
                                # throw away `r_p`'s result and return `r` which is done at this point
                                return r
                            elif isinstance(r_p, Partial):
                                return Partial(
                                    cont=lambda pos_p, action_p: _go_p(
                                        r_p=r_p.cont(pos_p, action_p)
                                    ),
                                    frontier_field=r_p.frontier_field,
                                    parent_pos=r_p.parent_pos,
                                )

                        # if the `end` continuation succeeds, commit to parsing `end`
                        return _go_p(r_p=end.cont(pos, action))
                    except ParseError:
                        # if `end` fails to parse at the first attempt, commit to parsing another `p`
                        return _go(r=self._append(r=r, r_p=p.cont(pos, action)))

                return Partial(
                    cont=_cont, frontier_field=p.frontier_field, parent_pos=p.parent_pos
                )
            elif isinstance(r, Partial):
                # recurse while `r` is not done
                return Partial(
                    cont=lambda pos, action: _go(r=r.cont(pos, action)),
                    frontier_field=r.frontier_field,
                    parent_pos=r.parent_pos,
                )

        # start with an empty tuple wrapped in a `Done` so that `end` will be attempted next
        return _go(r=Done(res=tuple()))

    def _many1_till(self, p: Partial[T], end: Partial[T_P],) -> Partial[Tuple[T, ...]]:
        return Partial(
            cont=lambda pos, action: self._prepend(
                r=self._many_till(p=p, end=end), r_p=p.cont(pos, action)
            ),
            frontier_field=p.frontier_field,
            parent_pos=p.parent_pos,
        )

    def _parse_primitive_type(
        self, frontier_field: Field, parent_pos: Pos
    ) -> Partial[str]:
        return self._parse_string(frontier_field=frontier_field, parent_pos=parent_pos)

    def _parse_field(self, field: Field, parent_pos: Pos) -> Result[RealizedField]:
        def _go(
            r: Result[
                Union[
                    str,
                    Tuple[str, ...],
                    AbstractSyntaxTree,
                    Tuple[AbstractSyntaxTree, ...],
                ]
            ]
        ) -> Result[RealizedField]:
            if isinstance(r, Done):
                return Done(
                    res=RealizedField(
                        name=field.name,
                        type=field.type,
                        cardinality=field.cardinality,
                        value=r.res,
                    )
                )
            elif isinstance(r, Partial):
                return Partial(
                    cont=lambda pos, action: _go(r=r.cont(pos, action)),
                    frontier_field=r.frontier_field,
                    parent_pos=r.parent_pos,
                )

        if isinstance(field.type, ASDLCompositeType):
            if field.cardinality == "single":
                return _go(
                    r=self._parse_ast(frontier_field=field, parent_pos=parent_pos)
                )
            elif field.cardinality == "optional":
                return _go(
                    r=self._or(
                        p=self._parse_reduce(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                        p_p=self._parse_ast(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                    )
                )
            elif field.cardinality == "multiple":
                return _go(
                    r=self._many_till(
                        p=self._parse_ast(frontier_field=field, parent_pos=parent_pos),
                        end=self._parse_reduce(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                    )
                )
            else:
                raise ValueError(
                    "Unexpected field cardinality {!r}".format(field.cardinality)
                )
        elif isinstance(field.type, ASDLPrimitiveType):
            if field.cardinality == "single":
                return _go(
                    r=self._parse_primitive_type(
                        frontier_field=field, parent_pos=parent_pos
                    )
                )
            elif field.cardinality == "optional":
                return _go(
                    r=self._or(
                        p=self._parse_reduce(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                        p_p=self._parse_primitive_type(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                    )
                )
            elif field.cardinality == "multiple":
                return _go(
                    r=self._many_till(
                        p=self._parse_primitive_type(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                        end=self._parse_reduce(
                            frontier_field=field, parent_pos=parent_pos
                        ),
                    )
                )
            else:
                raise ValueError(
                    "Unexpected field cardinality {!r}".format(field.cardinality)
                )

    def parse(self) -> Partial[AbstractSyntaxTree]:
        r"""
parse() -> Partial[AbstractSyntaxTree]

Returns a continuation for parsing an `AbstractSyntaxTree`.

What is a continuation and what does it have to do with parsing?

A continuation is a function, here `(Pos, Action) -> Result[T]`, where `T` is the type of the data class that we want
to construct from a sequence of `Action`s. `T` can be, for example, `str`, `Tuple[str, ...]`, `RealizedField`, or
`AbstractSyntaxTree`. `Result[T]` is a sum type with two inhabitants, `Done[T]` and `Partial[T]`. In other words, a
`Result[T]` is either `Done[T]` or `Partial[T]`, never both.

After supplying the parser with an Action, there are two mutually exclusive possibilities that are encoded by the two
inhabitants of `Result[T]`:

* Parsing is complete and a `T` can be created. In this case, parsing returns a `Done[T]`. `Done[T]` has one field,
  `res: T`. Thus, if we encounter a `Done[T]`, we can pull a completed `T` out by accessing `res`.

* Parsing is incomplete, and we need at least one more `Action` before we can create a `T`. In this second case,
  parsing returns a `Partial[T]`, where `Partial[T]` has the field `cont: Callable[[Pos, Action], Result[T]]`. In other
  words, we are returning yet another continuation and parsing can, well, continue.
"""
        return self._parse_ast(frontier_field=None, parent_pos=None)
