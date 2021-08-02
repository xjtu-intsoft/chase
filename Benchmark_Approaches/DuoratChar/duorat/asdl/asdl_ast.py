# coding=utf-8
from dataclasses import dataclass, replace
from typing import Tuple, Optional, Union
from io import StringIO

from duorat.asdl.asdl import ASDLProduction, Field, ASDLType, ASDLCompositeType


@dataclass(order=True, frozen=True)
class AbstractSyntaxTree(object):
    production: ASDLProduction
    fields: Tuple["RealizedField", ...]
    created_time: Optional[int] = None

    def sanity_check(self) -> None:
        if len(self.production.fields) != len(self.fields):
            raise ValueError("filed number must match")
        for field, realized_field in zip(self.production.fields, self.fields):
            assert field.name == realized_field.name
            assert field.type == realized_field.type
            assert field.cardinality == realized_field.cardinality
        for child in self.fields:
            for child_val in child.as_value_list:
                if isinstance(child_val, AbstractSyntaxTree):
                    child_val.sanity_check()

    def pretty(self, string_buffer: Optional[StringIO] = None) -> Optional[str]:
        is_root = False
        if string_buffer is None:
            is_root = True
            string_buffer = StringIO()

        string_buffer.write("(")
        string_buffer.write(self.production.constructor.name)

        for field in self.fields:
            string_buffer.write(" ")
            string_buffer.write("(")
            string_buffer.write(field.type.name)
            string_buffer.write(Field._get_cardinality_repr(field.cardinality))
            string_buffer.write("-")
            string_buffer.write(field.name)

            if field.value is not None:
                for val_node in field.as_value_list:
                    string_buffer.write(" ")
                    if isinstance(field.type, ASDLCompositeType):
                        val_node.pretty(string_buffer)
                    else:
                        string_buffer.write(str(val_node).replace(" ", "-SPACE-"))

            string_buffer.write(")")  # of field

        string_buffer.write(")")  # of node

        if is_root:
            return string_buffer.getvalue()

    @property
    def size(self) -> int:
        node_num = 1
        for field in self.fields:
            for val in field.as_value_list:
                if isinstance(val, AbstractSyntaxTree):
                    node_num += val.size
                else:
                    node_num += 1

        return node_num


@dataclass(order=True, frozen=True)
class RealizedField(Field):
    name: str
    type: ASDLType
    cardinality: str
    value: Union[
        None, str, Tuple[str, ...], AbstractSyntaxTree, Tuple[AbstractSyntaxTree, ...]
    ] = None

    def add_value(self, value: Union[str, AbstractSyntaxTree]) -> "RealizedField":
        if self.cardinality == "multiple":
            if self.value is None:
                return replace(self, value=(value,))
            else:
                return replace(self, value=self.value + (value,))
        else:
            return replace(self, value=value)

    @property
    def as_value_list(self) -> Union[Tuple[str, ...], Tuple[AbstractSyntaxTree, ...]]:
        """get value as an iterable"""
        if self.cardinality == "multiple":
            return self.value
        elif self.value is not None:
            return (self.value,)
        else:
            return tuple()
