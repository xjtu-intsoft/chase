import os
import random
import re
from collections import defaultdict
from typing import List, Callable, Optional, Sequence, Dict

import torch
from dataclasses import replace
from torchtext.vocab import Vocab

from duorat.datasets.spider import SpiderSchema
from duorat.asdl.transition_system import UnkAction
from duorat.types import SQLSchema, frozendict, ColumnId, TableId, T, OrderedFrozenDict
from duorat.utils import serialization


def pad_nd_tensor(
    tensors: Sequence[torch.Tensor],
    num_padding_dimensions: int,
    batch_first: bool = False,
    padding_value: float = 0,
):
    r"""Adapted from https://github.com/pytorch/pytorch/blob/v1.4/torch/nn/utils/rnn.py#L288-L344
    Pad a list of variable length Tensors with ``padding_value``
    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length along the first ``num_padding_dimensions`` (abbreviated ``n`` here)
    dimensions.
     For example, if the input is list of tensors with size ``L_{i, 1} x ... x L_{i, n} x *``
     and if batch_first is False, the output is a tensor with size ``T_1 x ... x T_n x B x *``.
    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T_k` is the largest `k`-th dimension size.
    `L_{i, k}` is size of the `i`-th tensor's `k`-th dimension.
    `*` is any number of trailing dimensions, including none.
    Note:
        This function returns a Tensor of size ``T_1 x ... x T_n x B x *`` or ``B x T_1 x ... x T_n x *``
        where `T_k` is the longest `k`-th dimension.
        This function assumes trailing dimensions and type of all the Tensors in sequences
        are same.
    Arguments:
        tensors (list[Tensor]): list of variable length nd-tensors.
        num_padding_dimensions (int): number of dimensions to pad.
        batch_first (bool, optional): output will be in ``B x T_1 x ... x T_n x *`` if True, or in
            ``T_1 x ... x T_n x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
    Returns:
        Tensor of size ``T_1 x ... x T_n x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T_1 x ... x T_n x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from tensors[0]
    max_size = tensors[0].size()
    trailing_dims = max_size[num_padding_dimensions:]
    max_lens = tuple(
        max([s.size(dim) for s in tensors]) for dim in range(num_padding_dimensions)
    )
    if batch_first:
        out_dims = (len(tensors),) + max_lens + trailing_dims
    else:
        out_dims = max_lens + (len(tensors),) + trailing_dims

    out_tensor = tensors[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        lengths = tensor.size()[:num_padding_dimensions]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[(i,) + tuple(slice(0, length) for length in lengths)] = tensor
        else:
            out_tensor[tuple(slice(0, length) for length in lengths) + (i,)] = tensor

    return out_tensor


class ActionVocab(Vocab):
    # FIXME: this class only exists because `UNK` is hard-coded in `Vocab`
    UNK = UnkAction()

    def __init__(
        self,
        counter,
        max_size=None,
        min_freq=1,
        specials=[UnkAction()],
        vectors=None,
        unk_init=None,
        vectors_cache=None,
        specials_first=True,
    ):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            # only extend max size if specials are prepended
            max_size = None if max_size is None else max_size + len(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if ActionVocab.UNK in specials:  # FIXME: hard-coded for now
            unk_index = specials.index(ActionVocab.UNK)  # FIXME: position in list
            # account for ordering of specials, set variable
            self.unk_index = unk_index if specials_first else len(self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(ActionVocab.UNK))


def preprocess_schema_uncached(
    schema: SpiderSchema,
    db_path: Optional[str],
    tokenize: Callable[[Optional[str], List[str], str], List[str]],
) -> SQLSchema:
    column_names = []
    tokenized_column_names = []
    original_column_names = []
    table_names = []
    tokenized_table_names = []
    original_table_names = []
    column_to_table = {}
    table_to_columns = {}
    foreign_keys = {}
    foreign_keys_tables = defaultdict(set)

    for i, column in enumerate(schema.columns):
        column_name = tokenize(column.type, column.name, column.unsplit_name)
        column_names.append(column.unsplit_name)
        tokenized_column_names.append(column_name)
        original_column_names.append(column.orig_name)

        table_id = None if column.table is None else column.table.id
        column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = table_to_columns.setdefault(str(table_id), [])
            columns.append(i)

        if column.foreign_key_for is not None:
            foreign_keys[str(column.id)] = column.foreign_key_for.id
            foreign_keys_tables[str(column.table.id)].add(
                column.foreign_key_for.table.id
            )

    for i, table in enumerate(schema.tables):
        table_names.append(table.unsplit_name)
        tokenized_table_names.append(tokenize(None, table.name, table.unsplit_name))
        original_table_names.append(table.orig_name)

    foreign_keys_tables = serialization.to_dict_with_sorted_values(foreign_keys_tables)
    primary_keys = [
        column.id for table in schema.tables for column in table.primary_keys
    ]

    return SQLSchema(
        column_names=OrderedFrozenDict(
            [(ColumnId(str(id)), name) for id, name in enumerate(column_names)]
        ),
        tokenized_column_names=OrderedFrozenDict(
            [
                (ColumnId(str(id)), tuple(tokenized_name))
                for id, tokenized_name in enumerate(tokenized_column_names)
            ]
        ),
        original_column_names=OrderedFrozenDict(
            [
                (ColumnId(str(id)), original_name)
                for id, original_name in enumerate(original_column_names)
            ]
        ),
        table_names=OrderedFrozenDict(
            [(ColumnId(str(id)), name) for id, name in enumerate(table_names)]
        ),
        tokenized_table_names=OrderedFrozenDict(
            [
                (ColumnId(str(id)), tuple(tokenized_name))
                for id, tokenized_name in enumerate(tokenized_table_names)
            ]
        ),
        original_table_names=OrderedFrozenDict(
            [
                (ColumnId(str(id)), original_name)
                for id, original_name in enumerate(original_table_names)
            ]
        ),
        column_to_table=frozendict(
            {
                ColumnId(column): TableId(str(table)) if table is not None else None
                for column, table in column_to_table.items()
            }
        ),
        table_to_columns=frozendict(
            {
                TableId(table): tuple(ColumnId(str(column)) for column in columns)
                for table, columns in table_to_columns.items()
            }
        ),
        foreign_keys=frozendict(
            {
                ColumnId(this): ColumnId(str(other))
                for this, other in foreign_keys.items()
            }
        ),
        foreign_keys_tables=frozendict(
            {
                TableId(this): tuple(TableId(str(other)) for other in others)
                for this, others in foreign_keys_tables.items()
            }
        ),
        primary_keys=tuple(ColumnId(str(column)) for column in primary_keys),
        db_id=schema.db_id,
        db_path=db_path,
    )


def shuffle_schema(schema: SQLSchema) -> SQLSchema:
    """
    Shuffles the order of column_names, tokenized_column_names, original_column_names on one side, and
    table_names, tokenized_table_names, original_table_names on the other side.
    """
    column_ids = list(schema.column_names.keys())
    assert (
        column_ids
        == list(schema.tokenized_column_names.keys())
        == list(schema.original_column_names.keys())
    )
    table_ids = list(schema.table_names.keys())
    assert (
        table_ids
        == list(schema.tokenized_table_names.keys())
        == list(schema.original_table_names.keys())
    )

    # Shuffle order of columns, and order of tables.
    random.shuffle(column_ids)
    random.shuffle(table_ids)

    shuffled_schema = replace(
        schema,
        column_names=OrderedFrozenDict(
            [(id, schema.column_names[id]) for id in column_ids]
        ),
        tokenized_column_names=OrderedFrozenDict(
            [(id, schema.tokenized_column_names[id]) for id in column_ids]
        ),
        original_column_names=OrderedFrozenDict(
            [(id, schema.original_column_names[id]) for id in column_ids]
        ),
        table_names=OrderedFrozenDict(
            [(id, schema.table_names[id]) for id in table_ids]
        ),
        tokenized_table_names=OrderedFrozenDict(
            [(id, schema.tokenized_table_names[id]) for id in table_ids]
        ),
        original_table_names=OrderedFrozenDict(
            [(id, schema.original_table_names[id]) for id in table_ids]
        ),
        table_to_columns=frozendict(
            {
                table_id: tuple(random.sample(column_ids, len(column_ids)))
                for table_id, column_ids in schema.table_to_columns.items()
            }
        ),
    )
    return shuffled_schema


def has_subsequence(seq: Sequence[T], subseq: Sequence[T]) -> bool:
    if not subseq:
        return True
    if len(subseq) > len(seq):
        return False
    start = 0
    while True:
        try:
            start = seq.index(subseq[0], start)
        except ValueError:
            # seq.index() did not match
            return False
        stop = start + len(subseq)
        if (
            stop - 1 < len(seq)
            and seq[stop - 1] == subseq[-1]
            and seq[start:stop] == subseq
        ):
            return True
        start += 1


def _process_name(name: str):
    # camelCase to spaces
    name = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", name)
    return name.replace("-", " ").replace("_", " ").lower()


def _prompt_table(table_name, prompt_user=False):
    table_name = _process_name(table_name)
    print(f"Current table name: {table_name}")
    new_name = (
        input("Type new name (empty to keep previous name): ") if prompt_user else ""
    )
    return new_name if new_name != "" else table_name


def _prompt_column(column_name, table_name, prompt_user=False):
    column_name = _process_name(column_name)
    print(f"Table {table_name}. Current col name: {column_name}")
    new_name = (
        input("Type new name (empty to keep previous name): ") if prompt_user else ""
    )
    return new_name if new_name != "" else column_name


def refine_schema_names(schema: Dict):
    new_schema = {
        "column_names": [],
        "column_names_original": schema["column_names_original"],
        "column_types": schema["column_types"],
        "db_id": schema["db_id"],
        "foreign_keys": schema["foreign_keys"],
        "primary_keys": schema["primary_keys"],
        "table_names": [],
        "table_names_original": schema["table_names_original"],
    }
    for table in schema["table_names_original"]:
        corrected = _prompt_table(table)
        new_schema["table_names"].append(corrected)
    for col in schema["column_names_original"]:
        t_id = col[0]
        column_name = col[1]
        corrected = _prompt_column(column_name, new_schema["table_names"][t_id])
        new_schema["column_names"].append([t_id, corrected])
    return new_schema
