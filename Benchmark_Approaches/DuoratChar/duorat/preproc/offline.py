import itertools
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional, Set, Union
from uuid import uuid4

from dataclasses import replace
from torchtext.vocab import Vocab, GloVe, Vectors

from duorat.datasets.spider import SpiderItem
from duorat.asdl.transition_system import (
    MaskAction,
    ApplyRuleAction,
    ReduceAction,
    TransitionSystem,
)
from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.asdl.lang.spider.spider_transition_system import (
    all_spider_gen_token_actions,
    SpiderTransitionSystem,
)
from duorat.preproc import abstract_preproc
from duorat.preproc.slml import SLMLParser
from duorat.preproc.utils import (
    ActionVocab,
    preprocess_schema_uncached,
    shuffle_schema,
)
from duorat.types import (
    RATPreprocItem,
    PreprocQuestionToken,
    SQLSchema,
    QuestionTokenId,
)
from duorat.utils import registry
from duorat.utils.schema_linker import AbstractSchemaLinker
from duorat.utils.tokenization import AbstractTokenizer, BERTTokenizer


class DuoRATPreproc(abstract_preproc.AbstractPreproc):
    def __init__(self, **kwargs) -> None:
        self.save_path = kwargs["save_path"]
        self.min_freq = kwargs["min_freq"]
        # Number of schema-shuffles, for data augmentation. 0 means no shuffling.
        self.train_num_schema_shuffles = kwargs.get("train_num_schema_shuffles", 0)
        self.val_num_schema_shuffles = kwargs.get("val_num_schema_shuffles", 0)

        # for production rules + ReduceAction + MaskAction + GenToken tokens
        self.target_vocab_counter = Counter()
        self.target_vocab_path = os.path.join(self.save_path, "target_vocab.pkl")
        self.target_vocab = None

        self.counted_db_ids: Set[int] = set()
        self.sql_schemas: Dict[str, SQLSchema] = {}

        self.tokenizer: AbstractTokenizer = registry.construct(
            "tokenizer", kwargs["tokenizer"]
        )

        self.transition_system: TransitionSystem = registry.construct(
            "transition_system", kwargs["transition_system"]
        )

        self.schema_linker: AbstractSchemaLinker = registry.construct(
            "schema_linker", kwargs["schema_linker"]
        )

        self.preproc_items: Dict[str, List[RATPreprocItem]] = {}

    def input_a_str_to_id(self, s: str) -> int:
        raise NotImplementedError

    def input_b_str_to_id(self, s: str) -> int:
        raise NotImplementedError

    def _schema_tokenize(
        self, type: Optional[str], something: List[str], name: str
    ) -> List[str]:
        raise NotImplementedError

    def validate_item(
        self, item: SpiderItem, section: str
    ) -> Tuple[bool, Optional[AbstractSyntaxTree]]:
        if item.spider_schema.db_id not in self.sql_schemas:
            self.sql_schemas[item.spider_schema.db_id] = preprocess_schema_uncached(
                schema=item.spider_schema,
                db_path=item.db_path,
                tokenize=self._schema_tokenize,
            )

        try:
            if isinstance(item, SpiderItem) and isinstance(
                self.transition_system, SpiderTransitionSystem
            ):
                asdl_ast = self.transition_system.surface_code_to_ast(
                    code=item.spider_sql
                )
            else:
                raise NotImplementedError
            return True, asdl_ast
        except Exception as e:
            if "train" not in section:
                raise e
                return True, None
            else:
                raise e

    def preprocess_item(
        self,
        item: SpiderItem,
        sql_schema: SQLSchema,
        validation_info: AbstractSyntaxTree,
    ) -> RATPreprocItem:
        raise NotImplementedError

    def add_item(
        self, item: SpiderItem, section: str, validation_info: AbstractSyntaxTree
    ) -> None:
        """Adds item and copies of it with shuffled schema if num_schema_shuffles > 0"""
        sql_schema = self.sql_schemas[item.spider_schema.db_id]
        preproc_item_no_shuffle = self.preprocess_item(
            item, sql_schema, validation_info
        )
        preproc_items = [preproc_item_no_shuffle]

        if "train" in section:
            num_schema_shuffles = self.train_num_schema_shuffles
        elif "val" in section:
            num_schema_shuffles = self.val_num_schema_shuffles
        else:
            num_schema_shuffles = 0
        for _ in range(num_schema_shuffles):
            shuffled_schema = shuffle_schema(sql_schema)
            preproc_items.append(
                replace(preproc_item_no_shuffle, sql_schema=shuffled_schema)
            )

        if section not in self.preproc_items:
            self.preproc_items[section] = []
        self.preproc_items[section] += preproc_items

        if "train" in section:
            self.update_vocab(item, preproc_item_no_shuffle)

    def clear_items(self) -> None:
        self.preproc_items: Dict[str, List[RATPreprocItem]] = {}

    def save_examples(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        for section, items in self.preproc_items.items():
            with open(os.path.join(self.save_path, section + ".pkl"), "wb") as f:
                pickle.dump(items, f)

    def save(self) -> None:
        self.save_examples()

        # production rules + Reduce + MASK + GenToken tokens that are *not* in the encoder sequence
        for element in itertools.chain(
            map(
                lambda production: ApplyRuleAction(production=production),
                self.transition_system.grammar.id2prod.values(),
            ),
            (ReduceAction(), MaskAction()),
        ):
            self.target_vocab_counter[element] = self.min_freq
        self.target_vocab = ActionVocab(
            counter=self.target_vocab_counter,
            max_size=50000,
            min_freq=self.min_freq,
            specials=[ActionVocab.UNK],
        )
        with open(self.target_vocab_path, "wb") as f:
            pickle.dump(self.target_vocab, f)

    def load(self) -> None:
        with open(self.target_vocab_path, "rb") as f:
            self.target_vocab = pickle.load(f)

    def dataset(self, section: str) -> List[RATPreprocItem]:
        with open(os.path.join(self.save_path, section + ".pkl"), "rb") as f:
            items = pickle.load(f)
        return items


class SingletonGloVe(Vectors):

    _glove = None

    def __init__(self):
        SingletonGloVe._load_if_needed()

    @property
    def vectors(self):
        return SingletonGloVe._glove.vectors

    def __getitem__(self, token):
        return SingletonGloVe._glove[token]

    @property
    def dim(self):
        return self._glove.dim

    @property
    def stoi(self):
        return SingletonGloVe._glove.stoi

    @staticmethod
    def _load_if_needed():
        if SingletonGloVe._glove is None:
            SingletonGloVe._glove = GloVe(
                name="42B",
                dim=300,
                cache=os.path.join(
                    os.environ.get("CACHE_DIR", os.getcwd()), ".vector_cache"
                ),
            )

    def __setstate__(self, state):
        assert len(state) == 0
        self._load_if_needed()


@registry.register("preproc", "TransformerDuoRAT")
class TransformerDuoRATPreproc(DuoRATPreproc):
    def __init__(self, **kwargs) -> None:
        super(TransformerDuoRATPreproc, self).__init__(**kwargs)

        self.use_full_glove_vocab = kwargs.get("use_full_glove_vocab", False)

        # for GloVe tokens that appear in the training data
        self.input_vocab_a_counter = Counter()
        self.input_vocab_a_vectors = SingletonGloVe()
        self.input_vocab_a_path = os.path.join(self.save_path, "input_vocab_a.pkl")
        self.input_vocab_a = None

        # for tokens that appear in the training data and are not in GloVe
        self.input_vocab_b_counter = Counter()
        self.input_vocab_b_path = os.path.join(self.save_path, "input_vocab_b.pkl")
        self.input_vocab_b = None

        # self.grouped_payload: Dict[str, List[dict]] = defaultdict(list)

    def input_a_str_to_id(self, s: str) -> int:
        return self.input_vocab_a.__getitem__(s)

    def input_b_str_to_id(self, s: str) -> int:
        return self.input_vocab_b.__getitem__(s)

    def _schema_tokenize(
        self, type: Optional[str], something: List[str], name: str
    ) -> List[str]:
        return (
            ["<type: {}>".format(type)] if type is not None else []
        ) + self.tokenizer.tokenize(name)

    def preprocess_item(
        self,
        item: SpiderItem,
        sql_schema: SQLSchema,
        validation_info: AbstractSyntaxTree,
    ) -> RATPreprocItem:
        slml_question: str = self.schema_linker.question_to_slml(
            question=item.question, sql_schema=sql_schema,
        ) if item.slml_question is None else item.slml_question

        item.slml_question = slml_question
        # self.grouped_payload[sql_schema.db_id].append(item.orig)

        parser = SLMLParser(sql_schema=sql_schema, tokenizer=self.tokenizer)
        parser.feed(data=slml_question)
        parser.close()

        question: Tuple[PreprocQuestionToken, ...] = parser.question_tokens

        asdl_ast = validation_info
        actions = (
            tuple(self.transition_system.get_actions(asdl_ast))
            if asdl_ast is not None
            else tuple()
        )
        return RATPreprocItem(question=question, sql_schema=sql_schema, actions=actions)

    def update_vocab(self, item: SpiderItem, preproc_item: RATPreprocItem):
        if item.spider_schema.db_id in self.counted_db_ids:
            # tokens_to_count: List[str] = [token.value for token in question]
            tokens_to_count: List[str] = self.tokenizer.tokenize(item.question)
        else:
            self.counted_db_ids.add(item.spider_schema.db_id)
            tokens_to_count: List[str] = list(
                itertools.chain(
                    # (token.value for token in question),
                    self.tokenizer.tokenize(item.question),
                    *preproc_item.sql_schema.tokenized_column_names.values(),
                    *preproc_item.sql_schema.tokenized_table_names.values()
                )
            )

        # add to first input vocab only what is in GLoVe
        self.input_vocab_a_counter.update(
            (
                token
                for token in tokens_to_count
                if token in self.input_vocab_a_vectors.stoi
            )
        )

        # add only to second input vocab what is *not* already in first input vocab (GLoVe)
        self.input_vocab_b_counter.update(
            (
                token
                for token in tokens_to_count
                if token not in self.input_vocab_a_vectors.stoi
            )
        )

        # add only GenToken tokens to target vocab that are *not* in the encoder sequence
        self.target_vocab_counter.update(
            (
                action
                for action in all_spider_gen_token_actions(preproc_item.actions)
                if action.token not in tokens_to_count
            )
        )

    def save(self) -> None:
        super(TransformerDuoRATPreproc, self).save()

        # GloVe tokens that appear in the training data
        self.input_vocab_a = Vocab(
            counter=self.input_vocab_a_counter,
            max_size=50000,
            min_freq=1,
            vectors=self.input_vocab_a_vectors,
            specials=["<unk>"],
            specials_first=True,
        )
        with open(self.input_vocab_a_path, "wb") as f:
            pickle.dump(self.input_vocab_a, f)

        # tokens that appear in the training data and are not in GloVe
        self.input_vocab_b = Vocab(
            counter=self.input_vocab_b_counter,
            max_size=5000,
            min_freq=self.min_freq,
            specials=["<unk>"],
            specials_first=True,
        )
        with open(self.input_vocab_b_path, "wb") as f:
            pickle.dump(self.input_vocab_b, f)

    def load(self) -> None:
        super(TransformerDuoRATPreproc, self).load()

        if self.use_full_glove_vocab:
            glove_with_fake_freqs = {
                token: len(self.input_vocab_a_vectors) - index
                for token, index in self.input_vocab_a_vectors.stoi.items()
            }
            # by setting ',' frequency to 0 we make it an unknown word
            # this is ATM necessary for backward compatibility
            glove_with_fake_freqs[","] = 0
            self.input_vocab_a = Vocab(
                Counter(glove_with_fake_freqs), specials=["<unk>"], specials_first=True,
            )
        else:
            with open(self.input_vocab_a_path, "rb") as f:
                self.input_vocab_a = pickle.load(f)
        with open(self.input_vocab_b_path, "rb") as f:
            self.input_vocab_b = pickle.load(f)


@registry.register("preproc", "BertDuoRAT")
class BertDuoRATPreproc(DuoRATPreproc):
    tokenizer: BERTTokenizer

    def __init__(self, **kwargs) -> None:
        super(BertDuoRATPreproc, self).__init__(**kwargs)
        self.add_cls_token = kwargs["add_cls_token"]
        self.add_sep_token = kwargs["add_sep_token"]
        assert isinstance(self.tokenizer, BERTTokenizer)

    def input_a_str_to_id(self, s: str) -> int:
        return self.tokenizer.convert_token_to_id(s)

    def input_b_str_to_id(self, s: str) -> int:
        return 0

    def _schema_tokenize(
        self, type: Optional[str], something: List[str], name: str
    ) -> List[str]:
        return (
            ([self.tokenizer.cls_token] if self.add_cls_token else [])
            + self.tokenizer.tokenize(
                ("{} ".format(type) if type is not None else "") + name
            )
            + ([self.tokenizer.sep_token] if self.add_sep_token else [])
        )

    def preprocess_item(
        self,
        item: SpiderItem,
        sql_schema: SQLSchema,
        validation_info: AbstractSyntaxTree,
    ) -> RATPreprocItem:
        slml_question: str = self.schema_linker.question_to_slml(
            question=item.question, sql_schema=sql_schema,
        ) if item.slml_question is None else item.slml_question
        item.slml_question = slml_question

        parser = SLMLParser(sql_schema=sql_schema, tokenizer=self.tokenizer)
        parser.feed(data=slml_question)
        parser.close()

        original_question: Tuple[PreprocQuestionToken, ...] = (
            (
                PreprocQuestionToken(
                    key=QuestionTokenId(uuid4()), value=self.tokenizer.cls_token
                ),
            )
            if self.add_cls_token
            else tuple()
        ) + parser.question_tokens + (
            (
                PreprocQuestionToken(
                    key=QuestionTokenId(uuid4()), value=self.tokenizer.sep_token
                ),
            )
            if self.add_sep_token
            else tuple()
        )
        question = tuple()
        start = 0
        for i in range(len(original_question)):
            if original_question[i].value == '>' and original_question[i - 1].value == '>' and original_question[i - 2].value == '>':
                assert i >= 2
                assert original_question[i].match_tags == tuple() and original_question[i - 1].match_tags == tuple() and original_question[i - 2].match_tags == tuple()
                end = i - 2
                question += original_question[start:end] + (
                    PreprocQuestionToken(
                        key=QuestionTokenId(uuid4()), value=self.tokenizer.sep_token
                    ),
                )
                start = i + 1

        asdl_ast = validation_info
        actions = tuple(self.transition_system.get_actions(asdl_ast))
        return RATPreprocItem(question=question, sql_schema=sql_schema, actions=actions)

    def update_vocab(self, item: SpiderItem, preproc_item: RATPreprocItem):
        if item.spider_schema.db_id in self.counted_db_ids:
            tokens_to_count: List[str] = [
                token.value for token in preproc_item.question
            ]
        else:
            self.counted_db_ids.add(item.spider_schema.db_id)
            tokens_to_count: List[str] = list(
                itertools.chain(
                    (token.value for token in preproc_item.question),
                    *preproc_item.sql_schema.tokenized_column_names.values(),
                    *preproc_item.sql_schema.tokenized_table_names.values()
                )
            )

        # add only GenToken tokens to target vocab that are *not* in the encoder sequence
        self.target_vocab_counter.update(
            (
                action
                for action in all_spider_gen_token_actions(preproc_item.actions)
                if action.token not in tokens_to_count
            )
        )

    def save(self) -> None:
        super(BertDuoRATPreproc, self).save()

    def load(self) -> None:
        super(BertDuoRATPreproc, self).load()
