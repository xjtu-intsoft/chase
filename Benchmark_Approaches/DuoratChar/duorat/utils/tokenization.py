import abc
from functools import lru_cache
from typing import List, Sequence, Tuple

import stanza
from transformers import BertTokenizerFast

from duorat.utils import registry, corenlp


class AbstractTokenizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tokenize(self, s: str) -> List[str]:
        pass

    @abc.abstractmethod
    def detokenize(self, xs: Sequence[str]) -> str:
        pass


@registry.register("tokenizer", "CoreNLPTokenizer")
class CoreNLPTokenizer(AbstractTokenizer):
    def __init__(self):
        pass

    @lru_cache(maxsize=1024)
    def _tokenize(self, s: str) -> List[str]:
        ann = corenlp.annotate(
            text=s,
            annotators=["tokenize", "ssplit"],
            properties={
                "outputFormat": "serialized",
                "tokenize.options": "asciiQuotes = false, latexQuotes=false, unicodeQuotes=false, ",
            },
        )
        return [tok.word for sent in ann.sentence for tok in sent.token]

    def tokenize(self, s: str) -> List[str]:
        return [token.lower() for token in self._tokenize(s)]

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        return [(token.lower(), token) for token in self._tokenize(s)]

    def detokenize(self, xs: Sequence[str]) -> str:
        return " ".join(xs)


@registry.register("tokenizer", "StanzaTokenizer")
class StanzaTokenizer(AbstractTokenizer):
    def __init__(self):
        stanza.download("en", processors="tokenize")
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize")

    @lru_cache(maxsize=1024)
    def tokenize(self, s: str) -> List[str]:
        doc = self.nlp(s)
        return [
            token.question for sentence in doc.sentences for token in sentence.tokens
        ]

    def detokenize(self, xs: Sequence[str]) -> str:
        return " ".join(xs)


@registry.register("tokenizer", "BERTTokenizer")
class BERTTokenizer(AbstractTokenizer):
    def __init__(self, pretrained_model_name_or_path: str):
        self._bert_tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

    def tokenize(self, s: str) -> List[str]:
        s = s.lower()
        return self._bert_tokenizer.tokenize(s)

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        # TODO: at some point, hopefully, transformers API will be mature enough
        # to do this in 1 call instead of 2
        s = s.lower()
        tokens = self._bert_tokenizer.tokenize(s)
        encoding_result = self._bert_tokenizer(s, return_offsets_mapping=True)
        assert len(encoding_result[0]) == len(tokens) + 2
        raw_token_strings = [
            s[start:end] for start, end in encoding_result["offset_mapping"][1:-1]
        ]
        raw_token_strings_with_sharps = []
        for token, raw_token in zip(tokens, raw_token_strings):
            assert (
                token == raw_token.lower()
                or token[2:] == raw_token.lower()
                or token[-2:] == raw_token.lower()
            )
            if token.startswith("##"):
                raw_token_strings_with_sharps.append("##" + raw_token)
            elif token.endswith("##"):
                raw_token_strings_with_sharps.append(raw_token + "##")
            else:
                raw_token_strings_with_sharps.append(raw_token)
        return zip(tokens, raw_token_strings_with_sharps)

    def detokenize(self, xs: Sequence[str]) -> str:
        """Naive implementation, see https://github.com/huggingface/transformers/issues/36"""
        text = " ".join([x for x in xs])
        fine_text = text.replace(" ##", "")
        return fine_text

    def convert_token_to_id(self, s: str) -> int:
        return self._bert_tokenizer.convert_tokens_to_ids(s)

    @property
    def cls_token(self) -> str:
        return self._bert_tokenizer.cls_token

    @property
    def sep_token(self) -> str:
        return self._bert_tokenizer.sep_token
