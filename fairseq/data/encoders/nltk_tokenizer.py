# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@register_tokenizer("nltk", dataclass=FairseqDataclass)
class NLTKTokenizer:
    """Tokenizer wrapper using NLTK's word_tokenize."""

    def __init__(self, *args, **kwargs) -> None:
        try:
            import nltk

            nltk.download("punkt", quiet=True)
            from nltk.tokenize import word_tokenize

            self._tokenize = word_tokenize
        except ImportError as e:
            raise ImportError("NLTK is not installed. Please install it via `pip install nltk`.") from e

    def encode(self, text: str) -> str:
        """Tokenizes input text into space-separated tokens."""
        tokens = self._tokenize(text)
        return " ".join(tokens)

    def decode(self, text: str) -> str:
        """Decoding is identity; assumes space-separated tokens."""
        return text
