# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class MosesTokenizerConfig(FairseqDataclass):
    source_lang: str = field(default="en", metadata={"help": "Source language code (e.g., 'en')"})
    target_lang: str = field(default="en", metadata={"help": "Target language code (e.g., 'de')"})
    moses_no_dash_splits: bool = field(
        default=False, metadata={"help": "Disable aggressive dash splitting (e.g., 'e-mail' → 'e - mail')"}
    )
    moses_no_escape: bool = field(
        default=False, metadata={"help": "Disable escaping of punctuation (e.g., quotes, apostrophes)"}
    )


@register_tokenizer("moses", dataclass=MosesTokenizerConfig)
class MosesTokenizer:
    """
    A wrapper around the SacreMoses tokenizer and detokenizer,
    supporting configurable dash-splitting and escaping.
    """

    def __init__(self, cfg: MosesTokenizerConfig) -> None:
        self.cfg = cfg

        try:
            from sacremoses import MosesDetokenizer as SacreMosesDetokenizer
            from sacremoses import MosesTokenizer as SacreMosesTokenizer
        except ImportError as e:
            raise ImportError("MosesTokenizer requires `sacremoses`. Install it via: pip install sacremoses") from e

        self._tokenizer = SacreMosesTokenizer(cfg.source_lang)
        self._detokenizer = SacreMosesDetokenizer(cfg.target_lang)

    def encode(self, text: str) -> str:
        """Tokenize input text using MosesTokenizer.

        Args:
            text (str): Raw input string.

        Returns:
            str: Tokenized string.
        """
        return self._tokenizer.tokenize(
            text,
            aggressive_dash_splits=not self.cfg.moses_no_dash_splits,
            return_str=True,
            escape=not self.cfg.moses_no_escape,
        )

    def decode(self, text: str) -> str:
        """Detokenize tokenized text back to natural string.

        Args:
            text (str): Tokenized string with space-separated tokens.

        Returns:
            str: Detokenized natural string.
        """
        tokens = text.split()
        return self._detokenizer.detokenize(tokens)
