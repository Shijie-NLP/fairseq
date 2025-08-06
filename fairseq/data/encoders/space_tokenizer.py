# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@register_tokenizer("space", dataclass=FairseqDataclass)
class SpaceTokenizer:
    """A simple whitespace-normalising tokenizer that replaces multiple
    whitespace characters with a single space during encoding."""

    def __init__(self, *args, **kwargs):
        self._space_pattern = re.compile(r"\s+")

    def encode(self, text: str) -> str:
        """Normalize whitespace in the input string.

        Args:
            text (str): Input string to encode.

        Returns:
            str: Encoded string with normalized spaces.
        """
        return self._space_pattern.sub(" ", text.strip())

    def decode(self, text: str) -> str:
        """Return the decoded string (identity function for this tokenizer).

        Args:
            text (str): Encoded string.

        Returns:
            str: Decoded string (same as input).
        """
        return text
