# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


SPACE = chr(32)
SPACE_ESCAPE = chr(9601)


@register_tokenizer("character", dataclass=FairseqDataclass)
class CharacterTokenizer:
    """A simple character-level tokenizer that splits text into individual characters
    and escapes whitespace using a special token."""

    def __init__(self, *args, **kwargs) -> None:
        # Placeholder for future extensibility
        pass

    @staticmethod
    def add_args(parser: Any) -> None:
        """Add tokenizer-specific arguments to the parser (currently unused)."""
        pass

    @staticmethod
    def encode(text: str) -> str:
        """Encode the input string into a space-separated character sequence,
        replacing spaces with a special escape token.

        Args:
            text (str): Input text to be encoded.

        Returns:
            str: Encoded character-level string.
        """
        escaped = text.replace(SPACE, SPACE_ESCAPE)
        return SPACE.join(list(escaped))

    @staticmethod
    def decode(text: str) -> str:
        """Decode a space-separated character sequence back to the original string,
        restoring escaped spaces.

        Args:
            text (str): Encoded character-level string.

        Returns:
            str: Decoded original string.
        """
        return text.replace(SPACE, "").replace(SPACE_ESCAPE, SPACE)
