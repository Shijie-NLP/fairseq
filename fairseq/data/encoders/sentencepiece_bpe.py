# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class SentencepieceConfig(FairseqDataclass):
    sentencepiece_model: str = field(default="???", metadata={"help": "Path to the SentencePiece model file (.model)"})
    sentencepiece_enable_sampling: bool = field(
        default=False, metadata={"help": "Enable sampling when encoding (for training or decoding variants)"}
    )
    sentencepiece_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Smoothing parameter for unigram sampling or merge dropout rate for BPE mode"},
    )


@register_bpe("sentencepiece", dataclass=SentencepieceConfig)
class SentencepieceBPE:
    def __init__(self, cfg: SentencepieceConfig) -> None:
        self.enable_sampling: bool = cfg.sentencepiece_enable_sampling
        self.alpha: Optional[float] = cfg.sentencepiece_alpha

        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "The `sentencepiece` library is required. Install it via: pip install sentencepiece"
            ) from e

        model_path = file_utils.cached_path(cfg.sentencepiece_model)
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.Load(model_path):
            raise RuntimeError(f"Failed to load SentencePiece model from: {model_path}")

    def encode(self, text: str) -> str:
        """Encodes a string into space-separated SentencePiece tokens."""
        tokens = self.sp.Encode(
            text,
            out_type=str,
            enable_sampling=self.enable_sampling,
            alpha=self.alpha if self.alpha is not None else 0.1,  # default fallback
        )
        return " ".join(tokens)

    def decode(self, text: str) -> str:
        """Decodes a space-separated SentencePiece token string back to plain text."""
        # "\u2581" = ▁ in Unicode, SentencePiece's marker for word boundaries
        return text.replace(" ", "").replace("\u2581", " ").strip()

    def is_beginning_of_word(self, token: str) -> bool:
        """Checks if a token is at the beginning of a word."""
        # Special tokens are treated as word-beginnings
        if token in {"<unk>", "<s>", "</s>", "<pad>"}:
            return True
        return token.startswith("\u2581")
