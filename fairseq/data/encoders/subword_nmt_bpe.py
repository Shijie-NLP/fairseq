# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class SubwordNMTBPEConfig(FairseqDataclass):
    bpe_codes: str = field(default="???", metadata={"help": "Path to Subword NMT BPE codes"})
    bpe_separator: str = field(default="@@", metadata={"help": "BPE separator symbol"})


@register_bpe("subword_nmt", dataclass=SubwordNMTBPEConfig)
class SubwordNMTBPE:
    """Subword NMT BPE encoder/decoder compatible with Fairseq."""

    def __init__(self, cfg: SubwordNMTBPEConfig) -> None:
        if not cfg.bpe_codes or cfg.bpe_codes == "???":
            raise ValueError("--bpe-codes is required for --bpe=subword_nmt")

        try:
            from subword_nmt import apply_bpe
        except ImportError as e:
            raise ImportError("subword-nmt is not installed. Please install via `pip install subword-nmt`.") from e

        codes_path = file_utils.cached_path(cfg.bpe_codes)
        parser = apply_bpe.create_parser()
        args = parser.parse_args(
            [
                "--codes",
                codes_path,
                "--separator",
                cfg.bpe_separator,
            ]
        )

        self.bpe = apply_bpe.BPE(
            codes=args.codes,
            merges=args.merges,
            separator=args.separator,
            vocab=None,
            glossaries=args.glossaries,
        )
        self.bpe_symbol: str = f"{args.separator} "

    def encode(self, text: str) -> str:
        """Apply BPE encoding to input text."""
        return self.bpe.process_line(text)

    def decode(self, text: str) -> str:
        """Remove BPE separators from encoded text."""
        return (text + " ").replace(self.bpe_symbol, "").rstrip()
