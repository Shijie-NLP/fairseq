# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os

from fairseq import registry


build_tokenizer, register_tokenizer, TOKENIZER_REGISTRY, _ = registry.setup_registry(
    "--tokenizer",
    default=None,
)


build_bpe, register_bpe, BPE_REGISTRY, _ = registry.setup_registry(
    "--bpe",
    default=None,
)


def import_all_encoders():
    current_dir = os.path.dirname(__file__)
    module_prefix = "fairseq.data.encoders"

    for filename in sorted(os.listdir(current_dir)):
        if filename.endswith(".py") and not filename.startswith("_"):
            module_name = filename[:-3]  # Strip ".py"
            importlib.import_module(f"{module_prefix}.{module_name}")


# Trigger automatic import
import_all_encoders()
