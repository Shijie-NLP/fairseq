# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from omegaconf import DictConfig

from fairseq import registry
from fairseq.criterions.fairseq_criterion import (  # noqa
    FairseqCriterion,
    LegacyFairseqCriterion,
)


(
    _build_criterion,
    register_criterion,
    CRITERION_REGISTRY,
    CRITERION_DATACLASS_REGISTRY,
) = registry.setup_registry("--criterion", base_class=FairseqCriterion, default="cross_entropy")


def build_criterion(cfg: DictConfig, task, from_checkpoint: bool = False):
    return _build_criterion(cfg, task, from_checkpoint=from_checkpoint)


def import_all_criterions():
    current_dir = os.path.dirname(__file__)
    module_prefix = "fairseq.criterions"

    for filename in sorted(os.listdir(current_dir)):
        if filename.endswith(".py") and not filename.startswith("_"):
            module_name = filename[:-3]  # Strip '.py'
            importlib.import_module(f"{module_prefix}.{module_name}")


# Import modules when this file is loaded
import_all_criterions()
