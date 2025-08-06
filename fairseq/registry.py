# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Union

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent


REGISTRIES = {}


def setup_registry(registry_name: str, base_class=None, default=None, required=False):
    assert registry_name.startswith("--"), "Registry name must start with '--'"
    registry_key = registry_name[2:].replace("-", "_")

    if registry_key in REGISTRIES:
        return  # Registry already exists

    registry = {}
    dataclass_registry = {}
    registered_class_names = set()

    REGISTRIES[registry_key] = {
        "registry": registry,
        "default": default,
        "dataclass_registry": dataclass_registry,
    }

    def build_x(cfg: Union[DictConfig, str, Namespace], *args, **kwargs):
        from_checkpoint = kwargs.pop("from_checkpoint", False)

        if isinstance(cfg, DictConfig):
            choice = cfg._name
            if choice and choice in dataclass_registry:
                dataclass = dataclass_registry[choice]
                cfg = merge_with_parent(dataclass(), cfg, remove_missing=from_checkpoint)
        elif isinstance(cfg, str):
            choice = cfg
            cfg = dataclass_registry[choice]() if choice in dataclass_registry else cfg
        else:
            choice = getattr(cfg, registry_key, None)
            if choice in dataclass_registry:
                cfg = dataclass_registry[choice].from_namespace(cfg)

        if not choice:
            if required:
                raise ValueError(f"{registry_key} is required!")
            return None

        cls = registry[choice]
        builder = getattr(cls, f"build_{registry_key}", cls)

        return builder(cfg, *args, **kwargs)

    def register_x(name, dataclass=None):
        def decorator(cls):
            if name in registry:
                raise ValueError(f"Duplicate registration of '{name}' in {registry_key}")
            if cls.__name__ in registered_class_names:
                raise ValueError(f"Class name '{cls.__name__}' already registered in {registry_key}")
            if base_class and not issubclass(cls, base_class):
                raise TypeError(f"{cls.__name__} must extend {base_class.__name__}")

            if dataclass:
                if not issubclass(dataclass, FairseqDataclass):
                    raise TypeError(f"Dataclass for '{name}' must extend FairseqDataclass")

                dataclass_registry[name] = dataclass

                node = dataclass()
                node._name = name
                ConfigStore.instance().store(name=name, group=registry_key, node=node, provider="fairseq")

            cls.__dataclass = dataclass
            registry[name] = cls
            registered_class_names.add(cls.__name__)
            return cls

        return decorator

    return build_x, register_x, registry, dataclass_registry
