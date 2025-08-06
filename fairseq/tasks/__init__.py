# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from hydra.core.config_store import ConfigStore

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent

from .fairseq_task import FairseqTask, LegacyFairseqTask  # noqa


# register dataclass
TASK_REGISTRY = {}
TASK_DATACLASS_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task(cfg: FairseqDataclass, **kwargs) -> FairseqTask:
    task_name = getattr(cfg, "task", None)
    task_cls = None

    if isinstance(task_name, str):
        # legacy tasks
        task_cls = TASK_REGISTRY.get(task_name)
        if task_name in TASK_DATACLASS_REGISTRY:
            dataclass = TASK_DATACLASS_REGISTRY[task_name]
            cfg = dataclass.from_namespace(cfg)
    else:
        task_name = getattr(cfg, "_name", None)
        if task_name and task_name in TASK_DATACLASS_REGISTRY:
            remove_missing = kwargs.get("from_checkpoint", False)
            dataclass = TASK_DATACLASS_REGISTRY[task_name]
            cfg = merge_with_parent(dataclass(), cfg, remove_missing=remove_missing)
            task_cls = TASK_REGISTRY.get(task_name)

    if task_cls is None:
        raise ValueError(
            f"Could not infer task from config: {cfg}. "
            f"Available argparse tasks: {list(TASK_REGISTRY.keys())}, "
            f"hydra tasks: {list(TASK_DATACLASS_REGISTRY.keys())}"
        )

    return task_cls.setup_task(cfg, **kwargs)


def register_task(name, dataclass=None):
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def decorator(cls):
        if name in TASK_REGISTRY:
            return TASK_REGISTRY[name]
        if not issubclass(cls, FairseqTask):
            raise TypeError(f"Task '{name}' must extend FairseqTask.")
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(f"Duplicate class name '{cls.__name__}' in task registry.")

        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        cls.__dataclass = dataclass

        if dataclass:
            if not issubclass(dataclass, FairseqDataclass):
                raise TypeError(f"Dataclass for '{name}' must extend FairseqDataclass.")

            TASK_DATACLASS_REGISTRY[name] = dataclass

            node = dataclass()
            node._name = name
            ConfigStore.instance().store(name=name, group="task", node=node, provider="fairseq")

        return cls

    return decorator


def get_task(name):
    return TASK_REGISTRY[name]


def _expose_parser(task_name: str):
    parser = argparse.ArgumentParser(add_help=False)
    group_task = parser.add_argument_group("Task name")
    group_task.add_argument("--task", metavar=task_name, help=f"Enable this task with: ``--task={task_name}``")
    group_args = parser.add_argument_group("Additional command-line arguments")
    TASK_REGISTRY[task_name].add_args(group_args)
    globals()[f"{task_name}_parser"] = parser


def import_tasks(tasks_dir, namespace):
    for filename in os.listdir(tasks_dir):
        if filename.startswith(("_", ".")):
            continue

        path = os.path.join(tasks_dir, filename)
        is_py_file = filename.endswith(".py")
        is_module_dir = os.path.isdir(path)

        if is_py_file or is_module_dir:
            module_name = filename[:-3] if is_py_file else filename
            importlib.import_module(f"{namespace}.{module_name}")

            # expose `task_parser` for sphinx
            if module_name in TASK_REGISTRY:
                _expose_parser(module_name)


# Automatically import all Python files in the current `tasks/` directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "fairseq.tasks")
