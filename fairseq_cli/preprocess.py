#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
from argparse import Namespace
from itertools import zip_longest
from typing import Optional

from fairseq import options, tasks, utils
from fairseq.binarizer import (
    AlignmentDatasetBinarizer,
    FileBinarizer,
    VocabularyDatasetBinarizer,
)
from fairseq.data import Dictionary


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")


def _train_path(lang: str, trainpref: str) -> str:
    return f"{trainpref}.{lang}" if lang else trainpref


def _file_name(prefix: str, lang: Optional[str]) -> str:
    return f"{prefix}.{lang}" if lang else prefix


def _dest_path(prefix: str, lang: Optional[str], destdir: str) -> str:
    return os.path.join(destdir, _file_name(prefix, lang))


def _dict_path(lang: str, destdir: str) -> str:
    return f"{_dest_path('dict', lang, destdir)}.txt"


def dataset_dest_prefix(args, output_prefix, lang):
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}"
        if lang
        else ("" if args.only_source else f".{args.source_lang}-{args.target_lang}")
    )
    return os.path.join(args.destdir, f"{output_prefix}{lang_part}")


def dataset_dest_file(args, output_prefix: str, lang: Optional[str], extension: str) -> str:
    return f"{dataset_dest_prefix(args, output_prefix, lang)}.{extension}"


def _build_dictionary(filenames, task, args, src=False):
    return task.build_dictionary(
        filenames,
        workers=args.workers,
        threshold=args.thresholdsrc if src else args.thresholdtgt,
        nwords=args.nwordssrc if src else args.nwordstgt,
        padding_factor=args.padding_factor,
    )


def _make_binary_dataset(
    vocab: Dictionary, input_prefix: str, output_prefix: str, lang: Optional[str], num_workers: int, args: Namespace
):
    logger.info(f"[{lang}] Dictionary: {len(vocab)} types")

    binarizer = VocabularyDatasetBinarizer(vocab, append_eos=True)
    input_file = f"{input_prefix}.{lang}" if lang else input_prefix
    full_output_prefix = dataset_dest_prefix(args, output_prefix, lang)

    summary = FileBinarizer.multiprocess_dataset(
        input_file,
        args.dataset_impl,
        binarizer,
        full_output_prefix,
        vocab_size=len(vocab),
        num_workers=num_workers,
    )

    logger.info(f"[{lang}] {input_file}: {summary} (by {vocab.unk_word})")


def _make_binary_alignment_dataset(input_prefix: str, output_prefix: str, num_workers: int, args: Namespace):
    binarizer = AlignmentDatasetBinarizer(utils.parse_alignment)
    full_output_prefix = dataset_dest_prefix(args, output_prefix, lang=None)

    summary = FileBinarizer.multiprocess_dataset(
        input_prefix,
        args.dataset_impl,
        binarizer,
        full_output_prefix,
        vocab_size=None,
        num_workers=num_workers,
    )

    logger.info(f"[alignments] {input_prefix}: parsed {summary.num_seq} alignments")


def _make_dataset(
    vocab: Dictionary, input_prefix: str, output_prefix: str, lang: Optional[str], args: Namespace, num_workers: int
):
    if args.dataset_impl == "raw":
        output_text_file = _dest_path(f"{output_prefix}.{args.source_lang}-{args.target_lang}", lang, args.destdir)
        shutil.copyfile(_file_name(input_prefix, lang), output_text_file)
    else:
        _make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, args)


def _make_all(lang: str, vocab: Dictionary, args: Namespace):
    if args.trainpref:
        _make_dataset(vocab, args.trainpref, "train", lang, args, args.workers)

    for k, validpref in enumerate(args.validpref.split(",") if args.validpref else []):
        prefix = f"valid{k}" if k else "valid"
        _make_dataset(vocab, validpref, prefix, lang, args, args.workers)

    for k, testpref in enumerate(args.testpref.split(",") if args.testpref else []):
        prefix = f"test{k}" if k else "test"
        _make_dataset(vocab, testpref, prefix, lang, args, args.workers)


def _make_all_alignments(args: Namespace):
    for prefix, name in zip(["train", "valid", "test"], [args.trainpref, args.validpref, args.testpref]):
        if name and os.path.exists(f"{name}.{args.align_suffix}"):
            _make_binary_alignment_dataset(f"{name}.{args.align_suffix}", f"{prefix}.align", args.workers, args)


def _align_files(args, src_dict, tgt_dict):
    if not args.trainpref:
        raise ValueError("--trainpref must be set if --alignfile is specified")

    src_file = _train_path(args.source_lang, args.trainpref)
    tgt_file = _train_path(args.target_lang, args.trainpref)
    freq_map = {}

    with (
        open(args.alignfile, encoding="utf-8") as a_file,
        open(src_file, encoding="utf-8") as s_file,
        open(tgt_file, encoding="utf-8") as t_file,
    ):
        for a, s, t in zip_longest(a_file, s_file, t_file):
            si = src_dict.encode_line(s, add_if_not_exist=False)
            ti = tgt_dict.encode_line(t, add_if_not_exist=False)
            for sai, tai in (x.split("-") for x in a.split()):
                srcidx, tgtidx = si[int(sai)], ti[int(tai)]
                if srcidx in {src_dict.unk(), src_dict.pad(), src_dict.eos()}:
                    continue
                if tgtidx in {tgt_dict.unk(), tgt_dict.pad(), tgt_dict.eos()}:
                    continue
                freq_map.setdefault(srcidx, {}).setdefault(tgtidx, 0)
                freq_map[srcidx][tgtidx] += 1

    align_dict = {src: max(tgt_freq, key=tgt_freq.get) for src, tgt_freq in freq_map.items()}
    out_path = os.path.join(args.destdir, f"alignment.{args.source_lang}-{args.target_lang}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for src, tgt in align_dict.items():
            print(f"{src_dict[src]} {tgt_dict[tgt]}", file=f)


def main(args: Namespace):
    utils.import_user_module(args)
    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(logging.FileHandler(os.path.join(args.destdir, "preprocess.log")))
    logger.info(args)

    if args.dataset_impl == "huffman":
        raise NotImplementedError("Huffman not supported. Use HuffmanCodeBuilder.")

    target = not args.only_source

    if not args.srcdict and os.path.exists(_dict_path(args.source_lang, args.destdir)):
        raise FileExistsError(_dict_path(args.source_lang, args.destdir))

    if target and not args.tgtdict and os.path.exists(_dict_path(args.target_lang, args.destdir)):
        raise FileExistsError(_dict_path(args.target_lang, args.destdir))

    task = tasks.get_task(args.task)

    if args.joined_dictionary:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            src_dict = _build_dictionary(
                {_train_path(lang, args.trainpref) for lang in [args.source_lang, args.target_lang]},
                task,
                args,
                src=True,
            )
        tgt_dict = src_dict
    else:
        src_dict = (
            task.load_dictionary(args.srcdict)
            if args.srcdict
            else _build_dictionary([_train_path(args.source_lang, args.trainpref)], task, args, src=True)
        )
        tgt_dict = (
            (
                task.load_dictionary(args.tgtdict)
                if args.tgtdict
                else _build_dictionary([_train_path(args.target_lang, args.trainpref)], task, args, src=False)
            )
            if target
            else None
        )

    src_dict.save(_dict_path(args.source_lang, args.destdir))
    if target and tgt_dict:
        tgt_dict.save(_dict_path(args.target_lang, args.destdir))

    if args.dict_only:
        return

    _make_all(args.source_lang, src_dict, args)
    if target and tgt_dict:
        _make_all(args.target_lang, tgt_dict, args)

    if args.align_suffix:
        _make_all_alignments(args)

    logger.info(f"Wrote preprocessed data to {args.destdir}")

    if args.alignfile:
        _align_files(args, src_dict, tgt_dict)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
