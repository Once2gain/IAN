#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

from fairseq_cli.preprocess import cli_main


if __name__ == "__main__":
    cli_main()



# --only-source
# --trainpref
# prepare/essay_big5_preprocess/new/fold-0/seg_dataset/train.input
# --validpref
# prepare/essay_big5_preprocess/new/fold-0/seg_dataset/valid.input
# --destdir
# DATA-bin/fold-0/seg_dataset/input
# --workers
# 60
# --srcdict
# gpt2_bpe/dict.txt
# --enc-document

# --only-source
# --trainpref
# prepare/essay_big5_preprocess/new/fold-0/seg_dataset/train.agr
# --validpref
# prepare/essay_big5_preprocess/new/fold-0/seg_dataset/valid.agr
# --destdir
# DATA-bin/fold-0/seg_dataset/agr
# --workers
# 60

# input

# --only-source
# --trainpref
# prepare/essay_big5_preprocess/finetune/fold-0/train/train.input
# --validpref
# prepare/essay_big5_preprocess/finetune/fold-0/valid/valid.input
# --destdir
# PD-bin/finetune/fold-0/input
# --workers
# 60
# --srcdict
# gpt2_bpe/dict.txt
# --enc-document

# opn

# --only-source
# --trainpref
# prepare/essay_big5_preprocess/finetune/fold-0/train/train.opn
# --validpref
# prepare/essay_big5_preprocess/finetune/fold-0/valid/valid.opn
# --destdir
# PD-bin/finetune/fold-0/opn
# --workers
# 60

# futher pretrain
# --only-source
# --trainpref
# prepare/essay_big5_preprocess/pretrain/train.bpe
# --validpref
# prepare/essay_big5_preprocess/pretrain/valid.bpe
# --destdir
# data-bin/finetune/big5
# --srcdict
# gpt2_bpe/dict.txt
# --workers
# 60