#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

from fairseq_cli.train import cli_main


if __name__ == "__main__":
    cli_main()

# sen2doc unfreeze top 4 layers

# PD-bin/
# --restore-file
# /tangqirui/fairseq/download/model.pt
# --batch-size
# 16
# --max-tokens
# 150000
# --task
# personality_detection
# --reset-optimizer
# --reset-dataloader
# --reset-meters
# --required-batch-size-multiple
# 1
# --arch
# sen2doc
# --criterion
# personality_detection
# --multi-label
# --num-classes
# 5
# --weight-decay
# 0.1
# --optimizer
# adam
# --adam-betas
# "(0.9, 0.98)"
# --adam-eps
# 1e-06
# --clip-norm
# 0.0
# --lr-scheduler
# polynomial_decay
# --lr
# 1e-04
# --total-num-update
# 900
# --warmup-updates
# 54
# --max-epoch
# 30
# --find-unused-parameters
# --best-checkpoint-metric
# ave-accuracy
# --maximize-best-checkpoint-metric
# --no-epoch-checkpoints
# --fp16
# --fp16-init-scale
# 4
# --threshold-loss-scale
# 1
# --fp16-scale-window
# 128
# --update-freq
# 4


