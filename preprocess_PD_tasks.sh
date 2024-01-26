#!/bin/bash

#ROOT="/data/tangqirui/fairseq"
SRC_DATA_DIR="DATA-bin/kaggle-8-randomx"
DEST_DATA_DIR="DATA-bin/kaggle-8-randomx"

DATASETS="doc_dataset"
for DATASET in $DATASETS
do
  echo "Preprocessing $DATASET"

  python preprocess.py \
    --only-source \
    --trainpref "$SRC_DATA_DIR/$DATASET/train.input" \
    --validpref "$SRC_DATA_DIR/$DATASET/valid.input" \
    --testpref "$SRC_DATA_DIR/$DATASET/test.input" \
    --destdir "$DEST_DATA_DIR/$DATASET/input" \
    --workers 60 \
    --srcdict "gpt2_bpe/dict.txt" \
    --enc-document;

  python preprocess.py \
    --only-source \
    --trainpref "$SRC_DATA_DIR/$DATASET/train.label" \
    --validpref "$SRC_DATA_DIR/$DATASET/valid.label" \
    --testpref "$SRC_DATA_DIR/$DATASET/test.label" \
    --destdir "$DEST_DATA_DIR/$DATASET/label" \
    --workers 60;

done