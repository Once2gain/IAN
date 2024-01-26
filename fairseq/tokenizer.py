# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_doc(doc):
    words = []
    str_list = doc.split('-100')
    for sent_str in str_list:
        sent_str = SPACE_NORMALIZER.sub(" ", sent_str)
        sent_str = sent_str.strip()
        words.append(sent_str.split())

    return words
