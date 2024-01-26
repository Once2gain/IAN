# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset
import numpy as np


class FeatureDataset(FairseqDataset):
    def __init__(self, vec_path, keep_raw=False):
        super().__init__()
        self.keep_raw = keep_raw
        if keep_raw:
            self.feature_vector = np.load(vec_path + '.npy', allow_pickle=True)
        else:
            self.feature_vector = np.load(vec_path + '.npy', allow_pickle=True).astype(np.float16)

    def __getitem__(self, index):
        return self.feature_vector[index]

    def __len__(self):
        return len(self.feature_vector)

    def collater(self, samples):
        if self.keep_raw:
            new_samples = []
            for item in samples:
                new_item = []
                for array in item:
                    new_item.append(torch.tensor(array.astype(np.float16), dtype=torch.float16))
                new_item = torch.stack(new_item)
                new_samples.append(new_item)
            return new_samples
        return torch.tensor(np.array(samples), dtype=torch.float16)


