# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class RawDataDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def collater(self, samples):
        # sen2doc
        return samples
