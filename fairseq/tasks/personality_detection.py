# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import contextlib
import os
from dataclasses import dataclass, field
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDocDataset,
    PrependTokenDataset,
    StripTokenDataset,
    data_utils,
    FeatureDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from omegaconf import MISSING, II, open_dict, OmegaConf
from fairseq.dataclass import ChoiceEnum
from typing import Any, List
import numpy as np

logger = logging.getLogger(__name__)
LABEL_CHOICES = ChoiceEnum(["ext", "neu", "agr", "con", "opn", "EI", "SN", "TF", "JP"])
TASK_CHOICES = ChoiceEnum(["seg", "doc"])
PERSONALITY_CHOICES = ChoiceEnum(["big5", "mbti"])


@dataclass
class PersonalityDetectionConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )
    max_positions: int = field(
        default=512,
        metadata={"help": "max tokens per example"},
    )
    classify_task: TASK_CHOICES = field(
        default="seg",
        metadata={
            "help": "if none, seg"
        },
    )
    feats_type: List[str] = field(
        default_factory=lambda: ['emotion', 'senticnet'],
        metadata={
            "help": "feats_type"
        },
    )
    vote: bool = field(
        default=False,
    )
    personality: PERSONALITY_CHOICES = field(
        default="big5",
        metadata={
            "help": "if none, big5"
        },
    )
    class_name: LABEL_CHOICES = field(
        default="EI",
        metadata={
            "help": "if none, the multi-targets"
        },
    )
    seed: int = II("common.seed")


@register_task("personality_detection", dataclass=PersonalityDetectionConfig)
class PersonalityDetectionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    def __init__(self, cfg: PersonalityDetectionConfig, data_dictionary, label_dictionary):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary

    def get_feats_type(self):
        return self.cfg.feats_type

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, cfg: PersonalityDetectionConfig, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(cfg.data, "input", "dict.txt"),
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        label_dict = cls.load_dictionary(
            os.path.join(cfg.data, "label", "dict.txt"),
        )
        logger.info("[label] dictionary: {} types".format(len(label_dict)))

        return cls(cfg, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_data_path(type, split):
            return os.path.join(self.cfg.data, type, split)

        def make_dataset(type, dictionary, impl='mmap'):
            split_path = get_data_path(type, split)
            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                dataset_impl=impl,
                combine=combine,
            )
            return dataset

        # add bos token for cls task
        if str(self.cfg.classify_task) == 'doc':
            input_data = make_dataset("input", self.source_dictionary, impl='doc')
            src_tokens = PrependTokenDocDataset(input_data, self.source_dictionary.bos())
        else:
            input_data = make_dataset("input", self.source_dictionary)
            src_tokens = RightPadDataset(
                PrependTokenDataset(input_data, self.source_dictionary.bos()),
                pad_idx=self.source_dictionary.pad(),
            )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        def make_np_dataset(f_type, fork_name=None, keep_raw=False):
            if fork_name:
                path = os.path.join(self.cfg.data, 'feats', fork_name, f_type, split)
            else:
                path = os.path.join(self.cfg.data, 'feats', f_type, split)

            return FeatureDataset(path, keep_raw=keep_raw)

        given_types = self.cfg.feats_type
        if len(given_types) > 0:
            append = {}
            if str(self.cfg.classify_task) == 'seg':
                for feature_type in ['mairesse', 'senticnet', 'emotion']:
                    if feature_type in given_types:
                        append[feature_type] = make_np_dataset(feature_type)
            else:
                for feature_type in ['doc_mairesse', 'doc_senticnet', 'doc_emotion', 'doc_senticnet_dis']:
                    if feature_type in given_types:
                        append[feature_type] = make_np_dataset(feature_type[4:], fork_name='doc')
                for feature_type in ['seg_mairesse', 'seg_senticnet', 'seg_emotion']:
                    if feature_type in given_types:
                        append[feature_type] = make_np_dataset(feature_type[4:], fork_name='seg', keep_raw=True)

            for k, v in append.items():
                dataset['net_input'].update(
                    {
                        k: v,
                     }
                )

        label_dataset = make_dataset("label", self.label_dictionary)
        assert label_dataset is not None
        dataset.update(
            target=OffsetTokensDataset(
                StripTokenDataset(
                    label_dataset,
                    id_to_strip=self.label_dictionary.eos(),
                ),
                offset=-self.label_dictionary.nspecial,
            )
        )

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if str(self.cfg.classify_task) == 'seg':
            with data_utils.numpy_seed(self.cfg.seed):
                shuffle = np.random.permutation(len(src_tokens))

            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )

            logger.info("shuffle in seg dataset")

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg, from_checkpoint=False):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self, from_checkpoint)
        if str(self.cfg.classify_task) == 'doc':
            model.register_classification_head(
                'document_classification_head',
                num_classes=self.cfg.num_classes,
            )

        model.register_classification_head(
            'segment_classification_head',
            num_classes=self.cfg.num_classes,
        )
        return model

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
