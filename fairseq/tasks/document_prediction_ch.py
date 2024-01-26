# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq.data import (
    TokenizerDictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    RawDataDataset,
    StripTokenDataset,
    data_utils,
    FeatureDataset,
    Dictionary,
)
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("document_prediction_ch")
class DocumentPredictionCHTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--add-prev-output-tokens",
            action="store_true",
            default=False,
            help="add prev_output_tokens to sample, used for encoder-decoder arch",
        )
        parser.add_argument(
            "--pretrained-model",
            default='/tangqirui/CPT/fairseq/checkpoints/Macbert/pytorch_model.bin',
        )
        parser.add_argument(
            "--class-name",
            default='multi',
        )

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    def class_name(self):
        return self.args.class_name

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if source:
            dictionary = TokenizerDictionary.load(filename)
        else:
            dictionary = Dictionary.load(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "vocab"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "label", "dict.txt"),
            source=False,
        )
        logger.info("[label] dictionary: {} types".format(len(label_dict)))

        return cls(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary, impl='mmap'):
            split_path = get_path(type, split)
            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                impl,
                combine=combine,
            )
            return dataset

        def get_data_path(type, split):
            return os.path.join(self.args.data, 'feat', type, split)

        def make_doc_feats_dataset():
            path = get_data_path('document', split)
            doc_feats_dataset = FeatureDataset(path)
            return doc_feats_dataset

        def make_sen_feats_dataset():
            path = get_data_path('sentence', split)
            sen_feats_dataset = FeatureDataset(path, keep_raw=True)
            return sen_feats_dataset

        input = make_dataset("input", self.source_dictionary, impl='doc')
        assert input is not None, "could not find dataset: {}".format(
            get_path(type, split)
        )

        assert input is not None, "could not find dataset: {}".format(
            get_path(type, split)
        )

        src_tokens = input

        doc_feats_input = make_doc_feats_dataset()
        sen_feats_input = make_sen_feats_dataset()

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RawDataDataset(src_tokens),
                "src_features": sen_feats_input,
                "features": doc_feats_input,
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        label_dataset = make_dataset("label", self.label_dictionary)
        if label_dataset is not None:
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

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        # only for ch
        model.copy_pretrained_params(args)

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
