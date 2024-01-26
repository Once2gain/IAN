# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .dictionary import Dictionary
from transformers import BertTokenizer
from fairseq.tokenizer import tokenize_doc


class TokenizerDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
            self,
            tokenizer,
    ):
        self.tokenizer = tokenizer

        self.count = []

        self.symbols = list(self.tokenizer.vocab.keys())
        self.indices = list(self.tokenizer.vocab.values())

        self.bos_index = tokenizer.cls_token_id
        self.pad_index = tokenizer.pad_token_id
        self.eos_index = tokenizer.sep_token_id
        self.unk_index = tokenizer.unk_token_id

        self.bos_word, self.unk_word, self.pad_word, self.eos_word = \
            self.tokenizer.cls_token, self.tokenizer.unk_token, self.tokenizer.pad_token, self.tokenizer.sep_token

        self.nspecial = 4

        # since we are leverating the bert's vocab, we cannot modify the symbols anymore.

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
            self,
            tensor,
            bpe_symbol=None,
            escape_unk=False,
            extra_symbols_to_ignore=None,
            unk_string=None,
            include_eos=False,  # is not used
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
    ):
        assert unk_string is None, "unk_string is not supported"
        assert bpe_symbol is None, "bpe_symbols is not supported"
        assert extra_symbols_to_ignore is None, "extra_symbols_to_ignore is not supported"
        assert escape_unk is False, "escape_unk is not supported, use skip_special_tokens=True"
        assert include_eos is False, "include_eos is not used"

        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces)
                for t in tensor
            )
        return self.tokenizer.decode(tensor, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        raise NotImplementedError("add_symbols should not be supported")

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        raise NotImplementedError("update should not be supported")

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        raise NotImplementedError("finalize should not be supported")

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        raise NotImplementedError("pad_to_multiple should not be supported")

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, model_path):
        """Loads the dictionary from the pretrained models' directory"""
        tokenizer = BertTokenizer.from_pretrained(model_path)
        d = cls(tokenizer=tokenizer)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        raise NotImplementedError("add_from_file should not be supported")

    def _save(self, f, kv_iterator):
        raise NotImplementedError("_save should not be supported")

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""

        raise NotImplementedError("save should not be supported")

    def dummy_sentence(self, length):
        raise NotImplementedError("dummy_sentence is not be supported yet")

    def encode_doc(
        self,
        doc_text,
        doc_tokenizer=tokenize_doc,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        already_numberized=False,
    ):
        doc_words = doc_tokenizer(doc_text)

        words_per_sent = max(len(sent_words) for sent_words in doc_words)
        sents_per_doc = len(doc_words)
        ids = torch.full([sents_per_doc, words_per_sent + 1 if append_eos else words_per_sent], self.pad_index, dtype=torch.int)
        for i, sent_words in enumerate(doc_words):
            nwords = len(sent_words)
            if already_numberized:
                id_list = list(map(int, sent_words))
                id_list = torch.tensor(id_list, dtype=torch.int)
                assert ids[i][:nwords].numel() == id_list.numel()
                ids[i][:nwords].copy_(id_list)
            else:
                for j, word in enumerate(sent_words):
                    if add_if_not_exist:
                        idx = self.add_symbol(word)
                    else:
                        idx = self.index(word)
                    if consumer is not None:
                        consumer(word, idx)
                    ids[i][j] = idx
            if append_eos:
                ids[i][nwords] = self.eos_index
        return ids


    def encode_line(
        self,
        line,
        line_tokenizer=None,
        add_if_not_exist=False,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = self.tokenizer.encode(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            ids[i] = word
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
            filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        raise NotImplementedError("_add_file_to_dictionary_single_worker is not supported")

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        raise NotImplementedError("add_file_to_dictionary_single_worker is not supported")