# import torch
# import re
# import numpy as np
# from fairseq.data import (
#     Dictionary,
# )
# from fairseq.data.encoders.gpt2_bpe import get_encoder
# import torch
#
# tokenizer = get_encoder("gpt2_bpe/encoder.json", "gpt2_bpe/vocab.bpe")
#
# from fairseq import hub_utils
#
# x = hub_utils.from_pretrained(
#     '/data/tangqirui/fairseq/outputs/2023-08-15/14-34-54/checkpoints',
#     'checkpoint_best.pt',
#     '/data/tangqirui/fairseq/DATA-bin/kaggle-mbti/mbti-3-5/fold-1/doc_dataset',
#     archive_map={},
#     bpe="gpt2",
#     load_checkpoint_heads=True,
# )
# model = x["models"][0]
#
# folder = '/data/tangqirui/fairseq/DATA-bin/kaggle-mbti/mbti-3-5/fold-1/doc_dataset'
# data = folder + '/input'
#
# dictionary = Dictionary.load(data + '/dict.txt')
# dictionary.add_symbol("<mask>")
#
# input_f = folder + '/train.input'
# f = open(input_f, 'r', encoding='utf-8')
# input_l = f.readlines()
# src_tokens = []
# for line in input_l[100:104]:
#     for seg in line.split('-100'):
#         seg = seg.strip().split()
#         seg = list(map(int, seg))
#         print(tokenizer.decode(seg))
#
#     item = dictionary.encode_doc(line.strip())
#     prepend = item.new(item.size(0), 1).fill_(dictionary.bos())
#     item = torch.cat([prepend, item], dim=1)
#     src_tokens.append(item)
#
# input_f = folder + '/train.label'
# f = open(input_f, 'r', encoding='utf-8')
# input_l = f.readlines()
# targets = []
# for line in input_l[100:104]:
#     targets.append(line.strip())
#
#
# doc_mairesse_f = folder + '/feats/doc/mairesse/train'
# doc_senticnet_f = folder + '/feats/doc/senticnet/train'
# doc_emotion_f = folder + '/feats/doc/emotion/train'
# doc_senticnet_dis_f = folder + '/feats/doc/senticnet_dis/train'
# seg_mairesse_f = folder + '/feats/seg/mairesse/train'
# seg_senticnet_f = folder + '/feats/seg/senticnet/train'
# seg_emotion_f = folder + '/feats/seg/emotion/train'
#
#
# def read_8(vec_path, keep_raw=False):
#     if keep_raw:
#         samples = np.load(vec_path + '.npy', allow_pickle=True)
#     else:
#         samples = np.load(vec_path + '.npy', allow_pickle=True).astype(np.float32)
#     samples = samples[100:104]
#     if keep_raw:
#         new_samples = []
#         for item in samples:
#             new_item = []
#             for array in item:
#                 new_item.append(torch.tensor(array.astype(np.float32), dtype=torch.float32))
#             new_item = torch.stack(new_item)
#             new_samples.append(new_item)
#         return new_samples
#     return torch.tensor(np.array(samples), dtype=torch.float32)
#
#
# doc_mairesse = read_8(doc_mairesse_f)
# doc_senticnet = read_8(doc_senticnet_f)
# doc_emotion = read_8(doc_emotion_f)
# doc_senticnet_dis = read_8(doc_senticnet_dis_f)
# seg_mairesse = read_8(seg_mairesse_f, keep_raw=True)
# seg_senticnet = read_8(seg_senticnet_f, keep_raw=True)
# seg_emotion = read_8(seg_emotion_f, keep_raw=True)
#
#
# model.eval()
# with torch.no_grad():
#     logits, extra = model(
#         src_tokens,  # batch_size_1(documents_num) * batch_size_2(sentence_num) * seq_length
#         doc_mairesse=doc_mairesse,
#         doc_senticnet=doc_senticnet,
#         doc_emotion=doc_emotion,
#         doc_senticnet_dis=doc_senticnet_dis,
#         seg_mairesse=seg_mairesse,
#         seg_senticnet=seg_senticnet,
#         seg_emotion=seg_emotion,
#     )
# preds = logits.argmax(dim=1)
# print(preds)
#
#
# tmp = []
# for target in targets:
#     tmp.append(target.split()[0])
# targets = torch.tensor(list(map(int, tmp)))
# print(targets)
# print((preds == targets).sum())

# import os
# import pickle
#
# feats_filename = os.path.join("prepare/tools/pandora_baseline/data/feats.pickle")
# vocab_filename = os.path.join("prepare/tools/pandora_baseline/data/vocab.pickle")
# unames_filename = os.path.join("prepare/tools/pandora_baseline/data/unames.pickle")
# gram_feats = pickle.load(open(feats_filename, "rb"))
# gram_feat_names = pickle.load(open(vocab_filename, "rb"))
# unames = pickle.load(open(unames_filename, "rb"))
# print(list(gram_feat_names))

import numpy as np
a = np.zeros([16, 16, 4])
print(a.shape)
print(a[..., 0].shape)

JP_pred = np.array(
    list(chain.from_iterable(log.get("JP_pred", []) for log in logging_outputs))
)
JP_targ = np.array(
    list(chain.from_iterable(log.get("JP_targ", []) for log in logging_outputs))
)
metrics.log_concat_tensor("JP_pred", torch.from_numpy(JP_pred), dim=0)
metrics.log_concat_tensor("JP_targ", torch.from_numpy(JP_targ), dim=0)