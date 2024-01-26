# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as _matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import numpy as np
from fairseq.logging.meters import safe_round
from itertools import chain
from omegaconf import II
import logging
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, labels=[0,1], average="macro", zero_division=1)

    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def matthews_corrcoef(preds, labels):
    # make it consistent with other metrics taking (preds, labels) as input
    mcc = _matthews_corrcoef(labels, preds)
    return mcc

from fairseq.dataclass import ChoiceEnum

@dataclass
class PersonalityDetectionConfig(FairseqDataclass):
    report_mcc: bool = False
    report_acc_and_f1: bool = True
    report_pearson_and_spearman: bool = False
    personality: str = II("task.personality")
    facol_func: bool = False


@register_criterion("personality_detection", dataclass=PersonalityDetectionConfig)
class PersonalityDetectionCriterion(FairseqCriterion):

    def __init__(self, cfg: PersonalityDetectionConfig, task):
        super().__init__(task)
        self.class_name = task.cfg.class_name
        self.keep_pred_and_targ = (
            cfg.report_mcc or cfg.report_acc_and_f1 or cfg.report_pearson_and_spearman
        )
        self.report_mcc = cfg.report_mcc
        self.report_acc_and_f1 = cfg.report_acc_and_f1
        self.report_pearson_and_spearman = cfg.report_pearson_and_spearman
        self.personality = str(cfg.personality)
        self.label_dict = task.label_dictionary
        self.index = {
                    'EI': 0,
                    'SN': 1,
                    'TF': 2,
                    'JP': 3,
                }
        self.kaggle_rate = {
                    'EI': [1, 3.3],
                    'SN': [1, 6.2],
                    'TF': [1, 1.1],
                    'JP': [1, 1.5],
                }
        self.pandora_rate = {
                    'EI': [1, 3.7],
                    'SN': [1, 7.7],
                    'TF': [1.8, 1],
                    'JP': [1, 1.4],
                    # 'EI': [1, 4],
                    # 'SN': [1, 7],
                    # 'TF': [2, 1],
                    # 'JP': [2, 3],
                }


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # logits: batch_size * num_classes
        # extra: batch_size * slice_size * num_classes
        logits, _ = model(
            **sample["net_input"],
            features_only=True,
        )
        targets = model.get_targets(sample, [logits])
        class_dict = self.index
        targets = targets[..., class_dict[self.class_name]].view(-1)
        loss = F.cross_entropy(input=logits, target=targets, weight=torch.tensor(self.pandora_rate[self.class_name]).cuda().half(), reduction="sum")
        sample_size = targets.numel()

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        preds = logits.argmax(dim=-1)  # bsz, num_seg, 1
        logging_output["ncorrect"] = (preds == targets).sum()

        if self.keep_pred_and_targ and not model.training:
            logging_output["pred"] = list(map(int, preds))
            logging_output["targ"] = list(map(int, targets))
            if self.report_mcc:
                logging_output["report_mcc"] = True
            if self.report_acc_and_f1:
                logging_output["report_acc_and_f1"] = True
            if self.report_pearson_and_spearman:
                logging_output["report_pearson_and_spearman"] = True

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

        pred = np.array(
            list(chain.from_iterable(log.get("pred", []) for log in logging_outputs))
        )
        targ = np.array(
            list(chain.from_iterable(log.get("targ", []) for log in logging_outputs))
        )
        if len(pred):
            metrics.log_concat_tensor("pred", torch.from_numpy(pred), dim=0)
            metrics.log_concat_tensor("targ", torch.from_numpy(targ), dim=0)
            if any("report_mcc" in log for log in logging_outputs):
                metrics.log_derived(
                    "mcc",
                    lambda meters: safe_round(
                        matthews_corrcoef(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )
                        * 100,
                        1,
                    ),
                )
            if any("report_acc_and_f1" in log for log in logging_outputs):
                metrics.log_derived(
                    "acc_and_f1",
                    lambda meters: safe_round(
                        acc_and_f1(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["acc_and_f1"]
                        * 100,
                        2,
                    ),
                )
                metrics.log_derived(
                    "f1",
                    lambda meters: safe_round(
                        acc_and_f1(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["f1"]
                        * 100,
                        2,
                    ),
                )

            if any("report_pearson_and_spearman" in log for log in logging_outputs):
                metrics.log_derived(
                    "pearson_and_spearman",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["corr"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "pearson",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["pearson"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "spearman",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["spearmanr"]
                        * 100,
                        1,
                    ),
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# import math
#
# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass, field
# from fairseq.dataclass import FairseqDataclass
# from fairseq import metrics, utils
# from fairseq.criterions import FairseqCriterion, register_criterion
# from sklearn.metrics import f1_score
# from sklearn.metrics import matthews_corrcoef as _matthews_corrcoef
# from scipy.stats import pearsonr, spearmanr
# import numpy as np
# from fairseq.logging.meters import safe_round
# from itertools import chain
# from omegaconf import II
# import logging
# logger = logging.getLogger(__name__)
#
#
# def simple_accuracy(preds, labels):
#     return (preds == labels).mean()
#
#
# def acc_and_f1(preds, labels):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds, labels=[0,1], average="macro", zero_division=1)
#
#     return {
#         "acc": acc,
#         "f1": f1,
#         "acc_and_f1": (acc + f1) / 2,
#     }
#
#
# def pearson_and_spearman(preds, labels):
#     pearson_corr = pearsonr(preds, labels)[0]
#     spearman_corr = spearmanr(preds, labels)[0]
#     return {
#         "pearson": pearson_corr,
#         "spearmanr": spearman_corr,
#         "corr": (pearson_corr + spearman_corr) / 2,
#     }
#
#
# def matthews_corrcoef(preds, labels):
#     # make it consistent with other metrics taking (preds, labels) as input
#     mcc = _matthews_corrcoef(labels, preds)
#     return mcc
#
# from fairseq.dataclass import ChoiceEnum
#
# @dataclass
# class PersonalityDetectionConfig(FairseqDataclass):
#     report_mcc: bool = False
#     report_acc_and_f1: bool = True
#     report_pearson_and_spearman: bool = False
#     personality: str = II("task.personality")
#     facol_func: bool = False
#
#
# @register_criterion("personality_detection", dataclass=PersonalityDetectionConfig)
# class PersonalityDetectionCriterion(FairseqCriterion):
#
#     def __init__(self, cfg: PersonalityDetectionConfig, task):
#         super().__init__(task)
#         self.class_name = task.cfg.class_name
#         self.vote = task.cfg.vote
#         self.keep_pred_and_targ = (
#             cfg.report_mcc or cfg.report_acc_and_f1 or cfg.report_pearson_and_spearman
#         )
#         self.report_mcc = cfg.report_mcc
#         self.report_acc_and_f1 = cfg.report_acc_and_f1
#         self.report_pearson_and_spearman = cfg.report_pearson_and_spearman
#         self.personality = str(cfg.personality)
#         self.label_dict = task.label_dictionary
#         self.facol_func = cfg.facol_func
#         self.big5 = {
#                     'ext': 0,
#                     'neu': 1,
#                     'agr': 2,
#                     'con': 3,
#                     'opn': 4,
#                 }
#         self.mbti = {
#                     'EI': 0,
#                     'SN': 1,
#                     'TF': 2,
#                     'JP': 3,
#                 }
#         self.kaggle_rate = {
#                     'EI': [1, 3.3],
#                     'SN': [1, 6.2],
#                     'TF': [1, 1.1],
#                     'JP': [1, 1.5],
#                 }
#         self.pandora_rate = {
#                     'EI': [1, 3.7],
#                     'SN': [1, 7.6],
#                     'TF': [1.8, 1],
#                     'JP': [1, 1.4],
#                 }
#
#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.
#
#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         # logits: batch_size * num_classes
#         # extra: batch_size * slice_size * num_classes
#         logits, _ = model(
#             **sample["net_input"],
#             features_only=True,
#         )
#         ensemble = False
#         if isinstance(logits, list):
#             ensemble = True
#             logits_for_pred = torch.stack(logits).transpose(0, 1).clone().detach()
#         targets = model.get_targets(sample, [logits])
#         if self.class_name == 'mbti-4':
#             criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([4.0, 7.0, 0.5, 1.5]).cuda())
#             loss = criterion(logits.float(), targets.float())
#             sample_size = targets.size()[0]
#         else:
#             if self.personality == 'big5':
#                 class_dict = self.big5
#             else:
#                 class_dict = self.mbti
#
#             targets = targets[..., class_dict[self.class_name]].view(-1)
#
#             if self.facol_func:
#                 ce_loss = F.cross_entropy(input=logits, target=targets, reduction='none')
#                 pt = torch.exp(-ce_loss)
#                 alpha = torch.tensor(0.75).cuda().half()
#                 gamma = 2
#                 ce_loss = alpha * (1-pt) ** gamma * ce_loss
#                 loss = ce_loss.sum()
#
#             else:
#                 loss = None
#                 if ensemble:
#                     for lgs in logits:
#                         ls = F.cross_entropy(input=lgs, target=targets,
#                                                    weight=torch.tensor(self.rate[self.class_name]).cuda().half(),
#                                                    reduction="sum")
#                         loss = ls if not loss else loss + ls
#                 else:
#                     loss = F.cross_entropy(input=logits, target=targets, weight=torch.tensor(self.rate[self.class_name]).cuda().half(), reduction="sum")
#                 # lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
#                 # loss = F.nll_loss(lprobs, targets, reduction="sum")
#
#             sample_size = targets.numel()
#
#             # if self.vote:
#             #     sub_logits_set = extra[0]
#             #     sigmas_dota = extra[1]
#             #
#             #     sub_loss = torch.tensor(0, dtype=loss.dtype, device=loss.device)
#             #     for i, sub_logits in enumerate(sub_logits_set):  # batch_size * [slice_size * num_classes]
#             #         sub_lprobs = F.log_softmax(sub_logits, dim=-1, dtype=torch.float32)
#             #         sub_targets = torch.stack([targets[i]] * sub_lprobs.shape[0], dim=0)
#             #         sub_loss = sub_loss + F.nll_loss(sub_lprobs, sub_targets, reduction="mean")
#             #
#             #     # loss = major_loss + sub_loss + sigmas_loss
#             #     # factor = torch.div(1.0, torch.mul(2.0, sigmas_dota))
#             #     # loss_part = torch.sum(torch.mul(factor, torch.stack([loss, sub_loss])))
#             #     # regular_part = torch.sum(torch.log(sigmas_dota))
#             #     # loss = loss_part + regular_part
#             #
#             #     # loss = 1.0 * major_loss + 0.1 * sub_loss
#             #     loss = torch.sum(torch.mul(sigmas_dota, torch.stack([loss, sub_loss])))
#
#         logging_output = {
#             "loss": loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample_size,
#             "sample_size": sample_size,
#         }
#         if ensemble:
#             logits = logits_for_pred
#         if self.class_name == 'mbti-4':
#             preds = logits.ge(0.5)
#             acc_matrix = (preds == targets)
#             assert isinstance(acc_matrix, torch.Tensor)
#             #
#             # 'EI': [1, 4],
#             # 'SN': [1, 7],
#             # 'TF': [2, 1],
#             # 'JP': [2, 3],
#
#             logging_output["sum_npreds"] = acc_matrix.numel()
#             logging_output["sum_ncorrect"] = int(torch.count_nonzero(acc_matrix))
#             logging_output["EI_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 0]))
#             logging_output["SN_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 1]))
#             logging_output["TF_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 2]))
#             logging_output["JP_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 3]))
#             # logging_output["ext_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 0]))
#             # logging_output["neu_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 1]))
#             # logging_output["agr_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 2]))
#             # logging_output["con_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 3]))
#             # logging_output["opn_ncorrect"] = int(torch.count_nonzero(acc_matrix[..., 4]))
#
#             ncorrect = sum(row.all().int().item() for row in acc_matrix)
#             logging_output["ncorrect"] = ncorrect
#
#         else:
#             preds = logits.argmax(dim=-1)  # bsz, num_seg, 1
#             if len(preds.shape) == 2:
#                 preds = torch.sum(preds, dim=1).ge(preds.shape[1]*0.5)
#             logging_output["ncorrect"] = (preds == targets).sum()
#
#             # if self.vote:
#             #     sub_logits_set = extra[0]
#             #     sigmas_dota, vote_weight = extra[1:]
#             #
#             #     sub_ncorrect = 0
#             #     sub_nsentences = 0
#             #     for i, sub_logits in enumerate(sub_logits_set):  # batch_size * [slice_size * num_classes]
#             #         sub_preds = sub_logits.argmax(dim=1)
#             #         # sub_ncorrect += (sub_preds == targets[i]).sum()
#             #         # sub_nsentences += sub_preds.shape[0]
#             #
#             #         # soft voting
#             #         # sub_logit = torch.sum(sub_logits.detach().clone(), dim=0)  # slice_size
#             #         # dom_logit = logits[i].detach().clone()
#             #         #
#             #         # sub_num = sub_logits.shape[0]
#             #         # vote_num = 1 + sub_num
#             #         #
#             #         # dom_weight = (1 + sub_num * vote_weight)/vote_num
#             #         # sub_weight = 1/vote_num
#             #         #
#             #         # vote_logit = sub_logit * sub_weight + dom_logit * dom_weight
#             #         #
#             #         # vote_lprob = F.log_softmax(vote_logit, dim=-1, dtype=torch.float32)
#             #         # loss += F.nll_loss(vote_lprob, targets[i], reduction="sum")
#             #         #
#             #         # preds.data[i] = vote_logit.argmax(dim=-1)
#             #
#             #         # hard voting
#             #         aux_pred_0 = torch.sum(sub_logits[:3, ], dim=0).argmax()
#             #         aux_pred_1 = torch.sum(sub_logits[3:, ], dim=0).argmax()
#             #
#             #         aux_pred_2 = torch.sum(sub_preds[:3]).ge(1.5)
#             #         aux_pred_3 = torch.sum(sub_preds[3:]).ge(1.5)
#             #
#             #         maj_pred = preds[i]
#             #
#             #         if aux_pred_0 == aux_pred_1 and aux_pred_1 == aux_pred_2 and aux_pred_2 == aux_pred_3 and aux_pred_3 != maj_pred:
#             #             preds.data[i] = aux_pred_0
#
#                     # ===========================
#                     # sub_logit = torch.sum((sub_logits == sub_logits.max(dim=-1, keepdim=True)[0])).int().detach().clone()
#                     # dom_logit = (logits[i] == logits[i].max(dim=-1, keepdim=True)[0]).int().detach().clone()
#                     #
#                     # sub_num = sub_logits.shape[0]
#                     # vote_num = 1 + sub_num
#                     #
#                     # dom_weight = (1 + sub_num * vote_weight)/vote_num
#                     # sub_weight = 1 / vote_num
#                     #
#                     # vote_logit = sub_logit * sub_weight + dom_logit * dom_weight
#                     # vote_pred = vote_logit.argmax(dim=0)
#                     #
#                     # loss += torch.abs(targets[i] - vote_pred)
#                     #
#                     # preds.data[i] = vote_pred
#
#                     # ==========================
#
#                     # verified
#
#                     # sub_logit = torch.sum((sub_logits == sub_logits.max(dim=-1, keepdim=True)[0]), dim=0).int().detach().clone()
#                     # dom_logit = (logits[i] == logits[i].max(dim=-1, keepdim=True)[0]).int().detach().clone()
#                     #
#                     # #  weight
#                     # # dom_weight = sub_logits.shape[0] * vote_weight + 1
#                     # dom_weight = 1 + vote_weight
#                     # vote_num = dom_weight + sub_logits.shape[0]
#                     #
#                     # #  vote_pred
#                     # vote_logit = (sub_logit + dom_logit * dom_weight) / vote_num
#                     # preds.data[i] = vote_logit.argmax(dim=-1)
#                     #
#                     # # loss = loss + vote_loss
#                     # vote_lprob = F.log_softmax(vote_logit, dim=-1, dtype=torch.float32)
#                     # loss += F.nll_loss(vote_lprob, targets[i], reduction="sum")
#
#
#                     # sub_logit = torch.sum((sub_logits == sub_logits.max(dim=-1, keepdim=True)[0])).int()
#                     # dom_logit = (logits[i] == logits[i].max(dim=-1, keepdim=True)[0]).int()
#                     #
#                     # dom_weight = sub_logits.shape[0] * vote_weight + 1
#                     #
#                     # vote_logit = (sub_logit + dom_logit*dom_weight) / (sub_logits.shape[0] + dom_weight)
#                     # vote_lprob = F.log_softmax(vote_logit, dim=-1, dtype=torch.float32)
#                     # loss += F.nll_loss(vote_lprob, targets[i], reduction="sum")
#                     #
#                     # preds[i] = vote_logit.argmax(dim=-1)
#
#                 # logging_output["sub_ncorrect"] = sub_ncorrect
#                 # logging_output["sub_nsentences"] = sub_nsentences
#                 # logging_output["vote_ncorrect"] = (preds == targets).sum()
#                 # logging_output["vote_weight"] = vote_weight.data
#                 # logging_output["sigmas_dota"] = sigmas_dota.data
#
#         # if self.keep_pred_and_targ and not model.training:
#         if self.keep_pred_and_targ and not model.training:
#             # remove offset `self.label_dict.nspecial` from OffsetTokensDataset
#             # preds = self.label_dict.string(preds + self.label_dict.nspecial).split()
#             # targets = self.label_dict.string(
#             #     targets + self.label_dict.nspecial
#             # ).split()
#             if self.class_name == 'mbti-4':
#                 logging_output["EI_pred"] = list(map(int, preds[:,0]))
#                 logging_output["EI_targ"] = list(map(int, targets[:,0]))
#                 logging_output["SN_pred"] = list(map(int, preds[:,1]))
#                 logging_output["SN_targ"] = list(map(int, targets[:,1]))
#                 logging_output["TF_pred"] = list(map(int, preds[:,2]))
#                 logging_output["TF_targ"] = list(map(int, targets[:,2]))
#                 logging_output["JP_pred"] = list(map(int, preds[:,3]))
#                 logging_output["JP_targ"] = list(map(int, targets[:,3]))
#             else:
#                 logging_output["pred"] = list(map(int, preds))
#                 logging_output["targ"] = list(map(int, targets))
#
#             if self.class_name == 'mbti-4':
#                 logging_output["multi_target"] = True
#             if self.report_mcc:
#                 logging_output["report_mcc"] = True
#             if self.report_acc_and_f1:
#                 logging_output["report_acc_and_f1"] = True
#             if self.report_pearson_and_spearman:
#                 logging_output["report_pearson_and_spearman"] = True
#
#         return loss, sample_size, logging_output
#
#     @staticmethod
#     def reduce_metrics(logging_outputs) -> None:
#         """Aggregate logging outputs from data parallel training."""
#         loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
#         ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
#         nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
#
#         metrics.log_scalar(
#             "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
#         )
#
#         if sample_size != ntokens:
#             metrics.log_scalar(
#                 "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
#             )
#
#         if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
#             ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
#             )
#
#         if len(logging_outputs) > 0 and "sub_ncorrect" in logging_outputs[0]:
#             sub_ncorrect = sum(log.get("sub_ncorrect", 0) for log in logging_outputs)
#             sub_nsentences = sum(log.get("sub_nsentences", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 "sub_accuracy", 100.0 * sub_ncorrect / sub_nsentences, sub_nsentences, round=1
#             )
#
#         if len(logging_outputs) > 0 and "vote_ncorrect" in logging_outputs[0]:
#             vote_ncorrect = sum(log.get("vote_ncorrect", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 "vote_accuracy", 100.0 * vote_ncorrect / nsentences, nsentences, round=1
#             )
#             # vote_num = sum(log.get("vote_num", 0) for log in logging_outputs)
#             # vote_correct = sum(log.get("vote_correct", 0) for log in logging_outputs)
#             # metrics.log_scalar(
#             #     "sentence_accuracy", 100.0 * vote_correct / vote_num, vote_num, round=1
#             # )
#
#         if len(logging_outputs) > 0 and "vote_weight" in logging_outputs[0]:
#             times = len(logging_outputs)
#             vote_weight = sum(log.get("vote_weight", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 "vote_weight", vote_weight / times, times, round=3
#             )
#             sigmas_dota = sum(log.get("sigmas_dota", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 "sigmas_dota", sigmas_dota / times, times, round=3
#             )
#
#         # special for muti-label pred
#         if len(logging_outputs) > 0 and "sum_ncorrect" in logging_outputs[0]:
#             sum_ncorrect = sum(log.get("sum_ncorrect", 0) for log in logging_outputs)
#             sum_npreds = sum(log.get("sum_npreds", 0) for log in logging_outputs)
#             EI_ncorrect = sum(log.get("EI_ncorrect", 0) for log in logging_outputs)
#             SN_ncorrect = sum(log.get("SN_ncorrect", 0) for log in logging_outputs)
#             TF_ncorrect = sum(log.get("TF_ncorrect", 0) for log in logging_outputs)
#             JP_ncorrect = sum(log.get("JP_ncorrect", 0) for log in logging_outputs)
#             # ext_ncorrect = sum(log.get("ext_ncorrect", 0) for log in logging_outputs)
#             # neu_ncorrect = sum(log.get("neu_ncorrect", 0) for log in logging_outputs)
#             # agr_ncorrect = sum(log.get("agr_ncorrect", 0) for log in logging_outputs)
#             # con_ncorrect = sum(log.get("con_ncorrect", 0) for log in logging_outputs)
#             # opn_ncorrect = sum(log.get("opn_ncorrect", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 "ave-accuracy", 100.0 * sum_ncorrect / sum_npreds, sum_npreds, round=1
#             )
#             # ext, neu, agr, con, opn
#             single_npreds = int(sum_npreds/4)
#             # metrics.log_scalar(
#             #     "ext-accuracy", 100.0 * ext_ncorrect / single_npreds, single_npreds, round=1
#             # )
#             # metrics.log_scalar(
#             #     "neu-accuracy", 100.0 * neu_ncorrect / single_npreds, single_npreds, round=1
#             # )
#             # metrics.log_scalar(
#             #     "agr-accuracy", 100.0 * agr_ncorrect / single_npreds, single_npreds, round=1
#             # )
#             # metrics.log_scalar(
#             #     "con-accuracy", 100.0 * con_ncorrect / single_npreds, single_npreds, round=1
#             # )
#             # metrics.log_scalar(
#             #     "opn-accuracy", 100.0 * opn_ncorrect / single_npreds, single_npreds, round=1
#             # )
#             metrics.log_scalar(
#                 "ei-accuracy", 100.0 * EI_ncorrect / single_npreds, single_npreds, round=1
#             )
#             metrics.log_scalar(
#                 "sn-accuracy", 100.0 * SN_ncorrect / single_npreds, single_npreds, round=1
#             )
#             metrics.log_scalar(
#                 "tf-accuracy", 100.0 * TF_ncorrect / single_npreds, single_npreds, round=1
#             )
#             metrics.log_scalar(
#                 "jp-accuracy", 100.0 * JP_ncorrect / single_npreds, single_npreds, round=1
#             )
#         if any("multi_target" in log for log in logging_outputs):
#             EI_pred = np.array(
#                 list(chain.from_iterable(log.get("EI_pred", []) for log in logging_outputs))
#             )
#             EI_targ = np.array(
#                 list(chain.from_iterable(log.get("EI_targ", []) for log in logging_outputs))
#             )
#             metrics.log_concat_tensor("EI_pred", torch.from_numpy(EI_pred), dim=0)
#             metrics.log_concat_tensor("EI_targ", torch.from_numpy(EI_targ), dim=0)
#
#             SN_pred = np.array(
#                 list(chain.from_iterable(log.get("SN_pred", []) for log in logging_outputs))
#             )
#             SN_targ = np.array(
#                 list(chain.from_iterable(log.get("SN_targ", []) for log in logging_outputs))
#             )
#             metrics.log_concat_tensor("SN_pred", torch.from_numpy(SN_pred), dim=0)
#             metrics.log_concat_tensor("SN_targ", torch.from_numpy(SN_targ), dim=0)
#
#             TF_pred = np.array(
#                 list(chain.from_iterable(log.get("TF_pred", []) for log in logging_outputs))
#             )
#             TF_targ = np.array(
#                 list(chain.from_iterable(log.get("TF_targ", []) for log in logging_outputs))
#             )
#             metrics.log_concat_tensor("TF_pred", torch.from_numpy(TF_pred), dim=0)
#             metrics.log_concat_tensor("TF_targ", torch.from_numpy(TF_targ), dim=0)
#
#             JP_pred = np.array(
#                 list(chain.from_iterable(log.get("JP_pred", []) for log in logging_outputs))
#             )
#             JP_targ = np.array(
#                 list(chain.from_iterable(log.get("JP_targ", []) for log in logging_outputs))
#             )
#             metrics.log_concat_tensor("JP_pred", torch.from_numpy(JP_pred), dim=0)
#             metrics.log_concat_tensor("JP_targ", torch.from_numpy(JP_targ), dim=0)
#
#             metrics.log_derived(
#                 "EI_f1",
#                 lambda meters: safe_round(
#                     acc_and_f1(
#                         meters["EI_pred"].tensor.numpy(),
#                         meters["EI_targ"].tensor.numpy(),
#                     )["f1"]
#                     * 100,
#                     1,
#                 ),
#             )
#             metrics.log_derived(
#                 "SN_f1",
#                 lambda meters: safe_round(
#                     acc_and_f1(
#                         meters["SN_pred"].tensor.numpy(),
#                         meters["SN_targ"].tensor.numpy(),
#                     )["f1"]
#                     * 100,
#                     1,
#                 ),
#             )
#             metrics.log_derived(
#                 "TF_f1",
#                 lambda meters: safe_round(
#                     acc_and_f1(
#                         meters["TF_pred"].tensor.numpy(),
#                         meters["TF_targ"].tensor.numpy(),
#                     )["f1"]
#                     * 100,
#                     1,
#                 ),
#             )
#             metrics.log_derived(
#                 "JP_f1",
#                 lambda meters: safe_round(
#                     acc_and_f1(
#                         meters["JP_pred"].tensor.numpy(),
#                         meters["JP_targ"].tensor.numpy(),
#                     )["f1"]
#                     * 100,
#                     1,
#                 ),
#             )
#         pred = np.array(
#             list(chain.from_iterable(log.get("pred", []) for log in logging_outputs))
#         )
#         targ = np.array(
#             list(chain.from_iterable(log.get("targ", []) for log in logging_outputs))
#         )
#         if len(pred):
#             metrics.log_concat_tensor("pred", torch.from_numpy(pred), dim=0)
#             metrics.log_concat_tensor("targ", torch.from_numpy(targ), dim=0)
#             if any("report_mcc" in log for log in logging_outputs):
#                 metrics.log_derived(
#                     "mcc",
#                     lambda meters: safe_round(
#                         matthews_corrcoef(
#                             meters["pred"].tensor.numpy(),
#                             meters["targ"].tensor.numpy(),
#                         )
#                         * 100,
#                         1,
#                     ),
#                 )
#             if any("report_acc_and_f1" in log for log in logging_outputs):
#                 # times = len(logging_outputs)
#                 # metrics.log_scalar(
#                 #     "pred_sum", sum(pred), times, round=3
#                 # )
#                 # metrics.log_scalar(
#                 #     "targ_sum", sum(targ), times, round=3
#                 # )
#                 metrics.log_derived(
#                     "acc_and_f1",
#                     lambda meters: safe_round(
#                         acc_and_f1(
#                             meters["pred"].tensor.numpy(),
#                             meters["targ"].tensor.numpy(),
#                         )["acc_and_f1"]
#                         * 100,
#                         1,
#                     ),
#                 )
#                 metrics.log_derived(
#                     "f1",
#                     lambda meters: safe_round(
#                         acc_and_f1(
#                             meters["pred"].tensor.numpy(),
#                             meters["targ"].tensor.numpy(),
#                         )["f1"]
#                         * 100,
#                         1,
#                     ),
#                 )
#
#             if any("report_pearson_and_spearman" in log for log in logging_outputs):
#                 metrics.log_derived(
#                     "pearson_and_spearman",
#                     lambda meters: safe_round(
#                         pearson_and_spearman(
#                             meters["pred"].tensor.numpy(),
#                             meters["targ"].tensor.numpy(),
#                         )["corr"]
#                         * 100,
#                         1,
#                     ),
#                 )
#                 metrics.log_derived(
#                     "pearson",
#                     lambda meters: safe_round(
#                         pearson_and_spearman(
#                             meters["pred"].tensor.numpy(),
#                             meters["targ"].tensor.numpy(),
#                         )["pearson"]
#                         * 100,
#                         1,
#                     ),
#                 )
#                 metrics.log_derived(
#                     "spearman",
#                     lambda meters: safe_round(
#                         pearson_and_spearman(
#                             meters["pred"].tensor.numpy(),
#                             meters["targ"].tensor.numpy(),
#                         )["spearmanr"]
#                         * 100,
#                         1,
#                     ),
#                 )
#
#     @staticmethod
#     def logging_outputs_can_be_summed() -> bool:
#         """
#         Whether the logging outputs returned by `forward` can be summed
#         across workers prior to calling `reduce_metrics`. Setting this
#         to True will improve distributed training speed.
#         """
#         return True
