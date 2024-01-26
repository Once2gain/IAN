import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
import math
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules import TransformerDocumentEncoder, LayerNorm
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch.nn.parameter import Parameter


logger = logging.getLogger(__name__)


@register_model("model_IAN")
class PDModel(FairseqEncoderModel):
    def __init__(self, args, encoder, pd_net):
        super().__init__(encoder)
        self.args = args
        self.classification_heads = nn.ModuleDict()
        self.pd_net = pd_net
        self.append_doc_psycho = args.append_doc_psycho
        if self.append_doc_psycho:
            self.append_dim = 16
            self.doc_psycho_dense = nn.Linear(113, self.append_dim)
        self.num_segments = args.num_segments
        self.ens_by_seg = args.ens_by_seg
        self.mean_by_seg = args.mean_by_seg
        if not self.mean_by_seg:
            self.doc_x_dense = nn.Linear(768*self.num_segments, 768)
        # self.apply(init_bert_params)

        self.input_dim = 768
        self.inner_dim = 128

        for param in self.parameters():
            param.param_group = "soft"
        for param in self.encoder.parameters():
            param.param_group = "solid"

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)

        pd_net = IAN(psycho_dim=113, semantic_dim=768, num_stack=args.num_stack)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder, pd_net)

    def forward(
        self,
        src_tokens,  # bsz * num_segment * seq_length
        features_only=False,
        return_all_hiddens=False,
        **kwargs
    ):
        if self.append_doc_psycho:
            doc_psycho = torch.concatenate(
                [kwargs['doc_mairesse'], kwargs['doc_senticnet'], kwargs['doc_emotion']], dim=-1)
            doc_psycho = self.doc_psycho_dense(doc_psycho)
        else:
            doc_psycho = None

        segments_psycho = []
        for mairesse, senticnet, emotion in zip(kwargs['seg_mairesse'], kwargs['seg_senticnet'], kwargs['seg_emotion']):
            seg_psycho = torch.concatenate([mairesse, senticnet, emotion], dim=-1)
            if seg_psycho.shape[0] != self.num_segments:
                times = self.num_segments // seg_psycho.shape[0] + 1
                seg_psycho = torch.concatenate([seg_psycho] * times, dim=0)[:self.num_segments]
                assert seg_psycho.shape[0] == self.num_segments
            segments_psycho.append(seg_psycho)
        segments_psycho = torch.stack(segments_psycho)  # bsz, num_seg, psycho_dim

        segments_semantic = []
        for x in src_tokens:  # num_segment * seq_length
            x, _ = self.encoder(x, features_only=True, return_all_hiddens=False, **kwargs)
            x = x[:, 0, :]  # num_segment * 768
            if x.shape[0] != self.num_segments:
                times = self.num_segments // x.shape[0] + 1
                x = torch.concatenate([x] * times, dim=0)[:self.num_segments]
                assert x.shape[0] == self.num_segments
            segments_semantic.append(x)
        segments_semantic = torch.stack(segments_semantic)  # bsz, num_seg, semantic_dim

        segments_x = self.pd_net(psychos=segments_psycho, semantics=segments_semantic)  # bsz, num_seg, semantic_dim

        if self.ens_by_seg:
            xs = []
            for i in range(segments_x.shape[1]):
                x = self.classification_heads['document_classification_head'](segments_x[:,i,:], doc_psycho=doc_psycho)
                xs.append(x)  # bsz, 2
            # num_seg, (bsz, 2)
            x = xs

        else:
            if self.mean_by_seg:
                x = torch.mean(segments_x, dim=1)
            else:
                x = self.doc_x_dense(segments_x)
            x = self.classification_heads['document_classification_head'](x, doc_psycho=doc_psycho)
        return x, None

    def _get_adaptive_head_loss(self):
        norm_loss = 0
        scaling = float(self.args.mha_reg_scale_factor)
        for layer in self.encoder.sentence_encoder.layers:
            norm_loss_layer = 0
            for i in range(layer.self_attn.num_heads):
                start_idx = i * layer.self_attn.head_dim
                end_idx = (i + 1) * layer.self_attn.head_dim
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.q_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.q_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.k_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.k_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.v_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.v_proj.bias[start_idx:end_idx])
                    )
                )

            norm_loss += norm_loss_layer
        return norm_loss

    def _get_adaptive_ffn_loss(self):
        ffn_scale_factor = float(self.args.ffn_reg_scale_factor)
        filter_loss = 0
        for layer in self.encoder.sentence_encoder.layers:
            filter_loss += torch.sum(
                torch.abs(layer.fc1.weight * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.weight * ffn_scale_factor))
            filter_loss += torch.sum(
                torch.abs(layer.fc1.bias * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.bias * ffn_scale_factor))
        return filter_loss

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log.log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        else:
            if self.append_doc_psycho:
                append_dim = self.append_dim
            else:
                append_dim = 0
            self.classification_heads[name] = RobertaClassificationHead(
                input_dim=self.input_dim,
                inner_dim=self.inner_dim,
                append_dim=append_dim,
                num_classes=num_classes,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                q_noise=self.args.quant_noise_pq,
                qn_block_size=self.args.quant_noise_pq_block_size,
                do_spectral_norm=self.args.spectral_norm_classification_head,
            )
            for param in self.classification_heads[name].parameters():
                param.param_group = "soft"

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder"):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            # if head_name == 'document_classification_head':
            #     keys_to_delete.append(k)
            #     continue

            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

        for k in list(state_dict.keys()):
            if '.lm_head.' in k:
                del state_dict[k]

        cur_state = self.state_dict()
        for k, v in cur_state.items():
            if k.startswith(prefix + "doc_x_dense"):
                state_dict[prefix + k] = v
                continue
            if k.startswith(prefix + "doc_psycho_dense"):
                state_dict[prefix + k] = v
                continue
            if k.startswith(prefix + "pd_net"):
                state_dict[prefix + k] = v
                continue


class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        def freeze_module_params(module, prefix):
            if module is not None:
                for name, param in module.named_parameters():
                    logger.info("freeze RobertaEncoder.sentence_encoder." + prefix + name)
                    param.requires_grad = False

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(self.sentence_encoder.layers[layer], prefix='layers.' + str(layer) + '.')

        if args.freeze_embeddings:
            freeze_module_params(self.sentence_encoder.embed_tokens, prefix='embed_tokens.')
            freeze_module_params(self.sentence_encoder.embed_positions, prefix='embed_positions.')
            freeze_module_params(self.sentence_encoder.layernorm_embedding, prefix='layernorm_embedding.')

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class SA(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.self_attn_layer_norm = LayerNorm(embed_dim)

        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.dropout_module = FairseqDropout(
            0.1, module_name=self.__class__.__name__
        )

    def forward(self, query, layer_norm=False):
        bsz = query.size()[0]  # bsz, num_seg, embed_dim
        residual = query

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.transpose(0, 1)  # num_seg, bsz, embed_dim
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q = (
            q.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)  # bsz * self.num_heads, num_seg, head_dim
        )
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)  # bsz, num_seg, num_seg
        attn = torch.bmm(attn_probs, v)  # bsz * self.num_heads, num_seg, head_dim

        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, self.embed_dim)  # num_seg, bsz, embed_dim

        x = self.out_proj(attn)
        x = x.transpose(0, 1)  # bsz, num_seg, embed_dim

        if layer_norm:
            x = self.self_attn_layer_norm(x)

        x = residual + x

        return x, attn_probs


class PsychoEncoder(nn.Module):
    def __init__(self, psycho_dim):
        super().__init__()
        self.sa = SA(embed_dim=psycho_dim, num_heads=1)

    def forward(self, psychos):
        return self.sa.forward(psychos, True)


class SemanticEncoder(nn.Module):
    def __init__(self, semantic_dim, post_sa=False):
        super().__init__()
        self.embed_dim = semantic_dim
        self.v_proj = nn.Linear(semantic_dim, semantic_dim, bias=True)
        self.out_proj = nn.Linear(semantic_dim, semantic_dim, bias=True)
        self.self_attn_layer_norm = LayerNorm(semantic_dim)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        self.post_sa = post_sa
        if post_sa:
            self.sa = SA(embed_dim=semantic_dim, num_heads=12)


    def forward(self, semantics, attn_probs):
        residual = semantics

        v = self.v_proj(semantics)
        attn = torch.matmul(attn_probs, v)  # bsz * self.num_heads, 113, 10
        x = self.out_proj(attn)

        x = x + residual
        x = self.self_attn_layer_norm(x)

        if self.post_sa:
            x ,_ = self.sa.forward(x, layer_norm=False)

        return x


class IndexAttnLayer(nn.Module):
    def __init__(self, psycho_dim, semantic_dim):
        super().__init__()
        self.pe = PsychoEncoder(psycho_dim)
        self.se = SemanticEncoder(semantic_dim)

    def forward(self, psychos, semantics):
        psychos, attn_probs = self.pe(psychos=psychos)
        semantics = self.se(semantics, attn_probs)
        return psychos, semantics


class IAN(nn.Module):
    def __init__(self, psycho_dim, semantic_dim, num_stack):
        super().__init__()
        net_stack = []
        out_net_stack = []
        # for _ in range(num_stack):
        #     net_stack.append(IndexAttnLayer(psycho_dim, semantic_dim))
        #     out_net_stack.append(SA(embed_dim=semantic_dim, num_heads=12))
        # self.net = nn.ModuleList(
        #     net_stack
        # )
        # self.out_net = nn.ModuleList(
        #     out_net_stack
        # )
        self.net = nn.ModuleList(
            [
                IndexAttnLayer(psycho_dim, semantic_dim),
                IndexAttnLayer(psycho_dim, semantic_dim),
                IndexAttnLayer(psycho_dim, semantic_dim),
                IndexAttnLayer(psycho_dim, semantic_dim),
            ]
        )
        self.out_net = nn.ModuleList(
            [
                SA(embed_dim=semantic_dim, num_heads=12),
                SA(embed_dim=semantic_dim, num_heads=12),
                # SA(embed_dim=semantic_dim, num_heads=12),
                # SA(embed_dim=semantic_dim, num_heads=12),
            ]
        )

    def forward(self, psychos, semantics):
        for ia in self.net:
            psychos, semantics = ia(psychos, semantics)
        x = semantics
        for sa in self.out_net:
            x, _ = sa(x)
        return x  # bsz, num_seg, dim


class RobertaClassificationHead(nn.Module):
    def __init__(
        self,
        num_classes,
        activation_fn,
        pooler_dropout,
        input_dim,
        inner_dim,
        append_dim,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.dense = nn.Linear(input_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim + append_dim, num_classes)
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, doc_reps, doc_psycho=None):
        x = self.dense(doc_reps)  #  bsz, 4, D
        if doc_psycho is not None:
            x = torch.concatenate((x, doc_psycho), dim=-1)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("model_IAN", "model_IAN")
def base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)
    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )

    # trick for finetune
    args.random_initial_layers = safe_getattr(
        args, "random_initial_layers", 0
    )
    args.freeze_encoder = safe_getattr(
        args, "freeze_encoder", False
    )
    args.n_trans_layers_to_freeze = safe_getattr(
        args, "n_trans_layers_to_freeze", 0
    )
    args.freeze_doc_encoder = safe_getattr(
        args, "freeze_doc_encoder", False
    )
    args.trait_decoder = safe_getattr(
        args, "trait_decoder", "cnn"
    )
    args.semantic_only = safe_getattr(
        args, "semantic_only", False
    )
    args.num_segments = safe_getattr(
        args, "num_segments", 10
    )
    args.num_segments = safe_getattr(
        args, "num_stack", 2
    )


#
# @register_model_architecture("seg2doc", "seg2doc_large")
# def seg_architecture(args):
#     args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
#     args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
#     args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
#     args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
#     base_architecture(args)
