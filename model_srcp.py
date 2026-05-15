# -*- coding: utf-8 -*-
"""
SilentReading Contrastive Pretraining (SRCP) Model

Multi-stream Transformer architecture for contrastive EEG-Text pretraining.
Combines masked autoencoding (MAE) for EEG signals, masked language modeling (MLM)
for text, and contrastive learning to align EEG and text representations.

Architecture:
  - EEG Stream: 6-layer Transformer encoder (8 heads, 2048 FFN)
  - Text Stream: Frozen BART-large encoder
  - Joint Stream: 1-layer Multi-Stream Transformer (16 heads, 4096 FFN)

Based on the methodology described in:
  Wang et al. (2024) "Enhancing EEG-to-Text Decoding through Transferable Representations
  from Pre-trained Contrastive EEG-Text Masked Autoencoder" (ACL 2024)
"""

import os
import random
import torch
import torch.nn as nn
import math
from transformers import BartModel, BartTokenizer, XLMRobertaTokenizer, XLMRobertaModel, T5Model, T5Tokenizer
import torch.nn.functional as F
import numpy as np
from Multi_Stream_TransformerEncoder import Multi_Stream_TransformerEncoder, Multi_Stream_TransformerEncoderLayer

def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"[WARNING] NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"[WARNING] Inf detected in {name}")

def Pooler(encoded_embedding, attention_mask):
    return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

def compute_sentencelevel_contrastive_logits(projection_embeddings, inputs_attn_mask_batch, target_input_ids_batch, text_llm):
    batch_size = projection_embeddings.shape[0]
    target_input_ids_batch = target_input_ids_batch
    EEG_features = Pooler(projection_embeddings, inputs_attn_mask_batch)
    # Get text feature embedding
    text_attention_mask = torch.clone(inputs_attn_mask_batch)
    # Learned temperature parameter
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    Text_features = text_llm(input_ids=target_input_ids_batch['input_ids'], attention_mask=target_input_ids_batch['attention_mask']).last_hidden_state
    text_attention_mask = target_input_ids_batch['attention_mask']
    Sentence_feature = Pooler(Text_features, text_attention_mask)
    # Normalized features
    EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True)
    Sentence_feature = Sentence_feature / Sentence_feature.norm(dim=-1, keepdim=True)
    # Cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_EEG = logit_scale * EEG_features @ Sentence_feature.t()
    logits_per_text = logit_scale * Sentence_feature @ EEG_features.t()

    labels = torch.arange(batch_size).long()
    total_loss = (F.cross_entropy(logits_per_EEG, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    return total_loss



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (5000, 840)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (5000, 1, 840)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, dim) — batch_first=True
        # self.pe shape: (5000, 1, dim) — index by seq_len, then transpose to (1, seq_len, dim)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)




class SRCPModel(nn.Module):
    """SilentReading Contrastive Pretraining (SRCP) Model"""
    # NOTE: embed_dim % num_heads == 0 : 1024%16 768%12
    # NOTE: decoder_embed_dim % decoder_num_heads == 0
    def __init__(self, embed_dim=1024, eeg_dim=840, multi_heads=8, feedforward_dim=2048, trans_layers=6, decoder_embed_dim=840, pretrain_path="./models/huggingface/bart-large",
                 norm_layer=nn.LayerNorm, device=0):
        super().__init__()
        print('SRCP Model — SilentReading Contrastive Pretraining')
        self.device = torch.device(device)
        self.tokenizer = BartTokenizer.from_pretrained(pretrain_path)

        self.fc_eeg = nn.Linear(eeg_dim, embed_dim)
        self.act = nn.GELU()

        self.pos_embed_e = PositionalEncoding(eeg_dim)

        # EEG branch — 6-layer Transformer encoder
        self.eeg_encoder_layer = nn.TransformerEncoderLayer(d_model=840, nhead=multi_heads, dim_feedforward=feedforward_dim, batch_first=True, norm_first=False)
        self.e_branch = nn.TransformerEncoder(self.eeg_encoder_layer, num_layers=trans_layers)
        
        # Text branch — frozen BART encoder
        self.t_branch = BartModel.from_pretrained(pretrain_path)
        self.t_branch_encoder = self.t_branch.get_encoder()
        for param in self.t_branch.parameters():
            param.requires_grad = False


        # Unified (joint) branch — Multi-Stream Transformer
        self.unify_encoder_layer = Multi_Stream_TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True, norm_first=False)
        self.unify_branch = Multi_Stream_TransformerEncoder(self.unify_encoder_layer, num_layers=1)

        # Token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_e = PositionalEncoding(eeg_dim)

        # Asymmetric EEG decoder (lightweight, 1 layer)
        self.eeg_decoder_layers = nn.TransformerEncoderLayer(d_model=840, nhead=multi_heads, dim_feedforward=feedforward_dim, batch_first=True, norm_first=False)
        self.eeg_decoder = nn.TransformerEncoder(self.eeg_decoder_layers, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Projection layer: 1024 -> 840
        self.decoder_embed_e = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pred_t = nn.Linear(embed_dim, 50265, bias=True)

        # MLM loss with ignore_index for padding
        self.loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def eeg_masking_preserve_order_last_position(self, x, mask_ratio, attention_mask):
        """
        Perform per-sample masking while ensuring the last position with attention is masked.
        x: [N, L, D], EEG embeddings
        attention_mask: [N, L], binary mask indicating attention areas
        """

        N, L, D = x.shape  # batch, length, dim

        # Calculate the effective length based on attention_mask for each sample
        len_keep = (torch.sum(attention_mask, dim=1) * (1 - mask_ratio)).int()

        # Keep the elements based on effective length without shuffling
        masked_x_list = []
        ids_keep_list = []
        ids_restore_list = []
        max_len = 0
        for i in range(N):
            # Find the index of the last position with attention
            last_attention_index = torch.nonzero(attention_mask[i]).squeeze(1)[-1]

            # Generate random indices excluding the last attention position
            rand_indices = torch.randperm(last_attention_index)
            # Unmasked tokens (kept visible)
            ids_keep_i = torch.nonzero(attention_mask[i]).squeeze(1)[rand_indices[:len_keep[i]]]


            ids_keep_i_sorted_indices = torch.argsort(ids_keep_i)
            ids_keep_i_sorted = ids_keep_i[ids_keep_i_sorted_indices]
            ids_keep_list.append(ids_keep_i_sorted)

            # Generate indices to restore the original order
            ids_remove_i = torch.nonzero(attention_mask[i]).squeeze(1)[rand_indices[len_keep[i]:]]
            # Ensure the last position with attention is masked
            ids_remove_i = torch.cat((ids_remove_i, torch.tensor([last_attention_index], device=x.device)))

            ids_remove_i_sorted_indices = torch.argsort(ids_remove_i)
            ids_remove_i_sorted = ids_remove_i[ids_remove_i_sorted_indices]
            ids_restore_list.append(ids_remove_i_sorted)

            masked_x_i = torch.index_select(x[i], dim=0, index=ids_keep_i_sorted)
            masked_x_list.append(masked_x_i)
            max_len = max(max_len, len(ids_keep_i_sorted))

        masked_x_list = [torch.nn.functional.pad(masked_x_i, (0, 0, 0, max_len - len(masked_x_i))) for masked_x_i in
                         masked_x_list]

        masked_x = torch.stack(masked_x_list, dim=0)

        # Generate the masked_attention_mask where the last attention position is always masked
        masked_attention_mask = torch.zeros((N, max_len), device=x.device)
        for i in range(N):
            masked_attention_mask[i, :len(ids_keep_list[i])] = 1

        masked_attention_mask_invert = torch.ones((N, max_len), device=x.device)
        for i in range(N):
            masked_attention_mask_invert[i, :len(ids_keep_list[i])] = 0

        return masked_x, ids_keep_list, ids_restore_list, masked_attention_mask, masked_attention_mask_invert


    def mask_batch_text_tokens(
            slef, inputs, tokenizer, mlm_probability=0.15, is_train=True):
        """ Modified from transformers.data.data_collator
        Args:
            inputs: (B, L), 2D torch.Tensor, does not work for 1D. It has already been padded.
            tokenizer:
            mlm_probability: float
            is_train: if True use random masking, else mask tokens at fixed position to remove randomness in evaluation.
        """
        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        labels_list = labels.tolist()
        # Sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer.pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            padding_mask = padding_mask.to(device=probability_matrix.device)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # Replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

        masked_indices = ~torch.eq(labels, tokenizer.pad_token_id)
        masked_indices &= (labels != -100)

        return inputs, labels, masked_indices




    def forward_encoder(self, e, e_attn_mask, t, mask_ratio_e, mlm_probability):


        e = self.pos_embed_e(e)

        e_masked, ids_keep, ids_restore, masked_attention_mask, masked_attention_mask_invert = self.eeg_masking_preserve_order_last_position(e, mask_ratio_e, e_attn_mask)

        text_mlm_input_ids, text_mlm_labels, mlm_indices= self.mask_batch_text_tokens(t['input_ids'], self.tokenizer, mlm_probability=mlm_probability)

        e_branch_embeddings = self.e_branch(e_masked, src_key_padding_mask=masked_attention_mask_invert.bool())
        e_branch_embeddings = self.act(self.fc_eeg(e_branch_embeddings))  # 840 -> 1024
        t_branch_embeddings = self.t_branch_encoder(input_ids=text_mlm_input_ids, attention_mask=t['attention_mask']).last_hidden_state
        unify_embeddings = torch.cat((e_branch_embeddings, t_branch_embeddings), dim=1)

        unify_attention_mask_invert = torch.cat((masked_attention_mask_invert, t['attention_mask_invert']), dim=1)

        unify_branch_embeddings = self.unify_branch(unify_embeddings, src_key_padding_mask=unify_attention_mask_invert.bool(), modality=None)

        _,  L_e,  _ = e_branch_embeddings.shape

        x_eeg =  unify_branch_embeddings[:, :L_e, :]
        x_text = unify_branch_embeddings[:, L_e:, :]

        ce = self.unify_branch(e_branch_embeddings, src_key_padding_mask=masked_attention_mask_invert.bool(), modality='e')

        text_attention_mask_invert = t['attention_mask_invert']
        text_attention_mask_invert_float = text_attention_mask_invert.to(torch.float32)
        ct = self.unify_branch(t_branch_embeddings, src_key_padding_mask=text_attention_mask_invert_float,  modality='t')

        return x_eeg, x_text, ids_restore, ids_keep, masked_attention_mask, text_mlm_input_ids, text_mlm_labels, mlm_indices, ce, ct

    def forward_decoder(self, masked_e, eeg_attn_mask_invert, ids_restore_list, ids_keep_list):

        e_decoder = self.act(self.decoder_embed_e(masked_e))  # 1024 -> 840

        N, _, D = e_decoder.shape
        L_full = eeg_attn_mask_invert.shape[1]  # original full sequence length

        # Reconstruct full-length sequence: place visible tokens at their
        # original positions, fill masked positions with the mask token
        full_seq = self.mask_token.expand(N, L_full, D).clone()
        for i in range(N):
            for j, idx in enumerate(ids_keep_list[i]):
                if j < e_decoder.shape[1]:
                    full_seq[i, idx] = e_decoder[i, j]

        e = full_seq + self.decoder_pos_embed_e(full_seq)

        e = self.eeg_decoder(e, src_key_padding_mask=eeg_attn_mask_invert.bool())
        e = self.decoder_norm(e)
        check_nan_inf(e, "decoder_eeg")
        return full_seq, e

    def Pooler(self, encoded_embedding, attention_mask):
        sum_mask = attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)
        return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / sum_mask

    def text_Pooler(self, encoded_embedding, attention_mask):
        sum_mask = attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)
        return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / sum_mask

    def masked_Pooler(self, encoded_embedding, attention_mask, masked_indices):
        masked_attention_mask = attention_mask.clone()
        masked_attention_mask[masked_indices] = 0  # Zero out masked positions
        sum_embed = (encoded_embedding * masked_attention_mask.unsqueeze(-1)).sum(1)
        sum_mask = masked_attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)
        pooled_output = sum_embed / sum_mask

        return pooled_output

    def compute_sentencelevel_contrastive_logits(self, eeg_embeddings, eeg_attention, text_embedddings, text_attention, masked_indices):
        batch_size = eeg_embeddings.shape[0]
        EEG_features = self.Pooler(eeg_embeddings, eeg_attention)
        logit_scale = torch.tensor(np.log(1 / 0.07), device=EEG_features.device).exp()
        Sentence_feature = self.masked_Pooler(text_embedddings, text_attention, masked_indices)
        # Normalized features
        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        Sentence_feature = Sentence_feature / Sentence_feature.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        logits_per_EEG = logit_scale * EEG_features @ Sentence_feature.t()
        logits_per_text = logit_scale * Sentence_feature @ EEG_features.t()

        labels = torch.arange(batch_size).long().to(EEG_features.device)
        total_loss = (F.cross_entropy(logits_per_EEG, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss


    def forward_contrastive(self, eeg_embeddings, text_embeddings, bidirect_contrast=False):
        # Calculate NCE loss for mean-visual representation and mean-audio representation

        eeg_embeddings = torch.nn.functional.normalize(eeg_embeddings, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

        total = torch.mm(eeg_embeddings, torch.transpose(text_embeddings, 0, 1)) / 0.05

        # By default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=eeg_embeddings.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=eeg_embeddings.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=eeg_embeddings.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc


    def forward_loss_eeg(self, eeg, pred, eeg_ids_restore_list):
        losses = []

        for i, sample_ids in enumerate(eeg_ids_restore_list):
            eeg_sample = eeg[i][sample_ids]
            pred_sample = pred[i][sample_ids]

            loss_sample = ((pred_sample - eeg_sample) ** 2).mean(dim=-1)


            losses.append(loss_sample)


        losses_tensor = torch.cat(losses)


        loss = losses_tensor.mean()

        return loss


    def forward(self, eeg, eeg_attn_mask, eeg_attn_mask_invert, text, mask_ratio_e=0.25, mlm_probability=0.5, mlm_loss_weight=0.5, mae_loss_weight=1.0, contrast_loss_weight=0.01, sim_loss_weight=0.0):


        latent_eeg, latent_text, eeg_ids_restore_list, eeg_ids_keep_list, masked_attention_mask, text_mlm_inputs_ids, text_mlm_labels, mlm_indices, latent_c_eeg, latent_c_text= self.forward_encoder(eeg, eeg_attn_mask, text, mask_ratio_e, mlm_probability)
        check_nan_inf(latent_eeg, "latent_eeg")
        project_e, pred_e = self.forward_decoder(latent_eeg, eeg_attn_mask_invert, eeg_ids_restore_list, eeg_ids_keep_list)
        check_nan_inf(latent_eeg, "latent_eeg")

        loss_mae_eeg = self.forward_loss_eeg(eeg, pred_e, eeg_ids_restore_list)
        loss_mae = mae_loss_weight * loss_mae_eeg

        mlm_logits = self.act(self.decoder_pred_t(latent_text))

        loss_mlm = self.loss_mlm(input=mlm_logits.view(-1, 50265), target=text_mlm_labels.view(-1))
        loss_mlm = mlm_loss_weight * loss_mlm


        eeg_embeddings_whole_words = self.Pooler(project_e, eeg_attn_mask)
        last_one_indices = (torch.sum(eeg_attn_mask, dim=1).long() - 1).clamp(min=0, max=57)

        # Get corresponding EEG embeddings via index
        eeg_sentence_embeddings = eeg[torch.arange(eeg.size(0)), last_one_indices]
        cos_sim = torch.nn.functional.cosine_similarity(eeg_embeddings_whole_words, eeg_sentence_embeddings, dim=1)
        loss_sim = 1 - cos_sim.mean()
        loss_sim = sim_loss_weight * loss_sim

        # Contrastive loss
        loss_c = self.compute_sentencelevel_contrastive_logits(latent_c_eeg, masked_attention_mask, latent_c_text, text['attention_mask'], mlm_indices)

        loss_c = contrast_loss_weight * loss_c


        loss = loss_mlm + loss_c + loss_mae
        check_nan_inf(loss, "loss")
        return loss_mae, loss_mlm, loss_c, loss_sim, loss
