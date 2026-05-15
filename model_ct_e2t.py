# -*- coding: utf-8 -*-
"""
SilentReading Chain-Thaw EEG-to-Text Decoding Model (CT-E2T)

Architecture:
  EEG → pos_embed → e_branch (6L) → fc_eeg+GELU → eeg_stream_encoder (1L) → BART decoder

This model loads pretrained EEG encoder weights from the SRCP (SilentReading Contrastive
Pretraining) phase and connects them to a BART-large decoder for text generation.

Based on the methodology described in:
  Wang et al. (2024) "Enhancing EEG-to-Text Decoding through Transferable Representations
  from Pre-trained Contrastive EEG-Text Masked Autoencoder" (ACL 2024)
"""
import torch
import torch.nn as nn
import math
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for EEG sequences."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class CTE2TModel(nn.Module):
    """
    Chain-Thaw EEG-to-Text (CT-E2T) Model
    
    Loads pretrained EEG encoder from SRCP and adds BART decoder for
    sequence-to-sequence EEG-to-text generation.
    
    Architecture:
      EEG → pos_embed → e_branch (6L) → fc_eeg+GELU → eeg_stream_encoder (1L) → BART decoder
    """
    def __init__(self, eeg_dim=840, embed_dim=1024, multi_heads=8, 
                 feedforward_dim=2048, trans_layers=6, 
                 pretrained_bart_path="./models/huggingface/bart-large",
                 device=0):
        super().__init__()
        print('CT-E2T Model for EEG-to-Text Generation')
        self.device = torch.device(device)
        
        # Positional encoding for EEG
        self.pos_embed_e = PositionalEncoding(eeg_dim)
        
        # EEG encoder (will be loaded from SRCP checkpoint)
        self.eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=eeg_dim, 
            nhead=multi_heads,
            dim_feedforward=feedforward_dim, 
            batch_first=True,
            norm_first=False
        )
        self.e_branch = nn.TransformerEncoder(
            self.eeg_encoder_layer, 
            num_layers=trans_layers
        )
        
        # Projection from EEG dim to BART dim
        self.fc_eeg = nn.Linear(eeg_dim, embed_dim)
        self.act = nn.GELU()
        
        # EEG stream transformer encoder (from SRCP's unify_branch)
        # 1 layer, 16 heads, 4096 FFN dim — processes projected 1024-dim EEG features
        eeg_stream_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,       # 1024
            nhead=16,
            dim_feedforward=4096,
            batch_first=True,
            norm_first=False
        )
        self.eeg_stream_encoder = nn.TransformerEncoder(
            eeg_stream_layer,
            num_layers=1,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # BART decoder (initialized from pretrained, then fine-tuned)
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_bart_path)
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_bart_path)
        
    def load_pretrained_eeg_encoder(self, checkpoint_path):
        """Load pretrained EEG encoder weights from SRCP checkpoint."""
        print(f'Loading pretrained EEG encoder from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        pretrained_dict = {}
        
        # Direct key matches: pos_embed_e, e_branch, fc_eeg
        for key, value in checkpoint.items():
            if key.startswith(('pos_embed_e.', 'e_branch.', 'fc_eeg.')):
                pretrained_dict[key] = value

        # Remap unify_branch → eeg_stream_encoder
        # SRCP's unify_branch has shared self-attention + modality-specific FFN/norms
        # We extract: shared self_attn + EEG-specific FFN (_e suffix) + EEG norm (_e suffix)
        unify_remap = {
            # Shared self-attention (used by both modalities in SRCP)
            'unify_branch.layers.0.self_attn.in_proj_weight': 'eeg_stream_encoder.layers.0.self_attn.in_proj_weight',
            'unify_branch.layers.0.self_attn.in_proj_bias':   'eeg_stream_encoder.layers.0.self_attn.in_proj_bias',
            'unify_branch.layers.0.self_attn.out_proj.weight': 'eeg_stream_encoder.layers.0.self_attn.out_proj.weight',
            'unify_branch.layers.0.self_attn.out_proj.bias':   'eeg_stream_encoder.layers.0.self_attn.out_proj.bias',
            # EEG-specific FFN (linear1_e, linear2_e → linear1, linear2)
            'unify_branch.layers.0.linear1_e.weight': 'eeg_stream_encoder.layers.0.linear1.weight',
            'unify_branch.layers.0.linear1_e.bias':   'eeg_stream_encoder.layers.0.linear1.bias',
            'unify_branch.layers.0.linear2_e.weight': 'eeg_stream_encoder.layers.0.linear2.weight',
            'unify_branch.layers.0.linear2_e.bias':   'eeg_stream_encoder.layers.0.linear2.bias',
            # EEG-specific layer norms (norm1_e, norm2_e → norm1, norm2)
            'unify_branch.layers.0.norm1_e.weight': 'eeg_stream_encoder.layers.0.norm1.weight',
            'unify_branch.layers.0.norm1_e.bias':   'eeg_stream_encoder.layers.0.norm1.bias',
            'unify_branch.layers.0.norm2_e.weight': 'eeg_stream_encoder.layers.0.norm2.weight',
            'unify_branch.layers.0.norm2_e.bias':   'eeg_stream_encoder.layers.0.norm2.bias',
            # EEG-specific final norm (norm_e → norm)
            'unify_branch.norm_e.weight': 'eeg_stream_encoder.norm.weight',
            'unify_branch.norm_e.bias':   'eeg_stream_encoder.norm.bias',
        }
        
        remap_count = 0
        for ckpt_key, model_key in unify_remap.items():
            if ckpt_key in checkpoint:
                pretrained_dict[model_key] = checkpoint[ckpt_key]
                remap_count += 1
        
        # Verification
        print(f'\n[DEBUG] Keys in checkpoint (first 10):')
        for k in list(checkpoint.keys())[:10]:
            print(f'  {k}')

        print(f'\n[DEBUG] Direct matches: {len(pretrained_dict) - remap_count}')
        print(f'[DEBUG] Remapped from unify_branch: {remap_count}')
        print(f'[DEBUG] Total keys to load: {len(pretrained_dict)}')
        
        print(f'\n[DEBUG] All keys being loaded:')
        for k, v in pretrained_dict.items():
            print(f'  [OK]  {k:60s}  shape: {list(v.shape)}')

        if len(pretrained_dict) == 0:
            print('\n  WARNING: No weights matched! Check key prefixes above.')
            print('  The model will train from random initialization -- expect poor results.')
        else:
            critical_keys = [
                'e_branch.layers.0.self_attn.in_proj_weight',
                'fc_eeg.weight',
                'pos_embed_e.pe',
                'eeg_stream_encoder.layers.0.self_attn.in_proj_weight',
                'eeg_stream_encoder.layers.0.linear1.weight',
                'eeg_stream_encoder.norm.weight',
            ]
            print('\n[DEBUG] Critical key check:')
            for ck in critical_keys:
                status = '[OK]' if ck in pretrained_dict else '[MISSING]'
                print(f'  {status}  {ck}')

        model_dict = self.state_dict()
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f'\n[INFO] Loaded {len(pretrained_dict)} pretrained parameters from SRCP')
    
    def get_eeg_embeddings(self, eeg, eeg_attn_mask):
        """Run EEG through our encoder pipeline, return projected embeddings."""
        eeg = self.pos_embed_e(eeg)
        eeg_attn_mask_invert = (eeg_attn_mask == 0).float()
        eeg_encoded = self.e_branch(eeg, src_key_padding_mask=eeg_attn_mask_invert.bool())
        eeg_projected = self.act(self.fc_eeg(eeg_encoded))
        eeg_projected = self.eeg_stream_encoder(
            eeg_projected, src_key_padding_mask=eeg_attn_mask_invert.bool()
        )
        return eeg_projected
        
    def forward(self, eeg, eeg_attn_mask, target_ids=None, target_mask=None):
        """
        Forward pass for CT-E2T model.
        
        Args:
            eeg: (batch, seq_len, 840) - EEG embeddings
            eeg_attn_mask: (batch, seq_len) - attention mask (1=attend, 0=ignore)
            target_ids: (batch, text_len) - target token IDs for training
            target_mask: (batch, text_len) - target attention mask
        
        Returns:
            loss (if training) or generated_ids (if inference)
        """
        # Apply positional encoding
        eeg = self.pos_embed_e(eeg)
        
        # EEG encoder (6 layers, 840-dim)
        eeg_attn_mask_invert = (eeg_attn_mask == 0).float()
        eeg_encoded = self.e_branch(eeg, src_key_padding_mask=eeg_attn_mask_invert.bool())
        
        # Project to BART dimension
        eeg_projected = self.act(self.fc_eeg(eeg_encoded))  # (batch, seq_len, 1024)
        
        # EEG stream transformer encoder (1 layer, 1024-dim)
        eeg_projected = self.eeg_stream_encoder(
            eeg_projected,
            src_key_padding_mask=eeg_attn_mask_invert.bool()
        )
        
        # Feed through BART's encoder via inputs_embeds
        # BART's encoder transforms EEG features into the distribution
        # its decoder's cross-attention was pretrained to understand
        if target_ids is not None:
            # Training / Validation mode: compute loss
            outputs = self.bart(
                inputs_embeds=eeg_projected,
                attention_mask=eeg_attn_mask,
                labels=target_ids,
                decoder_attention_mask=target_mask,
                return_dict=True
            )
            return outputs.loss
        else:
            # Inference mode: generate text
            generated_ids = self.bart.generate(
                inputs_embeds=eeg_projected,
                attention_mask=eeg_attn_mask,
                num_beams=5,
                max_length=60,
                min_length=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            return generated_ids
