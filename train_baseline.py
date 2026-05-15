# -*- coding: utf-8 -*-
"""
Baseline Training — Direct BART Fine-Tuning for EEG-to-Text
"""

import os
import argparse
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset
from model_ct_e2t import CTE2TModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


ACCUMULATION_STEPS = 2


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(train_dataloader, model, optimizer, tokenizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    batch_count = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_dataloader,
                                           desc='  Train', leave=False)):
        input_embeddings, _, input_attn_mask, input_attn_mask_invert, \
        target_ids, target_mask, target_tokenized, text = batch

        eeg = input_embeddings.to(device).float()
        eeg_attn_mask = input_attn_mask.to(device)
        target_ids = target_tokenized['input_ids'].to(device)
        target_mask = target_tokenized['attention_mask'].to(device)

        labels = target_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        loss = model(eeg, eeg_attn_mask, labels, target_mask) / ACCUMULATION_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            print(f'  [WARNING] Skipping batch {batch_idx} — NaN/Inf loss')
            optimizer.zero_grad()
            continue

        loss.backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            batch_count += 1
            total_loss += loss.item() * ACCUMULATION_STEPS

    # flush remaining gradients
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        batch_count += 1
        total_loss += loss.item() * ACCUMULATION_STEPS

    return total_loss / max(batch_count, 1)


@torch.no_grad()
def validate(valid_dataloader, model, tokenizer, device):
    """Run validation, return average loss."""
    model.eval()
    total_loss = 0
    batch_count = 0

    for batch in tqdm(valid_dataloader, desc='  Valid', leave=False):
        input_embeddings, _, input_attn_mask, _, \
        target_ids, target_mask, target_tokenized, text = batch

        eeg = input_embeddings.to(device).float()
        eeg_attn_mask = input_attn_mask.to(device)
        target_ids = target_tokenized['input_ids'].to(device)
        target_mask = target_tokenized['attention_mask'].to(device)

        labels = target_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        loss = model(eeg, eeg_attn_mask, labels, target_mask)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        batch_count += 1
        total_loss += loss.item()

    return total_loss / max(batch_count, 1)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline — Direct BART Fine-Tuning')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    args = vars(parser.parse_args())
    args = read_configuration(args['config'])

    init_logger(args)
    logger = getLogger()

    # Random seeds
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Device
    dev = args['cuda'] if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    print(f'[INFO] Using device: {dev}')

    # Tokenizer
    tokenizer = BartTokenizer.from_pretrained(args['pretrained_model'])

    # Datasets
    train_set = EEG_dataset(path=args['dataset_path'] + 'train')
    valid_set = EEG_dataset(path=args['dataset_path'] + 'valid')
    print(f'[INFO] Train: {len(train_set)} | Valid: {len(valid_set)}')

    train_dataloader = DataLoader(train_set, batch_size=args['batch_size'],
                                  shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_set, batch_size=args['batch_size'],
                                  shuffle=False, num_workers=0)

    # ── Model — NO CET-MAE pretraining, fully random EEG encoder ──
    model = CTE2TModel(
        eeg_dim=840,
        multi_heads=args['eeg_encoder_heads'],
        feedforward_dim=args['eeg_encoder_dim_feedforward'],
        trans_layers=args['eeg_encoder_layers'],
        pretrained_bart_path=args['pretrained_model'],
        device=device
    )
    # NOTE: We intentionally do NOT call model.load_pretrained_eeg_encoder()
    # The EEG encoder starts from random initialization.
    # BART encoder/decoder starts from pretrained bart-large weights.

    # ── Freeze all BART layers — only EEG components are trainable ──
    for name, param in model.named_parameters():
        if name.startswith('bart.'):
            param.requires_grad = False

    model.to(device)
    print(f'[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'[INFO] Trainable parameters: {count_trainable(model):,} (BART frozen)')

    os.makedirs(args['save_path'], exist_ok=True)
    checkpoint_path = os.path.join(args['save_path'], args['ct_e2t_checkpoint_name'])

    # ── Optimizer — single LR, all parameters ──
    lr = float(args['lr_finetune'])
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(args['weight_decay']),
        betas=(float(args['adam_beta1']), float(args['adam_beta2']))
    )

    num_epochs = int(args['num_epoch_fintune'])
    patience = int(args.get('patience', 5))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    logger.info(f'Baseline training started | lr={lr} | epochs={num_epochs} | '
                f'patience={patience} | batch_size={args["batch_size"]}')
    logger.info(f'No CET-MAE pretraining. No chain-thaw. Direct fine-tuning.')

    print(f'\n{"=" * 70}')
    print(f'  Baseline — Direct BART Fine-Tuning')
    print(f'  LR: {lr} | Epochs: {num_epochs} | Patience: {patience}')
    print(f'  No CET-MAE | No Chain-Thaw')
    print(f'{"=" * 70}\n')

    # ── Training loop ──
    best_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 40)

        # Train
        train_loss = train_one_epoch(train_dataloader, model, optimizer,
                                     tokenizer, device)
        print(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'Baseline | Epoch {epoch} Train Loss: {train_loss:.4f}')

        # Validate
        valid_loss = validate(valid_dataloader, model, tokenizer, device)
        print(f'  Valid Loss: {valid_loss:.4f}')
        logger.info(f'Baseline | Epoch {epoch} Valid Loss: {valid_loss:.4f}')

        scheduler.step()

        # Checkpoint + early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  [SAVED] Best checkpoint at epoch {epoch} '
                  f'(valid loss: {best_loss:.4f})')
            logger.info(f'Baseline | Saved best at epoch {epoch} '
                        f'| valid loss: {best_loss:.4f}')
        else:
            epochs_no_improve += 1
            print(f'  No improvement {epochs_no_improve}/{patience}')
            if epochs_no_improve >= patience:
                print(f'  [EARLY STOP] Best epoch: {best_epoch}')
                logger.info(f'Baseline | Early stop. Best epoch: {best_epoch} '
                            f'| best valid loss: {best_loss:.4f}')
                break

    print(f'\n[INFO] Baseline training complete!')
    print(f'[INFO] Best epoch: {best_epoch} | Best valid loss: {best_loss:.4f}')
    print(f'[INFO] Checkpoint: {checkpoint_path}')
    logger.info(f'Baseline training complete | best epoch: {best_epoch} '
                f'| best valid loss: {best_loss:.4f}')
