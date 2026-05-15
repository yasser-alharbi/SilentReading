# -*- coding: utf-8 -*-
"""
CT-E2T Chain-Thaw Fine-Tuning (Run 5)
Adapted from DeepMoji paper (Felbo et al., 2017).


Phase sequence:
  1. Phase 1 — Adapters (fc_eeg, eeg_stream_encoder)            (~5M)
  2. Phase 2 — BART Decoder + LM Head                    (~190M)
  3. Phase 3 — All Layers (full model, very low LR)               (~463M)
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


# ============================================================
# Layer group definitions
# Each entry: (phase_name, list of name-prefixes to unfreeze)
# ============================================================
CHAIN_THAW_PHASES = [
    (
        "Phase 1 — Adapters (fc_eeg, eeg_stream_encoder)",
        ['fc_eeg.', 'eeg_stream_encoder.']
    ),
    (
        "Phase 2 — BART Decoder + LM Head",
        ['bart.model.decoder.', 'bart.lm_head.']
    ),
    (
        "Phase 3 — All Layers (full model, gentle pass)",
        None
    ),
]

# LR for each phase: (new_lr, previously_unfrozen_lr)
PHASE_LRS = [
    (1e-4, None),    # Phase 1: adapters (random init)
    (5e-5, 1e-5),    # Phase 2: BART decoder + lm_head
    (1e-6, 1e-6),    # Phase 3: full model (gentle pass)
]

MAX_EPOCHS_PER_PHASE = 10
PATIENCE_PER_PHASE = 2
ACCUMULATION_STEPS = 2


def get_param_groups(model, current_phase_prefixes, previous_phase_prefixes,
                     new_lr, prev_lr):
    """
    Build optimizer param groups:
      - newly unfrozen params -> new_lr
      - previously unfrozen params -> prev_lr (if provided)
    """
    new_params = []
    prev_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = any(name.startswith(pfx) for pfx in current_phase_prefixes) \
                 if current_phase_prefixes else True
        if is_new:
            new_params.append(param)
        else:
            prev_params.append(param)

    param_groups = [{'params': new_params, 'lr': new_lr}]
    if prev_params and prev_lr is not None:
        param_groups.append({'params': prev_params, 'lr': prev_lr})

    return param_groups


def unfreeze_phase(model, phase_prefixes):
    """Unfreeze parameters matching phase_prefixes (cumulative — keeps previous unfrozen)."""
    if phase_prefixes is None:
        # Phase 5: unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in phase_prefixes):
            param.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_phase(train_dataloader, valid_dataloader, model, optimizer,
                    scheduler, tokenizer, phase_name, checkpoint_path,
                    checkpoint_name, max_epochs, patience):
    """Train model for one chain-thaw phase until convergence."""
    print(f'\n{"=" * 70}')
    print(f'  {phase_name}')
    print(f'  Trainable params: {count_trainable(model):,}')
    print(f'{"=" * 70}')
    logger.info(f'Starting {phase_name} | trainable: {count_trainable(model):,}')

    best_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    for epoch_idx in range(max_epochs):
        print(f'\nEpoch {epoch_idx}/{max_epochs - 1}')
        print('-' * 40)

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        train_batch_count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_dataloader,
                                               desc=f'  Train', leave=False)):
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
                train_batch_count += 1
                train_loss += loss.item() * ACCUMULATION_STEPS

        # flush remaining gradients
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_batch_count += 1
            train_loss += loss.item() * ACCUMULATION_STEPS

        scheduler.step()
        avg_train_loss = train_loss / max(train_batch_count, 1)
        print(f'  Train Loss: {avg_train_loss:.4f}')
        logger.info(f'{phase_name} | Epoch {epoch_idx} Train Loss: {avg_train_loss:.4f}')

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        valid_loss = 0
        valid_batch_count = 0

        with torch.no_grad():
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

                valid_batch_count += 1
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / max(valid_batch_count, 1)
        print(f'  Valid Loss: {avg_valid_loss:.4f}')
        logger.info(f'{phase_name} | Epoch {epoch_idx} Valid Loss: {avg_valid_loss:.4f}')

        # ── Checkpoint + early stopping ───────────────────────────────────
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_epoch = epoch_idx
            epochs_no_improve = 0
            saved_name = os.path.join(checkpoint_path, checkpoint_name)
            torch.save(model.state_dict(), saved_name)
            print(f'  [SAVED] Best checkpoint at epoch {epoch_idx} '
                  f'(valid loss: {best_loss:.4f})')
            logger.info(f'{phase_name} | Saved best at epoch {epoch_idx} '
                        f'| valid loss: {best_loss:.4f}')
        else:
            epochs_no_improve += 1
            print(f'  No improvement {epochs_no_improve}/{patience}')
            if epochs_no_improve >= patience:
                print(f'  [EARLY STOP] Phase complete. Best epoch: {best_epoch}')
                logger.info(f'{phase_name} | Early stop. Best epoch: {best_epoch} '
                            f'| best valid loss: {best_loss:.4f}')
                break

    logger.info(f'{phase_name} COMPLETE | best valid loss: {best_loss:.4f}')
    return best_loss


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT-E2T Chain-Thaw Fine-Tuning')
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

    # Model — start from scratch with SRCP pretrained EEG encoder
    model = CTE2TModel(
        eeg_dim=840,
        multi_heads=args['eeg_encoder_heads'],
        feedforward_dim=args['eeg_encoder_dim_feedforward'],
        trans_layers=args['eeg_encoder_layers'],
        pretrained_bart_path=args['pretrained_model'],
        device=device
    )

    # Load pretrained EEG encoder from SRCP (start from zero, no SecondBestModel)
    model.load_pretrained_eeg_encoder(args['srcp_checkpoint'])

    model.to(device)
    os.makedirs(args['save_path'], exist_ok=True)
    logger.info('Chain-thaw fine-tuning started from SRCP pretrained encoder')

    # ── Freeze all parameters to start ───────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False
    print(f'[INFO] All parameters frozen. Starting chain-thaw...')

    # Track all previously unfrozen prefixes (cumulative)
    all_unfrozen_prefixes = []

    # ── Chain-thaw loop ───────────────────────────────────────────────────
    for phase_idx, (phase_name, phase_prefixes) in enumerate(CHAIN_THAW_PHASES):
        new_lr, prev_lr = PHASE_LRS[phase_idx]

        # 1. Reload best checkpoint (critical — prevents carrying over bad weights)
        if phase_idx > 0:
            best_ckpt = os.path.join(args['save_path'], args['ct_e2t_checkpoint_name'])
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            print(f'\n[INFO] Reloaded best checkpoint before {phase_name}')
            # Re-freeze everything after reload
            for param in model.parameters():
                param.requires_grad = False
            # Re-unfreeze all previous phases cumulatively
            for prev_prefixes in [p for _, p in CHAIN_THAW_PHASES[:phase_idx]]:
                unfreeze_phase(model, prev_prefixes)

        # 2. Unfreeze current phase
        unfreeze_phase(model, phase_prefixes)

        # 3. Clear VRAM before building optimizer
        torch.cuda.empty_cache()
        current_prefixes = phase_prefixes if phase_prefixes else []
        param_groups = get_param_groups(
            model, current_prefixes, all_unfrozen_prefixes, new_lr, prev_lr
        )
        optimizer = AdamW(
            param_groups,
            weight_decay=float(args['weight_decay']),
            betas=(float(args['adam_beta1']), float(args['adam_beta2']))
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_PER_PHASE)

        # 4. Train this phase
        phase_best_loss = train_one_phase(
            train_dataloader, valid_dataloader, model, optimizer, scheduler,
            tokenizer, phase_name,
            checkpoint_path=args['save_path'],
            checkpoint_name=args['ct_e2t_checkpoint_name'],
            max_epochs=MAX_EPOCHS_PER_PHASE,
            patience=PATIENCE_PER_PHASE
        )

        # 5. Update cumulative prefix list
        if phase_prefixes:
            all_unfrozen_prefixes.extend(phase_prefixes)

        print(f'\n[INFO] {phase_name} complete. Best valid loss: {phase_best_loss:.4f}')

    print('\n[INFO] Chain-thaw training complete!')
    print(f'[INFO] Final checkpoint saved to: '
          f'{os.path.join(args["save_path"], args["ct_e2t_checkpoint_name"])}')
    logger.info('Chain-thaw training complete')
