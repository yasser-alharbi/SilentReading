# -*- coding: utf-8 -*-
"""
Baseline Decoding — 5 Specific Samples
=======================================
Generates teacher-forcing predictions from the baseline checkpoint
for the same 5 samples used in the E2T-PTR comparison.
"""
import os
import re
import torch
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from model_e2t_ptr import E2T_PTR
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset


# ── Config ──────────────────────────────────────────────────────
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "./datasets/data_word_sentence_5_pegasus/test"
PRETRAINED_BART = "./models/huggingface/bart-large"
BASELINE_CKPT = "./checkpoints/CT-E2T/baseline_frozen_ct_e2t_5_tasks_bart_large.pt"
OUTPUT_FILE = "./results/baseline/baseline_5_selected_samples.txt"

# The 5 specific dataset indices to evaluate
SAMPLE_INDICES = [1814, 1765, 1912, 2157, 1520]

# Model architecture (must match training)
EEG_DIM = 840
EEG_ENCODER_LAYERS = 6
EEG_ENCODER_HEADS = 8
EEG_ENCODER_DIM_FEEDFORWARD = 2048
# ────────────────────────────────────────────────────────────────


def clean_generated_text(text):
    """Post-process generated text: remove trailing g artifacts and extra spaces."""
    text = re.sub(r'g+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    device = torch.device(DEVICE)
    print(f"[INFO] Device: {device}")

    # Tokenizer
    tokenizer = BartTokenizer.from_pretrained(PRETRAINED_BART)

    # Dataset
    test_set = EEG_dataset(path=DATASET_PATH)
    print(f"[INFO] Test set size: {len(test_set)}")

    # Build model (same architecture as E2T-PTR)
    model = E2T_PTR(
        eeg_dim=EEG_DIM,
        multi_heads=EEG_ENCODER_HEADS,
        feedforward_dim=EEG_ENCODER_DIM_FEEDFORWARD,
        trans_layers=EEG_ENCODER_LAYERS,
        pretrained_bart_path=PRETRAINED_BART,
        device=device
    )

    # Load baseline checkpoint
    print(f"[INFO] Loading baseline checkpoint: {BASELINE_CKPT}")
    state_dict = torch.load(BASELINE_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.\n")

    # Evaluate each sample
    results = []
    for i, data_idx in enumerate(SAMPLE_INDICES):
        (input_embeddings, _, input_attn_mask, _,
         target_ids, target_mask, target_tokenized, text) = test_set[data_idx]

        eeg = input_embeddings.unsqueeze(0).to(device).float()
        eeg_attn_mask = input_attn_mask.unsqueeze(0).to(device)
        raw_ids = target_tokenized['input_ids'].unsqueeze(0).to(device)
        decoder_input_ids = shift_tokens_right(
            raw_ids,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=model.bart.config.decoder_start_token_id
        )

        with torch.no_grad():
            eeg_embeds = model.get_eeg_embeddings(eeg, eeg_attn_mask)
            outputs = model.bart(
                inputs_embeds=eeg_embeds,
                attention_mask=eeg_attn_mask,
                decoder_input_ids=decoder_input_ids,
            )
            predictions = outputs.logits[0].argmax(dim=-1)
            predicted_text = tokenizer.decode(predictions, skip_special_tokens=True)
            predicted_text = clean_generated_text(predicted_text)

        results.append({
            'sample_num': i + 1,
            'dataset_idx': data_idx,
            'ground_truth': text,
            'predicted': predicted_text,
        })

        print(f"Sample {i+1} (index {data_idx}):")
        print(f"  GT : {text}")
        print(f"  TF : {predicted_text}\n")

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('  Baseline (Frozen BART) — Teacher-Forcing Results\n')
        f.write('  5 Selected Samples for Comparison\n')
        f.write('=' * 80 + '\n\n')

        for r in results:
            f.write('-' * 80 + '\n')
            f.write(f'Sample {r["sample_num"]} (dataset index: {r["dataset_idx"]})\n')
            f.write('-' * 80 + '\n')
            f.write(f'  Ground Truth    : {r["ground_truth"]}\n')
            f.write(f'  TF Predicted    : {r["predicted"]}\n\n')

        f.write('=' * 80 + '\n')
        f.write('End of results\n')
        f.write('=' * 80 + '\n')

    print(f"[INFO] Results saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
