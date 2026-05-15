# -*- coding: utf-8 -*-
"""
CT-E2T Evaluation with Post-Processing Pipeline (Stage 1)
===========================================================
Runs teacher-forcing on the full test set, applies rule-based post-processing,
and computes BLEU / ROUGE / BERTScore on BOTH raw and cleaned predictions
to measure the exact improvement from post-processing alone.

Usage:
    python evaluate_with_postprocess.py -c config/eval_ct_e2t.yaml
"""
import os
import re
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from model_ct_e2t import CTE2TModel
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset
from utils import read_configuration


# ═══════════════════════════════════════════════════════════════
#  Stage 1 — Rule-Based Post-Processing
# ═══════════════════════════════════════════════════════════════

def postprocess_text(text):
    """
    Apply deterministic rule-based cleaning to a single prediction string.
    Rules are ordered from most aggressive (trailing junk) to most delicate.
    """
    original = text  # keep for debugging

    # ── Rule 1: Remove trailing gibberish g/i repetitions ──
    #   Matches patterns like: "gigigi", "ggggi", "gggg", "gigi"
    text = re.sub(r'[gi]{3,}\s*$', '', text)

    # ── Rule 2: Remove trailing repeated punctuation blocks ──
    #   Matches: ",,,,,,", "((((", "....", or mixed like ",,(("
    text = re.sub(r'[,.()\[\]]{3,}\s*$', '', text)

    # ── Rule 2b: Another pass — gibberish + punctuation combos ──
    #   Handles cases like "andgigigigi,,,,,,,,((" 
    text = re.sub(r'[gi,.()\[\]\s]{5,}$', '', text)

    # ── Rule 3: Remove standalone orphan fragments ──
    #   "ust", "ak", "pp", "gg", "gi" as isolated words
    text = re.sub(r'\b(ust|ak|pp|gg|gi|ggi|gigi)\b', '', text)

    # ── Rule 4: Remove trailing orphan "is" / "in" after period ──
    #   "sleeve.. is" → "sleeve."
    text = re.sub(r'\.\s*\b(is|in|of|it)\s*$', '.', text)

    # ── Rule 5: Collapse repeated consecutive words (3+ times) ──
    #   "and and and" → "and", "would would would" → "would"
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)

    # ── Rule 6: Collapse repeated "and" / "or" (2+ times) ──
    #   "and and" → "and"
    text = re.sub(r'\b(and|or|the|is|a)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # ── Rule 7: Fix multiple consecutive periods ──
    #   "...." → ".", ".. " → ". "
    text = re.sub(r'\.{2,}', '.', text)

    # ── Rule 8: Fix multiple consecutive commas ──
    #   ",,," → ","
    text = re.sub(r',{2,}', ',', text)

    # ── Rule 9: Remove dangling single-character words at end ──
    #   "...ch line. a" → "...ch line."
    text = re.sub(r'\s+[a-z]\s*$', '', text)

    # ── Rule 10: Clean orphan punctuation patterns ──
    #   Remove isolated commas/periods not attached to words
    text = re.sub(r'\s+[,.](\s|$)', r'\1', text)

    # ── Rule 11: Fix space before punctuation ──
    #   "word ." → "word.", "word ," → "word,"
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # ── Rule 12: Normalize whitespace ──
    text = re.sub(r'\s{2,}', ' ', text)

    # ── Rule 13: Final trim ──
    text = text.strip()

    # ── Rule 14: Remove trailing punctuation-only residue ──
    #   If after cleaning we end with just ",(" or similar
    text = re.sub(r'[,.()\[\]]+$', lambda m: '.' if '.' in m.group() else '', text)

    # ── Rule 15: Ensure text ends properly ──
    #   If text doesn't end with punctuation, add a period
    if text and text[-1] not in '.!?':
        text = text.rstrip(',;:') + '.'

    return text.strip()


# ═══════════════════════════════════════════════════════════════
#  Model Loading & Inference
# ═══════════════════════════════════════════════════════════════

def load_model(args, device):
    """Initialize model and load fine-tuned checkpoint."""
    model = CTE2TModel(
        eeg_dim=args['eeg_dim'],
        multi_heads=args['eeg_encoder_heads'],
        feedforward_dim=args['eeg_encoder_dim_feedforward'],
        trans_layers=args['eeg_encoder_layers'],
        pretrained_bart_path=args['pretrained_model'],
        device=device
    )
    checkpoint_path = args['ct_e2t_checkpoint']
    print(f'[INFO] Loading checkpoint from {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_teacher_forcing(model, dataset, tokenizer, device):
    """Run teacher-forcing on all samples, return lists of (reference, raw_prediction)."""
    references = []
    predictions = []

    print(f'\n[INFO] Running teacher-forcing on {len(dataset)} samples...')
    for idx in tqdm(range(len(dataset)), desc='Teacher-forcing'):
        (input_embeddings, _, input_attn_mask, _,
         target_ids, target_mask, target_tokenized, text) = dataset[idx]

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
            pred_ids = outputs.logits[0].argmax(dim=-1)
            predicted_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

        references.append(text)
        predictions.append(predicted_text)

    return references, predictions


# ═══════════════════════════════════════════════════════════════
#  Metrics Computation
# ═══════════════════════════════════════════════════════════════

def compute_bleu(references, predictions):
    """Compute corpus-level BLEU-1/2/3/4."""
    refs_tokenized = [[ref.split()] for ref in references]
    preds_tokenized = [pred.split() for pred in predictions]
    smoothing = SmoothingFunction().method1

    bleu1 = corpus_bleu(refs_tokenized, preds_tokenized,
                        weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(refs_tokenized, preds_tokenized,
                        weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(refs_tokenized, preds_tokenized,
                        weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(refs_tokenized, preds_tokenized,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    return {
        'BLEU-1': bleu1 * 100,
        'BLEU-2': bleu2 * 100,
        'BLEU-3': bleu3 * 100,
        'BLEU-4': bleu4 * 100,
    }


def compute_rouge(references, predictions):
    """Compute ROUGE-1 Precision/Recall/F1."""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    precisions, recalls, f1s = [], [], []

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        precisions.append(scores['rouge1'].precision)
        recalls.append(scores['rouge1'].recall)
        f1s.append(scores['rouge1'].fmeasure)

    return {
        'ROUGE-1 P': np.mean(precisions) * 100,
        'ROUGE-1 R': np.mean(recalls) * 100,
        'ROUGE-1 F': np.mean(f1s) * 100,
    }


def compute_bertscore(references, predictions):
    """Compute BERTScore Precision/Recall/F1."""
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=True)
    return {
        'BERTScore P': P.mean().item() * 100,
        'BERTScore R': R.mean().item() * 100,
        'BERTScore F': F1.mean().item() * 100,
    }


# ═══════════════════════════════════════════════════════════════
#  Results Display & Saving
# ═══════════════════════════════════════════════════════════════

def print_comparison_table(raw_metrics, pp_metrics, num_samples, output_path=None):
    """Print side-by-side comparison: Raw vs Post-Processed."""
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('  CT-E2T Post-Processing Impact Analysis')
    lines.append('=' * 80)
    lines.append(f'  Test samples: {num_samples}')
    lines.append('-' * 80)
    lines.append('')
    lines.append(f'  {"Metric":<34} {"Raw":>10} {"Post-Proc":>10} {"Delta":>10}')
    lines.append('  ' + '-' * 66)

    metric_names = {
        'BLEU-1': 'BLEU-1',
        'BLEU-2': 'BLEU-2',
        'BLEU-3': 'BLEU-3',
        'BLEU-4': 'BLEU-4',
        'ROUGE-1 P': 'ROUGE-1 Precision',
        'ROUGE-1 R': 'ROUGE-1 Recall',
        'ROUGE-1 F': 'ROUGE-1 F1-Score',
        'BERTScore P': 'BERTScore Precision',
        'BERTScore R': 'BERTScore Recall',
        'BERTScore F': 'BERTScore F1-Score',
    }

    for key in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        raw_val = raw_metrics['bleu'][key]
        pp_val = pp_metrics['bleu'][key]
        delta = pp_val - raw_val
        sign = '+' if delta >= 0 else ''
        lines.append(f'  {metric_names[key]:<34} {raw_val:>9.2f}% {pp_val:>9.2f}% {sign}{delta:>8.2f}%')

    lines.append('')

    for key in ['ROUGE-1 P', 'ROUGE-1 R', 'ROUGE-1 F']:
        raw_val = raw_metrics['rouge'][key]
        pp_val = pp_metrics['rouge'][key]
        delta = pp_val - raw_val
        sign = '+' if delta >= 0 else ''
        lines.append(f'  {metric_names[key]:<34} {raw_val:>9.2f}% {pp_val:>9.2f}% {sign}{delta:>8.2f}%')

    lines.append('')

    for key in ['BERTScore P', 'BERTScore R', 'BERTScore F']:
        raw_val = raw_metrics['bertscore'][key]
        pp_val = pp_metrics['bertscore'][key]
        delta = pp_val - raw_val
        sign = '+' if delta >= 0 else ''
        lines.append(f'  {metric_names[key]:<34} {raw_val:>9.2f}% {pp_val:>9.2f}% {sign}{delta:>8.2f}%')

    lines.append('')
    lines.append('=' * 80)

    output = '\n'.join(lines)
    print(output)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output + '\n')
        print(f'\n[INFO] Comparison results saved to {output_path}')


def save_sample_comparison(references, raw_preds, pp_preds, output_path, num_show=50):
    """Save side-by-side sample comparisons to see what changed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Pick evenly spaced samples
    total = len(references)
    indices = np.linspace(0, total - 1, min(num_show, total), dtype=int).tolist()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('=' * 90 + '\n')
        f.write('  Post-Processing Sample Comparison (Before → After)\n')
        f.write('=' * 90 + '\n\n')
        f.write(f'Showing {len(indices)} of {total} samples\n\n')

        for i, idx in enumerate(indices):
            f.write('-' * 90 + '\n')
            f.write(f'Sample {i+1} (index {idx})\n')
            f.write('-' * 90 + '\n')
            f.write(f'  Ground Truth : {references[idx]}\n')
            f.write(f'  Raw Model    : {raw_preds[idx]}\n')
            f.write(f'  Post-Proc    : {pp_preds[idx]}\n')

            # Highlight what changed
            if raw_preds[idx] != pp_preds[idx]:
                f.write(f'  [CHANGED] ✓\n')
            else:
                f.write(f'  [NO CHANGE]\n')
            f.write('\n')

        f.write('=' * 90 + '\n')

    print(f'[INFO] Sample comparisons saved to {output_path}')


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT-E2T Post-Processing Evaluation')
    parser.add_argument('-c', '--config', required=True, help='Path to eval config YAML')
    args = vars(parser.parse_args())
    args = read_configuration(args['config'])

    # ── Device ──
    device = torch.device(args.get('cuda', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # ── Load model and tokenizer ──
    tokenizer = BartTokenizer.from_pretrained(args['pretrained_model'])
    model = load_model(args, device)

    # ── Load test dataset ──
    test_set = EEG_dataset(path=args['dataset_path'] + 'test')
    print(f'[INFO] Test set size: {len(test_set)}')

    # ══════════════════════════════════════════════════════════
    #  Step 1: Run teacher-forcing (get raw predictions)
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  STEP 1: Running teacher-forcing inference...')
    print('=' * 60)

    start_time = time.time()
    references, raw_predictions = run_teacher_forcing(model, test_set, tokenizer, device)
    tf_time = time.time() - start_time
    print(f'[INFO] Teacher-forcing completed in {tf_time:.1f}s')

    # ══════════════════════════════════════════════════════════
    #  Step 2: Apply post-processing
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  STEP 2: Applying post-processing rules...')
    print('=' * 60)

    pp_predictions = [postprocess_text(pred) for pred in raw_predictions]

    # Count how many changed
    changed = sum(1 for r, p in zip(raw_predictions, pp_predictions) if r != p)
    print(f'[INFO] Post-processing modified {changed}/{len(raw_predictions)} predictions '
          f'({100*changed/len(raw_predictions):.1f}%)')

    # ══════════════════════════════════════════════════════════
    #  Step 3: Compute metrics on RAW predictions
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  STEP 3: Computing metrics on RAW predictions...')
    print('=' * 60)

    print('[INFO] Computing BLEU (raw)...')
    raw_bleu = compute_bleu(references, raw_predictions)

    print('[INFO] Computing ROUGE (raw)...')
    raw_rouge = compute_rouge(references, raw_predictions)

    print('[INFO] Computing BERTScore (raw)...')
    raw_bertscore = compute_bertscore(references, raw_predictions)

    raw_metrics = {'bleu': raw_bleu, 'rouge': raw_rouge, 'bertscore': raw_bertscore}

    # ══════════════════════════════════════════════════════════
    #  Step 4: Compute metrics on POST-PROCESSED predictions
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  STEP 4: Computing metrics on POST-PROCESSED predictions...')
    print('=' * 60)

    print('[INFO] Computing BLEU (post-processed)...')
    pp_bleu = compute_bleu(references, pp_predictions)

    print('[INFO] Computing ROUGE (post-processed)...')
    pp_rouge = compute_rouge(references, pp_predictions)

    print('[INFO] Computing BERTScore (post-processed)...')
    pp_bertscore = compute_bertscore(references, pp_predictions)

    pp_metrics = {'bleu': pp_bleu, 'rouge': pp_rouge, 'bertscore': pp_bertscore}

    # ══════════════════════════════════════════════════════════
    #  Step 5: Print comparison & save results
    # ══════════════════════════════════════════════════════════
    results_dir = os.path.dirname(args.get('output_file', 'results/metrics.txt'))
    os.makedirs(results_dir, exist_ok=True)

    comparison_path = os.path.join(results_dir, 'postprocess_comparison.txt')
    print_comparison_table(raw_metrics, pp_metrics, len(test_set), comparison_path)

    samples_path = os.path.join(results_dir, 'postprocess_samples.txt')
    save_sample_comparison(references, raw_predictions, pp_predictions, samples_path, num_show=50)

    print('\n[DONE] Post-processing evaluation complete!')
