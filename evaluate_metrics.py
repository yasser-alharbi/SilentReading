# -*- coding: utf-8 -*-
"""
E2T-PTR Full Test Set Metrics Evaluation
Computes BLEU-1/2/3/4, ROUGE-1 P/R/F1, and BERTScore P/R/F1
on all test samples using teacher-forcing.

Usage:
    python evaluate_metrics.py -c config/eval_e2t_ptr_3.yaml
"""
import os
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

from model_e2t_ptr import E2T_PTR
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset
from utils import read_configuration


def load_model(args, device):
    """Initialize model and load fine-tuned checkpoint."""
    model = E2T_PTR(
        eeg_dim=args['eeg_dim'],
        multi_heads=args['eeg_encoder_heads'],
        feedforward_dim=args['eeg_encoder_dim_feedforward'],
        trans_layers=args['eeg_encoder_layers'],
        pretrained_bart_path=args['pretrained_model'],
        device=device
    )
    checkpoint_path = args['e2t_ptr_checkpoint']
    print(f'[INFO] Loading checkpoint from {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_teacher_forcing(model, dataset, tokenizer, device):
    """Run teacher-forcing on all samples, return lists of (reference, prediction)."""
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


def compute_bleu(references, predictions):
    """Compute corpus-level BLEU-1/2/3/4."""
    # NLTK corpus_bleu expects: references = [[ref_tokens]], hypotheses = [hyp_tokens]
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


def print_results_table(bleu, rouge, bertscore, num_samples, output_path=None):
    """Print thesis-ready results table and optionally save to file."""
    lines = []
    lines.append('=' * 70)
    lines.append('  E2T-PTR Evaluation Results (Teacher-Forcing)')
    lines.append('=' * 70)
    lines.append(f'  Test samples: {num_samples}')
    lines.append('-' * 70)
    lines.append('')
    lines.append('  Metric                          Score')
    lines.append('  ' + '-' * 40)

    # BLEU
    for key in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        lines.append(f'  {key:<34} {bleu[key]:>6.2f}%')

    lines.append('')

    # ROUGE-1
    for key in ['ROUGE-1 P', 'ROUGE-1 R', 'ROUGE-1 F']:
        label = key.replace('ROUGE-1 P', 'ROUGE-1 Precision') \
                   .replace('ROUGE-1 R', 'ROUGE-1 Recall') \
                   .replace('ROUGE-1 F', 'ROUGE-1 F1-Score')
        lines.append(f'  {label:<34} {rouge[key]:>6.2f}%')

    lines.append('')

    # BERTScore
    for key in ['BERTScore P', 'BERTScore R', 'BERTScore F']:
        label = key.replace('BERTScore P', 'BERTScore Precision') \
                   .replace('BERTScore R', 'BERTScore Recall') \
                   .replace('BERTScore F', 'BERTScore F1-Score')
        lines.append(f'  {label:<34} {bertscore[key]:>6.2f}%')

    lines.append('')
    lines.append('=' * 70)

    output = '\n'.join(lines)
    print(output)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output + '\n')
        print(f'\n[INFO] Results saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E2T-PTR Metrics Evaluation')
    parser.add_argument('-c', '--config', required=True, help='Path to eval config YAML')
    args = vars(parser.parse_args())
    args = read_configuration(args['config'])

    # Device
    device = torch.device(args.get('cuda', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # Load model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(args['pretrained_model'])
    model = load_model(args, device)

    # Load test dataset
    test_set = EEG_dataset(path=args['dataset_path'] + 'test')
    print(f'[INFO] Test set size: {len(test_set)}')

    # Run teacher-forcing on full test set
    start_time = time.time()
    references, predictions = run_teacher_forcing(model, test_set, tokenizer, device)
    tf_time = time.time() - start_time
    print(f'[INFO] Teacher-forcing completed in {tf_time:.1f}s')

    # Compute metrics
    print('\n[INFO] Computing BLEU scores...')
    bleu = compute_bleu(references, predictions)

    print('[INFO] Computing ROUGE scores...')
    rouge = compute_rouge(references, predictions)

    print('[INFO] Computing BERTScore (this may take a minute)...')
    bertscore = compute_bertscore(references, predictions)

    # Output
    output_path = os.path.join(
        os.path.dirname(args.get('output_file', 'results/metrics.txt')),
        'metrics_results.txt'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print_results_table(bleu, rouge, bertscore, len(test_set), output_path)
