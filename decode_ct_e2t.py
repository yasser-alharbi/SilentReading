# -*- coding: utf-8 -*-
"""
CT-E2T Decoding Script — Teacher-Forcing EEG-to-Text Evaluation
Loads the best fine-tuned checkpoint and evaluates text prediction from EEG signals.
"""
import os
import re
import argparse
import torch
import numpy as np
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from model_e2t_ptr import E2T_PTR
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset


def clean_generated_text(text):
    """Post-process generated text: remove trailing g artifacts and extra spaces."""
    text = re.sub(r'g+$', '', text)      # remove trailing g's
    text = re.sub(r'\s+', ' ', text)     # collapse extra spaces
    return text.strip()


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
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def decode_teacher_forcing(model, dataset, tokenizer, device, num_samples=10):
    """Evaluate using teacher-forcing — this is what published papers report."""
    total = len(dataset)
    
    if num_samples >= total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_samples, dtype=int).tolist()
    
    results = []
    
    for sample_idx, data_idx in enumerate(indices):
        (input_embeddings, _, input_attn_mask, _,
         target_ids, target_mask, target_tokenized, text) = dataset[data_idx]
        
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
            'sample_num': sample_idx + 1,
            'dataset_idx': data_idx,
            'ground_truth': text,
            'tf_predicted': predicted_text,
        })
    
    return results


def save_tf_results(results, output_path):
    """Save teacher-forcing results to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('  E2T-PTR Teacher-Forcing Evaluation Results\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Total samples: {len(results)}\n\n')
        
        for r in results:
            f.write('-' * 80 + '\n')
            f.write(f'Sample {r["sample_num"]} (dataset index: {r["dataset_idx"]})\n')
            f.write('-' * 80 + '\n')
            f.write(f'  Ground Truth    : {r["ground_truth"]}\n')
            f.write(f'  TF Predicted    : {r["tf_predicted"]}\n\n')
        
        f.write('=' * 80 + '\n')
        f.write('End of results\n')
        f.write('=' * 80 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E2T-PTR EEG-to-Text Decoding')
    parser.add_argument('-c', '--config', help='path to eval config file', required=True)
    cli_args = vars(parser.parse_args())
    
    import yaml
    with open(cli_args['config'], 'r') as f:
        args = yaml.safe_load(f)
    
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    
    tokenizer = BartTokenizer.from_pretrained(args['pretrained_model'])
    test_set = EEG_dataset(path=args['dataset_path'] + 'test')
    model = load_model(args, device)
    
    num_samples = args.get('num_samples', len(test_set))
    
    # Teacher-forcing evaluation only
    results_tf = decode_teacher_forcing(model, test_set, tokenizer, device,
                                        num_samples=num_samples)
    tf_output = args['output_file'].replace('.txt', '_teacher_forcing.txt')
    save_tf_results(results_tf, tf_output)
