import os
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer

# Load BART tokenizer once at module level
TOKENIZER = BartTokenizer.from_pretrained('./models/huggingface/bart-large')

class EEG_dataset_add_sentence_mae(Dataset):
    """
    Dataset for CET-MAE pre-training.
    Loads pre-embedded EEG pickle files and returns EEG + BART-tokenized text.
    Each pickle file = one sentence read by one subject.
    """

    def __init__(self, path):
        self.path = path
        self.files = sorted([
            f for f in os.listdir(path)
            if f.endswith('.pickle')
        ])
        print(f'[INFO] Loaded dataset from {path}: {len(self.files)} samples')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, self.files[idx])

        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        # ── EEG ──────────────────────────────────────────────
        input_embeddings = sample['input_embeddings'].float()
        
        if 'non_normalized_input_embeddings' in sample:
            non_normalized = sample['non_normalized_input_embeddings'].float()
        else:
            non_normalized = input_embeddings.clone()

        input_attn_mask = sample['input_attn_mask'].long()
        input_attn_mask_invert = sample['input_attn_mask_invert'].long()

        # ── Text (original tokenization from dataset) ─────────
        target_ids = sample['target_ids'].long()
        target_mask = sample['target_mask'].long()

        # ── Raw text string ───────────────────────────────────
        text = sample.get('target_string', '')

        # ── Re-tokenize with BART ─────────────────────────────
        target_tokenized = TOKENIZER(
            text,
            padding='max_length',
            max_length=58,
            truncation=True,
            return_tensors='pt'
        )
        # Remove extra batch dim added by return_tensors='pt'
        target_tokenized = {k: v.squeeze(0) for k, v in target_tokenized.items()}

        return (
            input_embeddings,
            non_normalized,
            input_attn_mask,
            input_attn_mask_invert,
            target_ids,
            target_mask,
            target_tokenized,
            text
        )