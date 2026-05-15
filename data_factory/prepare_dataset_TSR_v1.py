from transformers import BartTokenizer
import codecs
import os
import numpy as np
import scipy.io as io
from glob import glob
from tqdm import tqdm
import pickle
import torch
from copy import deepcopy


def normalize_1d(input_tensor):
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std
    return input_tensor


def normalize_2d(input_matrix):
    flattened_tensor = input_matrix.view(-1)
    mean = flattened_tensor.mean()
    std = flattened_tensor.std()
    normalized_matrix = (input_matrix - mean) / std
    return normalized_matrix


def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands, dim):
    frequency_features = []
    for band in bands:
        frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type + band][0:dim])
    word_eeg_embedding = np.concatenate(frequency_features)
    if len(word_eeg_embedding) != dim * len(bands):
        print(
            f'expect word eeg embedding dim to be {dim * len(bands)}, but got {len(word_eeg_embedding)}, return None')
        return None, None
    assert len(word_eeg_embedding) == dim * len(bands)
    return_tensor = torch.from_numpy(word_eeg_embedding)
    return normalize_1d(return_tensor), return_tensor


def get_sent_eeg(sent_obj, bands):
    sent_eeg_features = []
    for band in bands:
        key = 'mean' + band
        sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
    sent_eeg_embedding = np.concatenate(sent_eeg_features)
    assert len(sent_eeg_embedding) == 105 * len(bands)
    return_tensor = torch.from_numpy(sent_eeg_embedding)
    normalize_1d_return_tensor = normalize_1d(return_tensor)
    return normalize_1d_return_tensor, return_tensor


def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                     max_len=58, dim=105, add_CLS_token=False):
    if sent_obj is None:
        return None

    input_sample = {}

    target_string = sent_obj['content']
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True,
                                 return_tensors='pt', return_attention_mask=True)
    input_sample['target_tokenized'] = target_tokenized
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = target_string.split()
    input_sample['target_string'] = target_string

    sent_level_eeg_tensor, non_normalized_sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None

    word_embeddings = []
    non_normalized_word_embeddings = []
    selected_words = []

    if len(sent_obj['word']) == 0:
        return None

    for word in sent_obj['word']:
        word_level_eeg_tensor, non_normalized_word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands, dim=dim)
        if word_level_eeg_tensor is None:
            return None
        if torch.isnan(word_level_eeg_tensor).any():
            return None
        word_embeddings.append(word_level_eeg_tensor)
        non_normalized_word_embeddings.append(non_normalized_word_level_eeg_tensor)
        selected_words.append(word["content"])

    input_sample["embeddings_for_vis"] = deepcopy(word_embeddings)
    input_sample["non_normalized_embeddings_for_vis"] = deepcopy(non_normalized_word_embeddings)
    input_sample["selected_words"] = selected_words

    word_embeddings.append(sent_level_eeg_tensor)
    non_normalized_word_embeddings.append(non_normalized_sent_level_eeg_tensor)

    non_normalized_word_embeddings = torch.stack(non_normalized_word_embeddings)
    normalized_word_sentence_embeddings = normalize_2d(non_normalized_word_embeddings)
    normalized_word_sentence_embeddings = list(torch.unbind(normalized_word_sentence_embeddings))

    seq_len = len(word_embeddings)

    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(dim * len(bands)))
        normalized_word_sentence_embeddings.append(torch.zeros(dim * len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    input_sample['normalized_input_embeddings'] = torch.stack(normalized_word_sentence_embeddings)

    input_sample['input_attn_mask'] = torch.zeros(max_len)
    input_sample['input_attn_mask'][:seq_len] = torch.ones(seq_len)

    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    input_sample['input_attn_mask_invert'][:seq_len] = torch.zeros(seq_len)

    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample


if __name__ == "__main__":
    version = 'v1'
    eeg_type = "GD"
    max_len = 58
    dim = 105
    bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
    task_name = "task3-TSR"          # ← only change from the v1 script

    print("load tokenizer......")
    tokenizer = BartTokenizer.from_pretrained('./models/huggingface/bart-large')

    print(f'start processing ZuCo-{version} {task_name}...')

    input_mat_files_dir = f"../zuco_dataset/{task_name}/Matlab_files"
    mat_files = glob(os.path.join(input_mat_files_dir, '*.mat'))
    mat_files = sorted(mat_files)

    dataset_dict = {}

    for mat_file in tqdm(mat_files):
        subject_name = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
        dataset_dict[subject_name] = []
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']

        for sent in matdata:
            word_data = sent.word
            if not isinstance(word_data, float):
                sent_obj = {'content': sent.content}
                sent_obj['sentence_level_EEG'] = {
                    'mean_t1': sent.mean_t1, 'mean_t2': sent.mean_t2,
                    'mean_a1': sent.mean_a1, 'mean_a2': sent.mean_a2,
                    'mean_b1': sent.mean_b1, 'mean_b2': sent.mean_b2,
                    'mean_g1': sent.mean_g1, 'mean_g2': sent.mean_g2
                }
                sent_obj['word'] = []
                word_tokens_has_fixation = []
                word_tokens_with_mask = []
                word_tokens_all = []

                for word in word_data:
                    word_obj = {'content': word.content}
                    word_tokens_all.append(word.content)
                    word_obj['nFixations'] = word.nFixations
                    n_fixations = np.asarray(word.nFixations).item() if hasattr(word.nFixations, '__len__') and len(word.nFixations) > 0 else (word.nFixations if isinstance(word.nFixations, (int, float)) else 0)
                    if n_fixations > 0:
                        word_obj['word_level_EEG'] = {
                            'FFD': {'FFD_t1': word.FFD_t1, 'FFD_t2': word.FFD_t2,
                                    'FFD_a1': word.FFD_a1, 'FFD_a2': word.FFD_a2,
                                    'FFD_b1': word.FFD_b1, 'FFD_b2': word.FFD_b2,
                                    'FFD_g1': word.FFD_g1, 'FFD_g2': word.FFD_g2},
                            'TRT': {'TRT_t1': word.TRT_t1, 'TRT_t2': word.TRT_t2,
                                    'TRT_a1': word.TRT_a1, 'TRT_a2': word.TRT_a2,
                                    'TRT_b1': word.TRT_b1, 'TRT_b2': word.TRT_b2,
                                    'TRT_g1': word.TRT_g1, 'TRT_g2': word.TRT_g2},
                            'GD':  {'GD_t1': word.GD_t1,   'GD_t2': word.GD_t2,
                                    'GD_a1': word.GD_a1,   'GD_a2': word.GD_a2,
                                    'GD_b1': word.GD_b1,   'GD_b2': word.GD_b2,
                                    'GD_g1': word.GD_g1,   'GD_g2': word.GD_g2}
                        }
                        sent_obj['word'].append(word_obj)
                        word_tokens_has_fixation.append(word.content)
                        word_tokens_with_mask.append(word.content)
                    else:
                        word_tokens_with_mask.append('[MASK]')
                        continue

                sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                sent_obj['word_tokens_all'] = word_tokens_all
                dataset_dict[subject_name].append(sent_obj)
            else:
                dataset_dict[subject_name].append(None)
                continue

        total_num_sentence = len(dataset_dict[subject_name])
        train_divider = int(0.8 * total_num_sentence)
        dev_divider = train_divider + int(0.1 * total_num_sentence)

        for i in range(train_divider):
            input_sample = get_input_sample(dataset_dict[subject_name][i], tokenizer, eeg_type,
                                            bands=bands, max_len=max_len, dim=dim)
            if input_sample is not None:
                output_name = f"../datasets/data_word_sentence_5_pegasus/train/{version}-{task_name}-{subject_name}-{i}.pickle"
                with codecs.open(output_name, 'wb') as handle:
                    pickle.dump(input_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(train_divider, dev_divider):
            input_sample = get_input_sample(dataset_dict[subject_name][i], tokenizer, eeg_type,
                                            bands=bands, max_len=max_len, dim=dim)
            if input_sample is not None:
                output_name = f"../datasets/data_word_sentence_5_pegasus/valid/{version}-{task_name}-{subject_name}-{i}.pickle"
                with codecs.open(output_name, 'wb') as handle:
                    pickle.dump(input_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(dev_divider, total_num_sentence):
            input_sample = get_input_sample(dataset_dict[subject_name][i], tokenizer, eeg_type,
                                            bands=bands, max_len=max_len, dim=dim)
            if input_sample is not None:
                output_name = f"../datasets/data_word_sentence_5_pegasus/test/{version}-{task_name}-{subject_name}-{i}.pickle"
                with codecs.open(output_name, 'wb') as handle:
                    pickle.dump(input_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)