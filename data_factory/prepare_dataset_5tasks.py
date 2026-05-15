from transformers import BartTokenizer
import codecs, os, numpy as np, scipy.io as io, h5py, pickle, torch
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import data_loading_helpers_modified as dh

# ── Paths ──────────────────────────────────────────────────────────────
ZUCO_ROOT    = "../zuco_dataset"
OUTPUT_ROOT  = "../datasets/data_word_sentence_5_pegasus"
BART_PATH    = "../models/huggingface/bart-large"

os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/valid", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/test",  exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
EEG_TYPE = "GD"
MAX_LEN  = 58
DIM      = 105
BANDS    = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']

# ── Helpers (same as your existing scripts) ────────────────────────────
def normalize_1d(t):
    return (t - t.mean()) / t.std()

def normalize_2d(m):
    f = m.view(-1)
    return (m - f.mean()) / f.std()

def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands, dim):
    feats = [word_obj['word_level_EEG'][eeg_type][eeg_type+b][0:dim] for b in bands]
    emb   = np.concatenate(feats)
    if len(emb) != dim * len(bands):
        return None, None
    t = torch.from_numpy(emb)
    return normalize_1d(t), t

def get_sent_eeg(sent_obj, bands):
    feats = [sent_obj['sentence_level_EEG']['mean'+b] for b in bands]
    emb   = np.concatenate(feats)
    t     = torch.from_numpy(emb)
    return normalize_1d(t), t

def get_input_sample(sent_obj, tokenizer):
    if sent_obj is None: return None
    text = sent_obj['content']
    text = text.replace('emp11111ty','empty').replace('film.1','film.')

    target_tokenized = tokenizer(text, padding='max_length', max_length=MAX_LEN,
                                 truncation=True, return_tensors='pt',
                                 return_attention_mask=True)

    sent_eeg, sent_eeg_raw = get_sent_eeg(sent_obj, BANDS)
    if torch.isnan(sent_eeg).any(): return None
    if len(sent_obj['word']) == 0:  return None

    word_embs, word_embs_raw = [], []
    for word in sent_obj['word']:
        we, we_raw = get_word_embedding_eeg_tensor(word, EEG_TYPE, BANDS, DIM)
        if we is None or torch.isnan(we).any(): return None
        word_embs.append(we); word_embs_raw.append(we_raw)

    word_embs.append(sent_eeg); word_embs_raw.append(sent_eeg_raw)
    seq_len = len(word_embs)

    raw_stack = torch.stack(word_embs_raw)
    norm_stack = list(torch.unbind(normalize_2d(raw_stack)))

    while len(word_embs)  < MAX_LEN: word_embs.append(torch.zeros(DIM*len(BANDS)))
    while len(norm_stack) < MAX_LEN: norm_stack.append(torch.zeros(DIM*len(BANDS)))

    attn      = torch.zeros(MAX_LEN); attn[:seq_len]  = 1
    attn_inv  = torch.ones(MAX_LEN);  attn_inv[:seq_len] = 0

    return {
        'target_tokenized':           target_tokenized,
        'target_ids':                 target_tokenized['input_ids'][0],
        'target_mask':                target_tokenized['attention_mask'][0],
        'target_string':              text,
        'seq_len':                    text.split(),
        'input_embeddings':           torch.stack(word_embs),
        'normalized_input_embeddings':torch.stack(norm_stack),
        'input_attn_mask':            attn,
        'input_attn_mask_invert':     attn_inv,
    }

def save_sample(sample, tag, subject, idx):
    if sample is None: return
    for split, folder in [('train','train'),('valid','valid'),('test','test')]:
        if tag == split:
            path = f"{OUTPUT_ROOT}/{folder}/{tag}-{subject}-{idx}.pickle"
            with open(path, 'wb') as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

def split_and_save(dataset_list, tokenizer, prefix):
    n = len(dataset_list)
    t_end = int(0.8 * n)
    v_end = t_end + int(0.1 * n)
    for i, sent_obj in enumerate(dataset_list):
        sample = get_input_sample(sent_obj, tokenizer)
        if sample is None: continue
        if   i < t_end: split = 'train'
        elif i < v_end: split = 'valid'
        else:           split = 'test'
        path = f"{OUTPUT_ROOT}/{split}/{prefix}-{i}.pickle"
        with open(path, 'wb') as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


# ══════════════════════════════════════════════════════════════════════
#  V1 SCIPY LOADER  (task1-SR, task2-NR, task3-TSR)
# ══════════════════════════════════════════════════════════════════════
def process_v1_task(task_name, tokenizer):
    print(f'\n=== Processing v1 {task_name} ===')
    mat_dir   = f"{ZUCO_ROOT}/{task_name}/Matlab_files"
    mat_files = sorted(glob(os.path.join(mat_dir, '*.mat')))

    for mat_file in tqdm(mat_files):
        subject = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']

        sent_list = []
        for sent in matdata:
            if isinstance(sent.word, float):
                sent_list.append(None); continue

            sent_obj = {
                'content': sent.content,
                'sentence_level_EEG': {
                    'mean_t1': sent.mean_t1, 'mean_t2': sent.mean_t2,
                    'mean_a1': sent.mean_a1, 'mean_a2': sent.mean_a2,
                    'mean_b1': sent.mean_b1, 'mean_b2': sent.mean_b2,
                    'mean_g1': sent.mean_g1, 'mean_g2': sent.mean_g2,
                },
                'word': []
            }
            for word in sent.word:
                try:
                    n_fix = int(np.asarray(word.nFixations).flatten()[0])
                except (IndexError, TypeError):
                    n_fix = 0
                if n_fix > 0:
                    sent_obj['word'].append({
                        'content': word.content,
                        'nFixations': n_fix,
                        'word_level_EEG': {
                            'GD':  {f'GD_{b[1:]}':  getattr(word, f'GD_{b[1:]}')  for b in BANDS},
                            'FFD': {f'FFD_{b[1:]}': getattr(word, f'FFD_{b[1:]}') for b in BANDS},
                            'TRT': {f'TRT_{b[1:]}': getattr(word, f'TRT_{b[1:]}') for b in BANDS},
                        }
                    })
            sent_list.append(sent_obj)

        split_and_save(sent_list, tokenizer, f"v1-{task_name}-{subject}")


# ══════════════════════════════════════════════════════════════════════
#  V2 H5PY LOADER  (task2-NR-2.0, task3-TSR-2.0)
# ══════════════════════════════════════════════════════════════════════
def process_v2_task(task_name, tokenizer):
    print(f'\n=== Processing v2 {task_name} ===')
    mat_dir   = f"{ZUCO_ROOT}/{task_name}/Matlab_files"
    mat_files = sorted(glob(os.path.join(mat_dir, '*.mat')))

    for mat_file in tqdm(mat_files):
        subject = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
        if subject == 'YMH': continue

        f  = h5py.File(mat_file, 'r')
        sd = f['sentenceData']

        band_keys = ['mean_t1','mean_t2','mean_a1','mean_a2','mean_b1','mean_b2','mean_g1','mean_g2']
        band_objs = {k: sd[k] for k in band_keys}

        sent_list = []
        for idx in range(len(sd['rawData'])):
            sent_string = dh.load_matlab_string(f[sd['content'][idx][0]])
            sent_obj = {
                'content': sent_string,
                'sentence_level_EEG': {k: np.squeeze(f[band_objs[k][idx][0]][()]) for k in band_keys},
                'word': []
            }
            word_data, _, _, _ = dh.extract_word_level_data(f, f[sd['word'][idx][0]])
            if not word_data:
                sent_list.append(None); continue

            for widx in range(len(word_data)):
                d = word_data[widx]
                if 'GD_EEG' not in d: continue
                gd, ffd, trt = d['GD_EEG'], d['FFD_EEG'], d['TRT_EEG']
                band_names = ['t1','t2','a1','a2','b1','b2','g1','g2']
                sent_obj['word'].append({
                    'content': d['content'],
                    'nFixations': d['nFix'],
                    'word_level_EEG': {
                        'GD':  {f'GD_{band_names[i]}':  gd[i]  for i in range(8)},
                        'FFD': {f'FFD_{band_names[i]}': ffd[i] for i in range(8)},
                        'TRT': {f'TRT_{band_names[i]}': trt[i] for i in range(8)},
                    }
                })
            sent_list.append(sent_obj)
        f.close()
        split_and_save(sent_list, tokenizer, f"v2-{task_name}-{subject}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Loading tokenizer...")
    tokenizer = BartTokenizer.from_pretrained(BART_PATH)

    # V1 tasks (scipy)
    process_v1_task("task1-SR",    tokenizer)
    process_v1_task("task2-NR",    tokenizer)
    process_v1_task("task3-TSR",   tokenizer)   # ← the missing one

    # V2 tasks (h5py)
    process_v2_task("task2-NR-2.0",  tokenizer)
    process_v2_task("task3-TSR-2.0", tokenizer)

    # Count results
    for split in ['train','valid','test']:
        n = len(glob(f"{OUTPUT_ROOT}/{split}/*.pickle"))
        print(f"{split}: {n} samples")
    print("Done!")