# <img width="80" height="80" alt="image" src="https://github.com/user-attachments/assets/0b4e2340-a61f-4030-ab2e-fd8a374ba8dd" />  |  SilentReading 


An EEG-to-text translation pipeline to help mute individuals communicate by translating brain signals into readable text.

This repository contains the official implementation of the **SilentReading** project, which uses a two-stage approach for high-fidelity brain-to-text decoding:
1. **SRCP (SilentReading Contrastive Pretraining):** A contrastive EEG-text masked autoencoder to align EEG representations with textual semantics.
2. **CT-E2T (Chain-Thaw EEG-to-Text):** A BART-based decoding model that iteratively unfreezes layers for optimal fine-tuning, translating the pre-trained EEG representations into fluent text.

## 🚀 Environment Setup

We recommend using Anaconda or Miniconda to set up the environment.

```bash
conda create -n silentreading python=3.8
conda activate silentreading
pip install -r requirements.txt
```

## 📊 Data Preparation

We use the [ZuCo (Zurich Cognitive Language Processing Corpus)](https://osf.io/q3zws/files/) benchmark dataset. 

1. Download the `Matlab files` for `task1-SR`, `task2-NR`, and `task3-TSR` from ZuCo v1.0. Place them in the respective directories under `./zuco_dataset/`.
2. Download ZuCo v2.0 `Matlab files` for `task1-NR` and place them under `./zuco_dataset/task2-NR-2.0/Matlab_files`.

Preprocess the data using the scripts in `data_factory/`:
```bash
python data_factory/data2pickle_v1.py
python data_factory/data2pickle_v2.py
```

## 🧠 Training & Evaluation

### 1. SRCP Pretraining (Stage 1)
Train the contrastive EEG-text masked autoencoder to align multimodal representations.
```bash
python train_srcp.py -c config/train_srcp.yaml
```

### 2. CT-E2T Fine-Tuning (Stage 2)
Train the EEG-to-text generation model using the Chain-Thaw methodology.
```bash
python train_ct_e2t.py -c config/train_ct_e2t.yaml
```

### 3. Baseline Training (Optional)
If you want to train the standard direct fine-tuning baseline for comparison.
```bash
python train_baseline.py -c config/train_baseline.yaml
```

### 4. Decoding and Inference
Generate text predictions from the fine-tuned model checkpoint.
```bash
python decode_ct_e2t.py -c config/eval_ct_e2t.yaml
```

### 5. Post-Processing & Evaluation
Calculate automated translation metrics (BLEU, ROUGE, BERTScore) or run the LLM-enhanced post-processing pipeline.
```bash
python evaluate_metrics.py
# Or with post-processing optimizations:
python evaluate_with_postprocess.py
```

## 📂 Repository Structure

- `config/`: YAML configuration files for all training, decoding, and evaluation tasks.
- `data_factory/`: Data loaders and preprocessing scripts for the ZuCo datasets.
- `contrastive_eeg_pretraining/`: Modules for the SRCP contrastive learning stage.
- `model_srcp.py` / `model_ct_e2t.py`: Core PyTorch architectures for the respective project phases.
- `train_*.py` / `decode_*.py`: Executable scripts for running the models.

## ⚖️ License
This project is released for academic and research purposes.
