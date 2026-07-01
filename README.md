<div align="center">

<img width="80" height="80" alt="SilentReading logo" src="https://github.com/user-attachments/assets/0b4e2340-a61f-4030-ab2e-fd8a374ba8dd" />

# SilentReading

**An EEG-to-Text translation pipeline that helps mute individuals communicate by turning brain signals into readable text.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-silentreading.ahhh.sa-2ea44f?style=flat-square)](https://silentreading.ahhh.sa/)
[![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)](#-license)

**🔗 Live Demo: [silentreading.ahhh.sa](https://silentreading.ahhh.sa/)**

</div>

---

## 📖 Overview

This repository contains the official implementation of **SilentReading**, a two-stage brain-to-text decoding system built for high-fidelity translation of EEG signals into fluent, readable text. The goal is to give non-verbal individuals a way to communicate directly from brain activity.

The pipeline works in two stages:

**1. SRCP — SilentReading Contrastive Pretraining**
A contrastive EEG-text masked autoencoder that aligns EEG representations with textual semantics.

<div align="center">
  <img width="864" height="576" alt="SRCP architecture" src="https://github.com/user-attachments/assets/74326de9-7ab0-42a4-998e-50b5800b9efb" />
</div>

**2. CT-E2T — Chain-Thaw EEG-to-Text**
A BART-based decoder that iteratively unfreezes layers for optimal fine-tuning, translating the pre-trained EEG representations into fluent text.

<div align="center">
  <img width="865" height="433" alt="CT-E2T architecture" src="https://github.com/user-attachments/assets/d704c325-cd5f-4388-9a93-f09cdb24b861" />
</div>

---

## ✨ Key Features

- **Two-stage decoding** — contrastive pretraining (SRCP) followed by Chain-Thaw fine-tuning (CT-E2T) for stronger semantic fidelity than direct fine-tuning.
- **5-task evaluation** — trained and evaluated across all 5 tasks of the ZuCo benchmark (v1.0 and v2.0).
- **LLM-enhanced post-processing** — a rule-based cleaner plus GPT-5.4-mini refinement stage that boosts grammatical fluency of raw predictions.
- **Reproducible configs** — every training, decoding, and evaluation run is driven by a YAML file under `config/`.

---

## 🚀 Environment Setup

We recommend [Anaconda](https://www.anaconda.com/) or Miniconda.

```bash
conda create -n silentreading python=3.8
conda activate silentreading
pip install -r requirements.txt
```

---

## 📊 Data Preparation

We use the [ZuCo (Zurich Cognitive Language Processing Corpus)](https://osf.io/q3zws/files/) benchmark, evaluating on all **5 tasks**.

1. Download the `Matlab files` for `task1-SR`, `task2-NR`, and `task3-TSR` from **ZuCo v1.0**. Place them in the respective directories under `./zuco_dataset/`.
2. Download **ZuCo v2.0** `Matlab files` for `task1-NR` and `task2-TSR`, and place them under `./zuco_dataset/task2-NR-2.0/Matlab_files` and `./zuco_dataset/task3-TSR-2.0/Matlab_files` respectively.

Preprocess all 5 tasks with the unified script:

```bash
python data_factory/prepare_dataset_5tasks.py
```

---

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
Train the standard direct fine-tuning baseline for comparison.
```bash
python train_baseline.py -c config/train_baseline.yaml
```

### 4. Decoding & Inference
Generate text predictions from a fine-tuned model checkpoint.
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

---

## 📈 Results

SilentReading significantly outperforms direct EEG-to-Text fine-tuning architectures. By combining **SRCP** contrastive pretraining with the **Chain-Thaw** fine-tuning methodology, the final model preserves semantic accuracy across noisy brainwaves.

| Metric | Baseline (Frozen BART) | Final CT-E2T Model | Absolute Improvement |
|---|---|---|---|
| **BLEU-1** | 25.87% | **48.88%** | +23.01% |
| **BLEU-4** | 2.06% | **35.54%** | +33.48% |
| **ROUGE-1 (F1)** | 31.03% | **50.53%** | +19.50% |
| **BERTScore (F1)** | 82.99% | **88.79%** | +5.80% |

<sub>Evaluated on 2,404 unseen test samples across 5 ZuCo tasks using teacher-forcing.</sub>

### Text Enhancement Module

A 2-stage enhancement pipeline (deterministic rule-based cleaning + GPT-5.4-mini grammatical refinement) further polishes the raw CT-E2T predictions:

| Metric | Raw CT-E2T | LLM-Enhanced |
|---|---|---|
| **BLEU-4** | 35.54% | **36.64%** |
| **ROUGE-1 (F1)** | 50.53% | **52.76%** |
| **BERTScore (F1)** | 88.79% | **90.52%** |

---

## 📝 Sample Predictions

| Ground Truth | Raw CT-E2T Output | LLM-Enhanced Output |
|---|---|---|
| *The book was awarded the 1957 Pulitzer Prize for Biography.* | film is awarded the 1957 Pulitzer Prize for Biography.,,,,,,,,,, | The film is awarded the 1957 Pulitzer Prize for Biography. |
| *He attended secondary school (Volksschule), and learned the trade of a joiner.* | was Florida school andVolksschule), and learned the trade of a joiner.rer....gigi | was Florida school and Volksschule, and learned the trade of a joiner. |

---

## 📂 Repository Structure

```
SilentReading/
├── config/                       # YAML configs for training, decoding, and evaluation
├── data_factory/                 # Data loaders and preprocessing for the ZuCo datasets
├── contrastive_eeg_pretraining/  # Modules for the SRCP contrastive learning stage
├── model_srcp.py                 # Core PyTorch architecture — SRCP stage
├── model_ct_e2t.py               # Core PyTorch architecture — CT-E2T stage
├── train_srcp.py                 # Stage 1 training script
├── train_ct_e2t.py               # Stage 2 training script
├── train_baseline.py             # Direct fine-tuning baseline
├── decode_ct_e2t.py              # Inference / decoding
├── evaluate_metrics.py           # Automated metrics (BLEU, ROUGE, BERTScore)
└── evaluate_with_postprocess.py  # Metrics with LLM-enhanced post-processing
```

---

## 🙏 Acknowledgements & Special Thanks

This project builds on foundational research in the brain-computer interface and NLP domains. Our deepest gratitude to:

- **Wang et al., 2024** — for their pioneering work on the *Contrastive EEG-Text Masked Autoencoder (CET-MAE)* and *E2T-PTR*, which heavily inspired the core SRCP methodology and data pipeline.
- **Felbo et al., 2017** — for their *DeepMoji* paper, which introduced the Chain-Thaw transfer learning strategy that enabled our CT-E2T phase.
- The creators of the **ZuCo Corpus** — for the high-quality, open-source EEG and eye-tracking dataset that made this research possible.

---

## 📌 Citation

If you use this work, please cite:

```bibtex
@misc{silentreading2026,
  title        = {SilentReading: An EEG-to-Text Translation Pipeline for Assisted Communication},
  author       = {Alharbi, Yasser},
  year         = {2026},
  howpublished = {\url{https://silentreading.ahhh.sa/}}
}
```

---

## ⚖️ License

This project is released for **academic and research purposes**.

<div align="center">
<sub>Built to give a voice to those who can't speak. 🧠 → 📝</sub>
</div>
