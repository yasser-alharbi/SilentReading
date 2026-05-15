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

We use the [ZuCo (Zurich Cognitive Language Processing Corpus)](https://osf.io/q3zws/files/) benchmark dataset, evaluating on all **5 tasks**. 

1. Download the `Matlab files` for `task1-SR`, `task2-NR`, and `task3-TSR` from ZuCo v1.0. Place them in the respective directories under `./zuco_dataset/`.
2. Download ZuCo v2.0 `Matlab files` for `task1-NR` and `task2-TSR` and place them under `./zuco_dataset/task2-NR-2.0/Matlab_files` and `./zuco_dataset/task3-TSR-2.0/Matlab_files` respectively.

Preprocess the data for all 5 tasks using the unified script in `data_factory/`:
```bash
python data_factory/prepare_dataset_5tasks.py
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

## 📈 Results

SilentReading significantly outperforms direct EEG-to-Text fine-tuning architectures. By utilizing **SRCP** (Contrastive Pretraining) and the **Chain-Thaw** fine-tuning methodology, our final model successfully preserves semantic accuracy across noisy brainwaves.

| Metric | Baseline (Frozen BART) | Final CT-E2T Model | Absolute Improvement |
|---|---|---|---|
| **BLEU-1** | 25.87% | **48.88%** | + 23.01% |
| **BLEU-4** | 2.06% | **35.54%** | + 33.48% |
| **ROUGE-1 (F1)** | 31.03% | **50.53%** | + 19.50% |
| **BERTScore (F1)** | 82.99% | **88.79%** | + 5.80% |

*(Evaluated on 2,404 unseen test samples across 5 ZuCo tasks using teacher-forcing).*

### Text Enhancement Module
Our 2-stage Text Enhancement pipeline (deterministic rule-based cleaning + GPT-5.4-mini grammatical refinement) further polishes the raw CT-E2T predictions to ensure grammatical fluency:
* **Raw BLEU-4:** 35.54% ➔ **LLM-Enhanced:** 36.64%
* **Raw ROUGE-1 (F1):** 50.53% ➔ **LLM-Enhanced:** 52.76%
* **Raw BERTScore (F1):** 88.79% ➔ **LLM-Enhanced:** 90.52%

## 📝 Sample Predictions

| Ground Truth (Original Text) | Raw CT-E2T Output | LLM-Enhanced Output |
|---|---|---|
| *The book was awarded the 1957 Pulitzer Prize for Biography.* | film is awarded the 1957 Pulitzer Prize for Biography.,,,,,,,,,, | The film is awarded the 1957 Pulitzer Prize for Biography. |
| *He attended secondary school (Volksschule), and learned the trade of a joiner.* | was Florida school andVolksschule), and learned the trade of a joiner.rer....gigi | was Florida school and Volksschule, and learned the trade of a joiner. |

## 📂 Repository Structure

- `config/`: YAML configuration files for all training, decoding, and evaluation tasks.
- `data_factory/`: Data loaders and preprocessing scripts for the ZuCo datasets.
- `contrastive_eeg_pretraining/`: Modules for the SRCP contrastive learning stage.
- `model_srcp.py` / `model_ct_e2t.py`: Core PyTorch architectures for the respective project phases.
- `train_*.py` / `decode_*.py`: Executable scripts for running the models.

## 🙏 Acknowledgements & Special Thanks

This project was built upon incredible foundational research in the brain-computer interface and NLP domains. We extend our deepest gratitude to the authors of:
* **[Wang et al., 2024]** for their pioneering work on the *Contrastive EEG-Text Masked Autoencoder (CET-MAE)* and *E2T-PTR*, which heavily inspired the core SRCP methodology and data processing pipeline of this project.
* **[Felbo et al., 2017]** for their *DeepMoji* paper, which introduced the powerful Chain-Thaw transfer learning strategy that enabled our CT-E2T phase.
* The creators of the **[ZuCo Corpus]**, for providing the high-quality, open-source EEG and eye-tracking dataset that made this research possible.

## ⚖️ License
This project is released for academic and research purposes.

