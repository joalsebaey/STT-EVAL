# 🎙️ STT EVAL — Arabic Speech-to-Text Evaluation & Robustness Kit

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/Colab-Notebook%20Ready-blue?logo=googlecolab)](https://colab.research.google.com/)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)]()
[![Language](https://img.shields.io/badge/Language-Arabic%20(العربية)-green.svg)]()

**STT EVAL** is a comprehensive benchmarking, diagnostics, and acoustic robustness evaluation toolkit for **Arabic Speech-to-Text (STT) models** (Whisper variants, local checkpoints, and custom architectures).

Evaluate and stress-test your models against clean speech, synthetic & real noise, telephony compression, speed variations, and reverberation with publication-grade metrics and error analysis.

---

## 🌟 Highlights

- **🤖 Model Agnostic**: Benchmark any HuggingFace model (`openai/whisper-small`, `large-v3-turbo`, fine-tuned models), local directory checkpoints, or custom Python inference functions (PyTorch, ONNX, Faster-Whisper, etc.).
- **🌍 Dialectal Arabic & Ingestion**: Stream the multi-dialect `oddadmix/dialectal-arabic-lahgtna-v2` dataset directly from HuggingFace, or drop in custom datasets formatted as CSV, TSV, JSON, JSONL, audio folders, or `.zip` archives. Full per-dialect WER & CER breakdown.
- **🔊 Acoustic Robustness & Stress-Testing**:
  - **White Noise**: Multi-level additive noise at 20 dB, 10 dB, and 5 dB SNR.
  - **Telephony Filter**: 300–3400 Hz bandpass, 8 kHz downsampling, and 8-bit $\mu$-law companding.
  - **Speed Perturbation**: Fast speech (1.15×) and slow speech (0.85×).
  - **Gain / Attenuation**: Extreme low volume (-25 dB).
  - **Room Reverberation**: Exponentially decaying synthetic room impulse response (RT60 = 0.5s).
  - **Real Noise**: Mix real-world environmental noise recordings (e.g., street, cafe).
- **📊 Rich Diagnostic Metrics**:
  - **WER & CER** with **Bootstrap 95% Confidence Intervals**.
  - **S/D/I Breakdown**: Substitutions (misheard words), Deletions (missed words), and Insertions (hallucinations).
  - **Real-Time Factor (RTF)** for speed/latency profiling.
  - **Failure Flags**: Automatically flags empty predictions, repetitive hallucination loops, excessively long generations, and foreign script / language identification failures.
  - **Top Confusions**: Word-level error alignment matrix (`top_confusions.csv`).
  - **🔤 Configurable Text Normalization**: Strips Arabic diacritics, unifies orthographic variants (alef, ya, ta marbuta), maps numerals, and automatically cleans audio annotation tags (e.g., `<music>`, `[applause]`, `(laughter)`).
- **📓 Google Colab & Jupyter Ready**: A self-contained, interactive notebook (`stt_eval_notebook.ipynb`) equipped with audio players, matplotlib dashboards, and auto-downloading reports.

---

## 📊 Benchmark & Model Comparison

This section presents empirical results from evaluating and comparing **`whisper-small`** against **`whisper-large-v3-turbo`** on the full Google FLEURS Egyptian Arabic test set ($n = 428$ utterances, fixed seed `42`, CUDA, identical text normalization).

### 🏆 Executive Head-to-Head Comparison

| Metric | `openai/whisper-small` | `openai/whisper-large-v3-turbo` | Improvement / Delta | Winner |
|---|:---:|:---:|:---:|:---:|
| **Clean WER** | `26.3%` | `10.1%` | **-16.2 pts** (-61.6% relative) | 🥇 `large-v3-turbo` |
| **Clean CER** | `9.4%` | `3.4%` | **-6.0 pts** (-63.8% relative) | 🥇 `large-v3-turbo` |
| **Speed (RTF)** *(lower is faster)* | `0.033` (~30× real-time) | `0.009` (~111× real-time) | **~3.5× faster** | 🥇 `large-v3-turbo` |
| **Substitution Rate (S)** | `22.6%` | `8.5%` | -14.1 pts | 🥇 `large-v3-turbo` |
| **Deletion Rate (D)** | `1.8%` | `1.1%` | -0.7 pts | 🥇 `large-v3-turbo` |
| **Insertion Rate (I)** | `1.5%` | `0.5%` | -1.0 pts (cut to 1/3) | 🥇 `large-v3-turbo` |
| **Paired Win Rate** | Baseline | **86.9% – 98.8%** | Loss rate only 0.5% – 2.6% | 🥇 `large-v3-turbo` |
| **Max Failure Flag Rate** | `2.1%` (speed 1.15) | `0.9%` (speed 1.15) | ~3× fewer hallucinations | 🥇 `large-v3-turbo` |

---

### 🔊 Condition-by-Condition Acoustic Robustness Matrix ($n = 428$)

All 10 acoustic conditions ranked by difficulty. `whisper-large-v3-turbo` outperformed `whisper-small` across **every single condition** with zero confidence interval overlap.

| Acoustic Condition | Description | `whisper-small` WER | `whisper-large-v3-turbo` WER | Absolute Gain ($\Delta$) | Utterances Where Large Wins |
|---|---|:---:|:---:|:---:|:---:|
| **`clean`** | Original studio-quality test set | 26.3% | **10.1%** | **-16.2 pts** | **86.9%** |
| **`low_volume_-25dB`** | Severe attenuation (-25 dB gain) | 26.6% | **10.2%** | **-15.9 pts** | **86.4%** |
| **`clean_auto_LID`** | Language auto-detection (no forced tag) | 26.3% | **10.4%** | **-15.8 pts** | **86.7%** |
| **`white_20dB`** | Mild synthetic white noise (SNR = 20 dB) | 29.7% | **11.0%** | **-18.8 pts** | **88.1%** |
| **`telephone`** | 300–3400 Hz bandpass + 8 kHz + $\mu$-law | 29.7% | **11.3%** | **-18.0 pts** | **90.7%** |
| **`white_10dB`** | Moderate white noise (SNR = 10 dB) | 40.6% | **16.3%** | **-24.3 pts** | **94.4%** |
| **`reverb_0.5s`** | Room reverberation impulse (RT60 = 0.5s) | 46.5% | **18.8%** | **-27.4 pts** | **94.9%** |
| **`speed_0.85`** | Slow speech (15% slower rate) | 54.8% | **23.8%** | **-29.7 pts** | **98.8%** |
| **`white_5dB`** | Severe white noise (SNR = 5 dB) | 51.4% | **25.6%** | **-25.7 pts** | **94.6%** |
| **`speed_1.15`** | Fast speech (15% faster rate) | 83.2% | **40.5%** | **-40.6 pts** | **96.7%** |

---

### 🔍 Key Diagnostic Insights & Production Recommendations

1. **No Speed vs. Accuracy Trade-Off**:
   - `whisper-large-v3-turbo` achieved **3.5× higher throughput** (`RTF = 0.009` vs `0.033`) while cutting WER by more than half. Its architectural optimizations (fewer decoder layers, pruned cross-attention) make it strictly superior in latency and accuracy.
2. **Resilience Grows Under Pressure**:
   - The accuracy gap between models widens dramatically under adverse conditions: `-16.2 pts` on clean audio vs **`-40.6 pts` on fast speech**, **`-29.7 pts` on slow speech**, and **`-27.4 pts` under reverberation**.
3. **Telephony Readiness**:
   - Telephony compression degraded WER by only **+1.2 pts** on `large-v3-turbo` (10.1% $\to$ 11.3%), confirming strong viability for call-center and VoIP pipelines.
4. **Where Large Still Fails (Failure-Mode Analysis)**:
   - **Explosive Repetition Loops**: Under extreme speech rates (`speed_1.15`), a single utterance (`...بعد حدث الأربعاء تنافس كاربانيدو`) triggered a severe repetitive decoding loop (`أن يطعأ أن يطعأ...`), reaching a WER of 17.7 (1,770%). *Mitigation:* Integrate our `flag_repetition` heuristic as an early-stop or temperature-fallback guardrail.
   - **Written Numbers vs. Digits**: Discrepancies like `"الخمس والعشرون"` vs `"25"` were counted as errors by strict word alignment. This is an orthographic normalization policy difference rather than an acoustic failure.
   - **Foreign Proper Names**: Both models struggle with non-Arabic technical acronyms and named entities (e.g., `"802.11a"` transcribed as `"102.11 تم"`).

**Final Verdict**: **`whisper-large-v3-turbo` is the clear choice for production.** Pair it with repetition-loop detection in post-processing for optimal reliability.

---

## 🚀 Quick Start

### 1. Using Google Colab (Recommended for GPU)
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`stt_eval_notebook.ipynb`](./stt_eval_notebook.ipynb).
3. Set your runtime to GPU: `Runtime` ➔ `Change runtime type` ➔ **T4 GPU**.
4. Run the cells step-by-step to:
   - Install dependencies.
   - Choose your model (HuggingFace, local, or custom).
   - Upload your data directly (`ZIP` or `CSV` + audio).
   - Run clean and robustness evaluations.
   - Inspect audio playback for errors and download your CSV/JSON reports.

### 2. Local Installation & Setup

Clone the repository and install the requirements:

```bash
git clone https://github.com/joalsebaey/STT-EVAL.git
cd STT-EVAL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify the installation (23 unit tests)
pytest tests -q
```

---

## 💻 CLI Usage (`run_eval.py`)

### Evaluate with Built-in FLEURS Dataset
```bash
# Quick test on 100 samples with Whisper Small
python run_eval.py --models openai/whisper-small --max-samples 100

# Full benchmark comparing two models
python run_eval.py --models openai/whisper-small openai/whisper-large-v3-turbo --max-samples 0

# Test specific robustness conditions
python run_eval.py --models openai/whisper-small --conditions clean telephone white_10dB reverb_0.5s
```

### Evaluate on Your Custom Dataset
```bash
# CSV metadata + audio folder
python run_eval.py \
  --models openai/whisper-small \
  --data ./my_dataset/metadata.csv \
  --audio-dir ./my_dataset/wavs/ \
  --max-samples 500

# Include real background noise files (e.g., street noise, cafe)
python run_eval.py \
  --models openai/whisper-small \
  --data ./my_dataset.zip \
  --noise-dir ./ambient_noises/
```

### CLI Arguments Overview
| Flag | Description | Default |
|---|---|---|
| `--models` | One or more HuggingFace model IDs or local checkpoint paths | `openai/whisper-small` |
| `--data` | Path to custom data (`.csv`, `.tsv`, `.json`, `.jsonl`, folder, or `.zip`) | `fleurs` |
| `--audio-dir` | Directory containing audio files (if relative paths in CSV) | `None` |
| `--conditions` | Robustness conditions to run (`clean`, `white_10dB`, `telephone`, etc.) | *All conditions* |
| `--noise-dir` | Folder containing real noise audio files to mix | `None` |
| `--max-samples` | Limit number of evaluated samples (`0` = all samples) | `100` |
| `--batch-size` | Batch size for inference | `16` |
| `--seed` | Random seed for sampling and perturbations | `42` |

---

## 📁 Custom Data Format Guidelines

The data loader (`stt_eval/dataset.py`) automatically maps column names. You can supply your data in any of the following formats:

### Supported Table Columns (CSV / TSV / JSON / JSONL)
- **Audio path**: `audio`, `audio_path`, `path`, `file`, `wav_path`, `file_name`, `filename`, `audio_file`, `wav`
- **Transcription**: `transcription`, `reference`, `text`, `ref`, `sentence`, `ground_truth`, `label`, `transcript`
- **Identifier** *(Optional)*: `id`, `uid`, `utterance_id`, `sample_id`, `file_id`, `name`
- **Speaker Gender** *(Optional)*: `gender`, `sex`, `speaker_gender`

### Example `metadata.csv`:
```csv
audio_path,transcription
wavs/sample_01.wav,ذهب الطالب إلى الجامعة صباح اليوم
wavs/sample_02.wav,سجلت الشركة نموا ملحوظا في الأرباح
```

### Or Folder Format:
Put audio files and transcripts with identical stems in one folder:
```
my_audio_folder/
├── sample_01.wav
├── sample_01.txt
├── sample_02.wav
└── sample_02.txt
```

---

## 📦 Output Artifacts

All evaluation artifacts are saved under `outputs/<model_name>/`:

| File | Description |
|---|---|
| `summary_by_condition.csv` | Summary table with WER, CER, 95% CI, S/D/I rates, RTF, and flag rates per condition. |
| `utterances_<condition>.csv` | Per-utterance details: reference, prediction, error counts, failure flags, and audio features. |
| `slices_clean.csv` | Sliced performance by gender and utterance duration. |
| `worst_30_clean.csv` | The 30 worst-performing utterances (highest WER) for manual inspection. |
| `top_confusions_clean.csv` | Most frequent word substitutions, deletions, and insertions. |
| `report.json` | Reproducible configuration snapshot (system info, seed, normalization rules, summary metrics). |
| `*_wer_by_condition.png` | Bar charts comparing error rates across acoustic conditions. |

---

## 🗂️ Project Architecture

```
STT-EVAL/
├── stt_eval/
│   ├── audio.py          # Acoustic features and perturbation engines (white noise, telephone, reverb...)
│   ├── dataset.py        # Universal data loader (CSV/JSON/ZIP/Folder, 16kHz mono resampling)
│   ├── evaluator.py      # High-level evaluation runner (EvalResult container)
│   ├── flags.py          # Failure flags detection (empty, repetition, length, foreign scripts)
│   ├── metrics.py        # jiwer word/character error scoring, bootstrap CI, top confusions
│   ├── normalize.py      # Configurable Arabic text normalizer (diacritics, alef, ya, digits)
│   └── pipeline.py       # Whisper inference engine & evaluation loops
├── tests/
│   ├── test_core.py      # Core metrics, audio perturbations, and normalization unit tests
│   └── test_dataset.py   # Dataset ingestion and file format compatibility tests
├── run_eval.py           # CLI evaluation runner
├── stt_eval_notebook.ipynb # Interactive Jupyter / Google Colab evaluation notebook
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (English)
```

---

## 🧪 Unit Testing

Run the test suite to verify pipeline integrity:

```bash
pytest tests -v
```

All 23 tests cover:
- Arabic text normalization rules and edge cases.
- Levenshtein alignment and substitution/deletion/insertion rates.
- Bootstrap confidence interval convergence.
- Audio perturbation SNR precision and shape preservation.
- File parser robustness across CSV, TSV, JSON, JSONL, folders, and ZIP archives.

---

## 📜 Arabic Documentation
The original Arabic documentation is available in [`README.ar.md`](./README.ar.md).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
