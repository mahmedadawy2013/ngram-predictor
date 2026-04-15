# N-Gram Next-Word Predictor

An n-gram language model that predicts the next word in a sequence using Maximum Likelihood Estimation (MLE) with stupid backoff. The model is trained on Project Gutenberg texts (Sherlock Holmes corpus) and supports configurable n-gram orders, vocabulary thresholding, and perplexity evaluation on a held-out corpus.

## Requirements

- **Python** 3.10+
- Dependencies listed in `requirements.txt` (install via pip — see Setup below)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ngram-predictor
```

### 2. Create and activate an Anaconda environment

```bash
conda create -n ngram python=3.12 -y
conda activate ngram
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure `config/.env`

Create (or verify) the file `config/.env` with the following variables:

```
TRAIN_RAW_DIR=data/raw/train/
EVAL_RAW_DIR=data/raw/eval/
TRAIN_TOKENS=data/processed/train_tokens.txt
EVAL_TOKENS=data/processed/eval_tokens.txt
MODEL=data/model/model.json
VOCAB=data/model/vocab.json
UNK_THRESHOLD=3
TOP_K=3
NGRAM_ORDER=4
LOG_LEVEL=INFO
```

| Variable | Description |
|---|---|
| `TRAIN_RAW_DIR` | Folder containing training `.txt` files |
| `EVAL_RAW_DIR` | Folder containing evaluation `.txt` files |
| `TRAIN_TOKENS` | Output path for tokenized training sentences |
| `EVAL_TOKENS` | Output path for tokenized evaluation sentences |
| `MODEL` | Output path for the trained model (`model.json`) |
| `VOCAB` | Output path for the vocabulary (`vocab.json`) |
| `UNK_THRESHOLD` | Minimum word count; rarer words become `<UNK>` |
| `TOP_K` | Number of top predictions returned |
| `NGRAM_ORDER` | Maximum n-gram order (e.g. 4 = up to 4-grams) |
| `LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### 5. Download raw text files

Place Project Gutenberg `.txt` files into the appropriate folders:

- **Training corpus** → `data/raw/train/` (e.g. Sherlock Holmes novels)
- **Evaluation corpus** → `data/raw/eval/` (e.g. *The Valley of Fear*, Gutenberg ID 3289)

## Usage

`main.py` is the single entry point. Use `--step` to run individual pipeline stages or the full pipeline:

```bash
# Step 1 — Data Preparation: normalize raw text → produce token files
python main.py --step dataprep

# Step 2 — Model Training: build vocab, count n-grams, compute MLE probabilities
python main.py --step model

# Step 3 — Interactive Inference: start the CLI prediction loop
python main.py --step inference

# Step 4 — Evaluation: compute perplexity on the held-out eval corpus
python main.py --step evaluate

# Run the full pipeline (dataprep → model → inference)
python main.py --step all
```

### Interactive Inference Example

```
$ python main.py --step inference
Type a sequence of words and press Enter to get predictions.
Type 'quit' or press Ctrl+C to exit.

> holmes looked at
Predictions: ['blessington']
> the game is
Predictions: ['up']
> quit
Goodbye.
```

### Running the UI

The `PredictorUI` class in `src/ui/app.py` provides a wrapper around the `Predictor`. To launch the interactive UI directly:

```python
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.ui.app import PredictorUI

model = NGramModel()
model.load("data/model/model.json", "data/model/vocab.json")
predictor = Predictor(model, Normalizer())
ui = PredictorUI(predictor)
ui.run()
```

### Running Tests

All tests are in the `tests/` folder and use **pytest**:

```bash
# Run all 42 tests
pytest tests/

# Verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_model.py -v
```

| Test file | Module tested | Coverage |
|---|---|---|
| `test_data_prep.py` | `Normalizer` | lowercase, punctuation, numbers, whitespace, normalize sequence, strip_gutenberg, sentence/word tokenize, FileNotFoundError |
| `test_model.py` | `NGramModel` | build_vocab with UNK, lookup (seen/unseen/empty), probabilities sum to 1, load FileNotFoundError, load JSONDecodeError |
| `test_inference.py` | `Predictor` | predict_next returns k results, sorted by probability, OOV context, empty input ValueError, map_oov |
| `test_ui.py` | `PredictorUI` | get_predictions returns strings, handles empty/None/whitespace input |
| `test_evaluation.py` | `Evaluator` | score_word negative float, score_word None for zero prob, perplexity > 1, evaluated count > 0 |

## Project Structure

```
ngram-predictor/
├── config/
│   └── .env                          # Environment variables
├── data/
│   ├── model/
│   │   ├── model.json                # Trained n-gram probability tables
│   │   └── vocab.json                # Vocabulary list
│   ├── processed/
│   │   ├── eval_tokens.txt           # Tokenized evaluation sentences
│   │   └── train_tokens.txt          # Tokenized training sentences
│   └── raw/
│       ├── eval/                     # Raw evaluation .txt files
│       └── train/                    # Raw training .txt files
├── src/
│   ├── data_prep/
│   │   └── normalizer.py             # Text loading, cleaning, tokenization
│   ├── evaluation/
│   │   └── evaluator.py              # Perplexity computation on eval corpus
│   ├── inference/
│   │   └── predictor.py              # Normalize → map OOV → lookup → top-k
│   ├── model/
│   │   └── ngram_model.py            # Vocab, n-gram counts, MLE probs, backoff
│   └── ui/
│       └── app.py                    # PredictorUI interactive CLI wrapper
├── tests/
│   ├── test_data_prep.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_ui.py
├── main.py                           # Single entry point (--step argument)
├── README.md
└── requirements.txt
```

## Logging

Controlled via `LOG_LEVEL` in `config/.env`:

| Level | What is logged |
|---|---|
| `DEBUG` | Every n-gram count, every backoff step during lookup |
| `INFO` | Module start/end, vocab size, total tokens, model save/load |
| `WARNING` | OOV words encountered at inference time, high skip ratio in evaluation |
| `ERROR` | Caught exceptions (missing files, malformed JSON, empty input) |

## Exception Handling

Structured exception handling with specific error messages — no bare `except:` clauses:

| Location | Exception | Message |
|---|---|---|
| `Normalizer.load()` | `FileNotFoundError` | Folder not found: {path}. Check TRAIN_RAW_DIR in config/.env. |
| `NGramModel.load()` | `FileNotFoundError` | model.json not found. Run the Model module first. |
| `NGramModel.load()` | `json.JSONDecodeError` | model.json is malformed. Re-run the Model module. |
| `main()` env loading | `KeyError` | Missing config variable: {key}. Check config/.env. |
| `Predictor.predict_next()` | `ValueError` | Input text is empty. Please type at least one word. |
