# NGram Project

Project structure for N-gram modeling.

## Configuration

Create a `config/.env` file with the following variables:

- `TRAIN_RAW_DIR`: Path to the directory containing training .txt files (e.g., `data/raw/train/`)
- `EVAL_RAW_DIR`: Path to the directory containing evaluation .txt files (e.g., `data/raw/eval/`)
- `TRAIN_TOKENS`: Path to the output file for training tokens (e.g., `data/processed/train_tokens.txt`)
- `EVAL_TOKENS`: Path to the output file for evaluation tokens (e.g., `data/processed/eval_tokens.txt`)
- `MODEL`: Path to the model JSON file (e.g., `data/model/model.json`)
- `VOCAB`: Path to the vocabulary JSON file (e.g., `data/model/vocab.json`)
- `UNK_THRESHOLD`: Threshold for unknown words (e.g., `3`)
- `TOP_K`: Number of top predictions to return (e.g., `3`)
- `NGRAM_ORDER`: Order of the n-gram model (e.g., `4`)

Example `.env` file:

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
```

## Modules

### Module 1 — Data Preparation (`src/data_prep/normalizer.py`)
Loads raw `.txt` files, strips Gutenberg headers/footers, normalizes text (lowercase, remove punctuation/numbers/whitespace), tokenizes into sentences and words, and saves to `train_tokens.txt` / `eval_tokens.txt`.

### Module 2 — Model (`src/model/ngram_model.py`)
Builds n-gram probability tables (1-gram through NGRAM_ORDER-gram) using MLE with stupid backoff. Vocabulary is built with an `UNK_THRESHOLD`; rare words are replaced with `<UNK>`. The `lookup()` method tries the highest-order context first and falls back to lower orders. Outputs `model.json` and `vocab.json`.

### Module 3 — Inference (`src/inference/predictor.py`)
Accepts a pre-loaded `NGramModel` and `Normalizer`, normalizes user input, maps OOV words to `<UNK>`, delegates backoff lookup to the model, and returns the top-k predicted next words sorted by probability.

### Module 4 — CLI (`main.py`)
Single entry point with `--step` argument to run individual pipeline stages or the full pipeline.

### Evaluator (`src/evaluation/evaluator.py`)
Computes perplexity on a held-out evaluation corpus using cross-entropy: H = -(1/N) Σ log₂ P(wᵢ | context), Perplexity = 2^H. Warns if >20% of words are skipped due to zero probability.

### UI (`src/ui/app.py`)
`PredictorUI` wraps `Predictor` with an interactive CLI loop and a `get_predictions()` method.

## Running

`main.py` accepts a `--step` argument to control which pipeline stage to run:

| Command | What it runs |
|---------|-------------|
| `python main.py --step dataprep` | Normalize raw text → produce `train_tokens.txt` |
| `python main.py --step model` | Build n-gram model → produce `model.json`, `vocab.json` |
| `python main.py --step inference` | Start interactive CLI prediction loop |
| `python main.py --step evaluate` | Compute perplexity on the eval corpus |
| `python main.py --step all` | Run dataprep → model → inference in sequence |

### Example Session

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

## Logging

Set `LOG_LEVEL` in `config/.env` to control verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

## Exception Handling

Structured exception handling with specific error messages for missing files, malformed JSON, missing config variables, and empty input.

## Testing

```bash
pytest tests/
```

42 tests covering Normalizer, NGramModel, Predictor, PredictorUI, and Evaluator.
