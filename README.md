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
