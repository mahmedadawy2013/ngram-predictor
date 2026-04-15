import os
import argparse
from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

load_dotenv('config/.env')

TRAIN_RAW_DIR     = os.environ.get('TRAIN_RAW_DIR')
EVAL_RAW_DIR      = os.environ.get('EVAL_RAW_DIR')
TRAIN_TOKENS_FILE = os.environ.get('TRAIN_TOKENS')
EVAL_TOKENS_FILE  = os.environ.get('EVAL_TOKENS')
MODEL_PATH        = os.environ.get('MODEL')
VOCAB_PATH        = os.environ.get('VOCAB')


def step_dataprep(normalizer):
    """Run data preparation: normalize raw text and produce token files."""
    def _process(raw_dir, output_file):
        text = normalizer.load(raw_dir)
        text = normalizer.strip_gutenberg(text)
        text = normalizer.lowercase(text)
        text = normalizer.remove_numbers(text)
        text = normalizer.remove_whitespace(text)
        sentences = normalizer.sentence_tokenize(text)
        tokenized_sentences = []
        for sent in sentences:
            sent = normalizer.remove_punctuation(sent)
            sent = normalizer.remove_whitespace(sent)
            tokens = normalizer.word_tokenize(sent)
            if tokens:
                tokenized_sentences.append(tokens)
        normalizer.save(tokenized_sentences, output_file)

    print("Running data preparation...")
    _process(TRAIN_RAW_DIR, TRAIN_TOKENS_FILE)
    print(f"  Saved training tokens to {TRAIN_TOKENS_FILE}")
    if os.path.exists(EVAL_RAW_DIR):
        _process(EVAL_RAW_DIR, EVAL_TOKENS_FILE)
        print(f"  Saved eval tokens to {EVAL_TOKENS_FILE}")


def step_model(model):
    """Build vocabulary, counts, probabilities, and save model files."""
    print("Building vocabulary...")
    model.build_vocab(TRAIN_TOKENS_FILE)
    print(f"  Vocabulary size: {len(model.vocab_list)}")

    print("Building counts and probabilities...")
    model.build_counts_and_probabilities(TRAIN_TOKENS_FILE)
    for order in range(1, model.ngram_order + 1):
        key = f"{order}gram"
        entries = model.model.get(key, {})
        if order == 1:
            print(f"  {key}: {len(entries)} entries")
        else:
            print(f"  {key}: {len(entries)} contexts")

    model.save_model(MODEL_PATH)
    print(f"  Saved model to {MODEL_PATH}")
    model.save_vocab(VOCAB_PATH)
    print(f"  Saved vocab to {VOCAB_PATH}")


def step_inference(model, normalizer):
    """Start the interactive CLI prediction loop."""
    if not model.model:
        model.load(MODEL_PATH, VOCAB_PATH)

    predictor = Predictor(model, normalizer)
    print("Type a sequence of words and press Enter to get predictions.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    try:
        while True:
            text = input("> ").strip()
            if text.lower() == "quit":
                print("Goodbye.")
                break
            if not text:
                continue
            predictions = predictor.predict_next(text)
            print(f"Predictions: {predictions}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


def main():
    parser = argparse.ArgumentParser(description="N-gram Predictor CLI")
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        default="all",
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    normalizer = Normalizer()
    model = NGramModel()

    if args.step in ("dataprep", "all"):
        step_dataprep(normalizer)

    if args.step in ("model", "all"):
        step_model(model)

    if args.step in ("inference", "all"):
        step_inference(model, normalizer)


if __name__ == '__main__':
    main()