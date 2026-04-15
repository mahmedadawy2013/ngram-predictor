import os
import sys
import logging
import argparse
from dotenv import load_dotenv

load_dotenv('config/.env')

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator

REQUIRED_KEYS = ['TRAIN_RAW_DIR', 'EVAL_RAW_DIR', 'TRAIN_TOKENS', 'EVAL_TOKENS', 'MODEL', 'VOCAB']


def get_env(key):
    """Get an environment variable or raise KeyError with a helpful message."""
    value = os.environ.get(key)
    if value is None:
        logger.error("Missing config variable: %s", key)
        raise KeyError(f"Missing config variable: {key}. Check config/.env.")
    return value


TRAIN_RAW_DIR     = get_env('TRAIN_RAW_DIR')
EVAL_RAW_DIR      = get_env('EVAL_RAW_DIR')
TRAIN_TOKENS_FILE = get_env('TRAIN_TOKENS')
EVAL_TOKENS_FILE  = get_env('EVAL_TOKENS')
MODEL_PATH        = get_env('MODEL')
VOCAB_PATH        = get_env('VOCAB')


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

    logger.info("Running data preparation...")
    _process(TRAIN_RAW_DIR, TRAIN_TOKENS_FILE)
    logger.info("Saved training tokens to %s", TRAIN_TOKENS_FILE)
    if os.path.exists(EVAL_RAW_DIR):
        _process(EVAL_RAW_DIR, EVAL_TOKENS_FILE)
        logger.info("Saved eval tokens to %s", EVAL_TOKENS_FILE)


def step_model(model):
    """Build vocabulary, counts, probabilities, and save model files."""
    model.build_vocab(TRAIN_TOKENS_FILE)
    model.build_counts_and_probabilities(TRAIN_TOKENS_FILE)

    model.save_model(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    model.save_vocab(VOCAB_PATH)
    logger.info("Saved vocab to %s", VOCAB_PATH)


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
                print("Input text is empty. Please type at least one word.")
                continue
            try:
                predictions = predictor.predict_next(text)
                print(f"Predictions: {predictions}")
            except ValueError as e:
                logger.error("%s", e)
                print(f"Error: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


def step_evaluate(model, normalizer):
    """Run evaluation on the held-out corpus."""
    if not model.model:
        model.load(MODEL_PATH, VOCAB_PATH)

    evaluator = Evaluator(model, normalizer)
    evaluator.run(EVAL_TOKENS_FILE)


def main():
    parser = argparse.ArgumentParser(description="N-gram Predictor CLI")
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "evaluate", "all"],
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

    if args.step == "evaluate":
        step_evaluate(model, normalizer)


if __name__ == '__main__':
    main()