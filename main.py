import os
from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel

# Load environment variables
load_dotenv('config/.env')

# Define paths from env
TRAIN_RAW_DIR      = os.environ.get('TRAIN_RAW_DIR')
EVAL_RAW_DIR       = os.environ.get('EVAL_RAW_DIR')
TRAIN_TOKENS_FILE  = os.environ.get('TRAIN_TOKENS')
EVAL_TOKENS_FILE   = os.environ.get('EVAL_TOKENS')
MODEL_PATH         = os.environ.get('MODEL')
VOCAB_PATH         = os.environ.get('VOCAB')

def process_data(raw_dir: str, output_file: str) -> None:
    """Process raw data from a directory and save tokenized output."""
    normalizer = Normalizer()

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

def train_model() -> None:
    """Build vocabulary, train n-gram model, and save outputs."""
    model = NGramModel()

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

    print(f"Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)

    print(f"Saving vocab to {VOCAB_PATH}...")
    model.save_vocab(VOCAB_PATH)

    print("Testing lookup...")
    results = model.lookup(["sherlock", "holmes", "said"])
    top = sorted(results.items(), key=lambda x: -x[1])[:5]
    for word, prob in top:
        print(f"  {word}: {prob:.4f}")

if __name__ == '__main__':
    # Process training data
    process_data(TRAIN_RAW_DIR, TRAIN_TOKENS_FILE)

    # Process evaluation data if eval folder exists
    if os.path.exists(EVAL_RAW_DIR):
        process_data(EVAL_RAW_DIR, EVAL_TOKENS_FILE)

    # Train and save model
    train_model()