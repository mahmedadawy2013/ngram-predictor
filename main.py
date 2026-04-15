import os
from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer

# Load environment variables
load_dotenv('config/.env')

# Define paths from env
TRAIN_RAW_DIR      = os.environ.get('TRAIN_RAW_DIR')
EVAL_RAW_DIR       = os.environ.get('EVAL_RAW_DIR')
TRAIN_TOKENS_FILE  = os.environ.get('TRAIN_TOKENS')
EVAL_TOKENS_FILE   = os.environ.get('EVAL_TOKENS')

def process_data(raw_dir: str, output_file: str) -> None:
    """Process raw data from a directory and save tokenized output."""
    normalizer = Normalizer()
    
    # Load raw text
    text = normalizer.load(raw_dir)
    
    # Strip Gutenberg
    text = normalizer.strip_gutenberg(text)
    
    # Partial normalize: lowercase, remove numbers, remove extra whitespace (keep punctuation for sentence splitting)
    text = normalizer.lowercase(text)
    text = normalizer.remove_numbers(text)
    text = normalizer.remove_whitespace(text)
    
    # Sentence tokenize (uses punctuation)
    sentences = normalizer.sentence_tokenize(text)
    
    # For each sentence: remove punctuation, remove whitespace, word tokenize
    tokenized_sentences = []
    for sent in sentences:
        sent = normalizer.remove_punctuation(sent)
        sent = normalizer.remove_whitespace(sent)
        tokens = normalizer.word_tokenize(sent)
        if tokens:  # Skip empty sentences
            tokenized_sentences.append(tokens)
    
    # Save
    normalizer.save(tokenized_sentences, output_file)

if __name__ == '__main__':
    # Process training data
    process_data(TRAIN_RAW_DIR, TRAIN_TOKENS_FILE)
    
    # Process evaluation data if eval folder exists
    if os.path.exists(EVAL_RAW_DIR):
        process_data(EVAL_RAW_DIR, EVAL_TOKENS_FILE)