import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Predictor:
    """
    Accepts a pre-loaded NGramModel and Normalizer via the constructor,
    normalizes input text, and returns the top-k predicted next words
    sorted by probability. Backoff lookup is delegated to NGramModel.lookup().
    """

    def __init__(self, model, normalizer):
        """
        Accept a pre-loaded NGramModel and Normalizer instance.

        Parameters:
            model (NGramModel): A pre-loaded n-gram model.
            normalizer (Normalizer): A Normalizer instance for text preprocessing.
        """
        load_dotenv('config/.env')
        self.model = model
        self.normalizer = normalizer
        self.top_k = int(os.environ.get('TOP_K', 3))

    def normalize(self, text):
        """
        Normalize the input text and extract the last NGRAM_ORDER-1 words as context.

        Parameters:
            text (str): Raw input text from the user.

        Returns:
            list[str]: The last NGRAM_ORDER-1 words from the normalized text.
        """
        normalized = self.normalizer.normalize(text)
        words = normalized.split()
        context_len = self.model.ngram_order - 1
        return words[-context_len:] if len(words) >= context_len else words

    def map_oov(self, context):
        """
        Replace out-of-vocabulary words with <UNK>.

        Parameters:
            context (list[str]): List of context words.

        Returns:
            list[str]: Context with OOV words replaced by <UNK>.
        """
        mapped = []
        for w in context:
            if w in self.model.vocab:
                mapped.append(w)
            else:
                logger.warning("OOV word encountered: '%s' -> <UNK>", w)
                mapped.append('<UNK>')
        return mapped

    def predict_next(self, text, k=None):
        """
        Orchestrate normalize -> map_oov -> NGramModel.lookup() -> return top-k words.

        Parameters:
            text (str): Raw input text from the user.
            k (int, optional): Number of top predictions to return. Defaults to TOP_K from config/.env.

        Returns:
            list[str]: Top-k predicted words sorted by probability (highest first).
                       Empty list if no predictions found.

        Raises:
            ValueError: If input text is empty.
        """
        if k is None:
            k = self.top_k

        if not text or not text.strip():
            logger.error("Empty input text received")
            raise ValueError("Input text is empty. Please type at least one word.")

        context = self.normalize(text)
        context = self.map_oov(context)
        candidates = self.model.lookup(context)

        if not candidates:
            return []

        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [word for word, prob in sorted_candidates[:k]]
