import os
from dotenv import load_dotenv


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
        return [w if w in self.model.vocab else '<UNK>' for w in context]

    def predict_next(self, text, k=None):
        """
        Orchestrate normalize -> map_oov -> NGramModel.lookup() -> return top-k words.

        Parameters:
            text (str): Raw input text from the user.
            k (int, optional): Number of top predictions to return. Defaults to TOP_K from config/.env.

        Returns:
            list[str]: Top-k predicted words sorted by probability (highest first).
                       Empty list if no predictions found.
        """
        if k is None:
            k = self.top_k

        context = self.normalize(text)
        context = self.map_oov(context)
        candidates = self.model.lookup(context)

        if not candidates:
            return []

        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [word for word, prob in sorted_candidates[:k]]
