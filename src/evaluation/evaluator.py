import math
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Computes perplexity on a held-out evaluation corpus using an NGramModel
    with backoff lookup and a Normalizer for text preprocessing.
    """

    def __init__(self, model, normalizer):
        """
        Accept a pre-loaded NGramModel and Normalizer instance.

        Parameters:
            model: A pre-loaded NGramModel instance.
            normalizer: A Normalizer instance for text preprocessing.
        """
        self.model = model
        self.normalizer = normalizer

    def score_word(self, word, context):
        """
        Return log2 P(word | context) via NGramModel.lookup().

        Parameters:
            word (str): The target word to score.
            context (list[str]): The preceding context words.

        Returns:
            float or None: log2 probability, or None if zero probability at all orders.
        """
        candidates = self.model.lookup(context)
        if not candidates:
            return None

        mapped_word = word if word in self.model.vocab else '<UNK>'
        prob = candidates.get(mapped_word)
        if prob is None or prob == 0:
            return None

        return math.log2(prob)

    def compute_perplexity(self, eval_file):
        """
        Compute perplexity over the full eval corpus.

        Parameters:
            eval_file (str): Path to the tokenized evaluation file.

        Returns:
            tuple: (perplexity, words_evaluated, words_skipped)
        """
        logger.info("Computing perplexity on %s", eval_file)
        total_log_prob = 0.0
        words_evaluated = 0
        words_skipped = 0

        with open(eval_file, 'r') as f:
            for line in f:
                words = line.strip().split()
                mapped = [w if w in self.model.vocab else '<UNK>' for w in words]
                for i in range(len(mapped)):
                    context = mapped[:i]
                    context = context[-(self.model.ngram_order - 1):]
                    score = self.score_word(mapped[i], context)
                    if score is None:
                        words_skipped += 1
                        logger.debug("Skipped word '%s' at position %d", mapped[i], i)
                    else:
                        total_log_prob += score
                        words_evaluated += 1

        if words_evaluated == 0:
            logger.error("No words evaluated — cannot compute perplexity")
            return float('inf'), 0, words_skipped

        total_words = words_evaluated + words_skipped
        skip_ratio = words_skipped / total_words if total_words > 0 else 0
        if skip_ratio > 0.2:
            logger.warning(
                "%.1f%% of words were skipped (zero probability) — perplexity may be unreliable",
                skip_ratio * 100
            )

        cross_entropy = -total_log_prob / words_evaluated
        perplexity = 2 ** cross_entropy

        logger.info("Perplexity: %.2f", perplexity)
        logger.info("Words evaluated: %d", words_evaluated)
        logger.info("Words skipped (zero probability): %d", words_skipped)

        return perplexity, words_evaluated, words_skipped

    def run(self, eval_file):
        """
        Orchestrate compute_perplexity and print results.

        Parameters:
            eval_file (str): Path to the tokenized evaluation file.
        """
        perplexity, evaluated, skipped = self.compute_perplexity(eval_file)
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Words evaluated: {evaluated:,}")
        print(f"Words skipped (zero probability): {skipped:,}")
