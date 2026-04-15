import os
import json
from collections import defaultdict, Counter
from dotenv import load_dotenv


class NGramModel:
    """
    N-gram language model implementation with backoff.

    This class builds, stores, and exposes n-gram probability tables and backoff lookup
    across all orders from 1 up to NGRAM_ORDER as specified in config/.env.
    """

    def __init__(self):
        load_dotenv('config/.env')
        self.ngram_order = int(os.environ.get('NGRAM_ORDER', 4))
        self.unk_threshold = int(os.environ.get('UNK_THRESHOLD', 3))
        self.vocab = set()
        self.vocab_list = []
        self.model = {}
        self.counts = {}

    def build_vocab(self, token_file):
        """
        Build vocabulary from a token file. Words appearing fewer than
        UNK_THRESHOLD times are replaced with <UNK>.

        Parameters:
            token_file (str): Path to the tokenized text file, one sentence per line.

        Returns:
            None. Sets self.vocab (set) and self.vocab_list (list).
        """
        word_counts = Counter()
        with open(token_file, 'r') as f:
            for line in f:
                words = line.strip().split()
                word_counts.update(words)

        self.vocab = set()
        for word, count in word_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(word)
        self.vocab.add('<UNK>')
        self.vocab_list = sorted(self.vocab)

    def _map_word(self, word):
        """Map a word to <UNK> if it is not in the vocabulary."""
        return word if word in self.vocab else '<UNK>'

    def _read_sentences(self, token_file):
        """Read token file and return list of sentences with UNK mapping applied."""
        sentences = []
        with open(token_file, 'r') as f:
            for line in f:
                words = line.strip().split()
                mapped = [self._map_word(w) for w in words]
                if mapped:
                    sentences.append(mapped)
        return sentences

    def build_counts_and_probabilities(self, token_file):
        """
        Count all n-grams at orders 1 through NGRAM_ORDER and compute MLE probabilities.

        Parameters:
            token_file (str): Path to the tokenized text file.

        Returns:
            None. Sets self.counts and self.model dicts keyed by "{order}gram".
        """
        sentences = self._read_sentences(token_file)

        self.counts = {}
        for order in range(1, self.ngram_order + 1):
            self.counts[order] = Counter()

        for sentence in sentences:
            for order in range(1, self.ngram_order + 1):
                for i in range(len(sentence) - order + 1):
                    ngram = tuple(sentence[i:i + order])
                    self.counts[order][ngram] += 1

        total_unigrams = sum(self.counts[1].values())

        self.model = {}
        for order in range(1, self.ngram_order + 1):
            key = f"{order}gram"
            self.model[key] = {}

            if order == 1:
                for (word,), count in self.counts[order].items():
                    self.model[key][word] = count / total_unigrams
            else:
                grouped = defaultdict(dict)
                for ngram, count in self.counts[order].items():
                    context = ngram[:-1]
                    word = ngram[-1]
                    context_count = self.counts[order - 1][context]
                    if context_count > 0:
                        grouped[' '.join(context)][word] = count / context_count
                self.model[key] = dict(grouped)

    def lookup(self, context):
        """
        Backoff lookup: try the highest-order context first, fall back to lower orders
        down to 1-gram. Return a dict of {word: probability} from the highest order
        that matches.

        Parameters:
            context (list[str]): List of preceding words.

        Returns:
            dict: {word: probability} from the first matching order, or empty dict.
        """
        mapped_context = [self._map_word(w) for w in context]

        for order in range(self.ngram_order, 0, -1):
            key = f"{order}gram"
            if order == 1:
                if self.model.get(key):
                    return dict(self.model[key])
            else:
                ctx_words = mapped_context[-(order - 1):]
                if len(ctx_words) < order - 1:
                    continue
                ctx_str = ' '.join(ctx_words)
                if ctx_str in self.model.get(key, {}):
                    return dict(self.model[key][ctx_str])

        return {}

    def save_model(self, model_path):
        """
        Save all probability tables to model.json.

        Parameters:
            model_path (str): Output path for model.json.

        Returns:
            None.
        """
        with open(model_path, 'w') as f:
            json.dump(self.model, f, indent=2)

    def save_vocab(self, vocab_path):
        """
        Save vocabulary list to vocab.json.

        Parameters:
            vocab_path (str): Output path for vocab.json.

        Returns:
            None.
        """
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab_list, f, indent=2)

    def load(self, model_path, vocab_path):
        """
        Load model.json and vocab.json into the instance.

        Parameters:
            model_path (str): Path to model.json.
            vocab_path (str): Path to vocab.json.

        Returns:
            None.
        """
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        with open(vocab_path, 'r') as f:
            self.vocab_list = json.load(f)
            self.vocab = set(self.vocab_list)
