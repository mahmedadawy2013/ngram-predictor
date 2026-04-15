import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('NGRAM_ORDER', '3')
os.environ.setdefault('UNK_THRESHOLD', '2')

from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.evaluation.evaluator import Evaluator


@pytest.fixture
def trained_model(tmp_path):
    f = tmp_path / "tokens.txt"
    f.write_text(
        "the cat sat on the mat\n"
        "the cat sat on the cat\n"
        "the dog sat on the mat\n"
        "the cat chased the dog\n"
    )
    m = NGramModel()
    m.ngram_order = 3
    m.unk_threshold = 2
    m.build_vocab(str(f))
    m.build_counts_and_probabilities(str(f))
    return m


@pytest.fixture
def evaluator(trained_model):
    return Evaluator(trained_model, Normalizer())


@pytest.fixture
def eval_file(tmp_path):
    f = tmp_path / "eval.txt"
    f.write_text(
        "the cat sat on the mat\n"
        "the dog sat on the cat\n"
    )
    return str(f)


class TestScoreWord:
    def test_seen_word_returns_negative_float(self, evaluator):
        score = evaluator.score_word("cat", ["the"])
        assert isinstance(score, float)
        assert score < 0

    def test_zero_probability_returns_none(self):
        m = NGramModel()
        m.ngram_order = 3
        m.unk_threshold = 2
        m.model = {"1gram": {"the": 0.5, "cat": 0.5}, "2gram": {}, "3gram": {}}
        m.vocab = {"the", "cat"}
        evaluator = Evaluator(m, Normalizer())
        score = evaluator.score_word("dog", ["zzz"])
        assert score is None


class TestComputePerplexity:
    def test_returns_positive_float_greater_than_one(self, evaluator, eval_file):
        perplexity, evaluated, skipped = evaluator.compute_perplexity(eval_file)
        assert isinstance(perplexity, float)
        assert perplexity > 1.0

    def test_evaluated_count_positive(self, evaluator, eval_file):
        perplexity, evaluated, skipped = evaluator.compute_perplexity(eval_file)
        assert evaluated > 0
