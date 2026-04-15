import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('NGRAM_ORDER', '3')
os.environ.setdefault('UNK_THRESHOLD', '2')
os.environ.setdefault('TOP_K', '3')

from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor


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
def predictor(trained_model):
    return Predictor(trained_model, Normalizer())


class TestPredictNext:
    def test_returns_k_predictions_for_seen_context(self, predictor):
        result = predictor.predict_next("the cat", k=2)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_sorted_by_probability(self, predictor):
        result = predictor.predict_next("the cat", k=3)
        candidates = predictor.model.lookup(predictor.normalize("the cat"))
        sorted_words = sorted(candidates, key=lambda w: -candidates[w])
        for i, word in enumerate(result):
            assert word == sorted_words[i]

    def test_handles_all_oov_context(self, predictor):
        result = predictor.predict_next("xyzzy qwerty zzzz")
        assert isinstance(result, list)

    def test_empty_input_raises_valueerror(self, predictor):
        with pytest.raises(ValueError, match="Input text is empty"):
            predictor.predict_next("")

    def test_whitespace_only_raises_valueerror(self, predictor):
        with pytest.raises(ValueError, match="Input text is empty"):
            predictor.predict_next("   ")


class TestMapOov:
    def test_known_words_unchanged(self, predictor):
        result = predictor.map_oov(["the", "cat"])
        assert result == ["the", "cat"]

    def test_unknown_words_replaced(self, predictor):
        result = predictor.map_oov(["xyzzy", "the"])
        assert result[0] == "<UNK>"
        assert result[1] == "the"
