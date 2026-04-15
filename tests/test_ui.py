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
from src.ui.app import PredictorUI


@pytest.fixture
def ui(tmp_path):
    f = tmp_path / "tokens.txt"
    f.write_text(
        "the cat sat on the mat\n"
        "the cat sat on the cat\n"
        "the dog sat on the mat\n"
    )
    m = NGramModel()
    m.ngram_order = 3
    m.unk_threshold = 2
    m.build_vocab(str(f))
    m.build_counts_and_probabilities(str(f))
    predictor = Predictor(m, Normalizer())
    return PredictorUI(predictor)


class TestGetPredictions:
    def test_returns_list_of_strings(self, ui):
        result = ui.get_predictions("the cat")
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)

    def test_handles_empty_input(self, ui):
        result = ui.get_predictions("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_handles_none_input(self, ui):
        result = ui.get_predictions(None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_handles_whitespace_only(self, ui):
        result = ui.get_predictions("   ")
        assert isinstance(result, list)
        assert len(result) == 0
