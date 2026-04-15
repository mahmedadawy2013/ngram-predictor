import os
import sys
import tempfile
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('NGRAM_ORDER', '3')
os.environ.setdefault('UNK_THRESHOLD', '2')

from src.model.ngram_model import NGramModel


@pytest.fixture
def token_file(tmp_path):
    f = tmp_path / "tokens.txt"
    f.write_text(
        "the cat sat on the mat\n"
        "the cat sat on the cat\n"
        "the dog sat on the mat\n"
        "a rare word appeared\n"
    )
    return str(f)


@pytest.fixture
def model(token_file):
    m = NGramModel()
    m.ngram_order = 3
    m.unk_threshold = 2
    m.build_vocab(token_file)
    m.build_counts_and_probabilities(token_file)
    return m


class TestBuildVocab:
    def test_unk_in_vocab(self, model):
        assert '<UNK>' in model.vocab

    def test_replaces_rare_words(self, model):
        assert 'rare' not in model.vocab
        assert 'word' not in model.vocab
        assert 'appeared' not in model.vocab

    def test_keeps_frequent_words(self, model):
        assert 'the' in model.vocab
        assert 'cat' in model.vocab
        assert 'sat' in model.vocab


class TestLookup:
    def test_seen_context_returns_nonempty(self, model):
        result = model.lookup(['the', 'cat'])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_unseen_context_falls_back_to_unigram(self, model):
        result = model.lookup(['xyzzy', 'qwerty'])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_orders_fail_returns_empty(self):
        m = NGramModel()
        m.ngram_order = 3
        m.unk_threshold = 2
        m.model = {"1gram": {}, "2gram": {}, "3gram": {}}
        m.vocab = set()
        result = m.lookup(['foo'])
        assert result == {}

    def test_probabilities_sum_to_one(self, model):
        result = model.lookup(['the'])
        if result:
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01


class TestLoadExceptions:
    def test_file_not_found(self):
        m = NGramModel()
        m.ngram_order = 3
        m.unk_threshold = 2
        with pytest.raises(FileNotFoundError, match="model.json not found"):
            m.load("/nonexistent/model.json", "/nonexistent/vocab.json")

    def test_malformed_json(self, tmp_path):
        model_f = tmp_path / "model.json"
        model_f.write_text("{bad json")
        vocab_f = tmp_path / "vocab.json"
        vocab_f.write_text("[]")
        m = NGramModel()
        m.ngram_order = 3
        m.unk_threshold = 2
        with pytest.raises(json.JSONDecodeError):
            m.load(str(model_f), str(vocab_f))
