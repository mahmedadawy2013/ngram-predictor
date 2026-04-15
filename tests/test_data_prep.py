import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_prep.normalizer import Normalizer


@pytest.fixture
def normalizer():
    return Normalizer()


class TestLowercase:
    def test_lowercases_text(self, normalizer):
        assert normalizer.lowercase("Hello WORLD") == "hello world"

    def test_already_lowercase(self, normalizer):
        assert normalizer.lowercase("hello") == "hello"


class TestRemovePunctuation:
    def test_removes_punctuation(self, normalizer):
        assert normalizer.remove_punctuation("hello, world!") == "hello world"

    def test_no_punctuation(self, normalizer):
        assert normalizer.remove_punctuation("hello world") == "hello world"


class TestRemoveNumbers:
    def test_removes_numbers(self, normalizer):
        assert normalizer.remove_numbers("chapter 12 begins") == "chapter  begins"

    def test_no_numbers(self, normalizer):
        assert normalizer.remove_numbers("hello world") == "hello world"


class TestRemoveWhitespace:
    def test_collapses_spaces(self, normalizer):
        assert normalizer.remove_whitespace("hello   world") == "hello world"

    def test_strips_edges(self, normalizer):
        assert normalizer.remove_whitespace("  hello  ") == "hello"


class TestNormalize:
    def test_full_pipeline(self, normalizer):
        result = normalizer.normalize("Hello, World! 123")
        assert result == "hello world"

    def test_sequence(self, normalizer):
        text = "  The 3rd TIME!  "
        result = normalizer.normalize(text)
        assert result == "the rd time"
        assert result == result.lower()
        assert "3" not in result
        assert "!" not in result


class TestStripGutenberg:
    def test_removes_header_and_footer(self, normalizer):
        text = (
            "Preamble\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
            "Actual content here.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
            "Postamble"
        )
        result = normalizer.strip_gutenberg(text)
        assert "Actual content here." in result
        assert "START OF THE PROJECT GUTENBERG" not in result
        assert "END OF THE PROJECT GUTENBERG" not in result

    def test_no_markers(self, normalizer):
        text = "Just some plain text."
        assert normalizer.strip_gutenberg(text) == text


class TestSentenceTokenize:
    def test_returns_list(self, normalizer):
        result = normalizer.sentence_tokenize("Hello world.")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_splits_sentences(self, normalizer):
        result = normalizer.sentence_tokenize("First. Second! Third?")
        assert len(result) == 3


class TestWordTokenize:
    def test_returns_list_of_strings(self, normalizer):
        result = normalizer.word_tokenize("hello world")
        assert result == ["hello", "world"]

    def test_no_empty_tokens(self, normalizer):
        result = normalizer.word_tokenize("hello world")
        assert all(len(t) > 0 for t in result)

    def test_empty_input(self, normalizer):
        result = normalizer.word_tokenize("")
        assert result == [] or result == [""]


class TestLoad:
    def test_folder_not_found(self, normalizer):
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            normalizer.load("/nonexistent/path/to/folder")
