import os
import re
import string
import logging

logger = logging.getLogger(__name__)

class Normalizer:
    """Text normalizer for preprocessing raw book text: loading, cleaning, tokenizing, and saving the corpus."""

    def load(self, folder_path: str) -> str:
        """Load all .txt files from a folder and concatenate their contents.

        Args:
            folder_path (str): Path to the folder containing .txt files.

        Returns:
            str: Concatenated text from all .txt files in the folder.

        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not os.path.isdir(folder_path):
            logger.error("Folder not found: %s", folder_path)
            raise FileNotFoundError(
                f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env."
            )
        text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    text += f.read() + "\n"
        return text

    def strip_gutenberg(self, text: str) -> str:
        """Remove Gutenberg header and footer from the text.

        Args:
            text (str): The raw text.

        Returns:
            str: Text with Gutenberg header and footer removed.
        """
        # Remove header
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
        text = re.sub(start_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove footer
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
        text = re.sub(end_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        return text

    def lowercase(self, text: str) -> str:
        """Lowercase all text.

        Args:
            text (str): The input text.

        Returns:
            str: Text converted to lowercase.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Remove all punctuation from the text.

        Args:
            text (str): The input text.

        Returns:
            str: Text with punctuation removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_numbers(self, text: str) -> str:
        """Remove all numbers from the text.

        Args:
            text (str): The input text.

        Returns:
            str: Text with numbers removed.
        """
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text: str) -> str:
        """Remove extra whitespace and blank lines.

        Args:
            text (str): The input text.

        Returns:
            str: Text with extra whitespace removed.
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove blank lines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """Apply all normalization steps in order: lowercase → remove punctuation → remove numbers → remove whitespace.

        Args:
            text (str): The input text.

        Returns:
            str: Normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text: str) -> list[str]:
        """Split text into a list of sentences.

        Args:
            text (str): The input text.

        Returns:
            list[str]: List of sentences.
        """
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def word_tokenize(self, sentence: str) -> list[str]:
        """Split a single sentence into a list of tokens.

        Args:
            sentence (str): A single sentence.

        Returns:
            list[str]: List of tokens.
        """
        return sentence.split()

    def save(self, sentences: list[list[str]], filepath: str) -> None:
        """Write tokenized sentences to output file.

        Args:
            sentences (list[list[str]]): List of tokenized sentences (each sentence is a list of tokens).
            filepath (str): Path to the output file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(' '.join(sentence) + '\n')
