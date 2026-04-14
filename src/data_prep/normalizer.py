class Normalizer:
    """Text normalizer for preprocessing raw book text."""

    def normalize(self, text: str) -> str:
        return text.lower().strip()
