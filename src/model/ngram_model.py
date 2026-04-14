class NGramModel:
    """N-gram language model implementation."""

    def __init__(self, n: int = 3):
        self.n = n
        self.model = {}

    def train(self, tokens: list[str]) -> None:
        pass

    def predict(self, context: tuple[str, ...]) -> str | None:
        return None
