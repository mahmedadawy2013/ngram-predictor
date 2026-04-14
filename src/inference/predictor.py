class Predictor:
    """Predictor interface for generating next-token suggestions."""

    def __init__(self, model):
        self.model = model

    def predict_next(self, context: list[str]) -> str | None:
        return None
