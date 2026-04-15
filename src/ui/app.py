import logging

logger = logging.getLogger(__name__)


class PredictorUI:
    """Simple UI wrapper for predictor interaction."""

    def __init__(self, predictor):
        """
        Parameters:
            predictor: A Predictor instance.
        """
        self.predictor = predictor

    def get_predictions(self, text):
        """
        Get predictions for the given text.

        Parameters:
            text (str): User input text.

        Returns:
            list[str]: Top-k predicted words, or empty list on error.
        """
        if not text or not text.strip():
            return []
        try:
            return self.predictor.predict_next(text)
        except ValueError:
            return []

    def run(self) -> None:
        """Start the interactive CLI prediction loop."""
        print("Type a sequence of words and press Enter to get predictions.")
        print("Type 'quit' or press Ctrl+C to exit.\n")
        try:
            while True:
                text = input("> ").strip()
                if text.lower() == "quit":
                    print("Goodbye.")
                    break
                predictions = self.get_predictions(text)
                if not predictions:
                    print("No predictions found.")
                else:
                    print(f"Predictions: {predictions}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
