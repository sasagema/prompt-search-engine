from typing import Sequence
import numpy as np


class Vectorizer:
    def __init__(self, model) -> None:
        """Initialize the vectorizer with a pre-trained embedding model.
        Args:
        model: The pre-trained embedding model to use for transforming
        prompts.
        """
        
        self.model = model

    def transform(self, prompts: Sequence[str]) -> np.ndarray:
        
        """Transform texts into numerical vectors using the specified
        model.
        Args:
        prompts: The sequence of raw corpus prompts. Returns:
        Vectorized
        prompts as a numpy array."""
        vectorized = self.model.encode(prompts, show_progress_bar=True)
        return vectorized