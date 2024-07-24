from typing import List, Sequence, Tuple
import numpy as np
from vectorizer import Vectorizer

def cosine_similarity(
        query_vector: np.ndarray,
        corpus_vectors: np.ndarray
    )-> np.ndarray:

        """Calculate cosine similarity between prompt vectors.
        Args:
        query_vector: Vectorized prompt query of shape (1, D).
        corpus_vectors: Vectorized prompt corpus of shape (N, D).
        Returns: The vector of shape (N,) with values in range [-1, 1] where 1
        is max similarity i.e., two vectors are the same.
        """
        dot_product = np.dot( corpus_vectors, query_vector)
        magnitude_A = np.linalg.norm(corpus_vectors, axis=1)
        magnitude_B = np.linalg.norm(query_vector)

        cosine_sim = dot_product / (magnitude_A * magnitude_B)
        return np.around(cosine_sim, 4)
        # return np.format_float_positional(cosine_sim, precision = 4)


class PromptSearchEngine:
    def __init__(self, prompts: Sequence[str], model) -> None:
        """Initialize search engine by vectorizing prompt corpus.
        Vectorized prompt corpus should be used to find the top n most
        similar prompts w.r.t. user’s input prompt.
        Args:
        prompts: The sequence of raw prompts from the dataset.
        """
        self.prompts = prompts
        self.vectorizer = Vectorizer(model)
        self.corpus_embeddings = self.vectorizer.transform(prompts)
    def most_similar(
    self,
    query: str,
    n: int = 5
    ) -> List[Tuple[float, str]]:
        """Return top n most similar prompts from corpus.
        Input query prompt should be vectorized with chosen Vectorizer.
        After
        that, use the cosine_similarity function to get the top n most
        similar
        prompts from the corpus.
        Args:
        query: The raw query prompt input from the user.
        n: The number of similar prompts returned from the corpus.
        Returns:
        The list of top n most similar prompts from the corpus along
        with similarity scores. Note that returned prompts are
        verbatim.
        """
        most_similar_prompts = []
        prompt_embedding = self.vectorizer.transform([query]).flatten()
        corpus_embeddings = self.corpus_embeddings

        result = cosine_similarity(prompt_embedding, corpus_embeddings)
        
        for i in range(len(self.prompts)):
            most_similar_prompts.append((result[i], self.prompts[i]))

        prompt_score_sorted = sorted(most_similar_prompts, key=lambda x: x[0], reverse=True)

        return prompt_score_sorted[0:n]
    def display_prompts(self, prompts):
        """Display the list of prompts with their similarity scores."""
        if prompts:
            for i, (score, prompt) in enumerate(prompts, 1):
                print(f"{i}. {prompt} (Similarity: {score:.4f})")
        else:
            print("No prompts found.")
    def stringify_prompts(self, prompts):
        """Save the list of prompts with their similarity scores."""
        strings = []
        if prompts:
            for i, (score, prompt) in enumerate(prompts, 1):
                strings.append(f"{i}. {prompt} (Similarity: {score:.4f})")
            return strings
        else:
            return []
