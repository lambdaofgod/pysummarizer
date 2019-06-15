from typing import List, Iterable

import numpy as np
from sklearn.feature_extraction.text import VectorizerMixin


class TokenWeighter:

    def token_weights(self, tokens: List[str], deduplicate_tokens: bool) -> Iterable[float]:
        pass


class UniformTokenWeighter(TokenWeighter):

    def token_weights(self, tokens, deduplicate_tokens=False) -> Iterable[float]:
        n = len(tokens)
        weights = np.ones((n,))
        return weights / n


class SklearnVectorizerTokenWeighter(TokenWeighter):

    def __init__(self, vectorizer: VectorizerMixin, input_text=None):
        if not hasattr(vectorizer, 'coeff_'):
            assert input_text is not None, 'Need to pass input text if vectorizer is not fitted'
            vectorizer.fit([input_text])
        self._vectorizer = vectorizer

    def token_weights(self, tokens, deduplicate_tokens=False) -> Iterable[float]:
        glued_text = ' '.join(tokens)
        text_vector = self._vectorizer.transform([glued_text])
        weights = [text_vector[0, self._vectorizer.vocabulary_[token]] for token in tokens]
        return np.array(weights) / sum(weights)