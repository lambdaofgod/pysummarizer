import numpy as np

from pysummarizer.token_weighter import UniformTokenWeighter, SklearnVectorizerTokenWeighter
from sklearn.feature_extraction.text import TfidfVectorizer


text = "woodchuck chucks wood what would woodchuck chuck"
tokens = text.split()


def test_uniform_weigher():

    weighter = UniformTokenWeighter()
    weights = weighter.token_weights(tokens)

    assert np.var(weights) < 1e-10
    assert np.isclose(weights[0], 1.0 / len(tokens))


def test_sklearn_vectorizer_weighter():

    weighter = SklearnVectorizerTokenWeighter(TfidfVectorizer(norm=None), text)
    weights = weighter.token_weights(tokens)

    woodchuck_index = weighter._vectorizer.vocabulary_['woodchuck']
    assert weighter._vectorizer.vocabulary_ == {'woodchuck': 4, 'chucks': 1, 'wood': 3, 'what': 2, 'would': 5, 'chuck': 0}
    assert np.isclose(weights[0], 0.22222)