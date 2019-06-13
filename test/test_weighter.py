import numpy as np

from pysummarizer.token_weighter import UniformTokenWeighter, SklearnVectorizerTokenWeighter
from sklearn.feature_extraction.text import TfidfVectorizer


text = "woodchuck chucks wood what would woodchuck chuck"
tokens = text.split()


def test_uniform_weigher():

    weighter = UniformTokenWeighter()
    weights = weighter.token_weights(tokens)

    assert np.var(weights) < 1e-10, 'weights are not the same'
    assert np.isclose(weights[0], 1.0 / len(tokens)), 'weights do not have appropriate value'


def test_sklearn_vectorizer_weighter():

    weighter = SklearnVectorizerTokenWeighter(TfidfVectorizer(norm=None), text)
    weights = weighter.token_weights(tokens)

    assert np.isclose(weights.sum(), 1), 'weights do not sum to one'
    assert np.isclose(weights[0], 0.22222), 'weights are not right'