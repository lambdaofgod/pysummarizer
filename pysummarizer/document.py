from collections import namedtuple
from functools import partial
from typing import List, Callable

import attr
import nltk
import numpy as np
from spacy.lang.en import English


SegmentData = namedtuple('SegmentData', ['word_weights', 'word_vectors'])


@attr.s
class SegmentableDocument:

    text: str = attr.ib()
    analyzer: Callable[[str], List[str]] = attr.ib()
    weighter: Callable[[str], np.ndarray] = attr.ib()
    vectorizer: Callable[[str], np.ndarray] = attr.ib()
    _segmenter: Callable[[str], List[str]] = attr.ib()

    def get_segment_data(self, text_segment: str) -> SegmentData:
        weights = self.weighter(text_segment)
        vectors = self.vectorizer(text_segment)
        return SegmentData(weights, vectors)

    def get_segments(self):
        return self._segmenter(self.text)


def document_factory_from_segmenter(segmenter):
    return partial(SegmentableDocument, segmenter=segmenter)


def spacy_segmented_document_factory(spacy_nlp=None):
    if spacy_nlp is None:
        spacy_nlp = English()
        sentencizer = spacy_nlp.create_pipe("sentencizer")
        spacy_nlp.add_pipe(sentencizer)

    def spacy_segmenter(text):
        return list(spacy_nlp(text).sents)
    return document_factory_from_segmenter(spacy_segmenter)


def nltk_segmented_document_factory():
    return document_factory_from_segmenter(nltk.tokenize.sent_tokenize)
