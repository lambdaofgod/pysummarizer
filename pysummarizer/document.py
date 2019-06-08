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

    def get_segment_data(self, text_segment: str, deduplicate_tokens=True) -> SegmentData:
        tokens = self.analyzer(text_segment)
        if deduplicate_tokens:
            tokens = _deduplicate(tokens)
        cleaned_text_segment = ' '.join(tokens)
        weights = self.weighter(cleaned_text_segment, deduplicate_tokens)
        vectors = self.vectorizer(cleaned_text_segment)
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


def _deduplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
