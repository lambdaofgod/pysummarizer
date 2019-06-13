from operator import itemgetter
from typing import Callable
from typing import Iterable

import attr
import nltk
import numpy as np
from mlutil.embeddings import WordEmbeddingsVectorizer, TextEncoderVectorizer
from spacy.lang.en import English

from pysummarizer.document import SegmentData
from pysummarizer.token_weighter import TokenWeighter


@attr.s
class GenericSummarizer:

    analyzer_document_factory = attr.ib()
    weighter: TokenWeighter = attr.ib()
    vectorizer: Callable[[str], np.ndarray] = attr.ib()
    segment_scorer = Callable[[SegmentData, Iterable[SegmentData]], Iterable[float]]

    def summarize(self, text: str, n_segments: int):
        document = self.analyzer_document_factory(text)
        whole_document_segment = document.get_segment_data(
            text,
            analyzer=self.analyzer,
            weighter=self.weighter,
            vectorizer=self.vectorizer
        )
        segments = [document.get_segment_data(segment_text) for segment_text in document.get_segments()]
        segment_scores = self.segment_scorer(whole_document_segment, segments)
        sorted_segments_with_scores = [
            (segment, score)
            for (segment, score) in
            sorted(zip(segments, segment_scores), key=itemgetter(1))
        ][:n_segments]
        sorted_segments, sorted_scores = zip(*sorted_segments_with_scores)
        return sorted_segments, sorted_scores


class Summarizer:

    def __init__(
            self,
            analyzer_type='spacy',
            vectorizer_type='gensim_word_embedding',
            weighter_type='tfidf',
            segmenter_type='spacy',
            sentence_encoder_type='large',
            gensim_embedding_model='glove-wiki-gigaword-50',
            spacy_nlp=None):
        pass

    @classmethod
    def _make_analyzer_factory(cls, analyzer_type, spacy_nlp):
        if analyzer_type == 'spacy':
            return 

    @classmethod
    def _spacy_analyzer(cls, nlp, text):
        return

    @classmethod
    def _make_vectorizer(cls, vectorizer_type, gensim_embedding_model, sentence_encoder_type):
        if vectorizer_type == 'gensim_word_embedding':
            return WordEmbeddingsVectorizer.from_gensim_embedding_model(gensim_embedding_model)
        elif vectorizer_type == 'universal_sentence_encoder':
            return TextEncoderVectorizer.from_tfhub_encoder(tfhub_encoder=sentence_encoder_type)
        else:
            raise NotImplementedError('Unknown vectorizer type: ' + vectorizer_type)

    @classmethod
    def _make_weighter(cls, weighter_type):
        pass

    @classmethod
    def _make_segmenter(cls, segmenter_type):
        pass


def _spacy_segmenter(spacy_nlp=None):
    if spacy_nlp is None:
        spacy_nlp = English()
        sentencizer = spacy_nlp.create_pipe("sentencizer")
        spacy_nlp.add_pipe(sentencizer)

    def segmenter(text):
        return list(spacy_nlp(text).sents)
    return segmenter

