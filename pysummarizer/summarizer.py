from operator import itemgetter
from typing import Callable, List
from typing import Iterable

import attr
import nltk
import numpy as np
from mlutil.embeddings import WordEmbeddingsVectorizer, TextEncoderVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.en import English

from pysummarizer.document import SegmentData, SegmentableDocument
from pysummarizer.token_weighter import TokenWeighter, UniformTokenWeighter, SklearnVectorizerTokenWeighter


@attr.s
class GenericSummarizer:

    analyzer = attr.ib()
    weighter: TokenWeighter = attr.ib()
    vectorizer: Callable[[str], np.ndarray] = attr.ib()
    scorer : Callable[[SegmentData, Iterable[SegmentData]], Iterable[float]] = attr.ib()
    segmenter: Callable[[str], List[str]] = attr.ib()

    def summarize(self, text: str, n_segments: int):
        document = SegmentableDocument(text, analyzer=self.analyzer, weighter=self.weighter, vectorizer=self.vectorizer, segmenter=self.segmenter)
        whole_document_segment = document.get_segment_data(
            text
        )
        segments = [document.get_segment_data(segment_text) for segment_text in document.get_segments()]
        segment_scores = self.scorer(whole_document_segment, segments)
        sorted_segments_with_scores = [
            (segment, score)
            for (segment, score) in
            sorted(zip(segments, segment_scores), key=itemgetter(1), reverse=True)
        ][:n_segments]
        sorted_segments, sorted_scores = zip(*sorted_segments_with_scores)
        return sorted_segments, sorted_scores


class Summarizer(GenericSummarizer):

    def __init__(
            self,
            analyzer_type='spacy',
            vectorizer_type=None,
            segmenter_type='spacy',
            weighter_type='uniform',
            sentence_encoder_type='large',
            scorer_type='lead',
            gensim_embedding_model='glove-wiki-gigaword-50',
            spacy_nlp=None):
        _analyzer = self._make_analyzer(analyzer_type, spacy_nlp)
        _vectorizer = self._make_vectorizer(vectorizer_type, gensim_embedding_model, sentence_encoder_type)
        _weighter = self._make_weighter(weighter_type)
        _segmenter = self._make_segmenter(segmenter_type, spacy_nlp)
        _scorer = self._make_scorer(scorer_type)

        super().__init__(analyzer=_analyzer, vectorizer=_vectorizer, weighter=_weighter, segmenter=_segmenter, scorer=_scorer)

    @classmethod
    def _make_analyzer(cls, analyzer_type, spacy_nlp):
        if analyzer_type == 'spacy':
            return _spacy_analyzer(spacy_nlp)
        elif analyzer_type == 'nltk':
            return _nltk_analyzer
        else:
            raise NotImplementedError('Unsupported analyzer: ' + analyzer_type)

    @classmethod
    def _make_vectorizer(cls, vectorizer_type, gensim_embedding_model, sentence_encoder_type):
        if vectorizer_type is None:
            return lambda x: 0
        elif vectorizer_type == 'gensim_word_embedding':
            return WordEmbeddingsVectorizer.from_gensim_embedding_model(gensim_embedding_model)
        elif vectorizer_type == 'universal_sentence_encoder':
            return TextEncoderVectorizer.from_tfhub_encoder(tfhub_encoder=sentence_encoder_type)
        else:
            raise NotImplementedError('Unknown vectorizer type: ' + vectorizer_type)

    @classmethod
    def _make_weighter(cls, weighter_type):
        if weighter_type == 'tfidf':
            return SklearnVectorizerTokenWeighter(vectorizer=TfidfVectorizer())
        elif weighter_type == 'count':
            return SklearnVectorizerTokenWeighter(vectorizer=CountVectorizer())
        elif type(weighter_type) in [CountVectorizer, TfidfVectorizer]:
            return SklearnVectorizerTokenWeighter(vectorizer=weighter_type)
        elif weighter_type =='uniform':
            return UniformTokenWeighter()
        else:
            raise NotImplementedError('Unknown weighter type: ' + weighter_type)

    @classmethod
    def _make_segmenter(cls, segmenter_type, spacy_nlp):
        if segmenter_type == 'spacy':
            return _spacy_segmenter(spacy_nlp)
        elif segmenter_type == 'nltk':
            return nltk.tokenize.sent_tokenize

    @classmethod
    def _make_scorer(cls, scorer_type):
        if scorer_type == 'lead':
            return lead_scorer
        else:
            raise NotImplementedError('Unknown scorer type: ' + scorer_type)


def _spacy_analyzer(spacy_nlp):
    def analyzer(text, spacy_nlp=spacy_nlp):
        if spacy_nlp is None:
            spacy_nlp = English()
        return [t.text for t in spacy_nlp(text)]
    return analyzer


def _nltk_analyzer(text):
    return nltk.tokenize.word_tokenize(text)


def _spacy_segmenter(spacy_nlp=None):
    if spacy_nlp is None:
        spacy_nlp = English()
        sentencizer = spacy_nlp.create_pipe("sentencizer")
        spacy_nlp.add_pipe(sentencizer)

    def segmenter(text):
        return [s.text for s in spacy_nlp(text).sents]
    return segmenter


def lead_scorer(whole_segment, other_segments):
    return len(other_segments) - np.arange(len(other_segments))
