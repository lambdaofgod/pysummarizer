import nltk
import spacy

from pysummarizer.document import SegmentableDocument
from pysummarizer.summarizer import _spacy_segmenter

text = "Colorless green ideas sleep furiously. Veni vidi vici"


def test_spacy_document():
    default_nlp_document = SegmentableDocument(
        segmenter=_spacy_segmenter(),
        analyzer=lambda s: s.split(), weighter=lambda s: 1, vectorizer=lambda s: 0, text=text
    )
    assert len(default_nlp_document.get_segments()) == 2


def test_nondefault_spacy_document():
    nlp = spacy.load("en_core_web_sm")
    english_nlp_document = SegmentableDocument(
        segmenter=_spacy_segmenter(nlp),
        analyzer=lambda s: s.split(), weighter=lambda s: 1, vectorizer=lambda s: 0, text=text
    )
    assert len(english_nlp_document.get_segments()) == 2


def test_nltk_document():

    nltk_document = SegmentableDocument(
        segmenter=nltk.tokenize.sent_tokenize,
        analyzer=lambda s: s.split(), weighter=lambda s: 1, vectorizer=lambda s: 0, text=text
    )

    assert len(nltk_document.get_segments()) == 2
