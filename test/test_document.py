import spacy

from pysummarizer.document import spacy_segmented_document_factory, nltk_segmented_document_factory


text = "Colorless green ideas sleep furiously. Veni vidi vici"


def test_spacy_document():
    default_nlp_document = spacy_segmented_document_factory()(
        analyzer=lambda s: s.split(), weighter=lambda s: 1, vectorizer=lambda s: 0, text=text
    )
    assert len(default_nlp_document.get_segments()) == 2


def test_nondefault_spacy_document():
    nlp = spacy.load("en_core_web_sm")
    english_nlp_document = spacy_segmented_document_factory(nlp)(
        analyzer=lambda s: s.split(), weighter=lambda s: 1, vectorizer=lambda s: 0, text=text
    )
    assert len(english_nlp_document.get_segments()) == 2


def test_nltk_document():

    nltk_document = nltk_segmented_document_factory()(
        analyzer=lambda s: s.split(), weighter=lambda s: 1, vectorizer=lambda s: 0, text=text
    )

    assert len(nltk_document.get_segments()) == 2
