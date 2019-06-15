from pysummarizer.summarizer import Summarizer


text = '''Twas bryllyg, and ye slythy toves.
            Did gyre and gymble in ye wabe.
            All mimsy were ye borogoves.
            And ye mome raths outgrabe.'''


def test_lead_summarizer():

    summarizer = Summarizer(scorer_type='lead')

    summary_data, summary_scores = summarizer.summarize(text, n_segments=3)

    assert summary_scores[0] > summary_scores[1] > summary_scores[2]