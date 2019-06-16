# pysummarizer

A Python library for text summarization.

### Goals

- consistent, scikit-learn like api for many existing summarization algorithms.
- easy to use summarization evaluation metrics
- example datasets for text summarization

### Summary

There are several straightforward extractive text summarization approaches that are implemented in a couple
of github projects. They all have different APIs, and also many of them can be sped up
using pretty straightforward modifications.

## Useful info on summarization

If you're new to summarization check out
[this blog post](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/).
It provides an introduction to summarization and explains abstractive vs extractive summarization.
A good concise review on extractive techniques can be found in [Unsupervised Extractive Summarization: A Comparative Study
](https://towardsdatascience.com/unsupervised-extractive-summarization-a-comparative-study-ca6ac2181d54) medium post.

This library will mostly tackle extractive summarization, because most algorithms for abstractive summarization need
to utilize some heavy language models - it seems that most useful abstractive summarization algorithms are based on deep
learning. They also need training, and they are dominantly supervised, what makes them less useful in general.
