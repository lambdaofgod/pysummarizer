from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='ot-summarizer',
    version='0.1',
    description='Extractive summarization using Optimal Transport and Word Embeddings',
    url='https://github.com/lambdaofgod/ot-summarizer',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements
)
