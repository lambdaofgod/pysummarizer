from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


dependency_links = [
    'git+https://github.com/lambdaofgod/mlutil.git#egg=mlutil'
]


setup(
    name='pysummarizer',
    version='0.1',
    description='Extractive summarization in Python',
    url='https://github.com/lambdaofgod/pysummarizer',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=dependency_links
)
