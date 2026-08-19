# CorpusExploration

Given:
- A model configuration for training
- A,B,...,X token orderings
- A set of lm evaluation tasks 

Goal:
- Determine if there is a meaningful difference in how each token ordering performs with training
- Do we see improvements in loss and validation?
- How are downstream tasks affected?


This repo consists of code for running on ALCF systems for pre-processing corpus data,
and clustering and ordering the data, and running lm\_eval benchmarkls. A separate torchtitan 
fork located at https://github.com/saforem2/torchtitan is used to run model pre-training.
