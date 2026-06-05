# CorpusExploration

Given: 
- pre-trained model, or a smaller version of a pre-trained model w/ the same weights. 
- Existing mixture model and dataset

Goal:
- How can we develop an alternate mixture model that improves the training rate and validation loss?

1) Start by exploring an existing dataset. Try clustering datasets, looking for similarities
within a dataset and across datasets.
    - How big is each dataset?
    - How many files in each dataset? 
    - Start with the raw dataset first before moving to the tokenized dataset.
    - In a given corpus:
        - cluster data based on various categories
        - Similarity of documents or code
        - size of data
        - number of tokens (depends on tokenizer)
        - Data sizes not necessarily equal to token size
        