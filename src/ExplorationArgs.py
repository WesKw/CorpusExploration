from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Literal

SUBSET_CHOICES=["algebraic-stack", "arxiv", "dclm", "open-web-math", "pes2o", "starcoder", "wiki"]
INFERENCE_CLUSTER=["sophia", "metis"]
AVAILABLE_MODELS=["openai/gpt-oss-120b", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it", "google/gemma-3-27b-it"]

# MODELS = {
#     "sophia": set(["openai/gpt-oss-120b", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it", "google/gemma-3-27b-it"]),
#     "metis": set(["gpt-oss-120b", "Llama-4-Maverick-17B-128E-Instruct", "gemma-4-31B-it"])
# }

@dataclass
class ExplorationArgs:
    # constant options
    MODELS = {
        "sophia": set(["openai/gpt-oss-120b", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it", "google/gemma-3-27b-it"]),
        "metis": set(["gpt-oss-120b", "Llama-4-Maverick-17B-128E-Instruct", "gemma-4-31B-it"])
    }

    # data processing arguments
    data: str # path to olmo-mix data
    tokenized_input: str="" # path to an input file containing pre-processed, tokenized input. If this is provided, processing the jsons is skipped.
    subset:list = field(default_factory=lambda: SUBSET_CHOICES) # Subsets in the olmo-mix data to use
    threads: int=1 # number of threads for data preprocessing
    sample_prob: float=1 # general probability for document sampling
    max_json_amt: int=0 # number of jsons from each subset to process

    # inference arguments
    inference_method:str="llm" # the type of inference method to use
    cluster:str="sophia"
    model:str="google/gemma-4-31B-it" # model name to use if using llm inference
    temperature:float=0 # model temperature
    max_doc_length:int=1250 # number of characters to pass to the inference method 
    categories:list=field(default_factory=lambda: ["Beginner", "Intermediate", "Advanced", "Expert"]) # classification categories
    n_categories:int=4 # number of categories for the LLM to write for each document
    subset_sample_prob_file:str="" # A json that includes the sample probability for each file in the dataset. Unspecified rates default to --sample-prob.

    # clustering arguments
    n_clusters: int=4
    weights_json: str="difficulty-bias.json"
    max_kmeans_points: int=10_000 # the maximum number of points for updating clusters. Once this limit is reached, centroids become static and new points are fit to the existing centroids.
    cluster_sample_rate_json: str="cluster-sample-rate.json" # sample rate for cluster labels (experimental)
    write_batches_to_file: bool=True # write the classified data to a file.

    # misc args
    outfile:str="output.txt"

    # checkpoint args
    resume_from_checkpoint: bool=False # Skip the unzipping and 
    checkpoint_dir:str="" # The directory to load the checkpoint from. Fails if the directory does not exist or is empty.
    reprocess_all:bool=True # reprocess all documents in each json, and wipe the existing documents.

@dataclass
class ClusterArgs:
    # input args
    llm_data_regex: str
    shard_data_regex: str
    sort:bool=False # Sort the input files before processing.

    # cluster args
    """
    Orderings:
        none -> Documents are collected into a single json with no specified ordering.
        difficulty -> Documents are clustered by difficulty metrics with *k* clusters, and sorted from least-difficult to most. Append -reverse to flip.
        stem -> Documents are ordered by how close they are to being about a STEM topic, least to most. Append -reverse to flip.
        code -> Documents are ordered by least to most amount of code to standard text. Append -reverse to flip.
    """
    method: Literal["none", "difficulty", "difficulty-reverse", "stem", "stem-reverse", "code", "code-reverse"] = "none" # The ordering method to use. None -> merges json files as is | difficulty -> Order by document difficulty. | topic -> Order by topic. If you would like the order reversed, append "-reverse" to the end of the ordering method name. Ex: "difficulty-reverse"
    reverse: bool=False # Reverse the ordering before writing.
    n_clusters: int=4 # number of clusters to create.
    seed:int=42 # The seed for k means.
    batch_size:int=50000 # batch size for k means.
    weights_json: str="difficulty.json" # the biases for each text attribute in the json. Non-topic weights are 0.5 by default.
    max_kmeans_points: int=-1 # the maximum number of points for updating clusters. Once this limit is reached, centroids become static and new points are fit to the existing centroids. -1 -> all points are used in clustering.
    # cluster_sample_rate_json: str="cluster-sample-rate.json" # sample rate for cluster labels (experimental)
    write_batches_to_file: bool=True # write the data to a file.

    # meta args
    outfile:str="output.json" # The final, merged json file name.

    # parser.add_argument("--ordering-method", help="The method to use when ordering clustered data.", choices=["none", "difficulty"])


@dataclass
class DashboardArgs:
    ezpz_logs_glob: str # glob pattern for the training logs
    output_dir: str # directory to save the dashboard files
    ezpz_metrics:list=field(default_factory=lambda: ["loss", "grad_norm"])
    # lm_eval_tasks:list=field(default_factory=lambda: ["hellaswag", "gsm8k", "mathqa", "code_eval"])
    lm_eval_tasks:list=field(default_factory=lambda: ["hellaswag"])