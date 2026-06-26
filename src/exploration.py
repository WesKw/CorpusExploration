import time
import sys
import json
import openai
import gzip
import subprocess
import math
import os
import multiprocessing
import zstandard as zstd
import random
import inference_auth_token
import shutil
import numpy as np

from itertools import islice
from mpi4py import MPI
from collections import OrderedDict as od
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from multiprocessing import Process,Pool,TimeoutError


# UNIT_CHOICES = {
#     "MB": math.pow(1024, 2),
#     "GB": math.pow(1024, 3),
#     "TB": math.pow(1024, 4),
# }
SUBSET_CHOICES=["algebraic-stack", "arxiv", "dclm", "open-web-math", "pes2o", "starcoder", "wiki"]
AVAILABLE_MODELS=["openai/gpt-oss-120b", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it", "google/gemma-3-27b-it"]
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# need a maximum length due to rate limiting
# use global vars for some parameters, yes I know this is bad practice I'm doing this for convenience. Would be better
# to just define a dataclass for all processing params
N_CATEGORIES=4
MAX_DOCUMENT_LENGTH=5000
DEFAULT_SAMPLE_PROBABILITY=1.0
OUTFILE_NAME="output.txt"
SAMPLE_RATES = {}
OUTFILE_LOG=""
OUTFILE="jsons_merged.txt"


def print_rank_log(msg):
    with open(OUTFILE_LOG, 'a') as log:
        log.write(msg + "\n")


def process_json_file(arg):
    path = arg
    # print("Worker started")
    """Threads process a json"""
    # print(f"[{RANK}] Processing {path}")
    print_rank_log(f"[{RANK}] Processing {path}\n")
    # print(path)
    # print(subset)
    
    # get the collection that the file is a part of
    parts = Path(path).parts
    collection = parts[parts.index("data") + 1] if "data" in parts else None

    # need to decompress and read the files
    chunk_size = 5000000
    bytes = 0
    data = bytearray() # use a byte array to avoid copying too often
    extension = Path(path).suffix
    if extension == ".gz":
        try:
            # decompress into memory and get the full size
            with gzip.open(path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    data.extend(chunk)
                    bytes += len(chunk)
                    # print(f"\033[KSize: {bytes / (1024 * 1024 * 1024):.02f} GB", end="\r", flush=True)
            # data = data.decode("utf-8")
            # print(f"Processed {path}")
            # bytes = int(result.stdout.decode("utf-8"))
        except Exception as exc:
            print(f"Error occurred while processing {path}")
            print(exc)
            bytes = 0

    elif extension == ".zstd":
        try:
            with open(path, 'rb') as f:
                # buf = io.BytesIO(f.read())
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    while chunk := reader.read(chunk_size):
                        data.extend(chunk)
                        bytes += len(chunk)
                # print(f"Uncompressed size: {size} bytes")
                # bytes = int(zstd_out.stderr.decode("utf-8").split(" ")[1])        
            # data = data.decode("utf-8")
            print(f"Processed {path}")
        except Exception as exc:
            print(f"Error occurred while processing {path}")
            print(exc)
            bytes = 0

    else: # probably don't process the file if it's not compressed 
        print(f"Skipped {path}")
        bytes=0

    documents = 0
    # print(data)
    # Source - https://stackoverflow.com/a/54666028
    # Posted by pschill, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-15, License - CC BY-SA 4.0
    
    # decode json data
    data = data.decode("utf-8")
    # print("decoded")
    decoder = json.JSONDecoder()
    jsons = []
    # jsons = json.loads(data)
    pos = 0 # use indexer so we don't have to slice the string
    while pos < len(data):
        value, end = decoder.raw_decode(data, pos)
        # print(value)
        pos = end
        while pos < len(data) and data[pos] in ' \t\n\r':
            pos += 1

        # print_rank_log(str(SAMPLE_RATES))
        # print_rank_log(collection)
        # print_rank_log(str(SAMPLE_RATES.get(collection.lower(), DEFAULT_SAMPLE_PROBABILITY)))
        # use probability to determine whether or not to sample document
        if random.random() < SAMPLE_RATES.get(collection.lower(), DEFAULT_SAMPLE_PROBABILITY):
            # if we're using a probability then add it to the json list.
            # Note: We cannot skip the processing step because we need to process a json to find the next one.
            #       This does, however, save on memory overall
            jsons.append(value)
        #     print_rank_log("Did save")
        # else:
        #     print_rank_log("Did not save")
        
    return (path, collection, bytes, len(jsons), jsons)
    # print(f"\033[K{json}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

    # # do some magic analysis here
    # print(f"Removing {new_location.replace('.gz', '')}")
    # os.remove(f"./{path.name.replace('.gz', '')}")


def cluster_with_llm(jsons: list, model: str, categories: list[str], batch_size: int=10, delay: float=1.0, temperature: float=0.2):
    """
    Use predefined labels to cluster documents according to labels. Otherwise
    the LLM will cluster based on similarity
    """
    # get authentication token
    # print("Documents:", jsons)
    print_rank_log(f"[{RANK}] Starting LLM Inference ({len(jsons)})")
    token = inference_auth_token.get_access_token()

    client = openai.OpenAI(
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        api_key=token
    )

    # build an arbitrary number of categories
    n_categories = N_CATEGORIES
    category_string = ""
    for i in range(1, n_categories+1):
        category_string += f'"<category{i}>": "<probability>", '
    
    # give concrete classifications for now
    # todo:: include the subsection of data that the document is from
    prompt = f"""You are a document classifier. Cluster documents by topic with probabilities. Add a content difficulty classification for each document into one of: {categories}. Give a prior knowledge rating for each document between 0 and 1, 0 is no prior knowledge and 1 is high domain knowledge. Include the number of white space separated tokens in the document. Do not use any previous JSON formats. Respond ONLY with a JSON array: [{{"title": "<title>", {category_string}"confidence": "<low|medium|high>", "difficulty": "<content_difficulty>", "prior_knowledge": "<prior_knowledge_value>", "language": "<language>": "tokens": "<number_of_tokens>", "location": "<collection>"}}]"""
    all_results = []
    all_results_count = 0

    out = open(f'{OUTFILE_NAME}', 'a')

    random.shuffle(jsons) # shuffle jsons to ensure we're not processing 1 subset at a time
    for i in range(0, len(jsons), batch_size):
        batch = jsons[i:i + batch_size]
        
        # Build a multi-document prompt
        doc_texts = []
        for doc in batch:
            # documents don't have titles, though all the jsons have a Text attribute
            doc_texts.append(f"[collection: {doc.get('collection', 'None')}, Content: {doc.get('text', '')[:MAX_DOCUMENT_LENGTH]}")

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": "\n\n".join(doc_texts)
                }
            ]
        )
        
        # if response.choices[0].message.content:
        try:
            batch_content = response.choices[0].message.content
            batch_content = batch_content.replace("```json", "").replace("```", "")
            batch_results = json.loads(batch_content)
            # ordered dict retains order
            # add word counts for each document into the json
            # for idx,data in enumerate(batch):
            #     print(data["text"])
            #     ws_tokens_in_doc = len(data["text"].split())
            #     list(batch_results.items())[idx]["word_count"] = ws_tokens_in_doc

            # all_results.extend(batch_results)
            all_results_count += len(batch_results)
            # write as we receive new results
            for result in batch_results:
                try:
                    out.write(json.dumps(result) + "\n")
                except Exception as exc:
                    print_rank_log(f"{exc}")
            out.flush()

        except Exception as exc:
            # ignore a bad batch of json responses
            print_rank_log(f"{exc}")
        
        print_rank_log(f"Processed batch {i // batch_size + 1} "
            f"({all_results_count}/{len(jsons)} docs)")
        
        time.sleep(delay)  # Rate limiting
        
    out.close()

    return all_results


def run_corpus_analysis(paths: list, nprocs: int, cluster_method: str, model: str, categories: list, sample: int, sample_probability: float|None, temperature: float, maximum_json_amt: int):
    """
    Path is the root directory of the training data
    """
    dataset_size = 0
    skipped_file_names = set()
    collection_sizes = {}
    all_jsons = []

    with Pool(processes=nprocs) as pool:
        results = pool.imap(process_json_file, paths, chunksize=1)
        for result in results:
            file,collection,size,num_docs,jsons = result

            # if the size is 0 we skipped the file
            if size == 0:
                skipped_file_names.add(file)

            if collection not in collection_sizes:
                collection_sizes[collection] = {"size": 0, "documents": 0, "processed": 0, "total": 0}
            collection_sizes[collection]["size"] += size
            collection_sizes[collection]["documents"] += num_docs
            collection_sizes[collection]["processed"] += 1 if size != 0 else 0
            collection_sizes[collection]["total"] += 1
            dataset_size += size

            for doc in jsons:
                doc["collection"] = collection

            # print(f"\033[K{file}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

            all_jsons.extend(jsons)

    sys.stdout.flush()
    print_rank_log("\nCollection totals:")
    for collection in sorted(collection_sizes.items(), key=lambda x: x[1]["size"], reverse=True):
        col = collection[0]
        print_rank_log(
            f"\t{col} -> {(collection_sizes[col]['size'] / math.pow(1024, 3)):.02f} GB | " \
            f"{(collection_sizes[col]['size'] / dataset_size) * 100:.02f}% | {collection_sizes[col]['processed']} of {collection_sizes[col]['total']} processed | {collection_sizes[col]['documents']} docs"
        )

    # json_sample = random.sample(jsons, sample)
    # now that we have a small subset of jsons we do the analysis with an LLM to start
    if cluster_method == "llm":
        # if we don't sample during processing... sample a subset afterwards.
        if sample_probability == None:
            all_jsons = random.sample(all_jsons, sample)

        # print_rank_log(f"Sample size: {len(all_jsons)}")
        results = cluster_with_llm(all_jsons, model, ["Beginner", "Intermediate", "Advanced", "Expert"], temperature=temperature)


def get_json_paths(root: Path, subsets: list, max_json_amount: int = 0):
    gz_str = str(root) + "/**/*.gz"
    gz_paths = glob(gz_str, recursive=True)
    zstd_str = str(root) + "/**/*.zstd"
    zstd_paths = glob(zstd_str, recursive=True)

    print_rank_log(f"Files found: {len(gz_paths) + len(zstd_paths)}")
    print_rank_log(f"\t{len(gz_paths)} gz files")
    print_rank_log(f"\t{len(zstd_paths)} zstd files")
    print_rank_log(f"Filtering with subsets {subsets} and {max_json_amount} per subset.")

    if subsets != None:
        paths = [path for path in gz_paths + zstd_paths if subsets and any([s in path for s in subsets])] 
    else:
        paths = gz_paths + zstd_paths

    random.shuffle(paths) # shuffle document paths to ensure we're not taking the same set of documents every time.
    maximum_paths = []
    if max_json_amount and max_json_amount > 0:
        document_numbers = {} # initialize the counts dict
        for path in paths:
            subset_first = path.replace(f"{root}/", "").split("/")[0]
            
            if subset_first not in document_numbers:
                document_numbers[subset_first] = 0

            if document_numbers[subset_first] < max_json_amount:
                document_numbers[subset_first] += 1
                maximum_paths.append(path)

        paths = maximum_paths


    print_rank_log(f"Total files after filtering: {len(paths)}")
    return paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Location")
    parser.add_argument("--subset", action="append", help="Data subset to process. If none are specified, all subsets are chosen.", choices=SUBSET_CHOICES, default=None)
    parser.add_argument("--threads", help="Number of processes", default=1)
    parser.add_argument("--cluster-method", help="The method of clustering to use.", choices=["llm", "transformer"], default="llm")
    parser.add_argument("--model", help="Available model to use", choices=AVAILABLE_MODELS, default="openai/gpt-oss-120b")
    parser.add_argument("--sample", help="The number of documents to sample.", default="100")
    parser.add_argument("--categories", action="append", help="Classification categories.", default=["Beginner", "Intermediate", "Advanced", "Expert"])
    parser.add_argument("--sample-prob", help="The probability of retaining a processed json. [0, 1]. If set, overrides the --sample argument.", default=None)
    parser.add_argument("--temperature", help="LLM temperature", default=0.2)
    parser.add_argument("--max-doc-length", help="Maximum number of tokens of each document to pass to the LLM." , default=MAX_DOCUMENT_LENGTH, type=int)
    parser.add_argument("--max-json-amt", help="The maximum number of json files to process from each subset of data. If 0, all jsons found are processed.", default=0, type=int)
    # parser.add_argument("--batch-rate")
    parser.add_argument("--outfile", default="output.txt")
    parser.add_argument("--subset-sample-prob", help="A json that includes the sample probability for each provided subset. Any rate not specified will default to --sample-prob", default=None)
    parser.add_argument("--n-categories", help="The number of categories for the LLM to write for each document.", defualt=4, type=int)

    args = parser.parse_args()

    subsets = args.subset
    if args.subset == None:
        subsets = SUBSET_CHOICES

    N_CATEGORIES = args.n_categories
    OUTFILE_NAME = f"{args.outfile}_rank{RANK}.txt"
    OUTFILE_LOG = f"{Path(args.outfile).parents[0]}/rank{RANK}.log"
    # clear any existing json data before the job starts
    if os.path.exists(OUTFILE_NAME):
        os.remove(OUTFILE_NAME)

    if os.path.exists(OUTFILE_LOG):
        os.remove(OUTFILE_LOG)

    if RANK == 0:
        print_rank_log("Running exploration with:")
        print_rank_log(f"\tdata -> {args.data}")
        print_rank_log(f"\tsubset -> {subsets}")
        print_rank_log(f"\tnthreads -> {args.threads}")
        print_rank_log(f"\tcluster method -> {args.cluster_method}")
        print_rank_log(f"\tmodel -> {args.model}")
        print_rank_log(f"\tmodel temperature -> {args.temperature}")
        print_rank_log(f"\tsample -> {args.sample}")
        print_rank_log(f"\tsample probability -> {args.sample_prob}")
        print_rank_log(f"\tmaximum doc length -> {args.max_doc_length}")
        print_rank_log(f"\tmaximum json amount -> {args.max_json_amt}")
        print_rank_log(f"\toutput file -> {args.outfile}")
        print_rank_log(f"\tsample rate file -> {args.subset_sample_prob}")
        print_rank_log(f"\tsample rates:")
    

    probability = None
    if args.sample_prob != None:
        probability = float(args.sample_prob)
        DEFAULT_SAMPLE_PROBABILITY = probability

    MAX_DOCUMENT_LENGTH = args.max_doc_length

    if args.subset_sample_prob:
        try:
            with open(args.subset_sample_prob) as sample_file:
                SAMPLE_RATES = json.load(sample_file)
                for collection,rate in SAMPLE_RATES.items():
                    print_rank_log(f"\t\t{collection} -> {rate}")
        except:
            print_rank_log(f"Could not locate sample file {args.subset_sample_prob}")

    if RANK == 0:
        paths = get_json_paths(Path(args.data), subsets, args.max_json_amt)
        random.shuffle(paths) # shuffle the array to attempt to get an even distribution of data for processes
        chunked_paths = np.array_split(paths, COMM.Get_size())
        paths = []
        for chunk in chunked_paths:
            paths.append(chunk)
    else: # other processes wait for jsons to process
        paths = None

    # print("rank ", rank)
    paths = COMM.scatter(paths, root=0)

    # get the corpus metadata
    run_corpus_analysis(paths, int(args.threads), args.cluster_method, args.model, args.categories, int(args.sample), probability, args.temperature, args.max_json_amt)

    # merge step (oh god I'm a physicist)
    for i in range(COMM.Get_size()):
        with open(OUTFILE, 'a') as combined:
            with open(OUTFILE_NAME, 'r') as rank_file:
                for line in rank_file.readlines():
                    combined.write(line)
