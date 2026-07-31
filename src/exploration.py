import time
import sys
import json
import openai
import gzip
import math
import os
import zstandard as zstd
import random
import inference_auth_token
import numpy as np
import asyncio
import orjson

from typing import overload
from simple_parsing import parse
from ExplorationArgs import ExplorationArgs
from itertools import islice
from mpi4py import MPI
from collections import OrderedDict as od
from pathlib import Path
from glob import glob
from multiprocessing import Process,Pool,TimeoutError
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
# need a maximum length due to rate limiting
# use global vars for some parameters, yes I know this is bad practice I'm doing this for convenience. Would be better
# to just define a dataclass for all processing params
N_CATEGORIES=4
DEFAULT_SAMPLE_PROBABILITY=1.0
OUTFILE_NAME="output.txt"
OUTFILE_LOG=""
OUTFILE="jsons_merged.txt"
FAILED_LOG=""
OUTFILE_DUMP=""
PER_FILE_SAMPLE_RATES={}
CLUSTERING_WEIGHTS={}
MAX_DOCUMENT_LENGTH=5000
BATCH_SIZE=5
CLUSTER_SIZE=500
MAX_KMEANS_POINTS=10_000
WRITE_BATCHES_TO_FILE=True

CONFIDENCE_ORDER = ["low", "medium", "high"]
DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced", "expert"]
RESERVED_KEYS = {"title", "confidence", "difficulty", "prior_knowledge", "tokens"}


INFERENCE_LINK="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"


def print_rank_log(msg):
    with open(OUTFILE_LOG, 'a') as log:
        log.write(msg + "\n")


def process_json_file(arg):
    path = arg
    """Threads process a json"""
    print_rank_log(f"[{RANK}] Processing {path}\n")
    
    # get the collection that the file is a part of
    parts = Path(path).parts
    collection = parts[parts.index("data") + 1] if "data" in parts else None

    # need to decompress and read the files
    chunk_size = 5000000
    bytes = 0
    data = bytearray() # use a byte array to avoid copying too often
    extension = Path(path).suffix
    if extension == ".gz":
        start_len = 0
        try:
            # decompress into memory and get the full size
            with gzip.open(path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    start_len = len(data)
                    data.extend(chunk)
                    bytes += len(chunk)
        except Exception as exc:
            print_rank_log(f"Error occurred while processing {path}")
            print_rank_log(exc)
            del data[start_len:]
            bytes = 0

    elif extension == ".zstd":
        start_len = 0
        try:
            with open(path, 'rb') as f:
                # buf = io.BytesIO(f.read())
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    while chunk := reader.read(chunk_size):
                        start_len = len(data)
                        data.extend(chunk)
                        bytes += len(chunk)     
            # print_rank_log(f"Processed {path}")
        except Exception as exc:
            print_rank_log(f"Error occurred while processing {path}")
            print_rank_log(exc)
            del data[start_len:]
            bytes = 0

    else: # probably don't process the file if it's not compressed 
        print_rank_log(f"Skipped {path}")
        bytes=0

    # Source - https://stackoverflow.com/a/54666028
    # Posted by pschill, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-15, License - CC BY-SA 4.0
    # decode json data
    data = data.decode("utf-8-sig")
    decoder = json.JSONDecoder()
    jsons = []
    pos = 0 # use indexer so we don't have to slice the string
    index = 1
    pid = os.getpid()
    while pos < len(data):
        value, end = (None, None)
        try: # attempt to get jsons
            value, end = decoder.raw_decode(data, pos)
        except json.decoder.JSONDecodeError as jde:
            print(jde)
            with open(FAILED_LOG + f"_{pid}.txt", 'a') as failed:
                failed.write(str(path) + f" | {jde.msg} [pos: {jde.pos}, lineno: {jde.lineno}, colno: {jde.colno}]\n")
            break # break out of the file if we fail to process a json, then log the file

        # print(value)
        pos = end
        while pos < len(data) and data[pos] in ' \t\n\r':
            pos += 1

        # use probability to determine whether or not to sample document
        if random.random() < PER_FILE_SAMPLE_RATES.get(collection, DEFAULT_SAMPLE_PROBABILITY):
            # chunk the document (we're only submitting X tokens anyway, saves memory)
            # dump json to some output file based on rank and pid so that we still have the processed data, just not in memory.
            # this is important for keeping a mapping for ordering since we don't keep all of the text in each document
            with open(f'{OUTFILE_DUMP}_pid{pid}.json', 'ab') as json_dump:
                json_dump.write(orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE))
            value["text"] = value.get('text', '')[:MAX_DOCUMENT_LENGTH] # after we write to a file, then chop the text to save memory for processing
            value["rank"] = RANK
            value["pid"] = pid
            value["row_index"] = index
            jsons.append(value)
            index += 1

    return (path, collection, bytes, len(jsons), jsons)

async def process_with_llm(jsons: list, model: str, categories: list[str], batch_size: int=BATCH_SIZE, delay: float=1.0, temperature: float=0, backoff_mult=1, max_delay=20.0, max_retries=5):
    """
    Use predefined labels to cluster documents according to labels. Otherwise
    the LLM will cluster based on similarity
    """
    # get authentication token
    # print("Documents:", jsons)
    print_rank_log(f"[{RANK}] Starting LLM Inference ({len(jsons)})")
    token = inference_auth_token.get_access_token()

    client = openai.AsyncOpenAI(
        base_url=INFERENCE_LINK,
        api_key=token
    )

    # build an arbitrary number of categories
    n_categories = N_CATEGORIES
    category_string = ""    
    for i in range(1, n_categories+1):
        category_string += f'"<category{i}>": "<probability>", '
    
    # give concrete classifications for now
    # todo:: include the subsection of data that the document is from
    prompt = f"""You are a document classifier. Cluster documents by topic with probabilities. Add a content difficulty classification for each document into one of: {categories}. Give a prior knowledge rating for each document between 0.001 and 1, 0 is little domain knowledge and 1 is high domain knowledge. vocab_complexity is from 0.001 to 1, 0.001 is simple words and 1 is high amounts of technical jargon or rare words. sentence_quality is a number from 0.001 to 1, indicating how well formed the average sentence is in each document. Include the number of white space separated tokens in the document. Do not use any previous JSON formats. Respond ONLY with a JSON array: [{{"title": "<title>", {category_string}"difficulty": "<content_difficulty>", "prior_knowledge": "<prior_knowledge_value>", "vocab_complexity": "<vocab_complexity>", "language": "<language>": "tokens": "<number_of_tokens>", "location": "<collection>", "sentence_quality": "<quality>", "idx": "<row_index>", "rank": "<rank>", "pid": "<pid>"}}]"""

    total=0
    # random.shuffle(jsons) # shuffle jsons to ensure we're not processing 1 subset at a time
    for i in range(0, len(jsons), batch_size):
        batch = jsons[i:i + batch_size]
        
        # Build a multi-document prompt
        doc_texts = []
        for idx,doc in enumerate(batch):
            # documents don't have titles, though all the jsons have a Text attribute
            doc_texts.append(f"collection: {doc.get('collection', 'None')}, idx: {doc.get('row_index', '')}, rank: {doc.get('rank', '')}, pid: {doc.get('pid', '')}, Content: {doc.get('text', '')[:MAX_DOCUMENT_LENGTH]}")

        retries=0
        mult=backoff_mult
        timeout=delay
        while True:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[   
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "\n\n".join(doc_texts)}
                    ]
                )

                try:
                    batch_content = response.choices[0].message.content
                    batch_content = batch_content.replace("```json", "").replace("```", "")
                    batch_results = json.loads(batch_content)
                    # for doc,result in zip(batch, batch_results):
                    #     result["text"] = doc["text"] # I have not noticed that the results are out of order in any capacity.

                    total += len(batch_results)
                    # write as we receive new results
                    yield batch_results
                    time.sleep(delay/2)

                except Exception as exc:
                    # just ignore a bad batch of json responses
                    print_rank_log(f"{exc}")
                    print_rank_log(f"{response.choices[0].message.content}")

                timeout=delay
                retries=0
                break 

            except Exception as exc: # client failed
                print_rank_log(f"Failure: {exc}")
                # try again with a delay
                timeout = math.pow(timeout + random.uniform(0, 2), backoff_mult)
                print_rank_log(f"Sleeping for {timeout}s")
                time.sleep(timeout)
                retries += 1
                mult+=1

                if retries > max_retries:
                    break
        
        print_rank_log(f"Processed batch {i // batch_size + 1} "
            f"({total}/{len(jsons)} docs)")
            


def do_preprocessing(paths: list, nprocs: int, maximum_json_amt: int):
    """
    Preprocesses (unzips and loads into memory) the jsons from the corpus
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

            all_jsons.extend(jsons)

    sys.stdout.flush()
    print_rank_log("\nCollection totals:")
    for collection in sorted(collection_sizes.items(), key=lambda x: x[1]["size"], reverse=True):
        col = collection[0]
        print_rank_log(
            f"\t{col} -> {(collection_sizes[col]['size'] / math.pow(1024, 3)):.02f} GB | " \
            f"{(collection_sizes[col]['size'] / dataset_size) * 100:.02f}% | {collection_sizes[col]['processed']} of {collection_sizes[col]['total']} processed | {collection_sizes[col]['documents']} docs"
        )

    return all_jsons


async def run_corpus_clustering_with_paths(paths: list, nprocs: int, inference_method: str, model: str, categories: list, temperature: float, maximum_json_amt: int, tokenized_input=None):
    """
    Path is the root directory of the training data
    """
    all_jsons = []
    # if not tokenized_input:
    all_jsons = do_preprocessing(paths, nprocs, maximum_json_amt)
    # else:
        # ... # do we need to do something with jsons here
    
    results=None
    # now that we have a small subset of jsons we do the analysis with an LLM to start
    if inference_method == "llm":
        # get generator
        results = process_with_llm(all_jsons, model, ["Beginner", "Intermediate", "Advanced", "Expert"])
    else:
        raise NotImplementedError("Non-llm analysis methods not implemented.")
    
    # One process should send seed documents to all other processes
    
    # overall container

    # some options that may turn into cli args
    docs_for_clustering = CLUSTER_SIZE
    idx = 0
    N_CLUSTERS = 4 # do difficulty clustering for now
    SEED = 42
    # kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=BATCH_SIZE, n_init="auto")
    # use generator to build clusters for each rank
    async for batch in results:
        # save batches to a file and do clustering in the separate merge step.
        if WRITE_BATCHES_TO_FILE:
            for result in batch:
                out = open(f'{OUTFILE_NAME}', 'a')
                try:
                    out.write(json.dumps(result) + "\n")
                except Exception as exc:
                    print_rank_log(f"{exc}")
                out.close()
        
        ...

        # update the clusters with the next batch of data
        # points,kmeans = cluster_step(kmeans, batch, 42)

        # # print centroids
        # print_rank_log("New cluster centers:")
        # print_rank_log(f"{kmeans.cluster_centers_}")

        # # check centroid labels
        # centroids = kmeans.cluster_centers_[:, -3:] # hardcode difficulty columns (for now)
        # feature_sums = np.sum(centroids, axis=1) # sum the features of each column to determine an outer label
        # sorted_centroids = sorted(zip(kmeans.labels_, centroids)) # centroids should converge on average difficulty levels if we have a large enough sample size
        # CENTROID_LABELS=["easy", "intermediate", "hard", "expert"]
        # doc_dict = {label: pts for idx,label in enumerate(CENTROID_LABELS, sorted_centroids)}


# def run_corpus_clustering_with_strings():
#     ...


def get_json_paths(root: Path, subsets: list, max_json_amount):
    print_rank_log(f"{max_json_amount}")
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
    if max_json_amount > 0:
        document_numbers = {} # initialize the counts dict
        for path in paths:
            subset_first = path.replace(f"{root}/", "").split("/")[0]
            # print_rank_log(subset_first)
            
            if subset_first not in document_numbers:
                document_numbers[subset_first] = 0

            if document_numbers[subset_first] < max_json_amount:
                document_numbers[subset_first] += 1
                maximum_paths.append(path)

        paths = maximum_paths


    print_rank_log(f"Total files after filtering: {len(paths)}")
    return paths


if __name__ == "__main__":
    args = parse(ExplorationArgs)

    subsets = args.subset
    N_CATEGORIES = args.n_categories
    OUTFILE_NAME = f"{args.outfile}_rank{RANK}.json" # data for each json row
    OUTFILE_LOG = f"{Path(args.outfile).parents[0]}/rank{RANK}.log"
    OUTFILE_DUMP = f"{args.outfile}_rank{RANK}_dump.json"
    FAILED_LOG = f"{Path(args.outfile).parents[0]}/FAILED_{RANK}"

    cluster = args.cluster
    model = args.model
    MAX_KMEANS_POINTS = args.max_kmeans_points

    if cluster == "sophia":
        INFERENCE_LINK="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    elif cluster == "metis":
        INFERENCE_LINK="https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    else:
        raise NotImplementedError("Only sophia and metis clusters supported")

    if model not in args.MODELS[cluster]:
        raise NotImplementedError(f"Available models on {cluster}: {args.MODELS[cluster]}")

    # clear any existing json data before the job starts
    if os.path.exists(OUTFILE_NAME):
        os.remove(OUTFILE_NAME)

    if os.path.exists(OUTFILE_LOG):
        os.remove(OUTFILE_LOG)

    if os.path.exists(OUTFILE_DUMP):
        os.remove(OUTFILE_DUMP)

    if args.subset_sample_prob_file:
        try:
            with open(args.subset_sample_prob_file) as sample_file:
                PER_FILE_SAMPLE_RATES = json.load(sample_file)
        except:
            print_rank_log(f"Could not locate sample file {args.subset_sample_prob_file}")

    if RANK == 0:
        using_preprocessed_input = False if args.tokenized_input == None else True
        print_rank_log("Running exploration with:")
        print_rank_log(f"\tdata -> {args.data}")
        print_rank_log(f"\tSkip data preprocessing? -> {using_preprocessed_input}")
        if using_preprocessed_input:
            print_rank_log(f"\t\tInput data file -> {args.tokenized_input}")
        print_rank_log(f"\tsubset -> {subsets}")
        print_rank_log(f"\tnthreads -> {args.threads}")
        print_rank_log(f"\tcluster method -> {args.inference_method}")
        print_rank_log(f"\tcluster -> {args.cluster}")
        print_rank_log(f"\tmodel -> {args.model}")
        print_rank_log(f"\tmodel temperature -> {args.temperature}")
        print_rank_log(f"\tmaximum doc length -> {args.max_doc_length}")
        print_rank_log(f"\tmaximum json amount -> {args.max_json_amt}")
        print_rank_log(f"\toutput file -> {args.outfile}")
        print_rank_log(f"\tdefault sample probability -> {args.sample_prob}")
        print_rank_log(f"\tsample rate file -> {args.subset_sample_prob_file}")
        print_rank_log(f"\tsample rates:")
        for collection,rate in PER_FILE_SAMPLE_RATES.items():
            print_rank_log(f"\t\t{collection} -> {rate}")
        print_rank_log(f"\tNumber of clusters -> {args.n_clusters}")
        print_rank_log(f"\tCategory weights for clustering -> {args.weights_json}")
        print_rank_log(f"\tMaximum cluster points -> {args.max_kmeans_points}")
    
    DEFAULT_SAMPLE_PROBABILITY = args.sample_prob
    MAX_DOCUMENT_LENGTH = args.max_doc_length
    WRITE_BATCHES_TO_FILE = args.write_batches_to_file

    paths = []
    # if not using_preprocessed_input: # if we're not using an existing pre-processed input then we need to load json paths
    if RANK == 0:
        paths = get_json_paths(Path(args.data), subsets, args.max_json_amt)
        random.shuffle(paths) # shuffle the array to attempt to get an even distribution of data for processes
        chunked_paths = np.array_split(paths, COMM.Get_size())
        paths = []
        for chunk in chunked_paths:
            paths.append(chunk)
        
    else: # other processes wait for jsons to process
        paths = []

    paths = COMM.scatter(paths, root=0)

    # get the corpus metadata
    asyncio.run(run_corpus_clustering_with_paths(paths, int(args.threads), args.inference_method, args.model, args.categories, args.temperature, args.max_json_amt, args.tokenized_input))

