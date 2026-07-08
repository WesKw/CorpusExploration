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
import torch.distributed as dist


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
PER_FILE_SAMPLE_RATES={}
CLUSTERING_WEIGHTS={}
MAX_DOCUMENT_LENGTH=5000
BATCH_SIZE=100
CLUSTER_SIZE=500

CONFIDENCE_ORDER = ["low", "medium", "high"]
DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced", "expert"]
RESERVED_KEYS = {"title", "confidence", "difficulty", "prior_knowledge", "tokens"}


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
        try:
            # decompress into memory and get the full size
            with gzip.open(path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    data.extend(chunk)
                    bytes += len(chunk)
        except Exception as exc:
            print_rank_log(f"Error occurred while processing {path}")
            print_rank_log(exc)
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
            print_rank_log(f"Processed {path}")
        except Exception as exc:
            print_rank_log(f"Error occurred while processing {path}")
            print_rank_log(exc)
            bytes = 0

    else: # probably don't process the file if it's not compressed 
        print_rank_log(f"Skipped {path}")
        bytes=0

    # Source - https://stackoverflow.com/a/54666028
    # Posted by pschill, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-15, License - CC BY-SA 4.0
    # decode json data
    data = data.decode("utf-8")
    decoder = json.JSONDecoder()
    jsons = []
    pos = 0 # use indexer so we don't have to slice the string
    while pos < len(data):
        value, end = decoder.raw_decode(data, pos)
        # print(value)
        pos = end
        while pos < len(data) and data[pos] in ' \t\n\r':
            pos += 1

        # use probability to determine whether or not to sample document
        if random.random() < PER_FILE_SAMPLE_RATES.get(collection, DEFAULT_SAMPLE_PROBABILITY):
            # chunk the document (we're only submitting X tokens anyway, saves memory)
            value["text"] = value.get('text', '')[:MAX_DOCUMENT_LENGTH] 
            jsons.append(value)
        
    return (path, collection, bytes, len(jsons), jsons)


def process_with_llm(jsons: list, model: str, categories: list[str], batch_size: int=BATCH_SIZE, delay: float=1.0, temperature: float=0, backoff_mult=1, max_delay=20.0, max_retries=5):
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
    prompt = f"""You are a document classifier. Cluster documents by topic with probabilities. Add a content difficulty classification for each document into one of: {categories}. Give a prior knowledge rating for each document between 0.001 and 1, 0 is little domain knowledge and 1 is high domain knowledge. vocab_complexity is from 0.001 to 1, 0.001 is simple words and 1 is high amounts of technical jargon or rare works. Include the number of white space separated tokens in the document. Do not use any previous JSON formats. Respond ONLY with a JSON array: [{{"title": "<title>", {category_string}"confidence": "<low|medium|high>", "difficulty": "<content_difficulty>", "prior_knowledge": "<prior_knowledge_value>", "vocab_complexity": "<vocab_complexity>", "language": "<language>": "tokens": "<number_of_tokens>", "location": "<collection>", "batch_id": "<batch_id>"}}]"""

    # random.shuffle(jsons) # shuffle jsons to ensure we're not processing 1 subset at a time
    for i in range(0, len(jsons), batch_size):
        batch = jsons[i:i + batch_size]
        
        # Build a multi-document prompt
        doc_texts = []
        for idx,doc in enumerate(batch):
            # documents don't have titles, though all the jsons have a Text attribute
            doc_texts.append(f"collection: {doc.get('collection', 'None')}, batch_id: {idx}, Content: {doc.get('text', '')[:MAX_DOCUMENT_LENGTH]}")

        total=0
        retries=0
        mult=backoff_mult
        timeout=delay
        while True:
            try:
                response = client.chat.completions.create(
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

                    total += len(batch_results)
                    # write as we receive new results
                    yield batch_results
                    time.sleep(delay/2)

                except Exception as exc:
                    # just ignore a bad batch of json responses
                    print_rank_log(f"{exc}")

                timeout=delay
                retries=0
                break 

            except: # client failed
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


def cluster_step(kmeans, batch: list, random_seed: int, save_batch_to_file:bool=True):
    def build_vocab(parsed):
        """Collect the set of all distinct category names across all documents."""
        vocab = set()
        for p in parsed:
            vocab.update(p["categories"].keys())
        return sorted(vocab)

    def parse_probability(value):
        """Coerce a probability field (often a string) into a float, or None."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def parse_int(value):
        """Coerce a count field (often a string, sometimes "1234.0") into an
        int, or None."""
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
        
    def difficulty_to_signed(difficulty):
        """beginner -> -1.0, ..., expert -> +1.0, unknown/missing -> 0.0.

        Centered rather than 0..1: with a plain 0..1 ordinal encoding, two
        "beginner" docs (both 0) contribute nothing to the cosine dot product
        on this dimension, while two "expert" docs (both 1) contribute the
        most - an asymmetry with nothing to do with how similar the documents
        actually are. Centering on 0 makes a match contribute the same amount
        regardless of which end of the scale it's at, and makes "unknown"
        truly neutral (0 contributes nothing when multiplied against anything)
        instead of silently behaving like "beginner"."""
        if difficulty in DIFFICULTY_ORDER:
            return (DIFFICULTY_ORDER.index(difficulty) / (len(DIFFICULTY_ORDER) - 1)) * 2 - 1
        return 0.0


    def prior_knowledge_to_signed(prior_knowledge):
        """0.0 prior knowledge -> -1.0, 1.0 -> +1.0, missing -> 0.0 (neutral).
        Same centering rationale as difficulty_to_signed."""
        if prior_knowledge is None:
            return 0.0
        return (prior_knowledge - 0.5) * 2
    
    def build_matrix(parsed, vocab):
        """Rows = documents, columns = [one per category, then difficulty,
        then prior_knowledge]. Category columns hold the assigned probability
        (0 if a document wasn't assigned that category). The two metadata
        columns are appended at the end, each centered to [-1, 1] (see
        difficulty_to_signed/prior_knowledge_to_signed) and multiplied by its
        `_weight`.

        Category columns are sparse - most documents only populate 2-4 of them
        out of possibly dozens - while difficulty/prior_knowledge are dense
        (present on every document). Left at full scale, two dense columns
        would dominate cosine similarity over many sparse ones; the weights
        default to 0.5 so difficulty/depth nudges which documents look close
        without overriding genuine topical overlap. Set a weight to 0 to drop
        that signal entirely (matching the old categories-only behavior)."""
        index = {name: i for i, name in enumerate(vocab)}
        n_cols = len(vocab) + 2
        difficulty_col, prior_knowledge_col,vocab = len(vocab), len(vocab) + 1, len(vocab) + 2

        matrix = np.zeros((len(parsed), n_cols))
        for row, p in enumerate(parsed):
            for name, prob in p["categories"].items():
                col = index.get(name)
                if col is not None:
                    matrix[row, col] = prob * CLUSTERING_WEIGHTS.get("category", 1.0)

            matrix[row, difficulty_col] = difficulty_to_signed(p.get("difficulty")) * CLUSTERING_WEIGHTS.get("difficulty", 0.5)
            matrix[row, prior_knowledge_col] = prior_knowledge_to_signed(p.get("prior_knowledge")) * CLUSTERING_WEIGHTS.get("prior_knowledge", 0.5)
            matrix[row, vocab] = prior_knowledge_to_signed(p.get("vocab_complexity", "None")) * CLUSTERING_WEIGHTS.get("vocab_complexity", 0.5) 
            # matrix[row, doc_id] = p.get("id", "None")
        return matrix
    
    def parse_records(records):
        """Turn raw JSON records into a flat list of dicts ready for plotting.

        Each result has a "categories" dict of {category_name: probability},
        sorted from most to least probable, so "rank 0" is always the
        document's primary category regardless of how many it has."""
        parsed = []
        for r in records:
            item = {
                "title": r.get("title", "Untitled"),
                "confidence": (r.get("confidence") or "unknown").lower(),
                "difficulty": (r.get("difficulty") or "unknown").lower(),
                "prior_knowledge": parse_probability(r.get("prior_knowledge")),
                "tokens": parse_int(r.get("tokens")),
                "location": r.get("location", "None"),
                "id": r.get("batch_id", "None"),
                "vocab_complexity": r.get("vocab_complexity", "None")
            }

            categories = {}
            for key, value in r.items():
                if key in RESERVED_KEYS:
                    continue
                prob = parse_probability(value)
                if prob is not None:
                    categories[key.lower()] = prob

            item["categories"] = dict(
                sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
            )
            parsed.append(item)
        return parsed


    parsed_batch = parse_records(batch)
    vocab = build_vocab(parsed_batch)
    matrix = build_matrix(parsed_batch, vocab)
    # first len(vocab) items in weights are for categories.
    # the rest of the array is specific 
    weights = ([CLUSTERING_WEIGHTS.get("category", 1.0)] * len(vocab)) + [
        CLUSTERING_WEIGHTS.get("difficulty", 0.5),
        CLUSTERING_WEIGHTS.get("prior_knowledge", 0.5)
    ]
    weights = np.array(weights)
    
    if save_batch_to_file:
        out = open(f'{OUTFILE_NAME}', 'a')
        for doc in batch:
            try:
                out.write(json.dumps(doc) + "\n")
            except Exception as exc:
                print_rank_log(f"{exc}")
        out.close()

    return kmeans.partial_fit(matrix, sample_weight=weights)


def run_corpus_clustering(paths: list, nprocs: int, inference_method: str, model: str, categories: list, temperature: float, maximum_json_amt: int, tokenized_input=None):
    """
    Path is the root directory of the training data
    """
    all_jsons = []
    if not tokenized_input:
        all_jsons = do_preprocessing(paths, nprocs, maximum_json_amt)
    else:
        ... # do we need to do something with jsons here
    
    results=None
    # now that we have a small subset of jsons we do the analysis with an LLM to start
    if inference_method == "llm":
        # get generator
        results = process_with_llm(all_jsons, model, ["Beginner", "Intermediate", "Advanced", "Expert"])

    else:
        raise NotImplementedError("Non-llm analysis methods not implemented.")
    
    # One process should send seed documents to all other processes

    # some options that may turn into cli args
    docs_for_clustering = CLUSTER_SIZE
    idx = 0
    N_CLUSTERS = 4 # do difficulty clustering for now
    SEED = 42
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=BATCH_SIZE, n_init="auto")    
    # use generator to build clusters for each rank
    for batch in results:
        # update the clusters with the next batch of data
        kmeans = cluster_step(kmeans, batch, 42)

        # print centroids
        print_rank_log("New cluster centers:")
        print_rank_log(f"{kmeans.cluster_centers_}")

        # get c 
        # processed = idx * len(batch)
        # if processed >= docs_for_clustering:
        #     # compute the documents closest to each centroid
        #     ...

        #     processed = len(batch)
        #     idx = 0



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
    args = parse(ExplorationArgs)

    subsets = args.subset
    N_CATEGORIES = args.n_categories
    OUTFILE_NAME = f"{args.outfile}_rank{RANK}.txt"
    OUTFILE_LOG = f"{Path(args.outfile).parents[0]}/rank{RANK}.log"

    # clear any existing json data before the job starts
    if os.path.exists(OUTFILE_NAME):
        os.remove(OUTFILE_NAME)

    if os.path.exists(OUTFILE_LOG):
        os.remove(OUTFILE_LOG)

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
    
    DEFAULT_SAMPLE_PROBABILITY = args.sample_prob
    MAX_DOCUMENT_LENGTH = args.max_doc_length

    paths = []
    # if not using_preprocessed_input: # if we're not using an existing pre-processed input then we need to load json paths
    if RANK == 0:
        paths = get_json_paths(Path(args.data), subsets, args.max_json_amt)
        random.shuffle(paths) # shuffle the array to attempt to get an even distribution of data for processes
        chunked_paths = np.array_split(paths, COMM.Get_size())
        paths = []
        for chunk in chunked_paths:
            paths.append(chunk)
        
        # print_rank_log(str(paths))
        # try using blendcorpus
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = '12355'
        # dist.init_process_group(backend="nccl", rank=0, world_size=1)
        # config = get_config()
        # print(config)
        # config.data_file_list=args.data
        # init_ret = init_distributed()
        # mpu.initialize_model_parallel()
        # config.data_file_list="/home/wkwiecinski/polaris_data/wiki2.txt"
        # config.seq_length=1000
        # train,valid,test = build_gpt_datasets(config)
        # print_rank_log(str(train))
    else: # other processes wait for jsons to process
        paths = []

    paths = COMM.scatter(paths, root=0)

    # get the corpus metadata
    clusters = run_corpus_clustering(paths, int(args.threads), args.inference_method, args.model, args.categories, args.temperature, args.max_json_amt, args.tokenized_input)

    # merge step (oh god I'm a physicist)
    # for i in range(COMM.Get_size()):
        # with open(OUTFILE, 'a') as combined:
            # with open(OUTFILE_NAME, 'r') as rank_file:
                # for line in rank_file.readlines():
                    # combined.write(line)
