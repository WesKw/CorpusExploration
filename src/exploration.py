import time
import json
import openai
import gzip
import os
import subprocess
import math
import multiprocessing
import zstandard as zstd
import random
import inference_auth_token

from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from multiprocessing import Process,Pool,TimeoutError


UNIT_CHOICES = {
    "MB": math.pow(1024, 2),
    "GB": math.pow(1024, 3),
    "TB": math.pow(1024, 4),
}


def process_json_file(path: str):
    # print("Worker started")
    """Threads process a json"""
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
            print(f"Processed {path}")
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
        jsons.append(value)
        # print(value.keys())
        
    return (path, collection, bytes, documents, jsons)
    # print(f"\033[K{json}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

    # # do some magic analysis here
    # print(f"Removing {new_location.replace('.gz', '')}")
    # os.remove(f"./{path.name.replace('.gz', '')}")


def cluster_with_argo(jsons: list, model: str, categories: list[str], batch_size: int=10, delay: float=1.0):
    """
    Use predefined labels to cluster documents according to labels. Otherwise
    the LLM will cluster based on similarity
    """
    # get authentication token
    # print("Documents:", jsons)
    print(f"Clustering sample-size: {len(jsons)}")
    token = inference_auth_token.get_access_token()
    print("Current token:", token)

    client = openai.OpenAI(
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        api_key=token
    )
    
    # give concrete classifications for now
    # todo:: include the subsection of data that the document is from
    prompt = f"""You are a document classifier. Cluster documents by topic similarity (top 3 topics with probabilities), and add a content difficulty classification for each document into one of: {categories}. Give a prior knowledge rating for each document between 0 and 1, 0 is no prior knowledge and 1 is high domain knowledge. Respond ONLY with a JSON array: [{{"title": "<title>", "<category1>": "<probability>", "<category2>": "<probability>", "<category3>": "<probability>", "confidence": "<high|medium|low>", "difficulty": "<content_difficulty>", "prior_knowledge": "<prior_knowledge_value>", "language": "<language>"}}]"""
    all_results = []
        
    for i in range(0, len(jsons), batch_size):
        batch = jsons[i:i + batch_size]
        
        # Build a multi-document prompt
        doc_texts = []
        for doc in batch:
            # documents don't have titles, though all the jsons have a Text attribute
            doc_texts.append(f"[Content: {doc.get('text', '')[:300]}")

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": "\n\n".join(doc_texts)
                }
            ],

        )
        
        # if response.choices[0].message.content:
        batch_results = json.loads(response.choices[0].message.content)
        all_results.extend(batch_results)
        
        print(f"Processed batch {i // batch_size + 1} "
            f"({len(all_results)}/{len(jsons)} docs)")
        
        time.sleep(delay)  # Rate limiting
        
    return all_results


def get_corpus_metadata(root: Path, units: str, subset: list, nprocs: int, cluster_method: str, model: str, categories: list, sample: int):
    """
    Path is the root directory of the training data
    """
    paths = get_json_paths(root, subset)
    random.shuffle(paths) # shuffle the array to attempt to get an even distribution of work for threads
    # paths = paths
    # paths = [path for path in paths if subset != None and subset in path]

    dataset_size = 0
    skipped_file_names = set()
    collection_sizes = {}
    print(f"Total files: {len(paths)}")

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

            print(f"\033[K{file}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

    print("\nCollection totals:")
    for collection in sorted(collection_sizes.items(), key=lambda x: x[1]["size"], reverse=True):
        col = collection[0]
        print(
            f"\t{col} -> {(collection_sizes[col]['size'] / UNIT_CHOICES[units]):.02f} {units} | " \
            f"{(collection_sizes[col]['size'] / dataset_size) * 100:.02f}% | {collection_sizes[col]['processed']} of {collection_sizes[col]['total']} processed | {collection_sizes[col]['documents']} docs"
        )

    # json_sample = random.sample(jsons, sample)
    # now that we have a small subset of jsons we do the analysis with an LLM to start
    if cluster_method == "llm":
        results = cluster_with_argo(random.sample(jsons, sample), model, ["Beginner", "Intermediate", "Advanced", "Expert"])
        with open("output.txt", 'w') as out:
           for result in results:
                try:
                   out.write(json.dumps(result) + "\n")
                except Exception as exc:
                   print(exc)


def get_json_paths(root: Path, subset: list):
    gz_str = str(root) + "/**/*.gz"
    gz_paths = glob(gz_str, recursive=True)
    zstd_str = str(root) + "/**/*.zstd"
    zstd_paths = glob(zstd_str, recursive=True)
    print(f"{len(gz_paths)} gz files")
    print(f"{len(zstd_paths)} zstd files")

    if subset != None:
        paths = [path for path in gz_paths + zstd_paths if subset and any([s in path for s in subset])] 
    else:
        paths = gz_paths + zstd_paths


    print(f"Using {subset} subsets")

    # print(paths)
    return paths
    # return gz_paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Location")
    parser.add_argument("--units", help="Unit for size output", choices=["MB", "GB", "TB"], default="GB")
    parser.add_argument("--subset", action="append", help="Data subset to process", choices=["algebraic-stack", "arxiv", "dclm", "open-web-math", "pes2o", "starcoder", "wiki"], default=[])
    parser.add_argument("--threads", help="Number of processes", default=1)
    parser.add_argument("--cluster-method", help="The method of clustering to use.", choices=["llm", "transformer"], default="llm")
    parser.add_argument("--model", help="Available model to use", choices=["openai/gpt-oss-120b", "google/gemma-4-26B-A4B-it"], default="openai/gpt-oss-120b")
    parser.add_argument("--sample", help="The number of documents to sample.", default="100")
    parser.add_argument("--categories", action="append", help="Classification categories.", default=["Beginner", "Intermediate", "Advanced", "Expert"])

    args = parser.parse_args()
    get_corpus_metadata(Path(args.data), args.units, args.subset, int(args.threads), args.cluster_method, args.model, args.categories, int(args.sample))
