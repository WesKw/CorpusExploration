#!/bin/bash -l
#PBS -l select=1:system=crux
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

. ~/.corpius/bin/activate
module load cray-python/3.11.7
cd /home/wkwiecinski/CorpusExploration/src

starting_location="/eagle/datascience_collab/wkwiecinski/runs/gemma-4-31B-it-237195.crux-pbs-0001.head.cm.crux.alcf.anl.gov"
llm_data_glob="$starting_location/merged.out_rank?.json"
shard_data_glob="$starting_location/merged.out_rank?_dump.json_pid*.json"
clusters=4
seed=42
batch_size=10000
weights="/home/wkwiecinski/CorpusExploration/src/json_configs/difficulty-bias.json"

for order in "none" "difficulty"; do
    outfile="$starting_location/textdata_$order.jsonl";
    # dataset_dir="$save_dir/";
    python ordering_step.py --llm_data_regex "$llm_data_glob" --shard_data_regex "$shard_data_glob" --method "$order" --sort --n_clusters $clusters --outfile "$outfile" --weights_json "$weights" --max_kmeans_points -1;

    if [[ $order != "none" ]]; then
        outfile="$starting_location/textdata_reverse_$order.json";
        # dataset_dir="$save_dir/";
        python ordering_step.py --llm_data_regex "$llm_data_glob" --shard_data_regex "$shard_data_glob" --method "$order" --sort --n_clusters $clusters --outfile "$outfile" --reverse --weights_json "$weights" --max_kmeans_points -1;
    fi
done

