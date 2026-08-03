#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

. /eagle/datascience_collab/wkwiecinski/torchtitan/.venv/bin/activate
SCRIPT=/home/wkwiecinski/CorpusExploration/src/ordering_step.py

# set up parameters
ORDER="${ORDER:-none}"
DATA_DIR="${START:-no_data}"
LLM_GLOB="$DATA_DIR/merged.out_rank?.json"
SHARD_GLOB="$DATA_DIR/merged.out_rank?_dump.json_pid*.json"
N_CLUSTERS="${N_CLUSTERS:-4}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-100000}"
WEIGHTS="/home/wkwiecinski/CorpusExploration/src/json_configs/$ORDER.json"
OUTFILE="${OUTFILE:-$DATA_DIR/$ORDER.jsonl}";

# call ordering step
python $SCRIPT --llm_data_regex "$LLM_GLOB" --shard_data_regex "$SHARD_GLOB" --method "$ORDER" --n_clusters $N_CLUSTERS --outfile "$OUTFILE" --weights_json "$WEIGHTS" --max_kmeans_points -1;

# if [[ $order != "none" ]]; then
#     outfile="$starting_location/textdata_reverse_$order.json";
#     # dataset_dir="$save_dir/";
#     python ordering_step.py --llm_data_regex "$llm_data_glob" --shard_data_regex "$shard_data_glob" --method "$order" --sort --n_clusters $clusters --outfile "$outfile" --weights_json "$weights" --max_kmeans_points -1;
# fi

