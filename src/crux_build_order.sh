#!/bin/bash -l
#PBS -l select=1:system=crux
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

# export OLMIX="/eagle/datascience_collab/venkatv/olmo-mix-1124/data/"
# cluster="metis"
# model="gemma-4-31B-it"
# model="openai/gpt-oss-120b"
# sampleprob=0.001
# temp=0
# threads=4
# max_jsons=25
# max_doc_length=2000
# outfile="out$PBS_JOBID.txt"
# sample_prob_json="./sample_rates.json"

. ~/.corpius/bin/activate
module load cray-python/3.11.7
# . ~/.bash_profile
# . $TRAIN
# . /eagle/datascience_collab/wkwiecinski/torchtitan/.venv/bin/activate
cd /home/wkwiecinski/CorpusExploration/src

# save_dir="$model-clustering-$sample-$sampleprob-$temp-maxjsons$max_jsons-maxlength$max_doc_length"
# save_dir="/eagle/datascience_collab/wkwiecinski/runs/$model-$PBS_JOBID"
# mkdir -p $save_dir
# out="$save_dir/merged.out"

# cp $sample_prob_json $save_dir

# python exploration.py $OLMIX --threads 4 --subset $subset --sample-prob $sampleprob --model $model
# mpiexec -n 10 python exploration.py \
    # --data $OLMIX --threads $threads --sample_prob $sampleprob --model $model \
    # --max_json_amt $max_jsons --max_doc_length $max_doc_length --outfile $out \
    # --subset_sample_prob_file $sample_prob_json --cluster $cluster

starting_location="/eagle/datascience_collab/wkwiecinski/runs/gemma-4-31B-it-237195.crux-pbs-0001.head.cm.crux.alcf.anl.gov"
# outputdir="$starting_location/textdata.json"

llm_data_glob="$starting_location/merged.out_rank?.json"
shard_data_glob="$starting_location/merged.out_rank?_dump.json_pid*.json"
clusters=4
seed=42
batch_size=10000
weights="./difficulty-bias.json"

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


# python visualize_clusters.py "$out" --out "cluster_dashboard_$PBS_JOBID.png"
# python document_similarity_graph.py $out --method "knn" --k "15" --out "similarity_graph_$PBS_JOBID.html"

# mv "cluster_dashboard_$PBS_JOBID.png" $save_dir
# mv "similarity_graph_$PBS_JOBID.html" $save_dir
# mv output.txt $save_dir
# mv cluster.log $save_dir
