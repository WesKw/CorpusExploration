#!/bin/bash -l
#PBS -l select=10:system=crux
#PBS -l place=scatter
#PBS -l walltime=14:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

export OLMIX="/eagle/datascience_collab/venkatv/olmo-mix-1124/data/"
cluster="metis"
model="gemma-4-31B-it"
# model="openai/gpt-oss-120b"
sampleprob=0.001
temp=0
threads=4
max_jsons=100
max_doc_length=2000
outfile="out$PBS_JOBID.txt"
sample_prob_json="./sample_rates.json"

. ~/.corpius/bin/activate
module load cray-python/3.11.7
# . ~/.bash_profile
# . $TRAIN
# . /eagle/datascience_collab/wkwiecinski/torchtitan/.venv/bin/activate
cd /home/wkwiecinski/CorpusExploration/src

# save_dir="$model-clustering-$sample-$sampleprob-$temp-maxjsons$max_jsons-maxlength$max_doc_length"
save_dir="/eagle/datascience_collab/wkwiecinski/runs/$model-$PBS_JOBID"
mkdir -p $save_dir
out="$save_dir/merged.out"

cp $sample_prob_json $save_dir

# python exploration.py $OLMIX --threads 4 --subset $subset --sample-prob $sampleprob --model $model
mpiexec -n 10 python exploration.py \
    --data $OLMIX --threads $threads --sample_prob $sampleprob --model $model \
    --max_json_amt $max_jsons --max_doc_length $max_doc_length --outfile $out \
    --subset_sample_prob_file $sample_prob_json --cluster $cluster

# python ordering_step.py "$save_dir/*_rank?.txt" --outfile "$out"
# python visualize_clusters.py "$out" --out "cluster_dashboard_$PBS_JOBID.png"
# python document_similarity_graph.py $out --method "knn" --k "15" --out "similarity_graph_$PBS_JOBID.html"

# mv "cluster_dashboard_$PBS_JOBID.png" $save_dir
# mv "similarity_graph_$PBS_JOBID.html" $save_dir
# mv output.txt $save_dir
# mv cluster.log $save_dir
