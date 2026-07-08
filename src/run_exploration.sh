#!/bin/bash -l
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=7:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

export OLMIX="/eagle/datascience_collab/venkatv/olmo-mix-1124/data/wiki/"
model="google/gemma-4-31B-it"
# model="openai/gpt-oss-120b"
sampleprob=0.01
temp=0
threads=4
max_jsons=1
max_doc_length=2000
outfile="out$PBS_JOBID.txt"
sample_prob_json="./sample_rates.json"

# . ~/.corpius/bin/activate
# module load cray-python/3.11.7
# . ~/.bash_profile
# . $TRAIN
cd /home/wkwiecinski/CorpusExploration/src

# save_dir="$model-clustering-$sample-$sampleprob-$temp-maxjsons$max_jsons-maxlength$max_doc_length"
save_dir="$model-$PBS_JOBID"
mkdir -p $save_dir
out="$save_dir/merged.out"

cp $sample_prob_json $save_dir

# python exploration.py $OLMIX --threads 4 --subset $subset --sample-prob $sampleprob --model $model
mpiexec -n 1 python exploration.py \
    --data $OLMIX --threads $threads --sample_prob $sampleprob --model $model \
    --max_json_amt $max_jsons --max_doc_length $max_doc_length --outfile $out \
    --subset_sample_prob_file $sample_prob_json --subset wiki

#python merge_step.py "$save_dir/*_rank?.txt" --outfile "$out"
#python visualize_clusters.py "$out" --out "cluster_dashboard_$PBS_JOBID.png"
#python document_similarity_graph.py $out --method "knn" --k "15" --out "similarity_graph_$PBS_JOBID.html"

#mv "cluster_dashboard_$PBS_JOBID.png" $save_dir
#mv "similarity_graph_$PBS_JOBID.html" $save_dir
#mv output.txt $save_dir
#mv cluster.log $save_dir
