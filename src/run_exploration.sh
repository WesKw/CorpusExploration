#!/bin/bash -l
#PBS -l select=1:system=crux
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -l filesystems=home:eagle
#PBS -q workq-route
#PBS -A datascience_collab

export OLMIX="/eagle/datascience_collab/venkatv/olmo-mix-1124/data/"
model="google/gemma-4-31B-it"
# model="openai/gpt-oss-120b"
subset=starcoder
sampleprob=0.025
sample=1000
temp=0.2
threads=8

. ~/.corpius/bin/activate
cd /home/wkwiecinski/CorpusExploration/src
# python exploration.py $OLMIX --threads 4 --subset $subset --sample-prob $sampleprob --model $model
python exploration.py $OLMIX --threads $threads --sample $sample --model $model --temperature $temp --subset $subset > cluster.log
python visualize_clusters.py output.txt
python document_similarity_graph.py output.txt --method "knn" --k "15"

save_dir="$model-clustering-$sample-$sampleprob-$temp-$subset"
mkdir -p $save_dir
mv cluster_dashboard.png $save_dir
mv similarity_graph.png $save_dir
mv output.txt $save_dir
mv cluster.log $save_dir