#!/bin/bash -l
#PBS -l select=10:system=crux
#PBS -l place=scatter
#PBS -l walltime=10:00:00
#PBS -l filesystems=home:eagle
#PBS -q workq-route
#PBS -A datascience_collab

#["algebraic-stack", "arxiv", "dclm", "open-web-math", "pes2o", "starcoder", "wiki", "pes2o"]

export OLMIX="/eagle/datascience_collab/venkatv/olmo-mix-1124/data/"
model="google/gemma-4-31B-it"
# model="openai/gpt-oss-120b"
sampleprob=0.01
temp=0
threads=4
max_jsons=10
max_doc_length=1250
outfile="out$PBS_JOBID.txt"
sample_prob_json="./sample_rates.json"

# . ~/.corpius/bin/activate
module load cray-python/3.11.7
cd /home/wkwiecinski/CorpusExploration/src

# save_dir="$model-clustering-$sample-$sampleprob-$temp-maxjsons$max_jsons-maxlength$max_doc_length"
save_dir="$model-$PBS_JOBID"
mkdir -p $save_dir
out="$save_dir/$outfile"

cp $sample_prob_json $save_dir

# python exploration.py $OLMIX --threads 4 --subset $subset --sample-prob $sampleprob --model $model
mpiexec -n 2 /opt/cray/pe/python/3.11.7/bin/python exploration.py \
    $OLMIX --threads $threads --sample-prob $sampleprob --model $model --temperature $temp \
    --max-json-amt $max_jsons --max-doc-length $max_doc_length --outfile $out \
    --subset-sample-prob $sample_prob_json --subset wiki

# /opt/cray/pe/python/3.11.7/bin/python visualize_clusters.py $out --out "cluster_dashboard_$PBS_JOBID.png"
# /opt/cray/pe/python/3.11.7/bin/python document_similarity_graph.py $out --method "knn" --k "15" --out "similarity_graph_$PBS_JOBID.html"

# mv "cluster_dashboard_$PBS_JOBID.png" $save_dir
# mv "similarity_graph_$PBS_JOBID.html" $save_dir
# mv output.txt $save_dir
# mv cluster.log $save_dir