#!/bin/bash -l
#PBS -l select=1:system=crux
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -l filesystems=home:eagle
#PBS -q workq-route
#PBS -A datascience_collab

OLMIX="/eagle/datascience_collab/venkatv/olmo-mix-1124/data/"

model="google/gemma-4-31B-it"
# model="openai/gpt-oss-120b"
subset="wiki"
sampleprob=0.2
sample=1000

. ~/.corpius/bin/activate
cd /home/wkwiecinski/CorpusExploration/src
# python exploration.py $OLMIX --threads 4 --subset $subset --sample-prob $sampleprob --model $model
python exploration.py $OLMIX --threads 2 --subset $subset --sample $sample --model $model