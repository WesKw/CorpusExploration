#!/bin/bash -l
#PBS -l select=1:system=polaris:ncpus=1:ngpus=4
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab
. /eagle/datascience_collab/wkwiecinski/torchtitan/.venv/bin/activate

cd /eagle/datascience_collab/wkwiecinski/torchtitan/ # go to run directory

CLUSTER_SCRIPT=/home/wkwiecinski/CorpusExploration/src/polaris_jobscripts/polaris_build_order.sh
JOBS="${JOBS:-/eagle/datascience_collab/wkwiecinski/torchtitan/config.json}"
MODEL="${MODEL:-2b}"
MODULE=ezpz.agpt
CONFIG=agpt_$MODEL
CACHE_PATH="/eagle/datascience_collab/wkwiecinski/cached/"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-3}"
HF_MODEL_PATH="/eagle/datascience_collab/wkwiecinski/torchtitan/assets/hf/gemma-7b"

if [[ $MODEL == "300m" || $MODEL == "proxy1b" || $MODEL == "proxy1bllama2" ]]; then
	BATCH_SIZE="2"
fi

if [[ $MODEL == "proxy1bllama2" ]]; then
	HF_MODEL_PATH="/eagle/datascience_collab/wkwiecinski/torchtitan/assets/hf/llama-7b"
	echo "Using $HF_MODEL_PATH tokenizer"
fi

# ORDERINGS="${ORDERINGS:-none difficulty difficulty-reverse}"
ORDERINGS_DEFAULT=("none" "difficulty" "difficulty-reverse")
ORDERINGS=(${ORDERINGS[@]:-${ORDERINGS_DEFAULT[@]}})

# remove any cached data
rm -rf $CACHE_PATH

echo "USING $CONFIG"

DATA_DIR="${DATA:-}"
if [[ -z $DATA_DIR ]]; then
	echo "DATA location not found."
	exit 1
fi

sum_processed=0
for file in $(ls $DATA_DIR/rank?.log); do
	echo "$file"
	files_processed=$(tail -n 1 $file | grep -oE '[0-9]+/[0-9]+' | head -n 1);
	if [[ -n $files_processed ]]; then
		IFS='/'
		read -r processed total <<< "$files_processed" && ((sum_processed += processed));
		unset IFS
	fi
done;

echo "PROCESSED DOCUMENTS: $sum_processed"
documents_per_step=40
training_steps=$((sum_processed / documents_per_step))
training_steps=$(( 15000 * EPOCHS ))
checkpoint_interval=1000
remainder=$(( training_steps % checkpoint_interval ))
last_checkpoint=$(( training_steps - remainder ))
echo "STEPS: $training_steps | Final Checkpoint: $last_checkpoint"

# once we have all the model data plot it here
PLOT_DIR=/home/wkwiecinski/training_results/$PBS_JOBID
mkdir -p $PLOT_DIR

for dataset in "${ORDERINGS[@]}"; do
	input_file="$DATA_DIR/$dataset.jsonl"
	echo $input_file
	echo $dataset

	checkpoint="$CONFIG/${training_steps}_${dataset}_model"

	if [[ $dataset == "none" ]]; then # order is not required, shuffle the data
		# run the ordering code
		(ORDER=$dataset START=$DATA_DIR OUTFILE=$input_file $CLUSTER_SCRIPT)

		# # create a shuffled version of a sorted dataset.
		shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"42" -nosalt </dev/zero 2>/dev/null) "$input_file" -o "$input_file"

		ezpz launch python3 -m torchtitan.experiments.ezpz.train \
			--module="${MODULE}" \
			--config="${CONFIG}" \
			--training.steps="${training_steps}" \
			--dataloader.dataset="$dataset" \
			--dataloader.dataset-path="${input_file}" \
			--checkpoint.folder "$checkpoint" \
			--dataloader.no-shuffle \
			--dataloader.no-shuffle-sample-in-corpus \
			--checkpoint.enable \
			--checkpoint.interval $checkpoint_interval \
			--training.local-batch-size=$BATCH_SIZE
		# echo "skipping training"
	else # ORDER IS REQUIRED
		# run the ordering here
		(ORDER=$dataset START=$DATA_DIR $CLUSTER_SCRIPT)

		# if we're doing no ordering shuffle data with a seed
		if [[ $dataset == "none" ]]; then
			shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"42" -nosalt </dev/zero 2>/dev/null) "$input_file" -o "$input_file"
		fi

		# then call the training pipeline with the clustering order
		ezpz launch python3 -m torchtitan.experiments.ezpz.train \
			--module="${MODULE}" \
			--config="${CONFIG}" \
			--training.steps="${training_steps}" \
			--dataloader.dataset="$dataset" \
			--dataloader.dataset-path="$input_file" \
			--checkpoint.folder "$checkpoint" \
			--dataloader.no-shuffle \
			--dataloader.no-shuffle-sample-in-corpus \
			--checkpoint.enable \
			--checkpoint.interval $checkpoint_interval \
			--training.local-batch-size=$BATCH_SIZE
	fi

	mkdir -p $PLOT_DIR/$checkpoint
	logdir=/eagle/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train
	# if config is 300m, log path changes to home directory
	if [[ $MODEL == "300m" ]]; then
		logdir=/home/wkwiecinski/logs/torchtitan.experiments.ezpz.train
	fi

	# if [[ $MODEL == "proxy1b" ]]; then
	# 	logdir=/home/wkwiecinski/logs/torchtitan.experiments.ezpz.train
	# fi

	# most recent log is the training we just did, move that to plot directory
	# logdir=/eagle/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train # for 2b config
	# logdir=/home/wkwiecinski/logs/torchtitan.experiments.ezpz.train
	most_recent_log=$(ls -t $logdir/ | head -n 1)
	cp $logdir/$most_recent_log $PLOT_DIR/${checkpoint}.jsonl

	# generate the huggingface model from the final checkpoint
	python /eagle/datascience_collab/wkwiecinski/torchtitan/torchtitan/experiments/ezpz/eval/convert_to_hf.py /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/step-$last_checkpoint/ /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model --hf_assets_path $HF_MODEL_PATH --model_name experiments.ezpz.agpt --model_flavor $MODEL --export_dtype float32

	# copy the tokenizer stuff over to the model
	cp /eagle/datascience_collab/wkwiecinski/torchtitan/torchtitan/experiments/ezpz/eval/configs/agpt_${MODEL}_config.json /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/config.json
	cp $HF_MODEL_PATH/*.json /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/
	cp $HF_MODEL_PATH/tokenizer.model /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/

	# evaluate with lm_eval
	echo "Beginning evaluation..."
	model_location=/eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/
	lm_eval --model hf --model_args pretrained=$model_location,tokenizer=$model_location,max_length=4096 --tasks hellaswag,arc_easy,openbookqa,lambada_openai,wikitext --device cuda:0 --output_path /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/results --batch_size 4 --log_samples
	cp /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/results/*/results_*.json $PLOT_DIR/${checkpoint}_lm_eval.json

done

echo "RUNNING: python3 /home/wkwiecinski/CorpusExploration/src/training_dashboard.py --ezpz_logs_glob "$PLOT_DIR/*.jsonl" --output_dir $PLOT_DIR"
# generate training dashboards
# python3 /home/wkwiecinski/CorpusExploration/src/training_dashboard.py --ezpz_logs_glob "$PLOT_DIR/*.jsonl" --output_dir $PLOT_DIR

