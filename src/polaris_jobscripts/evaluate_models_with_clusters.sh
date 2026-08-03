#!/bin/bash

#. /home/wkwiecinski/.bash_profile
#. "$TORCH_VENV"
. /eagle/datascience_collab/wkwiecinski/torchtitan/.venv/bin/activate

CLUSTER_SCRIPT=/home/wkwiecinski/CorpusExploration/src/polaris_jobscripts/polaris_build_order.sh
JOBS="${JOBS:-/eagle/datascience_collab/wkwiecinski/torchtitan/config.json}"
MODEL="${MODEL:-2b}"
MODULE=ezpz.agpt
CONFIG=agpt_$MODEL
CACHE_PATH="/eagle/datascience_collab/wkwiecinski/cached/"

# ORDERINGS="${ORDERINGS:-none difficulty difficulty-reverse}"
ORDERINGS_DEFAULT=("none" "difficulty" "difficulty-reverse")
ORDERINGS=(${ORDERINGS[@]:-${ORDERINGS_DEFAULT[@]}})
# echo ${ORDERINGS[@]}
# for order in "${ORDERINGS[@]}"; do
# 	echo "ORDER: $order"
# done

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
training_steps=$((sum_processed / 2 / documents_per_step))
training_steps=50
echo "STEPS: $training_steps"

# once we have all the model data plot it here
PLOT_DIR=/home/wkwiecinski/training_results/$PBS_JOBID
mkdir -p $PLOT_DIR

for dataset in "${ORDERINGS[@]}"; do
	input_file="$DATA_DIR/$dataset.jsonl"
	echo $input_file
	echo $dataset

	checkpoint="${training_steps}_${dataset}_model"
	rm -rf "./outputs/$checkpoint"

	if [[ $dataset == "none" ]]; then # order is not required, shuffle the data
		# run the ordering code
		(ORDER=$dataset START=$DATA_DIR OUTFILE=$input_file $CLUSTER_SCRIPT)

		# create a shuffled version of a sorted dataset.
		shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"42" -nosalt </dev/zero 2>/dev/null) "$input_file" -o "$input_file"

		ezpz launch python3 -m torchtitan.experiments.ezpz.train \
			--module="${MODULE}" \
			--config="${CONFIG}" \
			--training.steps="${training_steps}" \
			--dataloader.dataset="$dataset" \
			--dataloader.dataset-path="${input_file}" \
			--checkpoint.folder "$checkpoint" \
			--checkpoint.enable \
			--checkpoint.interval 5000 \
			--dataloader.no-infinite
	else # ORDER IS REQUIRED
		# run the ordering here
		(ORDER=$dataset START=$DATA_DIR $CLUSTER_SCRIPT)

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
			--checkpoint.interval 5000 \
			--dataloader.no-infinite
	fi

	# most recent log is the training we just did, move that to plot directory
	logdir=/eagle/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train/
	most_recent_log=$(ls -t $logdir/ | head -n 1)
	cp /eagle/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train/$most_recent_log $PLOT_DIR/${CONFIG}_${checkpoint}.jsonl

	# generate the huggingface model from the final checkpoint
	python /eagle/datascience_collab/wkwiecinski/torchtitan/torchtitan/experiments/ezpz/eval/convert_to_hf.py /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/step-$training_steps/ /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model --hf_assets_path /eagle/datascience_collab/wkwiecinski/torchtitan/assets/hf/gemma-7b/ --model_name experiments.ezpz.agpt --model_flavor 2b --export_dtype bfloat16

	# copy the tokenizer stuff over to the model
	cp /eagle/datascience_collab/wkwiecinski/torchtitan/torchtitan/experiments/ezpz/eval/configs/agpt_${MODEL}_config.json /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/config.json
	cp /eagle/datascience_collab/wkwiecinski/torchtitan/assets/hf/gemma-7b/*.json /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/
	cp /eagle/datascience_collab/wkwiecinski/torchtitan/assets/hf/gemma-7b/tokenizer.model /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/

	# evaluate with lm_eval
	lm_eval --model hf --model_args pretrained=/eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/model/ --tasks hellaswag --device cuda:0 --output_path /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/results --batch_size 8 --log_samples # 2>&1 | tee $PLOT_DIR/${CONFIG}_${checkpoint}_hellaswag.log

	cp /eagle/datascience_collab/wkwiecinski/torchtitan/outputs/$checkpoint/results/*/results_*.json $PLOT_DIR/${CONFIG}_${checkpoint}_lm_eval.json

done

# generate training dashboards
python3 /home/wkwiecinski/CorpusExploration/src/training_dashboard.py --ezpz_logs_glob "$PLOT_DIR/*.jsonl" --output_dir $PLOT_DIR

