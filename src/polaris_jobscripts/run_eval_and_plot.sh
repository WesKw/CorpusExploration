#!/bin/bash
#PBS -l select=1:system=polaris:ncpus=1:ngpus=4
#PBS -l place=scatter
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

. /eagle/datascience_collab/wkwiecinski/torchtitan/.venv/bin/activate
cd /eagle/datascience_collab/wkwiecinski/torchtitan/ # go to run directory

# PLOT_DIR=/home/wkwiecinski/training_results/$PBS_JOBID

STEPS=45000
LOG_LOCATION="/eagle/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train"
EVALCHECKPOINTS="{EVALCHECKPOINTS:-true}" # generate a HF model for each checkpoint and generate data for the eval suite.
METHODS_DEFAULT=("none" "difficulty" "difficulty-reverse", "stem", "stem-reverse", "code", "code-reverse") # the models to evaluate given a data ordering
METHODS=(${METHODS[@]:-${METHODS_DEFAULT[@]}})
HF_MODEL_PATH="/eagle/datascience_collab/wkwiecinski/torchtitan/assets/hf/llama-7b"
CONVERT_TO_HF="/eagle/datascience_collab/wkwiecinski/torchtitan/torchtitan/experiments/ezpz/eval/convert_to_hf.py"
GET_EZPZ_LOG="/home/wkwiecinski/CorpusExploration/src/get_most_recent_log_for_model.py"
MODEL="${MODEL:-proxy1bllama2}"
PLOT_DIR=/home/wkwiecinski/training_evaluations/$MODEL
mkdir -p $PLOT_DIR
MODEL_LOCATION="/eagle/datascience_collab/wkwiecinski/torchtitan/outputs/agpt_$MODEL"

# here I'm just gathering the jsons for each ordering just in case
for method in ${METHODS[@]}; do
    echo "Moving $method log file to $PLOT_DIR"
    python $GET_EZPZ_LOG $PLOT_DIR/agpt_${MODEL}_${method}.jsonl $method $LOG_LOCATION
done

for method in ${METHODS[@]}; do
    echo "Evaluating $method"

    # run the evaluation suite on each checkpoint for the model
    trained_model_location="${MODEL_LOCATION}/${STEPS}_${method}_model"
    for checkpt in $(ls -d $trained_model_location/step-*); do
        echo "Evaluating checkpoint [$checkpt]"
        # remove the existing results folder for the new checkpoint
        rm -rf $trained_model_location/results
        mkdir -p $trained_model_location/results

        # generate the huggingface model from the checkpoint
        echo "Converting to HF model"
        python $CONVERT_TO_HF $checkpt $checkpt/model --hf_assets_path $HF_MODEL_PATH --model_name experiments.ezpz.agpt --model_flavor $MODEL --export_dtype float32

        echo "Copying tokenizer files to checkpoint model location"
        cp /eagle/datascience_collab/wkwiecinski/torchtitan/torchtitan/experiments/ezpz/eval/configs/agpt_${MODEL}_config.json $checkpt/model/config.json
        cp $HF_MODEL_PATH/*.json $checkpt/model/
        cp $HF_MODEL_PATH/tokenizer.model $checkpt/model/

        # Evaluate checkpoint on a breadth of downstream tasks
        eval_model_path=$checkpt/model
        eval_results_path=$checkpt/results
        tasks=arc_easy,openbookqa,lambada_openai,wikitext,truthfulqa,gsm8k
        shots=5
        echo "Evaluating on $tasks with num shots $shots"
        lm_eval --model hf --model_args pretrained=$eval_model_path,tokenizer=$eval_model_path,max_length=4096 \
            --tasks $tasks --device cuda:0 --output_path $eval_results_path --batch_size 8 --log_samples \
            --num_fewshot 5

        # move results to final plotting directory
        echo "Copying results to plotting directory"
        stepname=$(basename "$checkpt")
        cp $eval_results_path/*/results_*.json $PLOT_DIR/agpt_${MODEL}_${method}_${stepname}_evaluation.json
    done

    # then, grab the most recent log file for the ordering method
    # python $GET_EZPZ_LOG $PLOT_DIR/agpt_${MODEL}_${method}.jsonl $method $LOG_LOCATION
done

# once all ordering methods have been evaluated, create the dashboard
# echo "RUNNING: python3 /home/wkwiecinski/CorpusExploration/src/training_dashboard.py --ezpz_logs_glob "$PLOT_DIR/*.jsonl" --output_dir $PLOT_DIR"
# # generate training dashboards
# python3 /home/wkwiecinski/CorpusExploration/src/training_dashboard.py --ezpz_logs_glob "$PLOT_DIR/*.jsonl" --output_dir $PLOT_DIR
