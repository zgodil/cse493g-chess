#!/bin/bash
# Run all experiments sequentially: train then test each before moving on.
# Usage: bash run_experiments.sh
# Run inside tmux so it survives SSH disconnections.

set -e  # exit immediately on any error

LOG="experiment_log_$(date +%Y%m%d_%H%M%S).txt"
EXPERIMENTS=("exp1" "exp2" "exp2b" "exp3" "ablation_occ")

echo "=====================================" | tee $LOG
echo "Starting all experiments: $(date)"    | tee -a $LOG
echo "=====================================" | tee -a $LOG

for exp in "${EXPERIMENTS[@]}"; do

    echo "" | tee -a $LOG
    echo "-------------------------------------" | tee -a $LOG
    echo "TRAIN: $exp  |  $(date)"              | tee -a $LOG
    echo "-------------------------------------" | tee -a $LOG
    python main.py --config configs/$exp.yaml train 2>&1 | tee -a $LOG

    echo "" | tee -a $LOG
    echo "TEST: $exp  |  $(date)"               | tee -a $LOG
    echo "-------------------------------------" | tee -a $LOG
    python main.py --config configs/$exp.yaml test \
        --checkpoint checkpoints/$exp/best_model.pt 2>&1 | tee -a $LOG

    echo "DONE: $exp  |  $(date)"               | tee -a $LOG

done

echo "" | tee -a $LOG
echo "=====================================" | tee -a $LOG
echo "All experiments finished: $(date)"    | tee -a $LOG
echo "=====================================" | tee -a $LOG
