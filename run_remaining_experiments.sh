#!/bin/bash
# Run remaining experiments (exp2b, exp3, ablation_occ) sequentially.
# Usage: bash run_remaining_experiments.sh
# Run inside tmux so it survives SSH disconnections.

set -e

LOG="experiment_log_$(date +%Y%m%d_%H%M%S).txt"
EXPERIMENTS=("exp2" "exp2b" "exp3" "ablation_occ")

echo "=====================================" | tee $LOG
echo "Starting remaining experiments: $(date)" | tee -a $LOG
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
echo "All remaining experiments finished: $(date)" | tee -a $LOG
echo "=====================================" | tee -a $LOG
