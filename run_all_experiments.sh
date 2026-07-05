#!/bin/bash
# Master runner for all HTDCNet experiments
# Runs sequentially to avoid GPU contention

PYBIN=~/miniconda3/envs/HTDCNet/bin/python
PROJDIR=/home/ubuntu/user_liu_20260403/python_projects/bibm2026
LOGDIR=$PROJDIR/results/logs
mkdir -p $LOGDIR

echo "============================================" > $LOGDIR/run_all.log
echo "HTDCNet Experiment Suite" >> $LOGDIR/run_all.log
echo "Started: $(date)" >> $LOGDIR/run_all.log
echo "============================================" >> $LOGDIR/run_all.log

# Experiment 1: BCI IV-2a LOSO
echo "" >> $LOGDIR/run_all.log
echo ">>> Starting BCI IV-2a LOSO..." >> $LOGDIR/run_all.log
echo "Start time: $(date)" >> $LOGDIR/run_all.log
$PYBIN $PROJDIR/htdcnet_bci2a.py > $LOGDIR/bci2a_output.log 2>&1
echo "BCI IV-2a finished: $(date)" >> $LOGDIR/run_all.log

# Experiment 2: t-SNE visualization
echo "" >> $LOGDIR/run_all.log
echo ">>> Starting t-SNE visualization..." >> $LOGDIR/run_all.log
echo "Start time: $(date)" >> $LOGDIR/run_all.log
$PYBIN $PROJDIR/run_tsne.py > $LOGDIR/tsne_output.log 2>&1
echo "t-SNE finished: $(date)" >> $LOGDIR/run_all.log

# Experiment 3: Ablation study
echo "" >> $LOGDIR/run_all.log
echo ">>> Starting Ablation Study..." >> $LOGDIR/run_all.log
echo "Start time: $(date)" >> $LOGDIR/run_all.log
$PYBIN $PROJDIR/run_ablation.py > $LOGDIR/ablation_output.log 2>&1
echo "Ablation finished: $(date)" >> $LOGDIR/run_all.log

echo "" >> $LOGDIR/run_all.log
echo "============================================" >> $LOGDIR/run_all.log
echo "ALL EXPERIMENTS COMPLETED: $(date)" >> $LOGDIR/run_all.log
echo "============================================" >> $LOGDIR/run_all.log
