#!/bin/bash
# Re-run fixed experiments: BCI IV-2a (larger model) + t-SNE
# Ablation already completed successfully

PYBIN=~/miniconda3/envs/HTDCNet/bin/python
PROJDIR=/home/ubuntu/user_liu_20260403/python_projects/bibm2026
LOGDIR=$PROJDIR/results/logs
mkdir -p $LOGDIR

echo "============================================" > $LOGDIR/run_v2.log
echo "HTDCNet Re-run (v3) - Larger model + fixed t-SNE" >> $LOGDIR/run_v2.log
echo "Started: $(date)" >> $LOGDIR/run_v2.log
echo "============================================" >> $LOGDIR/run_v2.log

# Experiment 1: BCI IV-2a LOSO (d_model=64, 200 epochs, warmup)
echo "" >> $LOGDIR/run_v2.log
echo ">>> Starting BCI IV-2a LOSO (v3: d_model=128, 300ep)..." >> $LOGDIR/run_v2.log
echo "Start time: $(date)" >> $LOGDIR/run_v2.log
PYTHONUNBUFFERED=1 $PYBIN $PROJDIR/htdcnet_bci2a.py > $LOGDIR/bci2a_v3_output.log 2>&1
echo "BCI IV-2a v2 finished: $(date)" >> $LOGDIR/run_v2.log

# Experiment 2: t-SNE visualization (fixed attribute name)
echo "" >> $LOGDIR/run_v2.log
echo ">>> Starting t-SNE visualization (v2: fixed)..." >> $LOGDIR/run_v2.log
echo "Start time: $(date)" >> $LOGDIR/run_v2.log
PYTHONUNBUFFERED=1 $PYBIN $PROJDIR/run_tsne.py > $LOGDIR/tsne_v3_output.log 2>&1
echo "t-SNE v2 finished: $(date)" >> $LOGDIR/run_v2.log

echo "" >> $LOGDIR/run_v2.log
echo "============================================" >> $LOGDIR/run_v2.log
echo "ALL EXPERIMENTS COMPLETED: $(date)" >> $LOGDIR/run_v2.log
echo "============================================" >> $LOGDIR/run_v2.log
