#!/bin/bash
PYBIN=~/miniconda3/envs/HTDCNet/bin/python
PROJDIR=/home/ubuntu/user_liu_20260403/python_projects/bibm2026
LOGDIR=$PROJDIR/results/logs
mkdir -p $LOGDIR

echo "============================================" > $LOGDIR/run_viz.log
echo "Visualization experiments" >> $LOGDIR/run_viz.log
echo "Started: $(date)" >> $LOGDIR/run_viz.log
echo "============================================" >> $LOGDIR/run_viz.log

echo ">>> t-SNE visualization..." >> $LOGDIR/run_viz.log
echo "Start: $(date)" >> $LOGDIR/run_viz.log
PYTHONUNBUFFERED=1 $PYBIN $PROJDIR/run_tsne.py > $LOGDIR/tsne_v4_output.log 2>&1
echo "t-SNE done: $(date)" >> $LOGDIR/run_viz.log

echo ">>> Confusion matrices (OpenBMI)..." >> $LOGDIR/run_viz.log
echo "Start: $(date)" >> $LOGDIR/run_viz.log
PYTHONUNBUFFERED=1 $PYBIN $PROJDIR/run_confusion_matrices.py > $LOGDIR/confmat_output.log 2>&1
echo "ConfMat done: $(date)" >> $LOGDIR/run_viz.log

echo "============================================" >> $LOGDIR/run_viz.log
echo "ALL VIZ COMPLETED: $(date)" >> $LOGDIR/run_viz.log
echo "============================================" >> $LOGDIR/run_viz.log
