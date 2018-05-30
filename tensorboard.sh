#!/bin/bash
ulimit -m 2097152 #-v 2097152

DIRS="logs/nn_logs/features_few/Rc*/z-score/*_ba5pct/struc*"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/??-??_ba0.5k/*"
#DIRS="logs/nn_logs/features_few/Rc2.73/z-score/*"
#DIRS=$DIRS" logs/nn_logs/features_few/Rc5/z-score/*"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/* logs/nn_logs/features_few/Rc2.7/z-score/*_ba1k"
#DIRS="logs/nn_logs/features_few/Rc*/z-score/100-100-100_ba1k*"
#DIRS="logs/nn_logs/features_few/Rc*/z-score/100-100_ba1k*"
#DIRS="logs/nn_logs/features_few/Rc2.7/z-score/55-* logs/nn_logs/features_few/Rc*/z-score/100-*"
#DIRS="logs/nn_logs/features_few/Rc*/z-score/100_ba*"
#DIRS="logs/nn_logs/features_few/Rc2.7/z-score/lr*/100"
#DIRS="logs/nn_logs/features_few/Rc2.7/z-score/*"
#tensorboard --logdir=$(i=0; for dir in logs/nn_logs/features_few/*/z-score/100_ba1k; do echo -n $dir:$dir, ; let i=$i+1; done; echo -n log/sentinel) --port 6007  &
#tensorboard --logdir=$(i=0; for dir in logs/nn_logs/features_few/*/z-score/100-100_ba1k; do echo -n $dir:$dir, ; let i=$i+1; done; echo -n log/sentinel) --port 6007  &
#tensorboard --logdir=$(i=0; for dir in logs/nn_logs/features_few/*/z-score/150_ba1k; do echo -n $dir:$dir, ; let i=$i+1; done; echo -n log/sentinel) --port 6007  &
#tensorboard --logdir=$(i=0; for dir in logs/nn_logs/features_few/Rc4.2/z-score/*; do echo -n $dir:$dir, ; let i=$i+1; done; echo -n log/sentinel) --port 6007  &
tensorboard --logdir=$(i=0; for dir in $DIRS; do echo -n $dir:$dir, ; let i=$i+1; done; echo -n log/sentinel) --port 6007  &
