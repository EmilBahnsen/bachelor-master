#!/bin/bash
ulimit -m 2097152 #-v 2097152

#DIRS="logs/nn_logs/features_few/Rc5/z-score/50-??-29_ba0.5k*/* logs/nn_logs/features_few/Rc5/z-score/??-29-29_ba0.5k/* logs/nn_logs/features_few/Rc5/z-score/29-??-29_ba0.5k/* logs/nn_logs/features_few/Rc5/z-score/29-29-??_ba0.5k/*"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/??-??-??-??_ba0.5k*/*"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/??-68_ba0.5k/* logs/nn_logs/features_few/Rc5/z-score/86-??_ba0.5k/*"
#DIRS="logs/mixed_log/*/features_few/Rc5/z-score/*_NEW"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba0.5k/*"
DIRS="logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba10k/*"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct_NEW/*"

#DIRS="logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba0.5k logs/nn_logs/features_few/Rc5/z-score/identity logs/nn_logs/features_few/Rc5/z-score/leaky_relu logs/nn_logs/features_few/Rc5/z-score/relu logs/nn_logs/features_few/Rc5/z-score/selu logs/nn_logs/features_few/Rc5/z-score/sigmoid logs/nn_logs/features_few/Rc5/z-score/sign logs/nn_logs/features_few/Rc5/z-score/softplus logs/nn_logs/features_few/Rc5/z-score/tanh"
#DIRS="logs/nn_logs/features_few/Rc5/z-score/*_ba0.5k/*"
#DIRS="logs/nn_logs/features_few/Rc*/z-score/*_ba5pct/struc*"
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
tensorboard --logdir=$(i=0; for dir in $DIRS; do echo -n $dir:$dir, ; let i=$i+1; done; echo -n log/sentinel) --port 6007
