#!/bin/bash
# Usage: . run_all_Rc.sh [test]
DIR=$(pwd)
cd .. # Back to root
for lr in $(seq -f %.5f 1e-5 1e-5 9e-5; echo 0.0001); do
	sed 's/RRR/'$lr'/g' run_scripts/params_run_all_learn_rate.txt > params.tmp
	#cat params.tmp
	. network_train.sh params.tmp $1
done
cd $DIR
