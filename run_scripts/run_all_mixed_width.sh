#!/bin/bash
# Usage: . run_all_Rc.sh [test]
DIR=$(pwd)
cd .. # Back to root
#for type in non_relaxed_single0.8_non_relaxed_single0.2 non_relaxed_single0.8_relaxed0.2 non_relaxed_double0.8_non_relaxed_double0.2 non_relaxed_double0.8_relaxed0.2; do
for type in non_relaxed_double0.8_relaxed0.2; do
#for type in multi_perturb; do
	#for struc_part in 0.01 0.005 0.001; do
	for node_num in $(seq 5 2 50); do
	#for node_num in $(seq 35 2 80); do
		sed 's/NNN/'$node_num'/g' run_scripts/params_run_all_mixed_width.txt > params.tmp
		sed -i 's/TTT/'$type'/g' params.tmp
		. network_train.sh params.tmp $1
	done
done
cd $DIR