#!/bin/bash
# Usage: . run_all_Rc.sh [test]
DIR=$(pwd)
cd .. # Back to root
#for type in non_relaxed_single0.8_non_relaxed_single0.2 non_relaxed_single0.8_relaxed0.2 non_relaxed_double0.8_non_relaxed_double0.2 non_relaxed_double0.8_relaxed0.2 multi_perturb; do
for type in multi_perturb; do
	#for struc_part in 1; do
	for struc_part in $(seq -f %g 0.05 0.05 1.0); do
		sed 's/NNN/'$struc_part'/g' run_scripts/params_run_all_mixed.txt > params.tmp
		batch_size=$(echo "500 $struc_part" | awk '{print int($1*$2)}')
		steps=$(echo "25000 $struc_part" | awk '{print int($1*$2)}')
		sed -i 's/TTT/'$type'/g' params.tmp
		sed -i 's/BBB/'$batch_size'/g' params.tmp
		sed -i 's/SSS/'$steps'/g' params.tmp
		# cat params.tmp
		# break
		. network_train.sh params.tmp $1
	done
done
cd $DIR
