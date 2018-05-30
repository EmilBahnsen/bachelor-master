#!/bin/bash
# usage: . run_all_nodes_depth.sh [test]
DIR=$(pwd)
cd .. # Back to root
for struc_part in $(seq -f %g 0.05 0.05 1.0); do
	sed 's/NNN/'$struc_part'/g' run_scripts/params_run_all_structures_number.txt > params.tmp
	batch_size=$(echo "1000 $struc_part" | awk '{print int($1*$2)}')
	sed -i 's/BBB/'$batch_size'/g' params.tmp
	#break
	. network_train.sh params.tmp $1
done
cd $DIR
