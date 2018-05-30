#!/bin/bash
# usage: . run_all_nodes_depth.sh [test]
DIR=$(pwd)
cd .. # Back to root
for layer_num in $(seq 1 1 30); do
	zzz=$(printf -v spaces '%*s' $layer_num ''; printf '%s\n' ${spaces// /0,})
	mmm=$(printf -v spaces '%*s' $layer_num ''; printf '%s\n' ${spaces// /5,})
	nnn=$layer_num
	sed 's/ZZZ/'$zzz'/g' run_scripts/params_run_all_nodes_depth.txt > params.tmp
	sed -i 's/MMM/'$mmm'/g' params.tmp
	sed -i 's/NNN/'$nnn'/g' params.tmp
	#. network_train.sh params.tmp $1
done
cd $DIR
