#!/bin/bash
# Usage: . run_all_Rc.sh [test]
DIR=$(pwd)
cd .. # Back to root
for Rc in 5; do
#for Rc in $(seq -f %g 2.5 0.1 5.5); do
	fea_list='feature_lists/f13_Rc'$Rc'.txt'
	echo $fea_list;
	Rc=${fea_list/feature_lists\/f13_Rc/};
	Rc=${Rc/.txt/};

	sed 's/CCC/'$Rc'/g' run_scripts/params_run_all_Rc.txt > params.tmp
	. network_train.sh params.tmp $1
done
cd $DIR
