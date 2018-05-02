#!/bin/bash
# Usage: . run_all_Rc.sh [test]
DIR=$(pwd)
cd .. # Back to root
for fea_list in feature_lists/f13_Rc*.txt; do
	echo $fea_list;
	Rc=${fea_list/feature_lists\/f13_Rc/};
	Rc=${Rc/.txt/};

	sed 's/CCC/'$Rc'/g' run_scripts/params_run_all_Rc.txt > params.tmp
	. network_train.sh params.tmp $1
done
cd $DIR
