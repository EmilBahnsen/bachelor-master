#!/bin/bash
# usage: . run_all_nodes_width.sh [test]
DIR=$(pwd)
cd .. # Back to root
#for node_num in $(seq -f %g 20 2 80); do
for node_num in $(seq 3 2 13); do
	sed 's/NNN/'$node_num'/g' run_scripts/$1 > params.tmp
	. network_train.sh params.tmp $2
done
cd $DIR
