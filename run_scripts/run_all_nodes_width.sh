#!/bin/bash
# usage: . run_all_nodes_width.sh [test]
DIR=$(pwd)
cd .. # Back to root
for node_num in $(seq 45 1 65); do
#for node_num in 57; do
	sed 's/NNN/'$node_num'/g' run_scripts/params_run_all_nodes_width.txt > params.tmp
	. network_train.sh params.tmp $1
done
cd $DIR
