#!/bin/bash
calculate_feature() {
  export EXP_NAME=$1
  export FEA=$2
  export ETA=$3
  export ZETA=$4
  export LAMB=$5
  export R_C=$6
  export R_S=$7
  export SAVE_DIR=$8
  export DATA_DIR=$9
  python feature_save.py && echo "Saved: "$EXP_NAME" at "$(date) || echo "Failed: "$EXP_NAME" at "$(date)
}
