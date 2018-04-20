#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for param_file in G*.txt; do
  echo '{
    "ntasks_per_node": 2, # Max 16 on q16
    "number_of_atoms": 24,
    "energy_interval": None,
    "train_part": 0.80,
    "n_epochs": 5000000,
    "summary_interval": 3000,
    "batch_size": 0, # 0: using all data, and uniform_batch has no effect
    "uniform_batch": False,
    "learning_rate": 1e-5,
    "feature_scaling":     True,
    "train_dropout_rate":  [0,0,0],
    "id":                  "'${param_file%.*}'",
    "hidden_neuron_count": [15,15,15],
    "log_root_dir": "./logs/nn_logs/features_few/tanh/feature_scaling/feature_num/3x15",
    "data_directory": "carbondata/bachelor2018-master/CarbonData",
    "feature_directory": "features",
    "feature_list_file": "'$DIR'/'$param_file'", # A feature file overwrites the feature criteria
  }
  ' > params.tmp
  cat params.tmp

  # Init train
  cd ~/carbon_nn
  . network_train.sh $DIR/params.tmp
  cd $DIR

  rm params.tmp
done
