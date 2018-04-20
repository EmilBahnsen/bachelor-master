for rate in $(seq 0.0010 0.00005 0.0015); do
  echo '{
    "number_of_atoms": 24,
    "train_part": 0.10,
    "n_epochs": 2500,
    "batch_size": 1000, # HAVE MODIFED TRAINER TO USE ALL USING get_all()
    "train_keep_probs": [0.7,0.7],
    "learning_rate": '$rate',
    "id": "rate_'$rate'",
    "data_directory": "carbondata/bachelor2018-master/CarbonData",
    "feature_directory": "features",
    #"feature_list_file": "features/f1.txt", # A feature file overwrites the feature criteria
    "feature_criteria": {"eta": [2,8,20,40,80], "R_s": [0,1,3,5,7], "R_c": [7], "zeta": [1,2,3], "lambda": [1,-1]},
    "hidden_neuron_count": [40],
    "log_root_dir": "./logs/nn_logs/rates_using_some_5"
  }' > "tmp/params_"$rate".txt"

  ./network_train.sh "tmp/params_"$rate".txt"
done
