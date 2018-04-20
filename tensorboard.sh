tensorboard --logdir=./logs/nn_logs/feature_full --port 6005 &
tensorboard --logdir=./logs/nn_logs/energy_interval_small --port 6006 &
tensorboard --logdir=./logs/nn_logs/features_few --port 6007 &
#tensorboard --logdir=./logs/nn_logs/layers --port 6008 &
tensorboard --logdir=./logs/nn_logs/energy_interval_deep --port 6008 &
tensorboard --logdir=./logs/nn_logs/layers_all --port 6009 &

