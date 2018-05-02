import os
import sys
import ast
from shutil import copyfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from feature_data_provider import *
import datetime
import scipy.stats

class NetworkTrainer:
    def __init__(self,params):
        def load_param(name):
            try:
                value = params[name]
            except KeyError:
                print("Error: Parameter", name, "not found!")
                return None
            print(name, "=", value)
            return value

        # Create the folder for the model output
        self.log_root_dir = load_param("log_root_dir")
        run_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
        self.run_name = run_timestamp
        self.log_dir = os.path.join(self.log_root_dir, self.run_name)
        print("Log dir:", self.log_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        f = open(os.path.join(self.log_dir, "params.txt"),"w+")
        f.write(str(params)) # save the params to a filePath
        f.close()

        # Load feature criteria if no feature_file_list is present
        import feature_pick
        feature_list_file = load_param("feature_list_file")
        feature_dir = load_param("feature_directory")
        if feature_list_file is None: # If not the critearia list is loaded
            print("... using criteria dictionary instead!")
            feature_criteria = load_param("feature_criteria")
            feature_file_list = feature_pick.pick_features(feature_dir, feature_criteria)
        else:
            feature_file_list = [x.strip() for x in open(feature_list_file, "r").readlines()]

        train_part = load_param("train_part")
        self.n_epochs   = load_param("n_epochs")
        self.batch_size = load_param("batch_size")
        self.train_dropout_rate = load_param("train_dropout_rate")
        self.n_atoms = load_param("number_of_atoms")
        data_dir = load_param("data_directory")
        self.checkpoint_path = load_param("checkpoint_path")
        self.hidden_neuron_count = load_param("hidden_neuron_count")
        self.learning_rate = load_param("learning_rate")
        self.summary_interval = load_param("summary_interval")
        feature_scaling = load_param("feature_scaling")
        self.max_checkpoints_keep = load_param("max_checkpoints_keep")

        print("All parameters loaded.")

        # Load features
        print("Loading features...")

        # Load the features specified
        self.carbon_data = CarbonData(data_dir = data_dir, structure_size = self.n_atoms)
        self.featureProvider = FeatureDataProvider(feature_file_list, self.carbon_data, trainPart = train_part, normalized_labels=False, feature_scaling=feature_scaling)
        print("Train samples:",len(self.featureProvider.train.labels))
        print("Test samples: ",len(self.featureProvider.test.labels))
        print("Labels shape:",self.featureProvider.train.labels.shape)
        print("Data shape:",self.featureProvider.train.data.shape)
        print("done.")

    def train(self):
        print("Training network...")
        # Reset tf
        tf.reset_default_graph()

        # Load the model
        import network_model
        m = network_model.Model2(self.featureProvider.train.data.shape[2], n_nodes_hl = self.hidden_neuron_count, n_atoms = self.n_atoms, learning_rate = self.learning_rate)

        # Create a Saver object
        saver = tf.train.Saver(max_to_keep=self.max_checkpoints_keep)

        # Create a session
        with tf.Session() as sess:
            # Create a FileWriter for the log. run 'tensorboard --logdir=./logs/nn_logs'
            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            merged = tf.summary.merge_all()

            # Init all variables
            if self.checkpoint_path is None:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            else:
                saver.restore(sess, self.checkpoint_path)

            # Train the model
            epoch_G, epoch_E = self.featureProvider.train.get_all()
            test_G, test_E = self.featureProvider.test.get_all()
            feed_dict = {m.G: epoch_G,
                         m.E: epoch_E,
                         m.train_dropout_rate: self.train_dropout_rate,
                         m.is_training: True,
                         m.G_test: test_G,
                         m.E_test: test_E}
            for epoch in range(self.n_epochs):
                #epoch_loss = 0

                if self.batch_size == 0:
                    sess.run(m.train_optimzer, feed_dict)
                else:
                    for i in range(int(self.carbon_data.numberOfStructures/self.batch_size)):
                        if self.uniform_batch is True:
                            epoch_G, epoch_E = self.featureProvider.train.next_uniform_batch(self.batch_size)
                        else:
                            epoch_G, epoch_E = self.featureProvider.train.next_batch(self.batch_size)
                        feed_dict[m.G] = epoch_G
                        feed_dict[m.E] = epoch_E
                        sess.run(m.train_optimzer, feed_dict)
                        #epoch_loss += lss

                # Write summary of every summary_interval step, and at the end
                if (epoch%self.summary_interval == 0 or epoch == self.n_epochs):
                    test_feed_dict = feed_dict.copy()
                    test_feed_dict[m.is_training] = False
                    summary= sess.run(merged, feed_dict)

                    print('Epoch', epoch, 'completed out of', self.n_epochs)
                    writer.add_summary(summary, epoch)
                    saver.save(sess, os.path.join(self.log_dir, 'model'), global_step=epoch)

            # Save the whole session?
            #saver.save(sess, os.path.join(self.log_dir, 'model'))

    def train_async(self):
        pass

if __name__ == "__main__":
    # Load parameters from dictianory literal in file of first argument
    params_file = str(sys.argv[1])
    params_dictionary = str(open(params_file, 'r').read())
    params = ast.literal_eval(params_dictionary)

    nt = NetworkTrainer(params)
    nt.train()
