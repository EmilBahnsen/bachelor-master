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

# Load parameters from dictianory literal in file of first argument
params_file = str(sys.argv[1])
params_dictionary = str(open(params_file, 'r').read())
params = ast.literal_eval(params_dictionary)

def load_param(name):
    try:
        value = params[name]
    except KeyError:
        print("Error: Parameter", name, "not found!")
        return None
    print(name, "=", value)
    return value

# Create the folder for the model output
run_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
model_dir = "./model_final/" + run_timestamp
os.makedirs(model_dir, exist_ok=True)
copyfile(params_file, os.path.join(model_dir, "params.txt")) # save the params dectionary-file

# Load feature criteria if no feature_file_list is present
import feature_pick
feature_list_file = load_param("feature_list_file")
if feature_list_file is None: # If not the critearia list is loaded
    print("... using criteria dictionary instead!")
    feature_criteria = load_param("feature_criteria")
    feature_file_list = feature_pick.pick_features(feature_criteria)

train_part = load_param("train_part")

n_epochs   = load_param("n_epochs")
batch_size = load_param("batch_size")
keep_probs = load_param("train_keep_probs")
train_keep_prob_h1 = keep_probs[0]
train_keep_prob_h2 = keep_probs[1]

n_nodes_hl = load_param("hidden_neuren_count")

print("All parametrs loaded.")

# Load features
print("Loading features...", end="", flush=True)

# Load the features specified
featureProvider = FeatureDataProvider(feature_file_list,trainPart=train_part, normalized=True)
print("Train samples:",len(featureProvider.train.labels))
print("Test samples: ",len(featureProvider.test.labels))
print("Labels shape:",featureProvider.train.labels.shape)
print("Data shape:",featureProvider.train.data.shape)
print("done.")

epoch_G, epoch_E = featureProvider.train.next_batch(2)
epoch_G_, epoch_E_ = featureProvider.train.next_batch(2)
# print(epoch_G)
# print(epoch_E)
# print("")
# print(epoch_G_)
# print(epoch_E_)
# exit()

print("Training network...")
# Reset tf
tf.reset_default_graph()

# Load the model
import network_model
m = network_model.Model2(featureProvider.train.data.shape[2], n_nodes_hl = n_nodes_hl)

# Create a Saver object
saver = tf.train.Saver()
log_dir = "./logs/nn_logs/" + run_timestamp + "/"
os.makedirs(log_dir, exist_ok=True)

# Create a session
with tf.Session() as sess:
    # Create a FileWriter for the log. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    merged = tf.summary.merge_all()

    # Init all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Train the model
    step = 0
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(int(len(featureProvider.train.data)/batch_size)):
            epoch_G, epoch_E = featureProvider.train.next_batch(batch_size)
            _, lss, summary = sess.run([m.train_optimzer, m.loss, merged], feed_dict={m.G: epoch_G, m.E: epoch_E, m.keep_prob_h1: train_keep_prob_h1, m.keep_prob_h2: train_keep_prob_h2, m.is_training: True})
            #print(lss)
            epoch_loss += lss
            if (step%100 == 0): # Write summary of every 100 step
                writer.add_summary(summary, step)
            step += 1

        if (epoch%5 == 0):
            print('Epoch', epoch, 'completed out of', n_epochs, 'epoch_loss:', epoch_loss)
            ex_G = featureProvider.train.data[0]
            ex_E = featureProvider.train.labels[0]
            print(m.E_G.eval(feed_dict={m.G: [ex_G], m.E: [ex_E], m.keep_prob_h1: 1, m.keep_prob_h2: 1, m.is_training: False})[0], ex_E)
            # Save model too far
            saver.save(sess, os.path.join(model_dir, 'model'), global_step=step)
            print("",flush=True)

    print("Examples:")
    test_G, test_E = featureProvider.test.next_batch(batch_size)
    test_predictions = m.E_G.eval({m.G: test_G, m.keep_prob_h1: 1, m.keep_prob_h2: 1, m.is_training: False})
    print("Predictions:")
    print(test_predictions)
    print("Actual values:")
    print(test_E)
    print("Diff:")
    print(test_predictions - test_E)

    # Save the whole session?
    saver.save(sess, os.path.join(model_dir, 'model'))

# covn2D/1D ...
