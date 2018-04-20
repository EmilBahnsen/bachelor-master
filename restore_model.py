import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import network_model
import pickle
import sys
import glob
import ast
import feature_pick
from feature_data_provider import *

model_dir = str(sys.argv[1])
list_of_files = glob.glob(os.path.join(model_dir, "*.meta")) # * means all if need specific format then *.csv
model_path = max(list_of_files, key=os.path.getctime)
#model_path = str(sys.argv[1]) # Its the .meta
model_dir = os.path.dirname(model_path)

# Open all the calculated features
# Load parameters from dictianory literal in file of first argument
params_file = os.path.join(model_dir, "params.txt")
params_dictionary = str(open(params_file, 'r').read())
params = ast.literal_eval(params_dictionary)

import feature_pick
def load_param(name):
    try:
        value = params[name]
    except KeyError:
        print("Error: Parameter", name, "not found!")
        return None
    print(name, "=", value)
    return value

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

data_dir = load_param("data_directory")
n_atoms = load_param("number_of_atoms")
train_part = load_param("train_part")
train_keep_probs = load_param("train_keep_probs")
data_directory = load_param("data_directory")

# Load the features specified
carbon_data = CarbonData(data_dir = data_dir, structure_size = n_atoms)
featureProvider = FeatureDataProvider(feature_file_list,carbon_data,trainPart=train_part, normalized_labels=False)
n_data = len(featureProvider.train.data)

# Reset tf
tf.reset_default_graph()

imported_meta = tf.train.import_meta_graph(model_path)

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()
    G = graph.get_tensor_by_name('G:0')
    E_G = graph.get_tensor_by_name('E_G:0') # Also called 'E_G:0'
    is_training = graph.get_tensor_by_name('is_training:0')
    keep_probs = graph.get_tensor_by_name('keep_probs:0')
    test_G, test_E = featureProvider.train.next_batch(n_data)
    feed_dict = {G: test_G, keep_probs: np.ones(np.size(train_keep_probs)), is_training: False}
    prediction = E_G.eval(feed_dict)
    errors = test_E - prediction

    energy_list_file = os.path.join(data_directory,"energy_lists.npy")
    print(energy_list_file)
    if os.path.isfile(energy_list_file):
        # Each atom energy intercept
        fc_out = graph.get_tensor_by_name('layer_out/fc_out/BiasAdd:0')
        atom_energies = fc_out.eval(feed_dict)
        energy_list = pickle.load(open(energy_list_file, "rb"))

        for i in range(n_data):
            ex_atom_energies = atom_energies[i]
            ex_energy_list = np.reshape(energy_list[i], (n_atoms,1))
            print("Atom energies:")
            print("Predicted:\n", ex_atom_energies)
            print("Actual:\n", ex_energy_list)
            print("Error:\n", (ex_atom_energies - ex_energy_list)/ex_energy_list)


    print("Prediction\t Actual value\t Abs. diff")
    for i in range(10): # Listing 10 first
        print("%.5f\t %.5f\t %.4f" % (prediction[i], test_E[i], errors[i]))


    # Show data
    print("Std. of: predic. - E:", np.std(prediction - test_E))
    print("E_mean = ", np.mean(test_E))

    plt.plot(prediction, test_E, '.b', alpha=.1)
    plt.xlabel("Prediction")
    plt.ylabel("Value")
    plt.text(0,1,"Nr. atoms: " + str(n_atoms),horizontalalignment='center', verticalalignment='center')
    p_line = [np.max(test_E),np.min(test_E)]
    plt.plot(p_line,p_line,'--k', alpha=.5)
    plt.title(model_dir)
    plt.savefig(os.path.join(model_dir, "out.png"))
    plt.show()
