from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import tensorflow as tf
import glob
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from feature_data_provider import FeatureDataProvider
from carbondata import CarbonData
import network_model

class ModelLoader:
    def __init__(self, log_dir):
        self.log_dir = os.path.abspath(log_dir)
        self.event_acc = EventAccumulator(log_dir)
        self.event_acc.Reload()

        # Load parametr file
        # Load parameters from dictianory literal in file of first argument
        params_file = os.path.join(self.log_dir, "params.txt")
        params_dictionary = str(open(params_file, 'r').read())
        self.params = ast.literal_eval(params_dictionary)

        # Get the latest model
        tf.reset_default_graph()
        try:
            list_of_files = glob.glob(os.path.join(self.log_dir, "*.meta"))
            self.model_path = max(list_of_files, key=os.path.getctime)
            self.meta_graph = tf.train.import_meta_graph(self.model_path)
        except ValueError:
            raise FileNotFoundError
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            with tf.Session() as sess:
                self.meta_graph.restore(sess, tf.train.latest_checkpoint(self.log_dir))
                self._graph = tf.get_default_graph()
        return self._graph

    def __restore_latest_checkpoint(self,sess):
        self.meta_graph.restore(sess, tf.train.latest_checkpoint(self.log_dir))

    def tags(self):
        return self.event_acc.Tags()

    def get_operators(self):
        return self.graph.get_operations()

    def get_scalar(self,name):
        w_times, step_nums, vals = zip(*self.event_acc.Scalars(name))
        return w_times, step_nums, vals

    def get_name_of_tensors(self):
        return [n.name for n in self.graph.as_graph_def().node]

    def get_tensor_by_name(self,name):
        return self.graph.get_tensor_by_name(name)

    def eval_tensor_by_name(self,name,feature_vectors):
        t = self.get_tensor_by_name(name)

        G = self.get_tensor_by_name("G:0")
        train_dropout_rate = self.get_tensor_by_name("train_dropout_rate:0")
        is_training = self.get_tensor_by_name("is_training:0")
        feed_dict = {G: feature_vectors,
                     train_dropout_rate: self.params["train_dropout_rate"],
                     is_training: False}

        with tf.Session() as sess:
            self.__restore_latest_checkpoint(sess)
            return sess.run(t, feed_dict=feed_dict)

    # Extract the energy of a new struture from the position of the atoms
    def get_energy_of_structures(self,structures,precision=tf.float64):
        here_path = os.getcwd()
        os.chdir('/home/bahnsen/carbon_nn') # Relative to the param-file paths

        feature_list_file   = self.params["feature_list_file"]
        data_dir            = self.params["data_directory"]
        n_atoms             = self.params["number_of_atoms"]
        structures_to_use   = self.params["structures_to_use"]
        train_part          = self.params["train_part"]
        feature_scaling     = self.params["feature_scaling"]
        hidden_neuron_count = self.params["hidden_neuron_count"]
        n_positions         = len(structures[0])

        feature_file_list = [x.strip() for x in open(feature_list_file, "r").readlines()]

        carbon_data = CarbonData(data_dir = data_dir, structure_size = n_atoms, structures_to_use = structures_to_use)
        feature_provider = FeatureDataProvider(feature_file_list, carbon_data, trainPart = train_part, normalized_labels=False, feature_scaling=feature_scaling)

        # Perferm features extraction
        # and scaling accoding to scaling method used in tarning
        feature_vectors = feature_provider.extract_features_from_structures(structures)

        # Reset tf
        tf.reset_default_graph()
        # Load the model
        m = network_model.Model2(feature_provider.train.data.shape[2], n_nodes_hl = hidden_neuron_count, n_atoms = n_positions, learning_rate = 0, precision=precision)
        saver = tf.train.Saver(max_to_keep=1)

        graph = tf.get_default_graph()
        E_G = graph.get_tensor_by_name('E_G:0')

        G = graph.get_tensor_by_name("G:0")
        train_dropout_rate = graph.get_tensor_by_name("train_dropout_rate:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        feed_dict = {G: feature_vectors,
                     train_dropout_rate: self.params["train_dropout_rate"],
                     is_training: False}

        with tf.Session(graph=graph) as sess:
            self.__restore_latest_checkpoint(sess)
            energies = sess.run(E_G, feed_dict=feed_dict)

        # Calc what structurs have features outside of trainnig set
        fea_min,fea_max = feature_provider.get_feature_space_hypercube_bonuds()
        out_of_train_index = ((feature_vectors < fea_min) | (feature_vectors > fea_max)).any(axis=2).sum(axis=1)

        # Back again
        os.chdir(here_path)
        
        return energies, out_of_train_index

    def get_forces_in_structures(self,structures,precision=tf.float64):
        n_structures = len(structures)
        n_atoms = self.params["number_of_atoms"]

        forces = np.ndarray((n_structures,n_atoms,3))

        # Perturb each atom in x,y,z and find the negative gradient as the force
        dl = 0.01
        pertubed_structures = np.ndarray(tuple([n_atoms,3]) + structures.shape)
        pertubed_structures[:,:,:] = np.broadcast_to(structures, pertubed_structures.shape)
        for i_atom in range(n_atoms):
            for i_xyz in range(3):
                pertubed_structures[i_atom,i_xyz,:,i_atom,i_xyz] += dl

        # Calculate the energies of these portubed strutures
        feed_structures = np.reshape(pertubed_structures, tuple([n_atoms*3*n_structures]) + structures.shape[1:3])
        # print(tuple([n_atoms*3*n_structures]) + structures.shape[1:3])
        # np.save("two.npy", feed_structures[:len(feed_structures)-1])
        # exit()
        E_structures,_ = self.get_energy_of_structures(feed_structures)
        E_structures = np.reshape(E_structures, (n_atoms,3,n_structures))

        E_references,_ = self.get_energy_of_structures(structures)

        # Find forces in x,y,z
        dE = E_structures - np.reshape(E_references, (1,1,len(E_references)))
        forces = np.moveaxis(-dE/dl,-1,0) # Make structure major array
        # for i_struc in range(n_structures):
        #     for i_atom in range(n_atoms):
        #         for i_xyz in range(3):
        #             dE = E_structures[i_atom,i_xyz,i_struc] - E_references[i_struc]
        #             forces[i_struc, i_atom, i_xyz] = -dE/dl # F = -grad(E)

        # print(forces)
        # np.save("F_one.npy", forces)
        # exit()

        return forces

    def get_all_variables(self):
        with self.graph.as_default():
            return tf.global_variables()

    def get_variable_by_name(self,name):
        all_vars = self.get_all_variables()
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                return all_vars[i]
        return None

    def get_values_of_variable_by_name(self,name):
        with tf.Session() as sess:
            self.__restore_latest_checkpoint(sess)
            return sess.run(self.get_variable_by_name(name))

    def get_AME_test(self):
        try:
            return self.get_scalar('loss/mean_absolute_error_test_1')[2]
        except KeyError:
            return [-1]

    def get_AME_train(self):
        try:
            return self.get_scalar('loss/mean_absolute_error_1')[2]
        except KeyError:
            return [-1]

    def get_steps(self):
        try:
            return self.get_scalar('loss/mean_absolute_error_1')[1]
        except KeyError:
            return [-1]

if __name__ == "__main__":
    model_dir = sys.argv[1]
    out_file = sys.argv[2]
    ml = ModelLoader(model_dir)
    _,step,AME      = ml.get_scalar('loss/mean_absolute_error_1')
    _,step,AME_test = ml.get_scalar('loss/mean_absolute_error_test_1')

    valiables = ml.get_all_variables()
    [print(var) for var in valiables]
    print(ml.get_variable_by_name('fc_in/weights:0'),"\n")

    ops = ml.get_operators()
    [print(op.name, "->",op.values()) for op in ops]

    os.makedirs(os.path.join(model_dir, "figures"), exist_ok=True)

    plt.figure(1, figsize=(17, 8.5))
    plt.suptitle(model_dir)
    plt.subplot(121)
    w_in = ml.get_values_of_variable_by_name('fc_in/weights:0')
    sns.heatmap(w_in, linewidth=0.5, cbar_kws={"orientation": "horizontal"})
    plt.title("$w_{in}$")

    plt.subplot(164)
    b_in = ml.get_values_of_variable_by_name('fc_in/biases:0')
    b_in = b_in.reshape((15,1))
    sns.heatmap(b_in, linewidth=0.5, xticklabels=False, cbar_kws={"orientation": "horizontal"})
    plt.title("$b_{in}$")

    plt.subplot(165)
    w_out = ml.get_values_of_variable_by_name('fc_out/weights:0')
    sns.heatmap(w_out, linewidth=0.5, xticklabels=False, cbar_kws={"orientation": "horizontal"})
    plt.title("$w_{out}$")

    plt.subplot(166)
    b_out = ml.get_values_of_variable_by_name('fc_out/biases:0')
    b_out = b_out.reshape((1,1))
    sns.heatmap(b_out, linewidth=0.5, xticklabels=False, yticklabels=False, annot=True, fmt="f", cbar=False)
    plt.title("$b_{out}$")

    plt.savefig(out_file)
