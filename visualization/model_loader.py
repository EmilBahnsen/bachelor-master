from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import tensorflow as tf
import glob
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ModelLoader:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.event_acc = EventAccumulator(log_dir)
        self.event_acc.Reload()

        # Load parametr file
        # Load parameters from dictianory literal in file of first argument
        params_file = os.path.join(log_dir, "params.txt")
        params_dictionary = str(open(params_file, 'r').read())
        self.params = ast.literal_eval(params_dictionary)

        # Get the latest model
        tf.reset_default_graph()
        list_of_files = glob.glob(os.path.join(self.log_dir, "*.meta"))
        model_path = max(list_of_files, key=os.path.getctime)
        self.meta_graph = tf.train.import_meta_graph(model_path)

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
