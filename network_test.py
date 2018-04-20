import network_model
import unittest
import tensorflow as tf
import numpy as np

class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = network_model.Model(2, n_nodes_hl=[100,100])
        self.sess = tf.Session()
        # Init all variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def test_output_is_same_for_all_atoms(self):
        m = self.m
        G = [np.tile([1,2], [24,1])]
        with self.sess.as_default() as sess:
            graph = tf.get_default_graph()
            out = graph.get_tensor_by_name("layer_out/out:0") # Get output layer
            result = out.eval(feed_dict={m.G: G, m.keep_prob_h1: 1, m.keep_prob_h2: 1})[0]
            self.assertEqual(result[0], result[1])
            self.assertEqual(result[5], result[10])

    def test_summation_of_atom_energies(self):
        m = self.m
        G = [np.tile([1,2], [24,1])]
        with self.sess.as_default() as sess:
            graph = tf.get_default_graph()
            out = graph.get_tensor_by_name("layer_out/out:0") # Get output layer
            feed_dict = {m.G: G, m.keep_prob_h1: 1, m.keep_prob_h2: 1}
            #E_partial = out.eval(feed_dict=feed_dict)[0]
            #E_sum  = m.E_G.eval(feed_dict=feed_dict)[0]
            #self.assertEqual(np.sum(E_partial), E_sum)

if __name__ == '__main__':
    unittest.main()
