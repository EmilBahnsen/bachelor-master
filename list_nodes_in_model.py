import tensorflow as tf
import network_model
import sys
import os
import glob

model_dir = str(sys.argv[1])
list_of_files = glob.glob(os.path.join(model_dir, "*.meta")) # * means all if need specific format then *.csv
model_path = max(list_of_files, key=os.path.getctime)
model_dir = os.path.dirname(model_path)

# Reset tf
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph(model_path)

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()
    [print(n.name) for n in graph.as_graph_def().node]
