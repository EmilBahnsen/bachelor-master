from visualization.model_loader import ModelLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from carbondata import CarbonData
import re
from datetime import datetime

# usage: python forces_save_test.py ../../logs/mixed_log/multi_perturb/features_few/Rc5/z-score/29-29-29_ba5pct/struc1/2018-06-06_10.13/ all_forces_test-29-29-29.npy

log_dir = sys.argv[1]
save_file = sys.argv[2]

match = re.search(r'\d{4}-\d{2}-\d{2}', log_dir)
precision_change_date = datetime(2018, 5, 27).date()
date = datetime.strptime(match.group(), '%Y-%m-%d').date()
print()
if date > precision_change_date:
	precision = tf.float64
else:
	precision = tf.float32
# precision = tf.float32

ml = ModelLoader(log_dir)
# [print(i) for i in ml.get_name_of_tensors()]
# exit()

cd = CarbonData(data_dir = os.path.join('/home/bahnsen/carbon_nn/',ml.params["data_directory"]), random_seed=None)

n_structures = len(cd.data_energies)
index_split = int((n_structures-1)*0.8)
print("Split at", index_split)
all_forces_test = ml.get_forces_in_structures(cd.data_positions[index_split:])
all_forces_test = np.array(all_forces_test)
np.save(save_file, all_forces_test)
print(all_forces_test)
print(all_forces_test.shape)

# os.chdir("/home/bahnsen/carbon_nn")
# E_known,E_outside = ml.get_energy_of_structures(cd.data_positions[0:5],precision=precision)
# os.chdir("/home/bahnsen/carbon_nn/visualization/forces")


