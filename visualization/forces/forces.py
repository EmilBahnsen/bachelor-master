# usage: python pos_E.py ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/25-25-25_ba1k/2018-05-29_18.11/
from visualization.model_loader import ModelLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from carbondata import CarbonData
import re
from datetime import datetime

log_dir = sys.argv[1]

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

cd = CarbonData(data_dir = "/home/bahnsen/carbon_nn/carbondata/bachelor2018-master/CarbonData", random_seed=None)



os.chdir("/home/bahnsen/carbon_nn")
E_known,E_outside = ml.get_energy_of_structures(cd.data_positions[0:5],precision=precision)
os.chdir("/home/bahnsen/carbon_nn/visualization/forces")


