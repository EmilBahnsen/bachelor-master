# usage: python E_ENN.py ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/25-25-25_ba1k/2018-05-29_18.11/
from visualization.model_loader import ModelLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from carbondata import CarbonData
import re
from datetime import datetime
from visualization.structure_heatmap import StructureEnergyMap

log_dir = sys.argv[1]
cd = CarbonData('/home/bahnsen/carbon_nn/carbondata/bachelor2018-master/CarbonData/')
E = cd.data_energies

sem = StructureEnergyMap(log_dir)
E_NN = np.sum(sem.structure_atom_energies, axis=1)

MAE = np.mean(np.absolute(E_NN - E))
print("MAE:", MAE)
exit()

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

cd = CarbonData(data_dir = "/home/bahnsen/carbon_nn/carbondata/bachelor2018-master/CarbonData")

os.chdir("/home/bahnsen/carbon_nn")
E_NN,_ = ml.get_energy_of_structures(cd.data_positions,precision=precision)
os.chdir("/home/bahnsen/carbon_nn/visualization/E_ENN")
np.save("E_NN.npy", E_NN)

MEA = np.mean(np.abs(E_NN - cd.data_enegies))

print(MAE)
