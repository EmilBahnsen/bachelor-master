from model_loader import *
import matplotlib.pyplot as plt
import ast
import numpy as np
from bash_file_tools import *

# usage: python Rc_MAE.py '../logs/nn_logs/features_few/Rc*/z-score/1x15/*' Rc_MAE/Rc_MAE_1x15.pdf "13-15-1 network after 500k itterations" 500000 <steps>
model_dirs_bash_path = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]
least_steps = int(sys.argv[4])
model_dirs = list_bash_files(model_dirs_bash_path)

mls = [ModelLoader(model_dir) for model_dir in model_dirs]

import re
Rc = [re.search("Rc(.+?)/", ml.params["log_root_dir"]).group(1) for ml in mls]
Rc = np.array(Rc)
Rc = Rc.astype(float)

# Get the MAE data and then the min of last 10 MAE
MAE_test_end_values = [np.min((ml.get_AME_test())[-10:-1]) for ml in mls]
steps_end			= [(ml.get_steps())[-1] for ml in mls]
print(MAE_test_end_values)

# At least 500 steps
MAE_test_end_values, Rc = zip(*[(MAE,Rc[i]) for i,MAE in enumerate(MAE_test_end_values) if steps_end[i] >= least_steps-1])

# Models which doesn't have loss function logged removed
MAE_test_end_values, Rc = zip(*[(MAE,Rc[i]) for i,MAE in enumerate(MAE_test_end_values) if MAE != -1])

Rc = np.array(Rc)
MAE_test_end_values = np.array(MAE_test_end_values)

# Calc mean and errorbars for points of multible runs
sorting_index = np.argsort(Rc) # Rc must be sorted
Rc = Rc[sorting_index]
MAE_test_end_values = MAE_test_end_values[sorting_index]
Rc_new = []
MAE_new = []
MAE_err = []
i = 0
while i<len(Rc)-1:
	print(i)
	for j in range(i,len(Rc)):
		#print(i,j, Rc[i], Rc[j])
		print(Rc[i], Rc[j])
		if Rc[i] == Rc[j] and j != len(Rc)-1:
			continue
		else:
			print(Rc[i])
			Rc_new.append(Rc[i])
			MAE_new.append(np.mean(MAE_test_end_values[i:j])) # Not including j
			MAE_err.append(np.std(MAE_test_end_values[i:j]))
			i = j
			break

print(Rc_new)
print(MAE_new)
print(MAE_err)

plt.errorbar(Rc_new,MAE_new,yerr=MAE_err,fmt='.b', markersize=0.5)
plt.ylim(0,2.5)
plt.title(title)
plt.xlabel("$R_c\, [Å]$")
plt.ylabel("$MAE\, [eV]$")
plt.savefig(out_file)
