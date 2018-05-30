from model_loader import *
import matplotlib.pyplot as plt
import ast
import numpy as np
from bash_file_tools import *
import re
from helper import *

# usage: python Rc_MAE.py '../logs/nn_logs/features_few/Rc*/z-score/1x15/*' Rc_MAE/Rc_MAE_1x15.pdf "13-15-1 network after 500k itterations" 500000 <steps>
model_dirs_bash_path = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]
last_step = int(sys.argv[4])
model_dirs = np.array(list_bash_files(model_dirs_bash_path))
[print(model_dir) for model_dir in model_dirs]

try:
	mls = [ModelLoader(model_dir) for model_dir in model_dirs]
except FileNotFoundError:
	pass

Rc = [re.search("Rc(.+?)/", ml.params["log_root_dir"]).group(1) for ml in mls]
Rc = np.array(Rc)
Rc = Rc.astype(float)

# Get min MEA below specified step
MAE_test_end_values,bad_index = get_min_MAEs(mls,last_step)
print("Bad dirs:",model_dirs[bad_index])
Rc = np.delete(Rc,bad_index)

# At least 500 steps
#MAE_test_end_values, Rc = zip(*[(MAE,Rc[i]) for i,MAE in enumerate(MAE_test_end_values) if steps_end[i] >= least_steps-1])

# Models which doesn't have loss function logged removed
MAE_test_end_values, Rc = zip(*[(MAE,Rc[i]) for i,MAE in enumerate(MAE_test_end_values) if MAE != -1])

Rc = np.array(Rc)
MAE_test_end_values = np.array(MAE_test_end_values)

# Calc mean and errorbars for points of multible runs
Rc_new,MAE_new,MAE_err = mean_and_error(MAE_test_end_values,Rc)

print(Rc_new)
print(MAE_new)
print(MAE_err)

plt.errorbar(Rc_new,MAE_new,yerr=MAE_err,fmt='.b', markersize=1)
plt.ylim(.45,0.75)
plt.title(title)
plt.xlabel("$R_c\, [Å]$")
plt.ylabel("$MAE\, [eV]$")
plt.savefig(out_file)

np.savetxt(out_file + '.txt', (Rc_new, MAE_new, MAE_err))
