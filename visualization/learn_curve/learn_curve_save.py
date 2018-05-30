from visualization.model_loader import *
import ast
import numpy as np
from visualization.bash_file_tools import *
import re
from visualization.helper import *

# usage: python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-*-*_ba0.5k/*00.28' n_MAE_HL3.txt 500000 <steps>
model_dirs_bash_path = sys.argv[1]
out_file = sys.argv[2]
last_step = int(sys.argv[3])
model_dirs = np.array(list_bash_files(model_dirs_bash_path))
[print(model_dir) for model_dir in model_dirs]

mls = []
for model_dir in model_dirs:
	try:
		ml = ModelLoader(model_dir)
		mls.append(ml)
	except FileNotFoundError:
        	pass

print(re.findall(r"struc(\d+\.\d|\d+)", mls[0].params["log_root_dir"])[0])

n = [re.findall(r"struc(\d+\.\d|\d+)", ml.params["log_root_dir"])[0] for ml in mls]
n = np.array(n)
n = n.astype(float)

# Get min MEA below specified step
MAE_test_end_values,bad_index = get_min_MAEs(mls,last_step)
print("Bad dirs:",model_dirs[bad_index])
n = np.delete(n,bad_index)

# At least 500 steps
#MAE_test_end_values, n = zip(*[(MAE,n[i]) for i,MAE in enumerate(MAE_test_end_values) if steps_end[i] >= least_steps-1])

# Models which doesn't have loss function logged removed
MAE_test_end_values, n = zip(*[(MAE,n[i]) for i,MAE in enumerate(MAE_test_end_values) if MAE != -1])

n = np.array(n)
MAE_test_end_values = np.array(MAE_test_end_values)

# Calc mean and errorbars for points of multible runs
n_new,MAE_new,MAE_err = mean_and_error(MAE_test_end_values,n)

print(n_new)
print(MAE_new)
print(MAE_err)

np.savetxt(out_file, (n_new, MAE_new, MAE_err))

