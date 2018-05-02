from model_loader import *
import matplotlib.pyplot as plt
import ast
import numpy as np
from bash_file_tools import *
import re

# usage: python n_MAE.py '../logs/nn_logs/features_few/Rc2.8/z-score/1x*/2018*' n_MAE/n_MAE.pdf "13-N-1 network after 300k itterations" 300000
model_dirs_bash_path = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]
last_steps = int(sys.argv[4])
model_dirs = list_bash_files(model_dirs_bash_path)

mls = [ModelLoader(model_dir) for model_dir in model_dirs]
# print(re.search("(?=1x)(\d+)", mls[0].params["log_root_dir"]).group(0), re.search("(1x)(\d+)", mls[0].params["log_root_dir"]).group(1), re.search("(1x)(\d+)", mls[0].params["log_root_dir"]).group(2))
# exit()

n = [re.search("(1x)(\d+)", ml.params["log_root_dir"]).group(2) for ml in mls]
n = np.array(n)
n = n.astype(float)

# Get the MAE data and then the min of last 10 MAE
steps_end			= [(ml.get_steps())[-1] for ml in mls]
MAE_test_end_values = [np.min((ml.get_AME_test()[-5:-1])) for ml in mls]
print(MAE_test_end_values)

# At least last_steps steps
MAE_test_end_values, n = zip(*[(MAE,n[i]) for i,MAE in enumerate(MAE_test_end_values) if steps_end[i] >= last_steps-1])

# Models which doesn't have loss function logged removed
MAE_test_end_values, n = zip(*[(MAE,n[i]) for i,MAE in enumerate(MAE_test_end_values) if MAE != -1])

n = np.array(n)
MAE_test_end_values = np.array(MAE_test_end_values)

# Calc mean and errorbars for points of multible runs
sorting_index = np.argsort(n) # n must be sorted
n = n[sorting_index]
MAE_test_end_values = MAE_test_end_values[sorting_index]
n_new = []
MAE_new = []
MAE_err = []
i = 0
while i<len(n)-1:
	print(i)
	for j in range(i,len(n)):
		#print(i,j, n[i], n[j])
		print(n[i], n[j])
		if n[i] == n[j] and j != len(n)-1:
			continue
		else:
			print(n[i])
			n_new.append(n[i])
			MAE_new.append(np.mean(MAE_test_end_values[i:j])) # Not including j
			MAE_err.append(np.std(MAE_test_end_values[i:j]))
			i = j
			break

print(n_new)
print(MAE_new)
print(MAE_err)

plt.errorbar(n_new[2:-1],MAE_new[2:-1],yerr=MAE_err[2:-1],fmt='.b', markersize=2)
plt.title(title)
plt.xlabel("$N$ [hidden nodes]")
plt.ylabel("$MAE\, [eV]$")
plt.savefig(out_file)
