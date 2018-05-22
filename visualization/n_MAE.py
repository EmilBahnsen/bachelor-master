from model_loader import *
import matplotlib.pyplot as plt
import ast
import numpy as np
from bash_file_tools import *
import re
from helper import *

# usage: python n_MAE.py '../logs/nn_logs/features_few/Rc2.8/z-score/1x*/2018*' n_MAE/n_MAE.pdf "13-N-1 network after 300k itterations" 300000
model_dirs_bash_path = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]
last_step = int(sys.argv[4])
model_dirs = np.array(list_bash_files(model_dirs_bash_path))
[print(model_dir) for model_dir in model_dirs]

mls = [ModelLoader(model_dir) for model_dir in model_dirs]
# print(re.search("(?=1x)(\d+)", mls[0].params["log_root_dir"]).group(0), re.search("(1x)(\d+)", mls[0].params["log_root_dir"]).group(1), re.search("(1x)(\d+)", mls[0].params["log_root_dir"]).group(2))
# exit()

n = [re.search("(1x)(\d+)", ml.params["log_root_dir"]).group(2) for ml in mls]
n = np.array(n)
n = n.astype(float)
n = np.array(n)

print("n=",n)

# Get min MEA below specified step
MAE_test_end_values,bad_index = get_min_MAEs(mls,last_step)
print("Bad dirs:",model_dirs[bad_index])
n = np.delete(n,bad_index)

# Calc mean and errorbars for points of multible runs
n_new,MAE_new,MAE_err = mean_and_error(MAE_test_end_values,n)

print("n=",n_new)
print("MAE=",MAE_new)
print("MAE_err=",MAE_err)

plt.errorbar(n_new,MAE_new,yerr=MAE_err,fmt='.b', markersize=2)
plt.title(title)
plt.ylim(0.28,0.40)
plt.xlabel("$N$ [hidden nodes]")
plt.ylabel("$MAE\, [eV]$")
plt.xticks(np.arange(10, 100, step=5))
plt.savefig(out_file)
