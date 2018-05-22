from model_loader import *
import matplotlib.pyplot as plt
import ast
import numpy as np
from bash_file_tools import *
import re

# usage: python loss.py '../logs/nn_logs/features_few/Rc*/z-score/58-58-58-58_*/*' loss/loss-58-58-58-58.pdf "13-58-58-58-58-1 network after 150k itterations" 150000
model_dirs_bash_path = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]
last_step = int(sys.argv[4])
model_dirs = np.array(list_bash_files(model_dirs_bash_path))

mls = [ModelLoader(model_dir) for model_dir in model_dirs]

# Plot accumulative min values of loss functions, to filuter out noice
plt.clf()
#for ml in [ml for i,ml in enumerate(mls) if ml.get_AME_test()[-250] < 0.4]:
for ml in mls:
	AME_test = ml.get_AME_test()
	accmin_AME = np.minimum.accumulate(AME_test)
	steps = ml.get_steps()
	plt.plot(steps, accmin_AME)

plt.savefig(out_file_MAE)
