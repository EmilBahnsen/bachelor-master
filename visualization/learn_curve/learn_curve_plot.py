# usage: python n_MAE_plot.py n_MAE_HL3.txt n_MAE_HL3.pdf "13-N-N-N-1 network after 500k itterations"
import matplotlib.pyplot as plt
import numpy as np
import sys

data_file = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]

n,MAE_new,MAE_err = np.loadtxt(data_file)

plt.errorbar(n,MAE_new,yerr=MAE_err,fmt='.b', markersize=2)
# plt.ylim(.35,0.80)
plt.title(title)
plt.xlabel("Fraction of training set")
plt.ylabel("$MAE\, [eV]$")
plt.savefig(out_file)
