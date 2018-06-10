# usage: python n_MAE_plot.py n_MAE_HL3.txt n_MAE_HL3.pdf "13-N-N-N-1 network after 500k itterations"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

data_file = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]

n,MAE_new,MAE_err = np.loadtxt(data_file)

matplotlib.rcParams.update({'font.size': 14})
plt.errorbar(n,MAE_new,yerr=MAE_err,fmt='.b', markersize=7)
print(np.mean(MAE_new[-23:-1]))
plt.ylim(.21,0.26)
plt.title(title)
plt.xlabel("$N$")
plt.ylabel("$MAE\, [eV]$")
plt.savefig(out_file)
