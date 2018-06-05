# usage: python Rc_MAE.py Rc_MAE_100.txt Rc_MAE_100.pdf "13-100-1 network after 150k itterations"
import matplotlib.pyplot as plt
import numpy as np
import sys

data_file = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]

Rc_new,MAE_new,MAE_err = np.loadtxt(data_file)

plt.errorbar(Rc_new,MAE_new,yerr=MAE_err,fmt='.b', markersize=2)
plt.xlim(2,10)
plt.ylim(0.2,0.55)
plt.title(title)
plt.xlabel("$R_c\, [A]$")
plt.ylabel("$MAE\, [eV]$")
plt.savefig(out_file)
