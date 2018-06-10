# usage: python n_MAE_plot.py n_MAE_HL3.txt n_MAE_HL3.pdf "13-N-N-N-1 network after 500k itterations"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

n_samples = 10808*0.8

out_file = sys.argv[1]

n1,MAE_new1,MAE_err1 = np.loadtxt('learn_curve-50.txt')
n2,MAE_new2,MAE_err2 = np.loadtxt('learn_curve-68-68.txt')
n3,MAE_new3,MAE_err3 = np.loadtxt('learn_curve-29-29-29.txt')

matplotlib.rcParams.update({'font.size': 14})

# plt.errorbar(n1,MAE_new1,yerr=MAE_err1,fmt='-b', label='13-50-1')
# plt.errorbar(n2,MAE_new2,yerr=MAE_err2,fmt='-r', label='13-68-68-1')
# plt.errorbar(n3,MAE_new3,yerr=MAE_err3,fmt='-g', label='13-29-29-29-1')
plt.plot(n1*n_samples,MAE_new1,'-r', label='13-50-1')
print("Fit 1:", np.polyfit(np.log(n1*n_samples), MAE_new1, 1))
plt.plot(n2*n_samples,MAE_new2,'-g', label='13-68-68-1')
print("Fit 2:", np.polyfit(np.log(n2*n_samples), MAE_new2, 1))
plt.plot(n3*n_samples,MAE_new3,'-b', label='13-29-29-29-1')
print("Fit 3:", np.polyfit(np.log(n3*n_samples), MAE_new3, 1))
plt.gca().legend()
plt.ylim(.2,0.50)
#plt.gca().set_xscale('log')
plt.xlim(0,n_samples)
plt.title("Learning curves for optimal networks")
plt.xlabel("Number of traning samples")
plt.ylabel("$MAE\, [eV]$")
plt.tight_layout()
plt.savefig(out_file)
