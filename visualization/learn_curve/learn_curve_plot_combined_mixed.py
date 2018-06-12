# usage: python n_MAE_plot.py n_MAE_HL3.txt n_MAE_HL3.pdf "13-N-N-N-1 network after 500k itterations"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

n_samples = 10808*0.8
n_samples_multi = 10000*0.8

out_file = sys.argv[1]

n_ss,MAE_new_ss,MAE_err_ss = np.loadtxt('learn_curve-29-29-29_single_point_single.txt')
n_sr,MAE_new_sr,MAE_err_sr = np.loadtxt('learn_curve-29-29-29_single_point_relaxed.txt')
n_dd,MAE_new_dd,MAE_err_dd = np.loadtxt('learn_curve-29-29-29_dual_point_dual.txt')
n_dr,MAE_new_dr,MAE_err_dr = np.loadtxt('learn_curve-29-29-29_dual_point_relaxed.txt')
n_mu,MAE_new_mu,MAE_err_mu = np.loadtxt('learn_curve-29-29-29_multi_perturbed.txt')
n_old,MAE_new_old,MAE_err_old = np.loadtxt('learn_curve-29-29-29.txt')
print("Multi last:",MAE_new_mu[-1])
print("Relax last:",MAE_new_old[-1])


matplotlib.rcParams.update({'font.size': 14})

plt.loglog(n_ss*n_samples,MAE_new_ss,'--b', label='single-point')
plt.loglog(n_sr*n_samples,MAE_new_sr,'-b', label='single-point/relax')
plt.loglog(n_dd*n_samples,MAE_new_dd,'--r', label='dual-point')
plt.loglog(n_dr*n_samples,MAE_new_dr,'-r', label='dual-point/relax')
plt.loglog(n_mu*n_samples,MAE_new_mu,'-k', label='multiple pertubations')
plt.loglog(n_old*n_samples,MAE_new_old,'--k', label='relaxed')

plt.gca().legend()
plt.ylim(0,6)
plt.xlim(100,8000)
#plt.gca().set_xscale('log')
plt.xlim(0,n_samples)
plt.title("Energy learning curves")
plt.xlabel("Number of traning samples")
plt.ylabel("$MAE\, [eV]$")
plt.tight_layout()
plt.savefig(out_file)
