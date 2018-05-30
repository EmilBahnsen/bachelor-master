# usage: python pos_E.py ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/25-25-25_ba1k/2018-05-29_18.11/
from visualization.model_loader import ModelLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from carbondata import CarbonData
import re
from datetime import datetime

log_dir = sys.argv[1]

match = re.search(r'\d{4}-\d{2}-\d{2}', log_dir)
precision_change_date = datetime(2018, 5, 27).date()
date = datetime.strptime(match.group(), '%Y-%m-%d').date()
print()
if date > precision_change_date:
	precision = tf.float64
else:
	precision = tf.float32
# precision = tf.float32

ml = ModelLoader(log_dir)
# [print(i) for i in ml.get_name_of_tensors()]
# exit()

cd = CarbonData(data_dir = "/home/bahnsen/carbon_nn/carbondata/bachelor2018-master/CarbonData")

n_structures = 1000
def create_structures(n_atoms,x_start,x_end,structure_fun):
	structures = np.ndarray((n_structures,n_atoms,3))
	indep_var = np.ndarray(n_structures)
	for n in range(n_structures):
		i_n = (x_end - x_start) * n/n_structures + x_start
		indep_var[n] = i_n
		structures[n,:,:] = structure_fun(i_n)
	return structures,indep_var

# Dimer and trimer structures
structures_dimer,R_dimer = create_structures(2,0,3,lambda x: [[0,0,0],[x,0,0]])
structures_trimer_linear,_ = create_structures(3,0,3,lambda x: [[0,0,0],[x,0,0],[2*x,0,0]])
R_trimer = 1.39
structures_trimer,angles = create_structures(3,0,2*np.pi,
	lambda x: [[0,0,0], [R_trimer,0,0], [R_trimer*np.cos(x),R_trimer*np.sin(x),0]]
)
def benzene_structure_fun(r):
	xyz = np.ndarray((6,3))
	for n,angle in enumerate(range(0,360,60)):
		xyz[n,:] = [r*np.cos(angle*180/np.pi), r*np.sin(angle*180/np.pi), 0]
	return xyz
structures_benzene_r,R_benzene = create_structures(6,0,3, benzene_structure_fun)

R_benzene_set = 1.39
def benzene_structure_fun(z):
	xyz = np.ndarray((6,3))
	for n,angle in enumerate(range(0,360,60)):
		xyz[n,:] = [R_benzene_set*np.cos(angle*180/np.pi), R_benzene_set*np.sin(angle*180/np.pi), z if n == 0 else 0]
	return xyz
structures_benzene_z,z_benzene = create_structures(6,-3,3, benzene_structure_fun)

os.chdir("/home/bahnsen/carbon_nn")
E_known,_ = ml.get_energy_of_structures(cd.data_positions[0:5],precision=precision)
print("Sanity check:")
print("Prediction: ",np.reshape(E_known,(-1)))
print("Actual: ",cd.data_energies[0:5])
E_dimer,dimer_outside = ml.get_energy_of_structures(structures_dimer,precision=precision)
print(dimer_outside)
E_trimer_linear,trimer_linear_outside = ml.get_energy_of_structures(structures_trimer_linear,precision=precision)
E_trimer,trimer_outside = ml.get_energy_of_structures(structures_trimer,precision=precision)
E_benzene_r,benzene_r_outside = ml.get_energy_of_structures(structures_benzene_r,precision=precision)
E_benzene_z,benzene_z_outside = ml.get_energy_of_structures(structures_benzene_z,precision=precision)
os.chdir("/home/bahnsen/carbon_nn/visualization/pos_E")

fig = plt.figure(0)
def plot_figure(x,y,outside,title, xlabel, ylabel, xlims):
	fig.clf()
	ax = fig.gca()
	j = 0
	for i in range(1,len(outside)):
		if outside[i] == outside[i-1] and i != len(outside)-1:
			continue
		style = '-b' if outside[i-1] == 0 else '--b'
		plt.plot(x[j:i],y[j:i],style)
		j = i
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(xlims)

plot_figure(R_dimer,E_dimer,dimer_outside,"$C_2$ dimer energy prediction", "R [Å]", "Predicted E [eV]", [0,3])
fig.savefig("pos_E_dimer.pdf")

plot_figure(R_dimer,E_trimer_linear,trimer_linear_outside,"$C_3$ linear trimer energy prediction", "R [Å]", "Predicted E [eV]", [0,3])
fig.savefig("pos_E_trimer_linear.pdf")
print("Min linear trimer distance:", R_dimer[np.argmin(E_trimer_linear)])

plot_figure(angles*180/np.pi,E_trimer,trimer_outside,"$C_3$ trimer energy prediction at R = %.2f Å" % R_trimer, "Angle [deg]", "Predicted E [eV]", [0,360])
fig.savefig("pos_E_trimer.pdf")

plot_figure(R_benzene,E_benzene_r,benzene_r_outside,"$C_6$ benzene energy prediction", "R [Å]", "Predicted E [eV]", [0,3])
fig.savefig("pos_E_benzene_r.pdf")

plot_figure(z_benzene,E_benzene_z,benzene_z_outside,"$C_6$ benzene energy prediction at R = %.2f Å" % R_trimer, "z-offset [Å]", "Predicted E [eV]", [-3,3])
fig.savefig("pos_E_benzene_z.pdf")
