import matplotlib
from visualization.structure_heatmap import *

# usage: python MAE_DFT_plot.py ../../logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct/struc1/2018-05-31_01.00/ MAE_DFT_29-29-29.pdf

model_dir = sys.argv[1]
save_dir = sys.argv[2]
sem = StructureEnergyMap(model_dir,with_forces=False,random_seed=0)

n_structures = sem.carbon_data.numberOfStructures
index_split = int(n_structures*0.8)
print(index_split)
energies_real_test = sem.carbon_data.data_energies[index_split:]
print(energies_real_test.shape)

energies_nn_test = np.sum(sem.structure_atom_energies[index_split:], axis=1)
print(energies_nn_test.shape)

delta_Es = np.absolute(energies_real_test - energies_nn_test)
MAE = np.mean(delta_Es)

print("Index with largest delta E:", index_split + np.argmax(delta_Es))

fig = plt.figure(0)
matplotlib.rcParams.update({'font.size': 14})
# text
plt.text(-189,-167,"MAE = %.2f" % MAE)

plt.plot(energies_real_test,energies_nn_test, '.b', alpha=.5)
p_line = np.array([-190, -165])
plt.plot(p_line,p_line, '--k')
plt.xlabel("DFT energy [eV]")
plt.ylabel("Cluster energy prediction [eV]")
plt.title("13-29-29-29-1 using all traning data")
plt.axis(np.append(p_line,p_line))
plt.tight_layout()
fig.savefig(save_dir)