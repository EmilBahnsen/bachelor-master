from carbondata import CarbonData
import numpy as np
import matplotlib.pyplot as plt

# Split the data, to accomade the CarbonData class
n_structures = 10808
positions_all = np.load('pPositions.npy')
positions_all = np.reshape(positions_all, (n_structures,3,24,3))

energies_all = np.load('pEnergy.npy')
energies_all = np.reshape(energies_all, (n_structures,3))

forces_all = np.load('pForces.npy')
forces_all = np.reshape(forces_all, (n_structures,3,24,3))

# Shuffle
indicies = np.arange(n_structures)
np.random.seed(0)			# Predictabilly random
#np.random.shuffle(indicies)
print(indicies)
positions_all = positions_all[indicies,:,:,:]
energies_all = energies_all[indicies,:]
forces_all = forces_all[indicies,:,:,:]

pos_relax = positions_all[:,0,:,:]
pos_non_relax1 = positions_all[:,1,:,:]
pos_non_relax2 = positions_all[:,2,:,:]
pos_non_relax_all = positions_all[:,1:3,:,:]
pos_non_relax_all = np.reshape(pos_non_relax_all, (2*n_structures,24,3))

E_relax = energies_all[:,0]
np.save("relax/energies.npy", E_relax)
E_non_relax1 = energies_all[:,1]
E_non_relax2 = energies_all[:,2]
E_non_relax_all = energies_all[:,1:3]
E_non_relax_all = np.reshape(E_non_relax_all, (2*n_structures))

# plt.figure(0)
# all_E = np.reshape(energies_all, (32424))
# plt.plot(range(len(all_E)), all_E, '-b')
# plt.plot(range(0,len(E_relax)*3,3), E_relax, '-k')
# # plt.plot(range(1,len(E_non_relax1)*3,3), E_non_relax1, '-r')
# # plt.plot(range(2,len(E_non_relax2)*3,3), E_non_relax2, '-g')
# plt.savefig('E_all.pdf')
# plt.show()
# input('')
# exit()

F_relax = forces_all[:,0,:,:]
np.save("relax/forces.npy", F_relax)
# print(F_relax[0])
# exit()
F_non_relax1 = forces_all[:,1,:,:]
F_non_relax2 = forces_all[:,2,:,:]
F_non_relax_all = forces_all[:,1:3,:,:]
F_non_relax_all = np.reshape(F_non_relax_all, (2*n_structures,24,3))

# plt.figure(0)
# forces = np.mean(np.linalg.norm(np.reshape(forces_all, (32424,24,3)), axis=2), axis=1)
# forces_r = np.mean(np.linalg.norm(F_relax, axis=2), axis=1)
# forces_r1 = np.mean(np.linalg.norm(F_non_relax1, axis=2), axis=1)
# forces_r2 = np.mean(np.linalg.norm(F_non_relax2, axis=2), axis=1)
# plt.plot(range(len(forces)), forces, '-b')
# plt.plot(range(0,len(forces_r)*3,3), forces_r, '-k')
# plt.plot(range(1,len(forces_r1)*3,3), forces_r1, '-r')
# plt.plot(range(2,len(forces_r2)*3,3), forces_r2, '-g')
# plt.show()
# input('')
# exit()

# np.save("relax/positions.npy", pos_relax)
# np.save("non_relax1/positions.npy", pos_non_relax1)
# np.save("non_relax2/positions.npy", pos_non_relax2)

# 80% sigle point non-relaxed / 20% single point non-relaxed
positions = np.reshape(pos_non_relax1, (n_structures*24,3))
energies = np.reshape(E_non_relax1, (n_structures))
forces = np.reshape(F_non_relax1, (n_structures*24,3))
np.save("non_relaxed_single0.8_non_relaxed_single0.2/positions.npy", positions)
np.save("non_relaxed_single0.8_non_relaxed_single0.2/energies.npy", energies)
np.save("non_relaxed_single0.8_non_relaxed_single0.2/forces.npy", forces)

# 80% sigle point non-relaxed / 20% relaxed
split = 0.8
split_index = int(split*n_structures)
positions = np.append(pos_non_relax1[:split_index,:,:], pos_relax[split_index:,:,:], axis=0)
energies = np.append(E_non_relax1[:split_index], E_relax[split_index:], axis=0)
forces = np.append(F_non_relax1[:split_index,:,:], F_relax[split_index:,:,:], axis=0)
positions = np.reshape(positions, (n_structures*24,3))
energies = np.reshape(energies, (n_structures))
forces = np.reshape(forces, (n_structures*24,3))
np.save("non_relaxed_single0.8_relaxed0.2/positions.npy", positions)
np.save("non_relaxed_single0.8_relaxed0.2/energies.npy", energies)
np.save("non_relaxed_single0.8_relaxed0.2/forces.npy", forces)

# 80% double point non-relaxed / 20% double point non-relaxed
positions = pos_non_relax_all
pos_non_relax = positions[:n_structures] # Half it to be camparable with the above

energies = E_non_relax_all
E_non_relax = energies[:n_structures] # Half it to be camparable with the above

forces = F_non_relax_all
F_non_relax = forces[:n_structures] # Half it to be camparable with the above

positions = np.reshape(pos_non_relax, (n_structures*24,3))
energies = np.reshape(E_non_relax, (n_structures))
forces = np.reshape(F_non_relax, (n_structures*24,3))
np.save("non_relaxed_double0.8_non_relaxed_double0.2/positions.npy", positions)
np.save("non_relaxed_double0.8_non_relaxed_double0.2/energies.npy", energies)
np.save("non_relaxed_double0.8_non_relaxed_double0.2/forces.npy", forces)

# 80% double point non-relaxed / 20% relaxed
split = 0.8
split_index = int(split*n_structures)
positions = np.append(pos_non_relax[:split_index,:,:], pos_relax[split_index:,:,:], axis=0)
energies = np.append(E_non_relax[:split_index], E_relax[split_index:], axis=0)
forces = np.append(F_non_relax[:split_index,:,:], F_relax[split_index:,:,:], axis=0)

positions = np.reshape(positions, (n_structures*24,3))
energies = np.reshape(energies, (n_structures))
forces = np.reshape(forces, (n_structures*24,3))

np.save("non_relaxed_double0.8_relaxed0.2/positions.npy", positions)
np.save("non_relaxed_double0.8_relaxed0.2/energies.npy", energies)
np.save("non_relaxed_double0.8_relaxed0.2/forces.npy", forces)

# Test
cd = CarbonData('non_relaxed_single0.8_non_relaxed_single0.2', random_seed=None, with_forces=True)
print(cd.data_positions.shape)
print(cd.data_energies.shape)
plt.figure(0)
forces_r = np.mean(np.linalg.norm(cd.data_forces, axis=2), axis=1)
plt.plot(range(0,len(forces_r)*3,3), forces_r, '-k')
plt.savefig('ss.pdf')

cd = CarbonData('non_relaxed_single0.8_relaxed0.2', random_seed=None, with_forces=True)
print(cd.data_positions.shape)
print(cd.data_energies.shape)
plt.clf()
forces_r = np.mean(np.linalg.norm(cd.data_forces, axis=2), axis=1)
plt.plot(range(0,len(forces_r)*3,3), forces_r, '-k')
plt.savefig('sr.pdf')

cd = CarbonData('non_relaxed_double0.8_non_relaxed_double0.2', random_seed=None, with_forces=True)
print(cd.data_positions.shape)
print(cd.data_energies.shape)
plt.clf()
forces_r = np.mean(np.linalg.norm(cd.data_forces, axis=2), axis=1)
plt.plot(range(0,len(forces_r)*3,3), forces_r, '-k')
plt.savefig('dd.pdf')

cd = CarbonData('non_relaxed_double0.8_relaxed0.2', random_seed=None, with_forces=True)
print(cd.data_positions.shape)
print(cd.data_energies.shape)
plt.clf()
forces_r = np.mean(np.linalg.norm(cd.data_forces, axis=2), axis=1)
plt.plot(range(0,len(forces_r)*3,3), forces_r, '-k')

energies_r = cd.data_energies
plt.plot(range(0,len(energies_r)*3,3), energies_r, '-b')
plt.savefig('dr.pdf')
# plt.show()

multi_forces = np.reshape(np.load('multi_perturb/forces.npy'), (10000,24,3))
plt.clf()
multi_forces = np.mean(np.linalg.norm(multi_forces, axis=2), axis=1)
plt.plot(range(len(multi_forces)),multi_forces,'-k')
plt.savefig('multi.pdf')