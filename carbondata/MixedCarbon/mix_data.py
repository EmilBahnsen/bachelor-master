from carbondata import CarbonData
import numpy as np

# Split the data, to accomade the CarbonData class
n_structures = 10808
positions_all = np.load('pPositions.npy')
positions_all = np.reshape(positions_all, (3,n_structures,24,3))
energies_all = np.load('pEnergy.npy')
energies_all = np.reshape(energies_all, (3,n_structures))
forces_all = np.load('pForces.npy')
forces_all = np.reshape(forces_all, (3,n_structures,24,3))

# Shuffle
indicies = np.arange(n_structures)
np.random.seed(0)			# Predictabilly random
np.random.shuffle(indicies)
print(indicies)
positions_all = positions_all[:,indicies,:,:]
energies_all = energies_all[:,indicies]
forces_all = forces_all[:,indicies,:,:]

pos_relax = positions_all[0,:,:,:]
pos_non_relax1 = positions_all[1,:,:,:]
pos_non_relax2 = positions_all[2,:,:,:]

E_relax = energies_all[0,:]
E_non_relax1 = energies_all[1,:]
E_non_relax2 = energies_all[2,:]

F_relax = forces_all[0,:,:,:]
F_non_relax1 = forces_all[1,:,:,:]
F_non_relax2 = forces_all[2,:,:,:]

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
positions = np.append(pos_non_relax1,pos_non_relax2, axis=0)
pos_non_relax = positions[:n_structures] # Half it to be camparable with the above

energies = np.append(E_non_relax1,E_non_relax2, axis=0)
E_non_relax = energies[:n_structures] # Half it to be camparable with the above

forces = np.append(F_non_relax1,F_non_relax2, axis=0)
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
cd = CarbonData('non_relaxed_single0.8_non_relaxed_single0.2')
print(cd.data_positions.shape)
print(cd.data_energies.shape)
cd = CarbonData('non_relaxed_single0.8_relaxed0.2')
print(cd.data_positions.shape)
print(cd.data_energies.shape)
cd = CarbonData('non_relaxed_double0.8_non_relaxed_double0.2')
print(cd.data_positions.shape)
print(cd.data_energies.shape)
cd = CarbonData('non_relaxed_double0.8_relaxed0.2')
print(cd.data_positions.shape)
print(cd.data_energies.shape)
