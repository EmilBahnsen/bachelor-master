import math
import itertools
import numpy as np
import pickle
import features as fea

def distance3D(p1,p2):
    return math.hypot(math.hypot(p1[0] - p2[0], p1[1] - p2[1]), p1[2] - p2[2])

def LD_pot(p1,p2):
    R = distance3D(p1,p2)
    return 1/(R**12) - 1/(R**6)

def LD_pot_tot(points):
    summed = 0
    energy_list = np.ndarray(np.shape(points)[0])
    for i,p1 in enumerate(points):
        atom_sum = 0
        for j,p2 in enumerate(points):
            if i is not j:
                energy = LD_pot(p1, p2)
                summed += energy
                atom_sum += energy
        energy_list[i] = atom_sum
    assert(round(summed,5) == round(np.sum(energy_list),5)) # check it's the same to 5 decimals
    return summed, energy_list

# Two atoms
n_values = 1000
positions = np.ndarray((1,3))
energies = np.ndarray((1))
energy_lists = np.ndarray((n_values), dtype=np.ndarray)
feature1 = np.ndarray((n_values), dtype=np.ndarray)
feature2 = np.ndarray((n_values), dtype=np.ndarray)

for i in range(n_values):
    z = (i/1000) * 2 + 0.99 # From R = .99 to 2

    position = [[0,0,0],[0,0,z],[0,2*z,0],[z,2*z,0],[z,0,0]]
    energy, energy_list = LD_pot_tot(position)

    energy_lists[i] = energy_list

    f1 = fea.G1(0, position, 2, 1, 7) # Atom 1
    f2 = fea.G1(1, position, 2, 1, 7) # Atom 2
    f3 = fea.G1(2, position, 2, 1, 7)
    f4 = fea.G1(3, position, 2, 1, 7)
    f5 = fea.G1(4, position, 2, 1, 7)

    feature1[i] = np.array([f1,f2,f3,f4,f5])
    feature2[i] = np.array([z,z,z,z,z])

    if i == 0:
        energies[0] = energy
        positions = np.array(position)
    else:
        energies = np.concatenate((energies, [energy]))
        positions = np.concatenate((positions, position))

pickle.dump(positions, open("carbondata/test_data/positions.npy","wb"))
pickle.dump(energies, open("carbondata/test_data/energies.npy","wb"))
pickle.dump(energy_lists, open("carbondata/test_data/energy_lists.npy","wb"))
pickle.dump(feature1, open("features/test_feature1.npy","wb"))
pickle.dump(feature1, open("features/test_feature2.npy","wb"))
