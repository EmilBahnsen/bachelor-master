import pickle
import carbondata as cd
import numpy as np

positions = np.load("carbondata/bachelor2018-master/CarbonData/positions.npy")
energies = np.load("carbondata/bachelor2018-master/CarbonData/energies.npy")

positions = positions.reshape((10808,24,3))
print(positions.shape)

# Get the sorting index and sort according to energy
sorted_index = np.argsort(energies)
energies = energies[sorted_index]
positions = positions[sorted_index]

max_samples = 100
E_min = min(energies)
E_max = max(energies)
keep_pattern = np.zeros(len(energies), dtype=bool)

n_bins = int(max_samples/10)
bin_indices = np.digitize(energies, np.linspace(E_min,E_max, num=n_bins))

collection_count = np.zeros(n_bins)
for i in range(len(energies)):
    if collection_count[bin_indices[i]-1] < 10:
        keep_pattern[i] = 1
        collection_count[bin_indices[i]-1] += 1
    else:
        keep_pattern[i] = 0
    if max_samples <= keep_pattern[keep_pattern == 1].shape[0]:
        break

ex_energies = energies[keep_pattern]
ex_positions = positions[keep_pattern]

ex_positions = ex_positions.reshape((-1,3))

file_energies = open("carbondata/CarbonDataSubset/energies.npy", "wb+")
file_positions = open("carbondata/CarbonDataSubset/positions.npy", "wb+")

np.save(file_energies, ex_energies)
np.save(file_positions, ex_positions)

print(int(ex_positions.shape[0]/24), "picked")
