import os
import numpy as np
import random

class DataProvider:
    batchPointer = 0
    def __init__(self, data, labels, normalized_labels = False, evenly_provided = False):
        self.data   = np.array(data)
        self.labels = np.array(labels)
        self.min_label = min(self.labels)
        self.max_label = max(self.labels)

        if normalized_labels:
            self.labels -= np.mean(self.labels)                 # Shift and normalize
            self.labels /= np.max(np.absolute(self.labels))     #

    # DOTO: Mini-batches
    def next_batch(self, batch_size: int):
        # If its asking for too much data, return it all
        if batch_size > len(self.data):
            return self.data, self.labels
        # If it askes for data past the array, wrap around
        # if self.batchPointer + batch_size - 1 > len(self.data) - 1:
        #     end_inverval = range(self.batchPointer,len(self.data))
        #     self.batchPointer = batch_size - len(end_inverval)
        #     start_inverval = range(0, self.batchPointer)
        #     return np.append(self.data[end_inverval],self.data[start_inverval]), np.append(self.labels[end_inverval],self.labels[start_inverval])
        # interval = range(self.batchPointer,self.batchPointer+batch_size)
        # self.batchPointer+=batch_size
        # Completely random sampling
        interval = random.sample(range(len(self.data)), batch_size)
        return self.data[interval], self.labels[interval]

    def next_uniform_batch(self, batch_size: int):
        return_index = np.ndarray(batch_size, dtype=int)
        bin_starts = np.linspace(self.min_label, self.max_label, batch_size)
        bin_size = (self.max_label-self.min_label)/batch_size
        for i in range(len(bin_starts)):
            # Get indices of the ones in the i'th bin
            bin_lower = bin_starts[i]
            bin_upper = bin_starts[i] + bin_size
            indices = np.where((self.labels >= bin_lower) & (self.labels < bin_upper))[0]

            # If there were no numbers in the interval
            # then get the index of the number closest to the upper bin limit
            # otherwise choose a random one from the bin
            if len(indices) == 0:
                index = min(range(len(self.labels)), key=lambda j: abs(self.labels[j]-bin_upper))
            else:
                index = np.random.choice(indices)
            return_index[i] = index

        return self.data[return_index], self.labels[return_index]

    def get_all(self):
        return self.data, self.labels


class CarbonData:
    fileDir = os.path.dirname(__file__)
    #dataDir = os.path.join(fileDir, '../tests/carbondata/data')

    def __init__(self, data_dir, structure_size = 24, energy_interval = None, structures_to_use = 1.0, random_seed = 0, with_forces=False):
        energyDataPath = os.path.join(data_dir, 'energies.npy')
        positionDataPath = os.path.join(data_dir, 'positions.npy')

        self.data_energies  = np.load(energyDataPath)
        self.numberOfStructures = len(self.data_energies)
        self.data_positions = np.load(positionDataPath)
        self.data_positions = np.reshape(self.data_positions, (self.numberOfStructures,structure_size,3))

        if with_forces:
            forcesDataPath = os.path.join(data_dir, 'forces.npy')
            self.data_forces = np.load(forcesDataPath)
            self.data_forces = np.reshape(self.data_forces, (self.numberOfStructures,structure_size,3))

        self.used_structures_index = np.arange(self.numberOfStructures)
        # Randomize data
        if (random_seed is not None):
            np.random.seed(seed=random_seed)
            np.random.shuffle(self.used_structures_index)
            self.data_energies = self.data_energies[self.used_structures_index]
            self.data_positions = self.data_positions[self.used_structures_index]

        # Use structures_to_use only
        if structures_to_use is not 1.0:
            self.numberOfStructures = int(np.floor(structures_to_use*self.numberOfStructures))
            self.used_structures_index = self.used_structures_index[0:self.numberOfStructures]
            self.data_energies = self.data_energies[0:self.numberOfStructures]
            self.data_positions = self.data_positions[0:self.numberOfStructures]

        if energy_interval is not None:
            raise ValueError('energy_interval not implemented: must leave None.')
            E_min = min(energy_interval)
            E_max = max(energy_interval)
            self.used_structures_index = np.where(np.logical_and(self.data_energies>=E_min, self.data_energies<=E_max))
            self.data_energies = self.data_energies[self.used_structures_index]
            self.data_positions = self.data_positions[self.used_structures_index]

        self.structure_size = structure_size

    def getStructure(self, i):
        return self.data_positions[i]

if __name__ == "__main__":

    dp = DataProvider([0,1,2,3,4], [1,2,3,4,5])
    print(dp.next_batch(3))
    print(dp.next_batch(5))
    print(dp.next_batch(3))
    exit()

    cd = CarbonData('bachelor2018-master/CarbonData/', structures_to_use=1000)
    import matplotlib.pyplot as plt
    np.random.seed(seed=0)
    #np.random.shuffle(cd.data_energies)
    plt.plot(range(len(cd.data_energies)), cd.data_energies, '.b')
    plt.show()
    #[print(e) for e in cd.data_energies]
