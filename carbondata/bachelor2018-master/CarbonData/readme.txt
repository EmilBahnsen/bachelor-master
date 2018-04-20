The data consist of 10808 structures of 24 carbon atoms
The energies are saved as a 1D numpy array of length 10808, with the first entry belonging to the first structure and so on.
The positions are saved as a 2D numpy array of length (10808 * 24, 3), i.e the first 24 rows belong to the first structure, the next 24 to the second structure and so on.

To load the data do

import numpy as np
energies = np.load('energies.npy')
positions = np.load('positions.npy')
