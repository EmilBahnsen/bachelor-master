from carbondata import *
import pickle

class FeatureDataProvider:
    def __init__(self, file_list, carbon_data, trainPart=0.80, normalized_labels=False, feature_scaling=False):
        # Remove commented out lines
        file_list = [aFile for aFile in file_list if aFile[0] != "#"]

        data         = carbon_data
        n_structures = data.numberOfStructures
        n_features   = len(file_list)
        energies     = np.reshape(data.data_energies, (n_structures,1))

        featureVectors = np.ndarray((n_structures,data.structure_size,n_features))
        for i,filePath in enumerate(file_list):
            filehandler = open(filePath, "rb")
            featureData = pickle.load(filehandler)
            featureData = np.array(featureData.tolist())
            featureData = featureData[carbon_data.used_structures_index]
            print(filePath)
            featureVectors[:,:,i] = featureData

        featureVectors = np.reshape(featureVectors, (n_structures,data.structure_size,n_features))
        if feature_scaling:
            featureVectors = self._feature_scaling(featureVectors)
        partTrain      = int(trainPart * n_structures)
        full_range     = range(data.numberOfStructures)
        train_range    = full_range[:partTrain]
        test_range     = full_range[partTrain:]
        train_vectors  = featureVectors[train_range]
        train_energies = energies[train_range]
        test_vectors   = featureVectors[test_range]
        test_energies  = energies[test_range]

        self.train = DataProvider(train_vectors, train_energies, normalized_labels = normalized_labels)
        self.test  = DataProvider(test_vectors, test_energies, normalized_labels = normalized_labels)

    # Feature scaling according to Behler eq. 21 and 22
    def _feature_scaling(self,featureVectors):
        (n_structures,structure_size,n_features) = featureVectors.shape
        n_atoms_total = n_structures*structure_size
        G_sum = np.sum(np.sum(featureVectors, axis=1),axis=0)
        G_mean = G_sum/n_atoms_total

        # Broadcast along 0 and 1 axis: eq. 21
        featureVectors = featureVectors - G_mean

        # Eq. 22
        S_min = -1
        S_max = 1
        G_min = featureVectors.min(axis=0, keepdims=True).min(axis=1, keepdims=True)
        G_max = featureVectors.max(axis=0, keepdims=True).max(axis=1, keepdims=True)
        featureVectors = (featureVectors - G_min)/(G_max - G_min) * (S_max - S_min) + S_min
        
        return featureVectors

# Test
if __name__ == "__main__":
    carbon_data = CarbonData(data_dir = "carbondata/bachelor2018-master/CarbonData",
                             structure_size = 24)
    featureProvider = FeatureDataProvider(["features/G1_0.05_0_2.pickle", "features/G1_20_0_2.pickle"],
                                          carbon_data,
                                          trainPart=0.9,
                                          normalized_labels=False,
                                          feature_scaling=True)
