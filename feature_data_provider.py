from carbondata import *
import pickle
import features as feature_funs
import re

f13_descriptors = [
    {"type": "G1", "eta": 0.05, "Rs": 0, "Rc": 2.7},
    {"type": "G1", "eta": 20, "Rs": 0, "Rc": 2.7},
    {"type": "G1", "eta": 2, "Rs": 0, "Rc": 2.7},
    {"type": "G1", "eta": 40, "Rs": 0, "Rc": 2.7},
    {"type": "G1", "eta": 4, "Rs": 0, "Rc": 2.7},
    {"type": "G1", "eta": 80, "Rs": 0, "Rc": 2.7},
    {"type": "G1", "eta": 8, "Rs": 0, "Rc": 2.7},
    {"type": "G2", "eta": 0.005, "zeta": 1, "lambda": -1, "Rc": 2.7},
    {"type": "G2", "eta": 0.005, "zeta": 1, "lambda": 1, "Rc": 2.7},
    {"type": "G2", "eta": 0.005, "zeta": 2, "lambda": -1, "Rc": 2.7},
    {"type": "G2", "eta": 0.005, "zeta": 2, "lambda": 1, "Rc": 2.7},
    {"type": "G2", "eta": 0.005, "zeta": 3, "lambda": -1, "Rc": 2.7},
    {"type": "G2", "eta": 0.005, "zeta": 3, "lambda": 1, "Rc": 2.7}
]

class FeatureDataProvider:
    def __init__(self, file_list, carbon_data, trainPart=0.80, trainFraction=1.0, normalized_labels=False, feature_scaling=None):
        # Remove commented out lines
        self.file_list = [aFile for aFile in file_list if aFile[0] != "#"]

        data         = carbon_data
        n_structures = data.numberOfStructures
        n_features   = len(self.file_list)
        energies     = np.reshape(data.data_energies, (n_structures,1))

        featureVectors = np.ndarray((n_structures,data.structure_size,n_features))
        for i,filePath in enumerate(self.file_list):
            filehandler = open(filePath, "rb")
            featureData = pickle.load(filehandler)
            featureData = np.array(featureData.tolist())
            featureData = featureData[carbon_data.used_structures_index]
            print(filePath)
            featureVectors[:,:,i] = featureData

        featureVectors = np.reshape(featureVectors, (n_structures,data.structure_size,n_features))
        self.feature_scaling = feature_scaling
        if feature_scaling is not None:
            print("Using feature scaling: ", feature_scaling['type'])
            if feature_scaling['type'] == 'min-max':
                featureVectors = self._feature_scaling_min_max(featureVectors, interval=feature_scaling['interval'])
            elif feature_scaling['type'] == 'z-score':
                self._feature_scaling_z_score_params(featureVectors,trainPart)
                featureVectors = self._feature_scaling_z_score(featureVectors)
            if np.isnan(featureVectors).any():
                raise ValueError('nan value encounted using feature scaling type: ' + feature_scaling['type'])


        partTrain      = int(trainPart * n_structures)
        full_range     = range(data.numberOfStructures)
        trainEnd       = int(trainFraction * partTrain)
        train_range    = full_range[:trainEnd]
        test_range     = full_range[partTrain:]
        train_vectors  = featureVectors[train_range]
        train_energies = energies[train_range]
        test_vectors   = featureVectors[test_range]
        test_energies  = energies[test_range]

        self.train = DataProvider(train_vectors, train_energies, normalized_labels = normalized_labels)
        if trainPart < 1:
            self.test = DataProvider(test_vectors, test_energies, normalized_labels = normalized_labels)

        self.featureVectors = featureVectors
        self.__feature_descriptors = None

    # Feature scaling according to Behler eq. 21 and 22
    def _feature_scaling_min_max(self,featureVectors,interval=[0,1]):
        (n_structures,structure_size,n_features) = featureVectors.shape
        n_atoms_total = n_structures*structure_size
        G_sum = np.sum(np.sum(featureVectors, axis=1),axis=0)
        G_mean = G_sum/n_atoms_total

        # Broadcast along 0 and 1 axis: eq. 21
        featureVectors = featureVectors - G_mean

        # Eq. 22
        S_min = interval[0]
        S_max = interval[1]
        G_min = featureVectors.min(axis=0, keepdims=True).min(axis=1, keepdims=True)
        G_max = featureVectors.max(axis=0, keepdims=True).max(axis=1, keepdims=True)
        featureVectors = (featureVectors - G_min)/(G_max - G_min) * (S_max - S_min) + S_min

        return featureVectors

    def _feature_scaling_z_score_params(self,featureVectors,trainPart):
        (n_structures,_,_) = featureVectors.shape
        partTrain = int(trainPart * n_structures)
        train_featureVectors = featureVectors[:partTrain]
        (n_structures,structure_size,n_features) = train_featureVectors.shape
        n_atoms_total = n_structures*structure_size
        G_sum = np.sum(np.sum(train_featureVectors, axis=1),axis=0)
        self.G_mean = G_sum/n_atoms_total
        self.G_std = np.sqrt(np.sum(np.sum((featureVectors - self.G_mean)**2,axis=1),axis=0)/(n_atoms_total - 1))

    def _feature_scaling_z_score(self,featureVectors):
        # Z-score normalization (https://docs.tibco.com/pub/spotfire/6.5.1/doc/html/norm/norm_z_score.htm, 22 apr.)
        featureVectors = (featureVectors - self.G_mean)/self.G_std

        # Sanitize if an entry of featureVectors was zero (results in nan's)
        featureVectors = np.nan_to_num(featureVectors)
        return featureVectors

    @property
    def feature_descriptors(self):
        if self.__feature_descriptors is None:
            self.__feature_descriptors = []
            for i,fea_file in enumerate(self.file_list):
                fea_file_name = fea_file.split('/')[-1]
                params = np.array(re.findall(r"[-+]?\d*\.\d+|[+-]?\d+", fea_file_name))
                fea_type = params[0]
                params = params[1:].astype(float)
                fea_dec = {}
                if fea_type == "1":
                    fea_dec = {"type": "G1", "eta": params[0], "Rs": params[1], "Rc": params[2]}
                elif fea_type == "2":
                    fea_dec = {"type": "G2", "eta": params[0], "zeta": params[1], "lambda": params[2], "Rc": params[3]}
                self.__feature_descriptors = np.append(self.__feature_descriptors, fea_dec)
        return self.__feature_descriptors


    def extract_features_from_structure(self,positions):
        feature_descriptors = self.feature_descriptors
        n_features = len(feature_descriptors)
        structure_size = len(positions)
        features = np.ndarray((1,structure_size,n_features))

        for i in range(len(positions)):
            for j,fea in enumerate(feature_descriptors):
                if fea["type"] == "G1":
                    features[0,i,j] = feature_funs.G1(i, positions, fea["eta"], fea["Rs"], fea["Rc"])
                elif fea["type"] == "G2":
                    features[0,i,j] = feature_funs.G2(i, positions, fea["eta"], fea["zeta"], fea["lambda"], fea["Rc"])

        # Scaling
        if self.feature_scaling is not None:
            if self.feature_scaling['type'] == 'z-score':
                features = self._feature_scaling_z_score(features)

        return features

    def extract_features_from_structures(self,structures):
        feature_descriptors = self.feature_descriptors
        n_structures = len(structures)
        structure_size = len(structures[0])
        n_features = len(feature_descriptors)
        features = np.ndarray((n_structures,structure_size,n_features))
        for i in range(len(structures)):
            positions = structures[i]
            features[i,:,:] = self.extract_features_from_structure(positions)
        return features

    def get_feature_space_hypercube_bonuds(self):
        fea_min = np.amin(self.featureVectors, axis=(0,1),keepdims=True)
        fea_max = np.amax(self.featureVectors, axis=(0,1),keepdims=True)
        return fea_min,fea_max


# Test
if __name__ == "__main__":
    carbon_data = CarbonData(data_dir = "carbondata/bachelor2018-master/CarbonData",
                             structure_size = 24)

    featureProvider = FeatureDataProvider([
            "features/G1_0.05_0_2.7.pickle",
            "features/G1_20_0_2.7.pickle",
            "features/G1_2_0_2.7.pickle",
            "features/G1_40_0_2.7.pickle",
            "features/G1_4_0_2.7.pickle",
            "features/G1_80_0_2.7.pickle",
            "features/G1_8_0_2.7.pickle",
            "features/G2_0.005_1_-1_2.7.pickle",
            "features/G2_0.005_1_1_2.7.pickle",
            "features/G2_0.005_2_-1_2.7.pickle",
            "features/G2_0.005_2_1_2.7.pickle",
            "features/G2_0.005_4_-1_2.7.pickle",
            "features/G2_0.005_4_1_2.7.pickle"
        ],
                                          carbon_data,
                                          trainPart=0.8,
                                          normalized_labels=False,
                                          feature_scaling={'type': 'z-score'})
    import numpy as np
    print("G_mean:",featureProvider.G_mean)
    print("G_std:",featureProvider.G_std)
