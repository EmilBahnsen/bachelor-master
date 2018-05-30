import features as fea
import carbondata as cd
import numpy as np
import os
import pickle

EXP_NAME = str(os.environ['EXP_NAME'])
FEA = str(os.environ['FEA'])
ETA = float(os.environ['ETA'])
ZETA = float(os.environ['ZETA'])
LAMB = float(os.environ['LAMB'])
R_C = float(os.environ['R_C'])
R_S = float(os.environ['R_S'])
SAVE_DIR = str(os.environ['SAVE_DIR'])
DATA_DIR = str(os.environ['DATA_DIR'])

carbonData = cd.CarbonData(DATA_DIR,random_seed=None) # DO NOT randomize when generating features

filehandler = open(SAVE_DIR + EXP_NAME, "wb")
filehandler_param = open(SAVE_DIR + "param_" + EXP_NAME, "wb")
n_structures = carbonData.numberOfStructures
#n_structures = 500
all_features = np.ndarray((n_structures), dtype=np.ndarray)
for i in range(n_structures):
    points = carbonData.getStructure(i)
    nPoints = len(points)
    features = np.ndarray((nPoints))
    for j in range(nPoints):
        if (FEA=="G1"):
            features[j] = fea.G1(j, points, ETA, R_S, R_C)
        elif (FEA=="G2"):
            features[j] = fea.G2(j, points, ETA, ZETA, LAMB, R_C)
    all_features[i] = features;

pickle.dump(all_features, filehandler)
pickle.dump({"exp_name": EXP_NAME,
             "feature": FEA,
             "eta": ETA,
             "zeta": ZETA,
             "LAMB": LAMB,
             "R_c": R_C,
             "R_s": R_S},filehandler_param)
filehandler.close()
filehandler_param.close()
