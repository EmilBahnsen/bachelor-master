import ast
import os
import sys
import pickle

def pick_features(feature_dir, criteria):
    param_files = [os.path.join(feature_dir, param_file) for param_file in os.listdir(feature_dir) if "param" in param_file]
    features_file_list = []
    for param_file in param_files:
        params = pickle.load(open(param_file, "rb"))
        if "G1" in param_file: # $eta"_"$R_s"_"$R_c"
            if params["eta"] in criteria["eta"] and params["R_s"] in criteria["R_s"] and params["R_c"] in criteria["R_c"]:
                features_file_list.append(param_file.replace('param_', ''))
        elif "G2" in param_file: # $eta"_"$zeta"_"$lambda"_"$R_c
            if params["eta"] in criteria["eta"] and params["zeta"] in criteria["zeta"] and params["LAMB"] in criteria["lambda"] and params["R_c"] in criteria["R_c"]:
                features_file_list.append(param_file.replace('param_', ''))
    return features_file_list
