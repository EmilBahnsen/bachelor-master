import pickle

filehandler = open("G1.pickle", "rb")
featureData = pickle.load(filehandler)
print(featureData)
