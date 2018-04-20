
all_features = np.ndarray((n_structures), dtype=np.ndarray)
for i in range(n_structures):
    points = carbonData.getStructure(i)
    nPoints = len(points)
    features = np.ndarray((nPoints))
    for j in range(nPoints):
        features[j] = distance3D(j, points, ETA, R_S, R_C)
    all_features[i] = features;

pickle.dump(all_features, filehandler)
