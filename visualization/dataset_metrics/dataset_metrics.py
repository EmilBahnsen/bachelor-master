from carbondata import CarbonData
import numpy as np
import scipy.spatial as spatial
from visualization.plot_structure import *
import matplotlib.pyplot as plt

cd = CarbonData('/home/bahnsen/carbon_nn/carbondata/bachelor2018-master/CarbonData/')

# # Properties of spacial representation of strutures
# positions = cd.data_positions # (10808,24,3) [nr. of structures, atoms in structure, (x,y,z)]
# (n_structures,n_atoms,_) = np.shape(positions)
# min_pos = np.min(positions,axis=1)
# max_pos = np.max(positions,axis=1)

# xyz_diff = max_pos - min_pos
# max_xyz 	= np.max(xyz_diff, axis=0)
# argmax_xyz 	= np.argmax(xyz_diff, axis=0)
# min_xyz 	= np.min(xyz_diff, axis=0)
# argmin_xyz 	= np.argmin(xyz_diff, axis=0)
# mean_xyz 	= np.mean(xyz_diff, axis=0)
# std_xyz 	= np.std(xyz_diff, axis=0)
# print("Enclosing zxy of structure:")
# print("Min:", min_xyz, "at index", argmin_xyz)
# print("Max:", max_xyz, "at index", argmax_xyz)
# print("Mean:", mean_xyz)
# print("Std:", std_xyz)
# print()

# volumes = np.prod(xyz_diff, axis=1)
# max_vol 	= np.max(volumes)
# argmax_vol 	= np.argmax(volumes)
# min_vol 	= np.min(volumes)
# argmin_vol 	= np.argmin(volumes)
# mean_vol 	= np.mean(volumes)
# std_vol 	= np.std(volumes)
# print("Enclosing volume of structure:")
# print("Min: %.2f" % min_vol, "at index", argmin_vol)
# print("Max: %.2f" % max_vol, "at index", argmax_vol)
# print("Mean: %.2f" % mean_vol)
# print("Std: %.2f" % std_vol)
# print()

# # Energy
# energies = cd.data_energies
# print("Energies:")
# print("Min: %.2f" % np.min(energies), "at index", np.argmin(energies))
# print("Max: %.2f" % np.max(energies), "at index", np.argmax(energies))
# print("Mean: %.2f" % np.mean(energies))
# print("Std: %.2f" % np.std(energies))
# print()

# # Of any structure pair of atoms farthest apart
# n_distances_per_structure = int((n_atoms)*(n_atoms-1)/2)
# dists = np.ndarray(n_structures * n_distances_per_structure)
# dist_counter = 0
# for struc_n,struc in enumerate(positions):
# 	for i in range(len(struc)):
# 		pi = struc[i]
# 		for j in range(i+1,len(struc)):
# 			pj = struc[j]
# 			dists[dist_counter] = np.linalg.norm(pi - pj)
# 			dist_counter+=1

# print("Interatomic distances:")
# print("Min: %.2f" % np.min(dists), "at index %i" % np.floor(np.argmin(dists)/n_distances_per_structure))
# print("Max: %.2f" % np.max(dists), "at index %i" % np.floor(np.argmax(dists)/n_distances_per_structure))
# print("Mean: %.2f" % np.mean(dists))
# print("Std: %.2f" % np.std(dists))
# print()

# # Minimal convex hull
# volumes = np.ndarray(n_structures)
# for i,coords in enumerate(positions):
# 	volumes[i] = spatial.ConvexHull(coords).volume

# print("Convex hull:")
# print("Min: %.2f" % np.min(volumes), "at index", np.argmin(volumes))
# print("Max: %.2f" % np.max(volumes), "at index", np.argmax(volumes))
# print("Mean: %.2f" % np.mean(volumes))
# print("Std: %.2f" % np.std(volumes))
# print()

sp = StructurePlotter()

fig = plt.figure()
def save_plot(index,title,file_name):
	fig.clf()
	sp.plot_structure(fig,index,global_axis=True)
	plt.title("Structure #" + str(index) + ": " + title + ", E = %.2f" % cd.data_energies[index] + " eV")
	fig.savefig(file_name)

save_plot(9344,"Min x-value","min_x.pdf")
save_plot(6417,"Min y-value","min_y.pdf")
save_plot(8907,"Min z-value","min_z.pdf")
save_plot(6983,"Max x-value","max_x.pdf")
save_plot(1590,"Max y-value","max_y.pdf")
save_plot(583, "Max z-value","max_z.pdf")

save_plot(8907,"Min enclosing volume","min_vol.pdf")
save_plot(8274,"Max enclosing volume","max_vol.pdf")

save_plot(7573,"Min energy","min_E.pdf")
save_plot(41,"Max energy","max_E.pdf")

save_plot(7028,"Min interatomic distance","min_dist.pdf")
save_plot(2073,"Max interatomic distance","max_dist.pdf")

save_plot(3577,"Min convex hull","min_convex_hull.pdf")
save_plot(5384,"Max convex hull","max_convex_hull.pdf")







