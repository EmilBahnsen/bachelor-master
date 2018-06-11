from visualization.model_loader import *
from feature_data_provider import *
from carbondata import *
import os
import matplotlib.pyplot as plt
from matplotlib.collections import CircleCollection

def absolute_path_wrt_parent(path):
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_file_dir, "..", path)

class StructureEnergyMap:
    def __init__(self, model_dir, with_forces, random_seed=None):
        self.ml = ModelLoader(log_dir=model_dir)

        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_data_dir_path = os.path.join(this_file_dir, "..", self.ml.params["data_directory"])
        self.carbon_data = CarbonData(data_dir = absolute_data_dir_path,
                                      structure_size = self.ml.params["number_of_atoms"],
                                      random_seed=random_seed,with_forces=with_forces)
        
        # Open feature file list and put it into a list
        feature_list_file = absolute_path_wrt_parent(self.ml.params["feature_list_file"])
        feature_file_list = [os.path.join(this_file_dir, "..", x.strip()) for x in open(feature_list_file, "r").readlines()]
        self.featureProvider = FeatureDataProvider(feature_file_list,
                                                   self.carbon_data, 
                                                   trainPart = self.ml.params["train_part"], 
                                                   normalized_labels=False, 
                                                   feature_scaling=self.ml.params["feature_scaling"])

        self._structure_atom_energies = None
        self._delta_z = None

    def get_energies_of_structure(self,n):
        return self.structure_atom_energies[n]

    # The lagest within a structure
    @property
    def delta_z(self):
        if self._delta_z == None:
            zs = self.carbon_data.data_positions[:,:,2]
            delta_zs = np.amax(zs, axis=0) - np.amin(zs, axis=0)
            self._delta_z = np.max(delta_zs)
        return self._delta_z

    @property
    def structure_atom_energies(self):
        if self._structure_atom_energies is None:
            #[print(x) for x in self.ml.get_name_of_tensors()]
            print(self.featureProvider.train.data.shape, self.featureProvider.test.data.shape)
            all_data = np.append(self.featureProvider.train.data, self.featureProvider.test.data, axis=0)
            print(all_data.shape)
            self._structure_atom_energies = self.ml.eval_tensor_by_name("layer_out/fc_out/Tensordot:0", all_data)
            self._structure_atom_energies = np.squeeze(self._structure_atom_energies)
        return self._structure_atom_energies

    def calc_sizes_using_z_depth(self, points):
        max_size = 70 # in points squared
        min_size = 30
        delta_s = max_size - min_size
        zs = points[:,2]
        return delta_s/self.delta_z * zs + min_size

    def structure_energy_map_figure_2D_2(self,fig,n):
        atom_positions = self.carbon_data.getStructure(n)
        structure_energy = self.carbon_data.data_energies[n]
        atom_energies = self.get_energies_of_structure(n)

        e_min = min(atom_energies)
        e_max = max(atom_energies)

        sizes = self.calc_sizes_using_z_depth(atom_positions)

        ax = fig.add_subplot(111)
        E_tot = np.sum(atom_energies)
        DeltaE = structure_energy - E_tot
        plt.title("Structure #" + str(n) + ", $E_N$={:.2f}eV".format(E_tot) + ", ΔE={:.2f}eV".format(DeltaE))

        cc = CircleCollection(sizes=sizes,offsets=atom_positions[:,0:2],transOffset=ax.transData)
        cc.set_array(atom_energies)
        ax.add_collection(cc)
        ax.autoscale_view()
        ax.axis("off")
        cbar = plt.colorbar(cc)
        cbar.set_label('Energy [eV]')

        x = atom_positions[:,0]
        y = atom_positions[:,1]
        ax.set_xlim(np.min(x) - 3, np.max(x) + 3)
        ax.set_ylim(np.min(y) - 3, np.max(y) + 3)

        return fig

    def add_forces(self,fig,n,network_forces=False,scaling=0.2,color='k'):
        positions = self.carbon_data.getStructure(n)
        if network_forces:
            #feed_positions = np.expand_dims(positions,axis=0)
            feed_positions = np.array([self.carbon_data.getStructure(n)])
            forces = self.ml.get_forces_in_structures(feed_positions)[0]
        else:
            forces = self.carbon_data.data_forces[n]

        forces = scaling*forces

        for i,xy in enumerate(positions[:,0:2]):
            x = xy[0]; y = xy[1]
            dx = forces[i,0]; dy = forces[i,1]
            plt.arrow(x,y,dx,dy,color=color,head_width=0.1)

# usage: python structure_heatmap.py ../logs/mixed_log/non_relaxed_double0.8_non_relaxed_double0.2/features_few/Rc5/z-score/29-29-29_ba5pct/struc1/2018-06-06_10.13/ structure_heatmaps_dual_dual/
# usage: python structure_heatmap.py ../logs/mixed_log/non_relaxed_double0.8_relaxed0.2/features_few/Rc5/z-score/29-29-29_ba5pct/struc1/2018-06-06_10.13/ structure_heatmaps_dual_relaxed/
# usage: python structure_heatmap.py ../logs/mixed_log/multi_perturb/features_few/Rc5/z-score/29-29-29_ba5pct/struc1/2018-06-06_10.13 structure_heatmaps_multi_perturb/
# usage: python structure_heatmap.py ../logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct/struc1/2018-05-31_01.00/ original_struc1/
if __name__ == "__main__":
    model_dir = sys.argv[1]
    save_dir = sys.argv[2]
    with_forces = True
    random_seed = None
    sem = StructureEnergyMap(model_dir,with_forces=with_forces,random_seed=random_seed)

    #print(sem.get_energies_of_structure(0))

    energies = sem.carbon_data.data_energies
    # plt.figure(0)
    # forces = sem.carbon_data.data_forces
    # print(forces.shape)
    # forces = np.mean(np.linalg.norm(forces, axis=2),axis=1)
    # plt.plot(range(len(forces)),forces,'-b')
    # plt.show()
    # input('')
    # exit()
    ascending_energy_index_list = np.argsort(energies)

    n_structures = sem.carbon_data.numberOfStructures
    index_split = int(n_structures*0.8)

    fig = plt.figure()
    #for i,n in enumerate(ascending_energy_index_list):
    #for i,n in enumerate(ascending_energy_index_list[0:10]):
    for i,n in enumerate(range(index_split,index_split+10)):
    #for i,n in enumerate(range(9217,9217+1)):
        print("Saveing index:", n)
        fig.clf()
        fig = sem.structure_energy_map_figure_2D_2(fig,n)
        if with_forces:
            sem.add_forces(fig,n,network_forces=False,color='k')
            sem.add_forces(fig,n,network_forces=True,color='r')
        file_name = os.path.join(save_dir, "structure_heatmap_" + str(i) + ".svg")
        fig.savefig(file_name)
        print(file_name, "saved.")

