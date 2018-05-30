from carbondata import *
import os
import matplotlib.pyplot as plt
from matplotlib.collections import CircleCollection
from matplotlib.patches import Circle

class StructurePlotter:

    def __init__(self):
        self.carbon_data = CarbonData(data_dir = '/home/bahnsen/carbon_nn/carbondata/bachelor2018-master/CarbonData/')
        self._delta_z = None

    # The lagest within a structure
    @property
    def delta_z(self):
        if self._delta_z == None:
            zs = self.carbon_data.data_positions[:,:,2]
            delta_zs = np.amax(zs, axis=0) - np.amin(zs, axis=0)
            self._delta_z = np.mean(delta_zs)
        return self._delta_z

    def calc_sizes_using_z_depth(self,points):
        max_size = 200 # in points squared
        min_size = 50
        delta_s = max_size - min_size
        zs = points[:,2]
        return delta_s/self.delta_z * zs + min_size

    def plot_structure(self,fig,n,global_axis=False):
        atom_positions = self.carbon_data.getStructure(n)
        sizes = self.calc_sizes_using_z_depth(atom_positions)
        
        ax = fig.add_subplot(111)
        # circles = np.ndarray(24)
        # for i in range(len(circles))
        #     circles[i] = Circle(atom_positions[:,0:2][i], radius = np.sqrt(sizes[i]))
        # pc = 
        cc = CircleCollection(sizes=sizes,offsets=np.sort(atom_positions[:,0:2],axis=2),transOffset=ax.transData,facecolors='gray',edgecolors='k')
        ax.add_collection(cc)
        if global_axis:
            xs = self.carbon_data.data_positions[:,:,0]
            ys = self.carbon_data.data_positions[:,:,1]
            ax.set_xlim(np.min(xs)-1, np.max(xs)+1)
            ax.set_ylim(np.min(ys)-1, np.max(ys)+1)
        else:
            ax.autoscale_view()
        ax.axis("off")

        return fig

