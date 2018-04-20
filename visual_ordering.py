from carbondata import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cd = CarbonData("carbondata/bachelor2018-master/CarbonData")

def getXYZ(points):
    return [point[0] for point in points], [point[1] for point in points], [point[2] for point in points]

def plotStructure(points,fig = plt.figure(), title = ""):
    ax = fig.add_subplot(111, projection='3d')

    x,y,z = getXYZ(points)
    ax.plot(x,y,z,'.k', markersize=2)
    zmin = ax.get_zlim()[0]
    ax.plot(x,y,zmin,'.g')
    plt.title(title)
    plt.show()

plotStructure(cd.getStructure(0), title=str(cd.data_energies[0]))
plotStructure(cd.getStructure(1))
