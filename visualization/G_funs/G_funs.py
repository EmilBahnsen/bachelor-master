import matplotlib
import matplotlib.pyplot as plt
import features as fea
import numpy as np

matplotlib.rcParams.update({'font.size': 15})

n_values = 100
R_end = 3.5
Rc = 3
R = np.ndarray((n_values))

eta_vals = [0.05,2,4,8,20,40,80]
G1 = np.ndarray((len(eta_vals),n_values))

angles = np.ndarray((n_values))
zeta_vals = [1,2,4]
lamb_vals = [1,-1]
G2 = np.ndarray((len(zeta_vals),len(lamb_vals),n_values))

for n in range(n_values):
	Rn = R_end/n_values * n
	p1 = [0,0,0]
	p2 = [Rn,0,0]
	R[n] = Rn
	for m in range(len(eta_vals)):
		eta = eta_vals[m]
		G1[m,n] = fea.G1(0,[p1,p2],eta,0,Rc)

	p3 = [0,0,0]
	p4 = [1,0,0]
	angle = np.pi/n_values * n
	p5 = [np.cos(angle),np.sin(angle),0]
	angles[n] = angle
	
	for z in range(len(zeta_vals)):
		zeta = zeta_vals[z]
		for l in range(len(lamb_vals)):
			lamb = lamb_vals[l]
			G2[z,l,n] = fea.G2(0,[p3,p4,p5],0.005,zeta,lamb,Rc)

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

fig1 = plt.figure(0)
plt.xlabel("$R_{ij}$ [Å]")
plt.ylabel("$G^{I}(R_s=0,R_c=3Å)$")
for m in range(len(eta_vals)):
	plt.plot(R,G1[m])
plt.legend([
	"$\eta = 0.05$",
	"$\eta = 2$",
	"$\eta = 4$",
	"$\eta = 8$",
	"$\eta = 20$",
	"$\eta = 40$",
	"$\eta = 80$",
	])
plt.xlim(xmin=0, xmax=3.5)
plt.tight_layout()
fig1.savefig("G1.pdf")

fig1 = plt.figure(1)
plt.xlabel("$\\theta_{ijk}$ [$\deg$]")
plt.ylabel("$G^{II}(\eta=0.005,R_c=3Å)$")
for l in range(len(lamb_vals)):
	for z in range(len(zeta_vals)):
		plt.plot(angles * 180/np.pi,G2[z,l])
plt.legend([
	"$\zeta = 1, \lambda = +1$",
	"$\zeta = 2, \lambda = +1$",
	"$\zeta = 4, \lambda = +1$",
	"$\zeta = 1, \lambda = -1$",
	"$\zeta = 2, \lambda = -1$",
	"$\zeta = 4, \lambda = -1$",
	])
plt.xticks(np.arange(0, 190, step=20))
plt.xlim(xmin=0, xmax=180)
plt.tight_layout()
fig1.savefig("G2.pdf")
