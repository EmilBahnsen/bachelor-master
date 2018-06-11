import matplotlib.pyplot as plt
import numpy as np

plt.figure(0)
x = np.linspace(-1,0,100)
plt.plot(x,x**2,'-k')
dual_point_coord = np.array([x[50],x[60]])
plt.plot(dual_point_coord,dual_point_coord**2,'+k',markersize=15)
plt.plot(x[-1],x[-1]**2,'.k',markersize=15)
plt.axis('off')
plt.savefig('relaxation_curve.pdf')