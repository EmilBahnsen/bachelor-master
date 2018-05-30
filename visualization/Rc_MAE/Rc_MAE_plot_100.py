# usage: python Rc_MAE.py Rc_MAE_100.txt Rc_MAE_100.pdf "13-100-1 network after 150k itterations"
import matplotlib.pyplot as plt
import numpy as np
import sys

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

data_file = sys.argv[1]
out_file = sys.argv[2]
title = sys.argv[3]

Rc_new,MAE_new,MAE_err = np.loadtxt(data_file)

fig = plt.figure()
ax = fig.gca()
ax.errorbar(Rc_new,MAE_new,yerr=MAE_err,fmt='.b', markersize=2)
# plt.ylim(.28,0.35)
# plt.xlim(2.6,3)
plt.title(title)
ax.set_xlabel("$R_c\, [A]$")
ax.set_ylabel("$MAE\, [eV]$")

rect = [0.25,0.5,0.7,0.45]
ax1 = add_subplot_axes(ax,rect)
s = int(np.where(Rc_new == 2.6)[0])
e = int(np.where(Rc_new == 3)[0])
ax1.errorbar(Rc_new[s:e],MAE_new[s:e],yerr=MAE_err[s:e],fmt='.b', markersize=2)

plt.savefig(out_file)
