import os
import numpy as np
import math
import carbondata as cd

# Feature vectors
def norm3D(v):
    return math.hypot(math.hypot(v[0], v[1]), v[2])

def distance3D(p1,p2):
    return math.hypot(math.hypot(p1[0] - p2[0], p1[1] - p2[1]), p1[2] - p2[2])

def dot3D(p1,p2):
    return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]

def f_c(R_ij, R_c):
    return 0.5*(math.cos(math.pi*R_ij/R_c) + 1) # ~4 times faster than np.cos!

def G1(i,allPoints, eta, R_s, R_c):
    sum = 0
    p_i = allPoints[i]
    restPoints = np.delete(allPoints,i,0) # Delete the i-th row. FIX: (allPoints,i) -> (allPoints,i,0)
    for j,p_j in enumerate(restPoints):
        R_ij = distance3D(p_i,p_j)
        if R_ij > R_c: # Due to f_c
            continue
        sum += math.exp(-eta*(R_ij - R_s)**2 / R_c**2) * f_c(R_ij, R_c)
    return sum

def G2(i, allPoints, eta, zeta, lamb, R_c): # lamb = +1 or -1
    sum = 0
    p_i = allPoints[i]
    restPoints = np.delete(allPoints,i,0) # Same as above
    for j,p_j in enumerate(restPoints):
        for k,p_k in enumerate(restPoints):
            vR_ij = p_i - p_j
            vR_ik = p_i - p_k
            R_ij = norm3D(vR_ij)
            R_ik = norm3D(vR_ik)
            R_jk = distance3D(p_j,p_k)
            if R_ij > R_c or R_ik > R_c or R_jk > R_c: # Due to f_c
                continue
            theta_ijk = dot3D(vR_ij, vR_ik) / (R_ij * R_ik)
            sum += (1 + lamb * math.cos(theta_ijk))**zeta \
                    * math.exp(-eta*(R_ij**2 + R_ik**2 + R_jk**2) / R_c**2) \
                    * f_c(R_ij,R_c)*f_c(R_ik,R_c)*f_c(R_jk,R_c)

    return 2**(1 - zeta) * sum
