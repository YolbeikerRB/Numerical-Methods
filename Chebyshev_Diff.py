"""
Author: Yolbeiker Rodríguez Baez
This file contains the functions to calculate 
the Chebyshev differentiation matrix
"""

import numpy as np
import matplotlib.pyplot as plt

def Chebyshev_DiffMatrix(xini, xmax, Num):
    """
    This function calculates the Chebyshev differentiation matrix
    """
    xgrid   = (xmax + xini)/2 + (xmax - xini)/2*np.cos(np.pi*np.arange(0, Num + 1)/Num)
    CheCons = np.zeros_like(xgrid)            # Coefficients of the Chebyshev polynomials

    for j in range(Num+1): 
        val = xgrid[j] - xgrid
        CheCons[j] = np.prod(val[val != 0.0])

    Dmatrix = np.zeros((Num+1, Num+1))      # Chebyshev differentiation matrix
    for i in range(Num+1):
        for j in range(Num+1):
            if i != j:
                Dmatrix[i,j] = CheCons[i]/(CheCons[j]*(xgrid[i] - xgrid[j]))
            else:
                val = xgrid[j] - xgrid
                Dmatrix[j,j] = np.sum(1/val[ val != 0 ])
    # The gris is computed from the largest to the smallest values.
    # We reverse the order of the grid wchich implies that we also cange the sign of the matrix
    return xgrid[::-1], -1*Dmatrix

##==================================================================================================
## Parameters of the grid
xmin = -1
xmax = 1
Nnum = 100

## Chebyshev differentiation matrix
xgrid, Dmatrix = Chebyshev_DiffMatrix(xmin, xmax, Nnum)
mygrid = np.linspace(xmin, xmax, Nnum+1)

## Plotting
plt.plot(xgrid, Dmatrix@np.sin(xgrid), label = r'$\partial_x \sin(x)$ by Chebyshev')
plt.plot(mygrid, np.cos(mygrid), label = r'$\cos(x)$')
plt.legend()
plt.tight_layout()
plt.show()