#/usr/bin/env python

import numpy as np
from itertools import combinations

def rownorm(mat):
    """Row normalization of a matrix"""
    return np.divide(mat.T, np.sum(mat, axis=1)).T
    
def colnorm(mat):
    """Column normalization of a matrix"""    
    return np.divide(mat, np.sum(mat, axis=0))

def safelog(vals):           
    with np.errstate(divide='ignore'):
        return np.log(vals)

def display_matrix(mat, rnames=None, cnames=None, title='', digits=4):
    """Utility function for displaying strategies to standard output."""
    mat = np.round(mat, digits)
    rowlabelwidth = 2 + max([len(x) for x in rnames] + [digits+2])
    cwidth = 2 + max([len(x) for x in cnames] + [digits+2])
    # Divider bar of the appropriate width:
    print "-" * ((cwidth * len(cnames)) + rowlabelwidth)
    print title
    # Matrix with even-width columns wide enough for the data:
    print ''.rjust(rowlabelwidth) + "".join([str(s).rjust(cwidth) for s in cnames])        
    for i in range(mat.shape[0]):  
        print str(rnames[i]).rjust(rowlabelwidth) + "".join(str(x).rjust(cwidth) for x in mat[i, :])

def powerset(x, minsize=1, maxsize=None):
    result = []
    if maxsize == None: maxsize = len(x)
    for i in range(minsize, maxsize+1):
        for val in combinations(x, i):
            result.append(list(val))
    return result
