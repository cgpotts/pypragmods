#!/usr/bin/env python

######################################################################
# Common strings and some nuts-and-bolts functions used in the paper.
######################################################################

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

######################################################################
# Plot set-up

plt.style.use('ggplot')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='times')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r'\boldmath']
matplotlib.rcParams['xtick.major.pad']='1'
matplotlib.rcParams['ytick.major.pad']='0'

# Colors that should be good for colorblind readers.
colors = [
    '#1B9E77', # green; human data
    '#666666', # gray; literal semantics
    '#A6761D', # brownish; RSA
    '#E7298A', # dark pink; unconstrained
    '#D95F02'] # orange; neo-gricean model

######################################################################
# High-level settings and notation:

# The null message:
NULL = 'NULL'

# For relating the experimental data to the logical grammar:
SENTENCES = {
    "Every player hit all of his shots": "every(player)(hit(every(shot)))",
    "Every player hit none of his shots": "every(player)(hit(no(shot)))",    
    "Every player hit some of his shots": "every(player)(hit(some(shot)))",    
    "Exactly one player hit all of his shots": "exactly_one(player)(hit(every(shot)))",
    "Exactly one player hit none of his shots": "exactly_one(player)(hit(no(shot)))",
    "Exactly one player hit some of his shots": "exactly_one(player)(hit(some(shot)))",
    "No player hit all of his shots": "no(player)(hit(every(shot)))",
    "No player hit none of his shots": "no(player)(hit(no(shot)))",
    "No player hit some of his shots": "no(player)(hit(some(shot)))"}

# Abbreviated plot titles:
TITLES = {
    "every(player)(hit(every(shot)))": 'every...all',
    "every(player)(hit(no(shot)))": 'every...none',
    "every(player)(hit(some(shot)))": 'every...some',
    "exactly_one(player)(hit(every(shot)))": 'exactly one...all',
    "exactly_one(player)(hit(no(shot)))": 'exactly one...none',
    "exactly_one(player)(hit(some(shot)))": 'exactly one...some',
    "no(player)(hit(every(shot)))": 'no...all',
    "no(player)(hit(no(shot)))": 'no...none',
    "no(player)(hit(some(shot)))": 'no...some'}

# Map from the experiment to our preferred notation for worlds/conditions:
CONDITION_MAP = {
    "none-none-none": "NNN",
    "none-none-half": "NNS",
    "none-none-all": "NNA",
    "none-half-half": "NSS",
    "none-half-all": "NSA",                         
    "none-all-all": "NAA",
    "half-half-half": "SSS",
    "half-half-all": "SSA",
    "half-all-all": "SAA",
    "all-all-all": "AAA" }

# Separate vector to ensure the desired ordering:
CONDITIONS = ("NNN", "NNS", "NNA", "NAA", "NSS", "NSA", "SSS", "SSA", "SAA", "AAA")

######################################################################
# Utility functions

def rownorm(mat):
    """Row normalization of a matrix"""
    return np.divide(mat.T, np.sum(mat, axis=1)).T
    
def colnorm(mat):
    """Column normalization of a matrix"""    
    return np.divide(mat, np.sum(mat, axis=0))

def safelog(vals):           
    with np.errstate(divide='ignore'):
        return np.log(vals)

def display_matrix(mat, display=True, rnames=None, cnames=None, title='', digits=4):
    """Utility function for displaying strategies to standard output.
    The display parameter saves a lot of conditionals in the important code"""
    if display:
        mat = np.round(mat, digits)
        rowlabelwidth = 2 + max([len(x) for x in rnames+cnames] + [digits+2])
        cwidth = 2 + max([len(x) for x in cnames] + [digits+2])
        # Divider bar of the appropriate width:
        print "-" * (cwidth * (max(len(cnames), len(rnames)) + 1))
        print title
        # Matrix with even-width columns wide enough for the data:
        print ''.rjust(rowlabelwidth) + "".join([str(s).rjust(cwidth) for s in cnames])        
        for i in range(mat.shape[0]):  
            print str(rnames[i]).rjust(rowlabelwidth) + "".join(str(x).rjust(cwidth) for x in mat[i, :])    

def powerset(x, minsize=0, maxsize=None):
    result = []
    if maxsize == None: maxsize = len(x)
    for i in range(minsize, maxsize+1):
        for val in itertools.combinations(x, i):
            result.append(list(val))
    return result

def mse(x, y):
    """Mean squared error"""
    err = np.sqrt(np.sum((x-y)**2)/len(x))
    return err

