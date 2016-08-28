#!/usr/bin/env python

"""
Common strings and some nuts-and-bolts stuff used in the paper.
"""

__author__ = "Christopher Potts"
__version__ = "2.0"
__license__ = "GNU general public license, version 3"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"


import matplotlib.pyplot as plt
import matplotlib

LIKERT_EXPERIMENT_SRC_FILENAME = 'pypragmods/embeddedscalars/embeddedscalars-experiment-results-likert.csv'
BINARY_EXPERIMENT_SRC_FILENAME = 'pypragmods/embeddedscalars/embeddedscalars-experiment-results-binary.csv'

######################################################################
# Model-theoretic entities:

a = 'a'; b = 'b'; c = 'c'
s1 = 's1' ; s2 = 's2'

######################################################################
# Plot set-up

plt.style.use('pypragmods/embeddedscalars/embimp.mplstyle')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='times')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r'\boldmath']

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

