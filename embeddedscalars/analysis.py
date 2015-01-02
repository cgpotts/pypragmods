#!/usr/bin/env python

######################################################################
# Functions for analyzing models against experimental data using the
# techniques described in the paper. For examples, see paper.py
######################################################################

import sys
import csv
from copy import copy
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats import spearmanr, pearsonr
from settings import *
sys.path.append('../')
from utils import *

######################################################################

class Analysis:
    def __init__(self, experiment=None, models=None):
        self.experiment = experiment
        self.models = models
        self.messages = copy(self.models[0].messages)
        self.worlds = self.models[0].states
        self.literal_listener, rsa_spk, self.rsa_listener = self.models[0].rsa()
        self.uncertainty_listeners = [mod.final_listener for mod in self.models]
        self.listeners = [self.literal_listener, self.rsa_listener] + self.uncertainty_listeners
        self.modnames = ['Literal', 'Fixed lexicon'] + [mod.name for mod in self.models]
        if NULL in self.messages:
            self.messages.remove(NULL)
            for i, lis in enumerate(self.listeners):
                self.listeners[i] = lis[: -1]
        self.expmat = np.array(self.experiment.target_means2matrix(self.messages, self.worlds))
        self.rescale_experiment()        

    def rescale_experiment(self):
        self.expmat = rownorm(self.expmat-1.0)
        
    def overall_analysis(self, digits=4):
        expvec = self.expmat.flatten()
        rows = []
        for i, lis in enumerate(self.listeners):
            lisvec = lis.flatten()
            pearson, pearson_p = pearsonr(expvec, lisvec)
            spearman, spearman_p = spearmanr(expvec, lisvec)
            err = mse(expvec, lisvec)            
            rows.append(np.array([pearson, pearson_p, spearman, spearman_p, err]))        
        display_matrix(np.array(rows), rnames=self.modnames, cnames=['Pearson', 'Pearson p', 'Spearman', 'Spearman p', 'MSE'], digits=digits)
	
    def analysis_by_message(self, digits=4):
        rows = []
        msglen = max([len(x) for x in self.messages])
        modlen = max([len(x) for x in self.modnames])
        rnames = [msg.rjust(msglen)+" "+ mod.rjust(modlen) for msg, mod in product(self.messages, self.modnames)]
        for i, msg in enumerate(self.messages):        
            expvec = self.expmat[i]
            for j, lis in enumerate(self.listeners):
                lisvec = lis[i]
                pearson, pearson_p = pearsonr(expvec, lisvec)
                spearman, spearman_p = spearmanr(expvec, lisvec)
                err = mse(expvec, lisvec)            
                rows.append(np.array([pearson, pearson_p, spearman, spearman_p, err]))        
        display_matrix(np.array(rows), rnames=rnames, cnames=['Pearson', 'Pearson p', 'Spearman', 'Spearman p', 'MSE'], digits=digits)

    def comparison_plot(self, width=0.2, output_filename=None):
        # Preferred: human left, then models from best to worse, informally:
        listeners = copy(self.listeners)[::-1]
        modnames = copy(self.modnames)[::-1]
        # Plot dimensions:              
        nrows = len(self.messages)
        ncols = len(self.listeners)+1
        # Basic set-up:
        fig, axarray = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_figheight(nrows*4)
        fig.set_figwidth(ncols*4)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.text(0.5, 0.08, 'Probability', ha='center', va='center', fontsize=30)
        fig.text(0.08, 0.5, 'World', ha='center', va='center', rotation='vertical', fontsize=30)
        # Human column, then model columns:
        self.model_comparison_plot(axarray[:,0], self.expmat, width=width, color=colors[0], modname='Human', left=True, right=False)
        for i, lis in enumerate(listeners):
            self.model_comparison_plot(axarray[: , i+1], lis, width=width, color=colors[-(i+1)], modname=modnames[i], left=False, right=i==ncols-2)
        # Output:
        if output_filename:
            plt.savefig(output_filename, bbox_inches='tight')
        else:
            plt.show()

    def model_comparison_plot(self, axarray, modmat, width=1.0, color='black', modname=None, left=False, right=False):
        # Preferred ordering puts the embedded 'some' sentences last:
        message_ordering_indices = [0,3,6,1,4,7,2,5,8]
        msgs = [self.messages[i] for i in message_ordering_indices]
        titles = [TITLES[msg] for msg in msgs]
        titles = [r"\textbf{\emph{%s}}" % t for t in titles]
        # Sizing:
        title_size = 30
        xtick_labelsize = 16
        ytick_labelsize = 16
        # Orientation:
        pos = np.arange(0.0, len(self.worlds)*width, width)
        ylim = [0.0, len(self.worlds)*width]
        yticks = pos+(width/2.0)
        ytick_labels = [r'\texttt{%s}' % s for s in self.worlds]
        ytick_labels = ytick_labels[::-1] # Reverse for preferred ordering.       
        xlim = [0.0, 1.0]
        xticks = [0.0, 0.25, 0.5, 0.75, 1.0]            
        xtick_labels = ["0", ".25", ".5", ".75", "1"]        
        # Axes:
        modmat = modmat[message_ordering_indices, : ]
        for j, ax in enumerate(axarray):
            msg = msgs[j]            
            row = modmat[j]
            row = row[::-1] # Reversal for preferred ordering.
            ax.tick_params(axis='both', which='both', bottom='off', left='off', top='off', right='off')            
            ax.barh(pos, row, width, color=color)
            # title as model name:
            if j == 0:
                ax.set_title(r"\textbf{%s}" % modname, fontsize=title_size, color=color, fontweight='bold')                
            # x-axis
            ax.set_xlim(xlim)            
            ax.set_xticks(xticks)
            if j == len(axarray)-1:
                ax.set_xticklabels(xtick_labels, fontsize=xtick_labelsize, color='black')
            else:
                ax.set_xticklabels([])
            # y-axis:
            if right:
                ax.yaxis.set_label_position("right")                
                ax.set_ylabel(titles[j], fontsize=title_size, color='black')
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
            if left:
                ax.set_yticklabels(ytick_labels, fontsize=ytick_labelsize, color='black')
            else:
                ax.set_yticklabels([])            

######################################################################    
    
if __name__ == '__main__':

    # For examples, see paper.py
    pass
