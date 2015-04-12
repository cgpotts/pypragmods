#/usr/bin/env python

######################################################################
# Examples from 
#
# Potts, Christopher and Roget Levy. 2015. Negotiating lexical
# uncertainty and speaker expertise with disjunction. Proceedings of
# the 41st Annual Meeting of the Berkeley Linguistics Society.
#
# and
#
# Levy, Roger and Christopher Potts. 2015. Negotiating lexical
# uncertainty and expertise with disjunction. Poster presented at the
# 89th Meeting of the Linguistic Society of America, Portland, OR,
# January 8-11.
#
# ---Christopher Potts
#
######################################################################

import numpy as np
import sys
from collections import defaultdict
import cPickle as pickle
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('../')
from lexica import Lexica, NULL_MSG, DISJUNCTION_SIGN
from pragmods import Pragmod
from utils import display_matrix

plt.style.use('bls41.mplstyle')
COLORS = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666']

######################################################################
##### Illustrative examples
    
w1 = '1'; w2 = '2'; w3 = '3'
p = 'p'; q = 'q'; porq = 'p v q'; pandq = 'p & q'

def simple_disjunction():    
    lexicon = np.array([[1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0]])
    mod = Pragmod(
        lexica=None,
        messages=[p, q, pandq, porq, NULL_MSG],
        states=[w1, w2, w3],
        costs=np.array([0.0, 0.0, 1.0, 1.0, 5.0]),
        prior=np.repeat(1.0/4.0, 3),
        lexprior=None,
        temperature=1.0,
        alpha=1.0,
        beta=0.0)
    mod.run_base_model(lexicon, n=2, display=True, digits=2)

def basic_scalar():
    lexica = [np.array([[1.0,1.0], [1.0, 0.0], [1.0, 1.0]]),
              np.array([[1.0,0.0], [1.0, 0.0], [1.0, 1.0]]),
              np.array([[0.0,1.0], [1.0, 0.0], [1.0, 1.0]])]    
    mod = Pragmod(lexica=lexica,
                  messages=['cheap', 'free', NULL_MSG],
                  states=[w1, w2],
                  costs=np.array([0.0, 0.0, 5.0]),
                  prior=np.repeat(1.0/2.0, 2),
                  lexprior=np.repeat(1.0/3.0, 3),
                  temperature=1.0,
                  alpha=1.0,
                  beta=2.0)
    for lex in lexica:
        print "=" * 70
        print mod.lex2str(lex)
        mod.run_base_model(lex, n=2, display=True, digits=2)      
    mod.run_expertise_model(n=3, display=True, digits=2)
          
def compositional_disjunction():
    lexica = Lexica(
        baselexicon={p: [w1, w2], q: [w1, w3], pandq:[w1]},
        costs={p:0.0, q:0.0, pandq:1.0},
        join_closure=True,
        nullsem=True,
        nullcost=5.0,
        disjunction_cost=1.0)
    lexica.display()
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  states=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=1.0,
                  alpha=1.0,
                  beta=1.0)
    mod.run_expertise_model(n=2, display=True, digits=2)

######################################################################
##### Hurfordian and disjunction examples

def generic_disjunction_example(
        alpha=1.0,
        beta=1.0,
        disjunction_cost=1.0,
        n=2,
        fulldisplay=False,
        unknown_word=None):
    """Common code for our two illustrative examples, which
    differ only in the above keyword parameters. Increase n to see
    greater depths of recursion. use fulldisplay=True to see more
    details."""    
    # Use the lexicon generation convenience function to
    # generate all the join-closure lexica and calculate
    # the necessary message costs:
    lexica = Lexica(
        baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']}, 
        costs={'A':0.0, 'B':0.0, 'X':0.0},
        join_closure=True,
        nullsem=True,
        nullcost=5.0,
        disjunction_cost=disjunction_cost,
        unknown_word=unknown_word)
    # Lexical matrices:
    lexmats = lexica.lexica2matrices()         
    # Pragmatic models for the above lexical space.
    mod = Pragmod(
        lexica=lexmats,
        messages=lexica.messages,
        states=lexica.states,
        costs=lexica.cost_vector(),
        prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
        lexprior=np.repeat(1.0/len(lexmats), len(lexmats)),
        temperature=1.0,
        alpha=alpha,
        beta=beta)
    if fulldisplay:
        lexica.display()
        # Run the base model on the individual lexica so we can show those lower steps:
        for lex in lexmats:
            print "=" * 70
            print mod.lex2str(lex)
            mod.run_base_model(lex, n=2, display=True, digits=2)         
    ## Run the anxious experts model - fulldisplay=True for a fuller picture:
    langs = mod.run_expertise_model(n=n, display=fulldisplay, digits=2)
    # Look at the specific table we care about:
    msg_index = mod.messages.index('A v X')
    final_listener = langs[-1]
    mod.display_joint_listener(final_listener[msg_index], digits=2)
    return langs
        
def hurfordian_example(n=2, fulldisplay=False):
    generic_disjunction_example(alpha=2.0, beta=1.0, disjunction_cost=1.0, n=n, fulldisplay=fulldisplay)
        
def definitional_example(n=2, fulldisplay=False):
    generic_disjunction_example(alpha=5.0, beta=7.0, disjunction_cost=0.01, n=n, fulldisplay=fulldisplay)

def focal_definitional_example(n=2, fulldisplay=False):
    generic_disjunction_example(alpha=5.0, beta=7.0, disjunction_cost=0.01, n=n, fulldisplay=fulldisplay, unknown_word='X')

######################################################################

def Q_implicature_simulation(output_filename="Q-implicature-simulation"):
    ##### General set-up:
    # Messages:
    GENERAL_MSG = 'general'
    SPECIFIC_MSG = 'specific'
    DISJ_MSG = GENERAL_MSG + DISJUNCTION_SIGN + SPECIFIC_MSG
    # States:
    GENERAL_ONLY_REF = r'w_{\textsc{general-only}}'
    SPECIFIC_REF = r'w_{\textsc{specific}}'
    DISJ_REF = r'%s v %s' % (GENERAL_ONLY_REF, SPECIFIC_REF)
    # Common structures:
    BASELEXICON = {GENERAL_MSG: [GENERAL_ONLY_REF, SPECIFIC_REF], SPECIFIC_MSG: [SPECIFIC_REF]}
       
    ##### General function for getting data points:
    def Q_implicature_simulation_datapoint(specific_cost, dcost=1.0, alpha=2.0):
        # Values to obtain:
        is_max = False
        listener_val = None
        speaker_val = None
        # Set-up:
        lexica = Lexica(baselexicon=BASELEXICON, costs={GENERAL_MSG: 0.0, SPECIFIC_MSG: specific_cost}, join_closure=True, nullsem=True, nullcost=5.0, disjunction_cost=dcost)
        ref_probs = np.repeat(1.0/len(lexica.states), len(lexica.states))
        lexprior = np.repeat(1.0/len(lexica.lexica2matrices()), len(lexica.lexica2matrices()))
        # Run the model:
        mod = Pragmod(lexica=lexica.lexica2matrices(), messages=lexica.messages, states=lexica.states, costs=lexica.cost_vector(), lexprior=lexprior, prior=ref_probs, alpha=alpha)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        # Get the values we need:
        speaker = mod.speaker_lexical_marginalization(langs[-2])
        listener = mod.listener_lexical_marginalization(langs[-3])
        general_msg_index = lexica.messages.index(GENERAL_MSG)
        general_only_state = lexica.states.index(GENERAL_ONLY_REF)
        disj_state_index = lexica.states.index(DISJ_REF)
        disj_msg_index = lexica.messages.index(DISJ_MSG)
        speaker_val = speaker[disj_state_index, disj_msg_index]
        listener_val = listener[general_msg_index, general_only_state]
        # Determine whether max, with a bit of rounding to avoid spurious mismatch diagnosis:
        maxspkval = np.max(speaker[disj_state_index])
        is_max = np.round(speaker_val, 10) == np.round(maxspkval, 10)
        # Return values:
        return (listener_val, speaker_val, is_max)

    ##### Plot creation:
    matplotlib.rc('font', family='serif', serif='times') # Not sure why this has to be set to get the legend font to change.
    # Values to vary:
    specific_costs = [0.0,1.0,2.0,3.0,4.0]
    disjunction_costs = np.arange(0.0, 5.0, 1)
    alphas = np.array([1.0, 2.0, 3.0, 4.0])    
    # Panels:
    variable_lookup = {r'C(\textit{or})': disjunction_costs, r'\alpha': alphas}
    variable_filename_suffixes = ['alphas', 'or']
    ylims = {r'C(\textit{or})':  [-0.05, 0.45], r'\alpha': [0.15, 0.75]}    
    for variable_name, suffix in zip(variable_lookup, variable_filename_suffixes):
        # Figure set-up:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(7)
        fig.set_figwidth(8)        
        variables = variable_lookup[variable_name]
        ann_index = 0
        ann_adj = 1.01
        ha = 'left'
        va = 'top'
        if variable_name == r'\alpha':
            ha = 'right'
            va = 'center'
            ann_index = -1
            ann_adj = 0.98
        for color, variable in zip(COLORS, variables):
            dcost = variable
            alpha = 2.0            
            if variable_name == r'\alpha':                
                dcost = 1.0
                alpha = variable
            vals = []
            for cost in specific_costs:
                vals.append(Q_implicature_simulation_datapoint(cost, dcost=dcost, alpha=alpha))
            listener_vals, speaker_vals, _ = zip(*vals)                
            max_booleans = [(i, j) for i, j, is_max in vals if is_max]            
            # Plotting (multiple lines with max-value annotations)
            ax.plot(listener_vals, speaker_vals, color=color, linewidth=2)
            if max_booleans:
                maxx, maxy = zip(*max_booleans)
                ax.plot(maxx, maxy, linestyle=':', linewidth=6, color=color)
            ax.annotate(r'$%s = %s$' % (variable_name, variable), xy=(listener_vals[ann_index]*ann_adj, speaker_vals[ann_index]), fontsize=16, ha=ha, va=va, color=color)
        # Axes:
        ax.set_xlabel(r'$L_1(%s \mid \textit{%s})$' % (GENERAL_ONLY_REF, GENERAL_MSG), fontsize=18)
        ax.set_ylabel(r'$S_2(\textit{%s} \mid %s)$' % (DISJ_MSG.replace(' v ', r' or '), DISJ_REF.replace(' v ', r' \vee ')), fontsize=18)        
        ax.set_xlim([0.2, 1.05])   
        ax.set_ylim([0.0, 1.05])
        # Save the panel:
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.savefig("%s-%s.pdf" % (output_filename, suffix), bbox_inches='tight')
    
######################################################################

def I_implicature_simulation(output_filename="I-implicature-simulation"):
    ##### General set-up:
    # Messages:
    SUPERKIND_MSG = r'general'
    COMMON_MSG = r'unmarked\_specific'
    UNCOMMON_MSG = r'marked\_specific'
    DISJ_MSG = "%s%s%s" % (SUPERKIND_MSG, DISJUNCTION_SIGN, UNCOMMON_MSG)
    # Referents:
    COMMON_REF = r'r_{\textsc{COMMON}}'
    UNCOMMON_REF = r'r_{\textsc{UNCOMMON}}'
    DISJ_REF = "%s%s%s" % (COMMON_REF, DISJUNCTION_SIGN, UNCOMMON_REF)
    # Common structures:
    BASELEXICON = {SUPERKIND_MSG: [UNCOMMON_REF, COMMON_REF], COMMON_MSG: [COMMON_REF], UNCOMMON_MSG: [UNCOMMON_REF]}
    LEXICAL_COSTS = {SUPERKIND_MSG: 0.0, COMMON_MSG: 0.0, UNCOMMON_MSG: 0.0}
   
    ##### General function for getting data points:
    def I_implicature_simulation_datapoint(common_ref_prob, dcost=1.0, alpha=2.0):
        # Values to obtain:
        is_max = False
        listener_val = None
        speaker_val = None
        # Set-up:
        lexica = Lexica(baselexicon=BASELEXICON, costs=LEXICAL_COSTS, join_closure=True, nullsem=True, nullcost=5.0, disjunction_cost=dcost)
        ref_probs = np.array([common_ref_prob, (1.0-common_ref_prob)/2.0, (1.0-common_ref_prob)/2.0])
        lexprior = np.repeat(1.0/len(lexica.lexica2matrices()), len(lexica.lexica2matrices()))
        # Run the model:
        mod = Pragmod(lexica=lexica.lexica2matrices(), messages=lexica.messages, states=lexica.states, costs=lexica.cost_vector(), lexprior=lexprior, prior=ref_probs, alpha=alpha)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        # Get the values we need:
        speaker = mod.speaker_lexical_marginalization(langs[-2])
        listener = mod.listener_lexical_marginalization(langs[-3])
        superkind_term_index = mod.messages.index(SUPERKIND_MSG)
        common_state_index = mod.states.index(COMMON_REF)
        disj_term_index = mod.messages.index(DISJ_MSG)
        disj_state_index = mod.states.index(DISJ_REF)
        # Fill in listener_val and speaker_val:
        listener_val = listener[superkind_term_index, common_state_index]
        speaker_val = speaker[disj_state_index, disj_term_index]
        # Determine whether max, with a bit of rounding to avoid spurious mismatch diagnosis:
        maxspkval = np.max(speaker[disj_state_index])
        is_max = np.round(speaker_val, 10) == np.round(maxspkval, 10)
        # Return values:
        return (listener_val, speaker_val, is_max)

    ##### Plot creation:
    matplotlib.rc('font', family='serif', serif='times') # Not sure why this has to be set to get the legend font to change.
    # Values to vary:
    common_ref_probs = np.arange(1.0/3.0, 1.0/1.0, 0.01)
    disjunction_costs = np.arange(1.0, 5.0, 1)
    alphas = np.array([1.06, 2.0, 3.0, 4.0])    
    # Panels:
    variable_lookup = {r'C(\textit{or})': disjunction_costs, r'\alpha': alphas}
    variable_filename_suffixes = ['alphas', 'or']
    ylims = {r'C(\textit{or})':  [-0.05, 0.45], r'\alpha': [0.15, 0.75]}
    for variable_name, suffix in zip(variable_lookup, variable_filename_suffixes):
        # Figure set-up:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(7)
        fig.set_figwidth(8)
        variables = variable_lookup[variable_name]
        ha = 'left'
        va = 'top'
        if variable_name == r'\alpha':
            ha = 'right'
            va = 'top'
        for color, variable in zip(COLORS, variables):
            dcost = variable
            alpha = 2.0            
            if variable_name == r'\alpha':                
                dcost = 1.0
                alpha = variable
            vals = []
            for ref_prob in common_ref_probs:
                vals.append(I_implicature_simulation_datapoint(ref_prob, dcost=dcost, alpha=alpha))
            listener_vals, speaker_vals, _ = zip(*vals)                
            max_booleans = [(i, j) for i, j, is_max in vals if is_max]            
            # Plotting (multiple lines with max-value annotations)
            ax.plot(listener_vals, speaker_vals, color=color, linewidth=2)
            if max_booleans:
                maxx, maxy = zip(*max_booleans)
                ax.plot(maxx, maxy, linestyle=':', linewidth=6, color=color)
            # Annotation:
            if variable_name == r'\alpha' and variable == variables[-1]: # Avoid label overlap for alpha=3 and alpha=4.
                va = 'bottom'
            ax.annotate(r'$%s = %s$' % (variable_name, variable), xy=(listener_vals[0]*0.98, speaker_vals[0]), fontsize=16, ha=ha, va=va, color=color)
            ax.set_ylim(ylims[variable_name])
        # Axes:
        ax.set_xlabel(r'$L_1(%s \mid \textit{%s})$' % (COMMON_REF, SUPERKIND_MSG), fontsize=18)
        ax.set_ylabel(r'$S_2(\textit{%s} \mid %s)$' % (DISJ_MSG.replace(' v ', r' or '), DISJ_REF.replace(' v ', r' \vee ')), fontsize=18)
        ax.set_xlim([0.0,1.0])
        # Save the panel:
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.savefig("%s-%s.pdf" % (output_filename, suffix), bbox_inches='tight')
    
######################################################################
##### Parameter exploration

class ListenerParameterExperiment:
    def __init__(self,
            baselexicon={'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},
            lexical_costs={'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0},
            unknown_word=None,
            dcosts=np.arange(0.0, 0.21, 0.01),
            alphas=np.arange(0.0, 15.0, 1),
            betas=np.arange(0.0, 15.0, 1),
            depths=[10],
            results_filename=None,
            results=None,
            plot_filename=None,
            logx=True,
            xlim=(-3.0, 3.0)):
        # Parameter exploration parameters:
        self.results = results                   # Optionally read in existing results for plotting.
        self.results_filename = results_filename # Output filename for the pickled results.
        self.baselexicon = baselexicon           # The basic lexicon to explore.
        self.lexical_costs = lexical_costs       # Costs for the basic lexicon.
        self.unknown_word = unknown_word         # If present, then it is presumed by Lexica to be constrained to an atomic meaning.
        self.dcosts = dcosts                     # Vector of disjunction costs to explore.
        self.alphas = alphas                     # Vector of alpahs to explore.
        self.betas = betas                       # Vector of betas to explore.
        self.depths = depths                     # Vector of integer depths to explore.
        # Plotting parameters:
        self.plot_filename = plot_filename       # Optional output filename for the plot; if None, then plt.show().
        self.logx = logx                         # Should the beta/alpha be put on the log scale (default: True)
        self.xlim = xlim                         # The current default is good for the current settings.
        # Parameters not exposed because assumed
        # fixed; these are here mainly to avoid
        # mistakes and facilitate future work:
        self.msg = 'A v X'
        self.left_disjunct = 'A'
        self.right_disjunct = 'X'
        self.hurford_state = '1 v 2'
        self.definitional_state = '1'
        self.hurford_cls = 'H'
        self.definitional_cls = 'D'
        self.other_cls = 'O'

    def run(self):
        """The main method: parameter exploration if it's not already done, and then the plot"""
        # If the results are not already computed, get them:
        if not self.results:
            self.results = self.explore_listener_parameters()
            pickle.dump(self.results, file(self.results_filename, 'w'), 2)
        # Classify the parameters into Hurfordian and definitional
        hc = self._get_cls_params(self.hurford_cls, self.hurford_state)
        defin = self._get_cls_params(self.definitional_cls, self.definitional_state)
        # Create the plot:
        self.alpha_beta_cost_scatterplot(hc, defin)
    
    def explore_listener_parameters(self):
        """Explore a large parameter space, classifying the parameter vectors
        based on the max listener <world,lex> inference given self.msg""" 
        results = defaultdict(list)
        for dcost, alpha, beta, depth in product(self.dcosts, self.alphas, self.betas, self.depths):
            params = {'alpha': alpha, 'beta': beta, 'depth': depth, 'disjunction_cost': dcost}
            lexica = Lexica(
                baselexicon=self.baselexicon, 
                costs=self.lexical_costs,
                join_closure=True,
                nullsem=True,
                nullcost=5.0,
                disjunction_cost=dcost,
                unknown_word=self.unknown_word)
            lexmats = lexica.lexica2matrices()
            mod = Pragmod(
                lexica=lexmats,
                messages=lexica.messages,
                states=lexica.states,
                costs=lexica.cost_vector(),
                prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                lexprior=np.repeat(1.0/len(lexmats), len(lexmats)),
                temperature=1.0,
                alpha=alpha,
                beta=beta)
            # Run the model:
            langs = mod.run_expertise_model(n=depth, display=False)
            # Get the listener's joint probability table for this message:
            msg_index = mod.messages.index(self.msg)
            prob_table = langs[-1][msg_index]        
            sorted_probs = sorted(prob_table.flatten())
            max_pair = None
            max_prob = sorted_probs[-1]
            # No ties allowed!
            if max_prob != sorted_probs[-2]:
                for i, j in product(range(prob_table.shape[0]), range(prob_table.shape[1])):
                    if prob_table[i, j] == max_prob:
                        max_pair = (i, mod.states[j])
            # Add the target probability:
            params['prob'] = max_prob
            # Print to show progress:
            print max_pair, params
            # Store this dictionary of results -- parameters plus the predicted probability
            # max_pair is a lexicon index and state name.
            results[max_pair].append(params)
        return results
    
    def _classify_lexica(self):
        classified = []
        lexica = Lexica(baselexicon=self.baselexicon, join_closure=True, nullsem=True)
        for i, lex in enumerate(lexica.lexica2matrices()):
            a_index = lexica.messages.index(self.left_disjunct)
            a_sem = set([s for j, s in enumerate(lexica.states) if lex[a_index][j] > 0.0])
            x_index = lexica.messages.index(self.right_disjunct)
            x_sem = set([s for j, s in enumerate(lexica.states) if lex[x_index][j] > 0.0])
            cls = self.other_cls
            if len(a_sem & x_sem) == 0:
                cls = self.hurford_cls            
            elif a_sem == x_sem:
                cls = self.definitional_cls
            classified.append((i, cls))
        return classified                        

    def _get_cls_params(self, cls, state):
        cls_lex = []
        for lexindex, c in self._classify_lexica():
            if c == cls:
                cls_lex += self.results[(lexindex, state)]
        return cls_lex

    def alpha_beta_cost_scatterplot(self, hc, defin):
        """Create the scatterplot with beta/alpha on the x-axis and disjunction costs on the y-axis"""
        # Set-up:
        matplotlib.rc('font', family='serif', serif='times') # Not sure why this has to be set to get the legend font to change.
        transform = np.log if self.logx else (lambda x : x)    
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        labsize = 24
        markersize = 15
        ticksize = 18
        colors = ['#1B9E77', '#E6AB02', '#000000'] # '#7570B3']
        # Hurfordian:
        h_ba = [r['beta']/r['alpha'] for r in hc]
        h_g = [r['disjunction_cost'] for r in hc]
        # Definitional:
        d_ba = [r['beta']/r['alpha'] for r in defin]
        d_g = [r['disjunction_cost'] for r in defin]        
        # Both (if any):
        both = [(x,y) for x, y in zip(d_ba, d_g) if (x,y) in zip(h_ba, h_g)]
        b_ba = []; b_g = []
        if both:
            b_ba, b_g = zip(*both)
        # Remove overlap:
        h_ba, h_g = zip(*[(x,y) for x,y in zip(h_ba, h_g) if (x,y) not in both])
        d_ba, d_g = zip(*[(x,y) for x,y in zip(d_ba, d_g) if (x,y) not in both])                        
        # Plotting:
        for i, vals in enumerate(((h_ba, h_g, 'Hurfordian'), (d_ba, d_g, 'Definitional'), (b_ba, b_g, 'Both'))):
            x, y, label = vals
            ax.plot(self.jitter(transform(x)), self.jitter(y), linestyle="", marker=".", label=label, markersize=markersize, color=colors[i])
        # Labeling:
        xlab = r"\beta/\alpha"    
        if self.logx:
            xlab = r"\log(%s)" % xlab
        ax.set_xlabel(r'$%s$' % xlab, fontsize=labsize, color='black')
        ax.set_ylabel(r'$C(or)$', fontsize=labsize, color='black')
        ax.legend(loc='upper left', bbox_to_anchor=(0,1.1), ncol=3, fontsize=labsize)
        # Ticks and bounds:
        if self.xlim:
            x1,x2,y1,y2 = ax.axis()
            x1, x2 = self.xlim
            ax.axis((x1, x2, y1, y2))
        plt.setp(ax.get_xticklabels(), fontsize=ticksize)
        plt.setp(ax.get_yticklabels(), fontsize=ticksize)
        if self.plot_filename:
            plt.savefig(self.plot_filename, bbox_inches='tight')
        else:
            plt.show()

    def jitter(self, x):
        """Jitter while respecting the data bounds"""
        mu = 0.0
        sd = 0.001
        j = x + np.random.normal(mu, sd, len(x))
        j = np.maximum(j, np.min(x))
        j = np.minimum(j, np.max(x))
        return j

######################################################################
        
if __name__ == '__main__':

    ## Some simple warm-up examples:
    simple_disjunction()
    # basic_scalar()
    # compositional_disjunction()

    ## From the poster:
    # hurfordian_example(n=2, fulldisplay=False)
    # definitional_example(n=3, fulldisplay=True)
    # focal_definitional_example(n=2, fulldisplay=False)

    ## Implicature blocking
    # Q_implicature_simulation()
    # I_implicature_simulation()        
    
    # Parameter exploration with a large lexicon; this takes a long time to run!
    # ListenerParameterExperiment(
    #     results_filename='paramexplore-lex5.pickle',
    #     # If the results are already computed:
    #     results=pickle.load(file("paramexplore-lex5.pickle")),
    #     plot_filename='paramexplore-lex5.pdf').run()

    # # Parameter exploration as above but constraining the unknown word X to an atomic meaning:
    # ListenerParameterExperiment(
    #     results_filename='paramexplore-lex5-focal.pickle',
    #     # If the results are already computed:
    #     results=pickle.load(file("paramexplore-lex5-focal.pickle")),
    #     plot_filename='paramexplore-lex5-focal.pdf',
    #     unknown_word='X').run()

