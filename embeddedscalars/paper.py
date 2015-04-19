#!/usr/bin/env python

######################################################################
# Functions for all of the figures and tables in the paper. The
# functions themselves are called as part of the main method, but
# only simple_scalar_inference_example is currently commented in.
######################################################################

import csv
import sys
sys.path.append('../')
import numpy as np
from copy import copy
from itertools import product
from collections import defaultdict
from operator import itemgetter
from grammar import UncertaintyGrammars
from pragmods import Pragmod
from settings import BINARY_EXPERIMENT_SRC_FILENAME, LIKERT_EXPERIMENT_SRC_FILENAME
from settings import a, b, c, s1, s2 # model-theoretic entities
from fragment import *
from experiment import Experiment
from analysis import Analysis
from utils import *

######################################################################
# Figure 2

def simple_scalar_inference_example():

    def lexicon_iterator():
        mats = [
            np.array([[0., 1., 1.], [0., 0., 1.], [1., 1., 1.]]),
            np.array([[0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
            np.array([[0., 0., 1.], [0., 0., 1.], [1., 1., 1.]])]
        for mat in mats:
            yield mat

    mod = Pragmod(
        lexica=lexicon_iterator,
        messages=['A scored', 'A aced', 'NULL'],
        states=['N', 'S', 'A'],
        temperature=1.0,
        nullmsg=True,
        nullcost=5.0)

    mod.stream_lexical_uncertainty(n=0)

    for lex in lexicon_iterator():
        print "=" * 70
        print "Messages"
        display_matrix(lex, rnames=mod.messages, cnames=mod.states, digits=2)
        print 'l0'
        display_matrix(mod.l0(lex), rnames=mod.messages, cnames=mod.states, digits=2)
        print 's1'
        display_matrix(mod.s1(lex), rnames=mod.states, cnames=mod.messages, digits=2)
        print 'l1'
        display_matrix(mod.L(mod.S(lex)), rnames=mod.messages, cnames=mod.states, digits=2)        

    display_matrix(mod.final_listener,  rnames=mod.messages, cnames=mod.states, digits=2)

######################################################################
# Table 2: Nested scalar terms (from Sauerland's work):

def scalar_disjunction_example(refinable={}):
    players = [a]
    shots = ['1', '2', 'f']
    regular_shots = ['1', '2']    
    worlds = [("-",), ('f',), ('1',), ('2',), ('f1',), ('f2',), ('12',), ('f12',)]    
    lexicon = {
        "player": players,
        "PlayerA":  [[a]],
        "shot1": [['1']],
        "shot2": [['2']],
        "the_freethrow": [X for X in powerset(shots) if 'f' in X],
        "hit":    [[w, x, y] for w, x, y in product(worlds, players, shots) if y in set(w[players.index(x)])],
        "some_shot": [Y for Y in powerset(shots) if len(set(regular_shots) & set(Y)) > 0],
        "only_some_shot": [Y for Y in powerset(shots) if len(set(regular_shots) & set(Y)) == 1],
        "every_shot": [Y for Y in powerset(shots) if set(regular_shots) <= set(Y)],
        "not_every_shot": [Y for Y in powerset(set(shots)) if not(set(regular_shots) <= set(Y))],
        "AND": [[X, Y, [z for z in X if z in Y]] for X, Y in product(list(powerset(powerset(shots))), repeat=2)],
        "OR": [[X, Y, [z for z in X]+[z for z in Y if z not in X]] for X, Y in product(list(powerset(powerset(shots))), repeat=2)],
        "XOR": [[X, Y, [z for z in X if z not in Y]+[z for z in Y if z not in X]] for X, Y in product(list(powerset(powerset(shots))), repeat=2)]}

    # Import the new lexicon into the namespace:
    for word, sem in lexicon.items():
        setattr(sys.modules[__name__], word, sem)
    new_worldnames = (lambda x : "".join(map(str,x)))
    worldnames = [new_worldnames(w) for w in worlds]

    # Messages:
    messages = [
        ("PlayerA hit the freethrow", "iv(PlayerA, tv(hit, the_freethrow, self.worlds, player))"),
        ("PlayerA hit every shot", "iv(PlayerA, tv(hit, every_shot, self.worlds, player))"),
        ("PlayerA hit some shot", "iv(PlayerA, tv(hit, some_shot, self.worlds, player))"),
        ("PlayerA hit some shot or the freethrow", "iv(PlayerA, tv(hit, coord(OR, some_shot, the_freethrow), self.worlds, player))"),
        ("PlayerA hit some shot and the freethrow", "iv(PlayerA, tv(hit, coord(AND, some_shot, the_freethrow), self.worlds, player))"),
        ("PlayerA hit every shot or the freethrow", "iv(PlayerA, tv(hit, coord(OR, every_shot, the_freethrow), self.worlds, player))"),
        ("PlayerA hit every shot and freethrow", "iv(PlayerA, tv(hit, coord(AND, every_shot, the_freethrow), self.worlds, player))")]
    temperature = 1.0
    nullmsg = True
    nullcost = 5.0

    # Refinement model:
    gram = UncertaintyGrammars(
        baselexicon=copy(lexicon),
        messages=copy(messages),
        worlds=copy(worlds),        
        refinable=refinable,
        nullmsg=nullcost)
    mod = Pragmod(
        lexica=gram.lexicon_iterator,
        baselexicon=gram.baselexicon_mat,
        messages=gram.messages,
        states=worldnames,
        temperature=temperature,
        nullmsg=nullmsg,
        nullcost=nullcost)

    # Model run:
    mod.stream_lexical_uncertainty(n=0)
    display_matrix(mod.final_listener,  rnames=mod.messages, cnames=mod.states, digits=2, latex=True)
    
######################################################################
# Tables 3 and 4

def complex_example():
    # General settings:
    players = [a,b]
    shots = [s1,s2]
    worlds = get_worlds(basic_states=(0,1,2), length=2, increasing=False)
    baselexicon = define_lexicon(player=players, shot=shots, worlds=worlds)   
    subjs = ('PlayerA', 'PlayerB', 'some_player', 'every_player', 'no_player')
    preds = ('scored', 'aced')
    temperature = 1.0
    nullmsg = True
    nullcost = 5.0 
    worldnames = [worldname(w) for w in worlds]
    messages = []
    for subj, pred in product(subjs, preds):
        msg = "%s %s" % (subj.replace("_", " "), pred)        
        formula = "iv(%s, %s)" % (subj, pred)
        messages.append((msg, formula))

    # Neo-Gricean refinement:
    neogram = UncertaintyGrammars(
        baselexicon=copy(baselexicon),
        messages=copy(messages),
        worlds=worlds,
        refinable={'some_player': ['exactly_one_player'], 'PlayerA': ['only_PlayerA'], 'PlayerB': ['only_PlayerB'], 'scored': ['scored_not_aced'], 'aced': []},
        nullmsg=nullmsg)
    neomod = Pragmod(
        lexica=neogram.lexicon_iterator,
        baselexicon=neogram.baselexicon_mat,
        messages=neogram.messages,
        states=worldnames,
        temperature=temperature,
        nullmsg=nullmsg,
        nullcost=nullcost)
    
    # Run and report:
    neomod.stream_lexical_uncertainty(n=0)
    neomod.listener_report(digits=2)

    # Unconstrained refinement:
    ucgram = UncertaintyGrammars(
        baselexicon=copy(baselexicon),
        messages=copy(messages),
        worlds=worlds,
        refinable={'some_player': [], 'PlayerA': [], 'PlayerB': [], 'scored': [], 'aced': []},
        nullmsg=nullmsg)
    ucmod = Pragmod(
        lexica=ucgram.lexicon_iterator,
        baselexicon=ucgram.baselexicon_mat,
        messages=ucgram.messages,
        states=worldnames,
        temperature=temperature,
        nullmsg=nullmsg,
        nullcost=nullcost)

    # Run and report:
    ucmod.stream_lexical_uncertainty(n=0)
    ucmod.listener_report(digits=2)  

######################################################################
# An example a reviewer asked about:

def embedded_disjunction_example(refinable={}):
    players = [a,b]
    shots = [s1,s2]
    worlds = get_worlds(basic_states=(0,1,2,3), length=2, increasing=False)
    baselexicon = define_lexicon(player=players, shot=shots, worlds=worlds)
    subjs = ('every_player', )
    preds = ('hit_shot1', 'hit_shot2', 'hit_shot1_or_shot2', 'hit_shot1_and_shot2')
    temperature = 1.0
    nullmsg = True
    nullcost = 5.0
    # N: no shots; 1: made just shot1; 2: made just shot2; B: made both shot1 and shot2.
    new_worldnames = (lambda w : "".join(["N12B"[i] for i in w]))
    worldnames = [new_worldnames(w) for w in worlds]
    messages = []
    for subj, pred in product(subjs, preds):
        msg = "%s %s" % (subj.replace("_", " "), pred)        
        formula = "iv(%s, %s)" % (subj, pred)
        messages.append((msg, formula))
    # Refinement model:
    gram = UncertaintyGrammars(
        baselexicon=copy(baselexicon),
        messages=copy(messages),
        worlds=copy(worlds),        
        refinable=refinable,
        nullmsg=nullcost)
    mod = Pragmod(
        lexica=gram.lexicon_iterator,
        baselexicon=gram.baselexicon_mat,
        messages=gram.messages,
        states=worldnames,
        temperature=temperature,
        nullmsg=nullmsg,
        nullcost=nullcost)
    mod.stream_lexical_uncertainty(n=0)
    display_matrix(mod.final_listener,  rnames=mod.messages, cnames=mod.states, digits=2, latex=True)
           
######################################################################
# Figure 4 and figure 7
            
def experiment_plot_and_report_binary():    
    experiment_plot_and_report(        
        src_filename=BINARY_EXPERIMENT_SRC_FILENAME,
        output_filename=BINARY_EXPERIMENT_SRC_FILENAME.replace('.csv', '.pdf'),
        response_transformation=(lambda x : 1.0 if x=='T' else 0.0),
        plot_keywordargs={'xlim':[0.0,1.0], 'xlabel':'Percentage True responses', 'xticks':np.arange(0.2, 1.2, .2)})
    
def experiment_plot_and_report_likert():    
    experiment_plot_and_report(        
        src_filename=LIKERT_EXPERIMENT_SRC_FILENAME,
        output_filename=LIKERT_EXPERIMENT_SRC_FILENAME.replace('.csv', '.pdf'),
        response_transformation=(lambda x : int(x)),
        plot_keywordargs={'xlabel':'Mean Likert response'})

def experiment_plot_and_report(
        src_filename=None,
        output_filename=None,
        response_transformation=None,
        plot_keywordargs={}):
    exp = Experiment(src_filename=src_filename, response_transformation=response_transformation)
    exp.experimental_report()
    exp.plot_targets(output_filename=output_filename, **plot_keywordargs)
    # Selected pairwise tests:
    cmps = {
        'Every player hit some of his shots': [('SSS', 'SSA'), ('SSS', 'SAA'), ('SSA', 'AAA'), ('SAA', 'AAA')],                                                   
        'Exactly one player hit some of his shots': [('NSA', 'SSA'), ('SAA', 'SSA')],                                                         
        'No player hit some of his shots': list(product(('NNS', 'NSA'), ('AAA', 'NNA', 'NAA'))) + [(('NNS','NSA'), ('AAA','NNA','NAA'))]}
    for sent, pairs in cmps.items():
        print sent
        for w1, w2 in pairs:            
            coef, p = exp.pairwise_comparison_test(sent, w1, w2)
            print "\t%s, %s: W = %s; p = %s" % (w1, w2, np.round(coef, 2), np.round(p, 4))
                                                    
######################################################################
# Assessment

# Figure 5 and tables 5 and 6
def experimental_assessment_binary():
    experimental_assessment(
        experiment_src=BINARY_EXPERIMENT_SRC_FILENAME,
        plot_output_filename="allmodels-binary.pdf",
        response_transformation=(lambda x : 1.0 if x=='T' else 0.0),
        rescaler=0.0)

# Table 8 (but does much more if run):
def experimental_assessment_likert():
    experimental_assessment(
        experiment_src=LIKERT_EXPERIMENT_SRC_FILENAME,
        plot_output_filename="allmodels-likert.pdf",
        response_transformation=(lambda x : int(x)),
        rescaler=1.0)

# Figure 6
def experimental_assessment_binary_critical_optimal_params():
    experimental_assessment(
        experiment_src=BINARY_EXPERIMENT_SRC_FILENAME,
        plot_output_filename='allmodels-paramexplore-binary.pdf',
        response_transformation=(lambda x : 1.0 if x=='T' else 0.0),
        rescaler=0.0,
        uctemp=0.1,
        ucnullcost=1.0,
        ngtemp=0.1,
        ngnullcost=1.0,
        nrows=3)        

# General function for the above:
def experimental_assessment(experiment_src=None,
                            plot_output_filename=None,
                            response_transformation=None,
                            rescaler=None,
                            uctemp=1.0,
                            ucnullcost=5.0,
                            ngtemp=1.0,
                            ngnullcost=5.0,
                            nrows=None):                            
    # General settings:
    subjs= ('every_player', 'exactly_one_player', 'no_player')
    objs = ('every_shot', 'no_shot', 'some_shot')
    players = [a,b,c]
    shots = [s1,s2]
    basic_states = (0,1,2)
    worlds = get_worlds(basic_states=(0,1,2), length=3, increasing=True)
    baselexicon = define_lexicon(player=players, shot=shots, worlds=worlds)
    temperature = 1.0
    nullmsg = True
    nullcost = 5.0       
    worldnames = [worldname(w) for w in worlds]
    messages = []
    for d1, d2 in product(subjs, objs):
        subj = d1.replace("_player", "(player)")
        obj = d2.replace("_shot", "(shot)")
        msg = "%s(hit(%s))" % (subj, obj)
        formula = "iv(%s, tv(hit, %s, self.worlds, player))" % (d1,  d2)       
        messages.append((msg, formula))    
     
    # Unconstrained refinement:
    ucgram = UncertaintyGrammars(
        baselexicon=copy(baselexicon),
        messages=copy(messages),
        worlds=copy(worlds),        
        refinable={'some_player': [], 'some_shot': []},
        nullmsg=nullmsg)    
    ucmod = Pragmod(
        name="Unconstrained",
        lexica=ucgram.lexicon_iterator,
        baselexicon=ucgram.baselexicon_mat,
        messages=ucgram.messages,
        states=worldnames,
        temperature=uctemp,
        nullmsg=nullmsg,
        nullcost=ucnullcost)

    # Neo-Gricean refinement:
    neogram = UncertaintyGrammars(
        baselexicon=copy(baselexicon),
        messages=copy(messages),
        worlds=copy(worlds),        
        refinable={'some_player': ['exactly_one_player'], 'some_shot': ['exactly_one_shot']},
        nullmsg=nullmsg)    
    neomod = Pragmod(
        name="Neo-Gricean",
        lexica=neogram.lexicon_iterator,
        baselexicon=neogram.baselexicon_mat,
        messages=neogram.messages,
        states=worldnames,
        temperature=ngtemp,
        nullmsg=nullmsg,
        nullcost=ngnullcost)
        
    # Run the models, going only to the first uncertainty listener (n=0):
    ucmod.stream_lexical_uncertainty(n=0)
    neomod.stream_lexical_uncertainty(n=0)
            
    # The analysis:
    analysis = Analysis(experiment=Experiment(src_filename=experiment_src, response_transformation=response_transformation), models=[ucmod, neomod], rescaler=rescaler)
    analysis.overall_analysis()
    analysis.analysis_by_message()
    analysis.comparison_plot(output_filename=plot_output_filename, nrows=nrows)
    
######################################################################
# Table 7
# Grid search over parameter space in response to a reviewer request.

def parameter_exploration_binary():
    parameter_exploration(
        experiment_src=BINARY_EXPERIMENT_SRC_FILENAME,
        rescaler=0.0,
        output_filename='embeddedscalars-paramexplore-binary.csv',
        response_transformation=(lambda x : 1.0 if x=='T' else 0.0))

# Not used in the paper:
def parameter_exploration_likert():
    parameter_exploration(
        experiment_src=LIKERT_EXPERIMENT_SRC_FILENAME,
        rescaler=1.0,
        output_filename='embeddedscalars-paramexplore-binary.csv',
        response_transformation=(lambda x : int(x)))

# General function for the above:
def parameter_exploration(
        experiment_src=None,
        rescaler=None,
        output_filename=None,
        response_transformation=None):
    writer = csv.DictWriter(file(output_filename, 'w'), fieldnames=['Experiment', 'Listener', 'lambda', 'depth', 'nullcost'] + ['Pearson', 'Pearson p', 'Spearman', 'Spearman p', 'MSE'])
    writer.writeheader()    
    # Parameter space:
    temps = np.concatenate((np.arange(0.1, 2.1, 0.1), np.arange(2.0, 6.0, 1)))
    depths = [0, 1, 2, 3, 4, 5]
    nullcosts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]    
    # General settings:
    subjs= ('every_player', 'exactly_one_player', 'no_player')
    objs = ('every_shot', 'no_shot', 'some_shot')
    players = [a,b,c]
    shots = [s1,s2]
    basic_states = (0,1,2)
    worlds = get_worlds(basic_states=(0,1,2), length=3, increasing=True)
    baselexicon = define_lexicon(player=players, shot=shots, worlds=worlds)
    nullmsg = True
    worldnames = [worldname(w) for w in worlds]
    messages = []
    for d1, d2 in product(subjs, objs):
        subj = d1.replace("_player", "(player)")
        obj = d2.replace("_shot", "(shot)")
        msg = "%s(hit(%s))" % (subj, obj)
        formula = "iv(%s, tv(hit, %s, self.worlds, player))" % (d1,  d2)       
        messages.append((msg, formula))
    # Runs:
    for temperature, n, nullcost in product(temps, depths, nullcosts):
        print temperature, n, nullcost
        neogram = UncertaintyGrammars(
            baselexicon=copy(baselexicon),
            messages=copy(messages),
            worlds=copy(worlds),        
            refinable={'some_player': ['exactly_one_player'], 'some_shot': ['exactly_one_shot']},
            nullmsg=nullcost)    
        neomod = Pragmod(
            name="NeoGricean",
            lexica=neogram.lexicon_iterator,
            baselexicon=neogram.baselexicon_mat,
            messages=neogram.messages,
            states=worldnames,
            temperature=temperature,
            nullmsg=nullmsg,
            nullcost=nullcost)
        ucgram = UncertaintyGrammars(            
            baselexicon=copy(baselexicon),
            messages=copy(messages),
            worlds=copy(worlds),        
            refinable={'some_player': [], 'some_shot': []},
            nullmsg=nullcost)    
        ucmod = Pragmod(
            name="Unconstrained",
            lexica=ucgram.lexicon_iterator,
            baselexicon=ucgram.baselexicon_mat,
            messages=ucgram.messages,
            states=worldnames,
            temperature=temperature,
            nullmsg=nullmsg,
            nullcost=nullcost)        
        ucmod.stream_lexical_uncertainty(n=n)
        neomod.stream_lexical_uncertainty(n=n)        
        analysis = Analysis(experiment=Experiment(src_filename=experiment_src, response_transformation=response_transformation), models=[ucmod, neomod], rescaler=rescaler)
        results = analysis.numeric_analysis()
        for key, vals in results.items():
            vals['Listener'] = key
            vals['Experiment'] = experiment_src
            vals['lambda'] = temperature
            vals['depth'] = n
            vals['nullcost'] = nullcost        
            writer.writerow(vals)

def parameter_exploration_summary(src_filename=None):
    reader = csv.DictReader(file(src_filename))
    listeners = defaultdict(list)    
    for d in reader:
        listeners[d['Listener']].append(d)
    rows = [['Listener', 'Measure', 'Value', 'Nullcost', 'Lambda', 'Depth']]
    lisnames = ['Literal', 'Fixed lexicon', 'Unconstrained', 'NeoGricean']
    measures = ['Pearson', 'Spearman', 'MSE']    
    for lisname in lisnames:
        lis = listeners[lisname]
        summary = process_listener_by_param(lis)
        for measure in measures:
            params = summary[measure]
            rows.append([lisname, measure] + [y for x, y in sorted(params.items(), reverse=True)])
    print "\\\\\n".join([" & ".join(map(str, row)) for row in rows])
            
def process_listener_by_param(lisdict):
    measures = [
        ('Pearson', max, 2),
        ('Spearman', max, 2),
        ('MSE', min, 4) ]
    summary = {}
    for measure, func, digits in measures:
        vals = [float(d[measure]) for d in lisdict]
        best = func(vals)
        vals = [(float(d['nullcost']), float(d['lambda']), float(d['depth'])) for d in lisdict if float(d[measure]) == best]
        vals = sorted(vals, key=itemgetter(0, 1, 2))
        vals = vals[0]
        summary[measure] = {'value': round(best, digits), 'nullcost': vals[0], 'lambda': vals[1], 'depth':vals[2]}
    return summary
                          
######################################################################

if __name__ == '__main__':

    ## Figure 2
    simple_scalar_inference_example()

    ## Table 2, with and without refinement:
    # scalar_disjunction_example(refinable={'some_shot': ['only_some_shot'], 'OR':['XOR'] })
    # scalar_disjunction_example(refinable={})    

    ## Reviewer request (not in the paper):
    # embedded_disjunction_example(refinable={})
    # embedded_disjunction_example(refinable={'hit_shot1_or_shot2': ['hit_shot1_and_shot2']})

    ## Tables 3 and 4
    # complex_example()

    ## Figure 4
    # experiment_plot_and_report_binary()

    ## Figure 7
    # experiment_plot_and_report_likert()
    
    ## Takes a long time to run the first; 'embeddedscalars-paramexplore-binary.csv' is included:
    ## parameter_exploration_binary()
    # parameter_exploration_summary(src_filename='embeddedscalars-paramexplore-binary.csv')
        
    ## Figure 5 and tables 5 and 6
    # experimental_assessment_binary()

    ## Table 8
    # experimental_assessment_likert()

    ## Figure 6
    # experimental_assessment_binary_critical_optimal_params()
