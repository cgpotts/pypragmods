#/usr/bin/env python

######################################################################
# Examples for the LSA 2015 poster
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
sys.path.append('../')
from lexica import Lexica
from pragmods import Pragmod

######################################################################
##### Illustrative examples

def generic_example(alpha=1.0, beta=1.0, disjunction_cost=1.0):
    """Common code for our two illustrative examples, which
    differ only in the above keyword parameters."""    
    # Use the lexicon generation convenience function to
    # generate all the join-closure lexica and calculate
    # the necessary message costs:
    lexica = Lexica(
        baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']}, 
        costs={'A':0.0, 'B':0.0, 'X':0.0},
        join_closure=True,
        nullsem=True,
        nullcost=5.0,
        disjunction_cost=disjunction_cost)
    # Lexical matrices:
    lexmats = lexica.lexica2matrices()         
    # Pragmatic models for the above lexical space.
    mod = Pragmod(
        lexica=lexmats,
        messages=lexica.messages,
        meanings=lexica.states,
        costs=lexica.cost_vector(),
        prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
        lexprior=np.repeat(1.0/len(lexmats), len(lexmats)),
        temperature=1.0,
        alpha=alpha,
        beta=beta)
    ## Uncomment these lines for a fuller picture:
    ## Show the lexica:
    # lexica.display()
    ## Run the base model on the individual lexica so we can show those lower steps:
    # for lex in lexmats:
    #     print "======================================================================"
    #     print mod.lex2str(lex)
    #     mod.run_base_model(lex, n=2, display=True, digits=2)             
    ## Run the anxious experts model - display=True for a fuller picture:
    langs = mod.run_expertise_model(n=2, display=False, digits=2)
    # Look at the specific table we care about:
    msg_index = mod.messages.index('A v X')
    final_listener = langs[-1]
    mod.display_joint_listener(final_listener[msg_index], digits=2)
    return langs
        
def Hurfordian_Contexts():
    generic_example(alpha=2.0, beta=1.0, disjunction_cost=1.0)
        
def Definitional_Contexts():
    generic_example(alpha=5.0, beta=7.0, disjunction_cost=0.01)

    
if __name__ == '__main__':

    Hurfordian_Contexts()
    Definitional_Contexts()
    
