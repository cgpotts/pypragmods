#!/usr/bin/env python

######################################################################
# Compositional lexical uncertainty functions.
######################################################################

import sys
import itertools
import numpy as np
from fragment import *
from settings import NULL
sys.path.append('../')
from utils import display_matrix, powerset

######################################################################

class UncertaintyGrammars:
    def __init__(self,
                 baselexicon=None,
                 messages=[],
                 worlds=[],
                 refinable={},
                 nullmsg=True):
        self.worlds = worlds
        self.refinable = refinable
        self.nullmsg = nullmsg
        if self.nullmsg:
            messages.append(("NULL", None))
        self.messages, self.formulae = zip(*messages)
        self.messages = list(self.messages)
        self.formulae = list(self.formulae)
        self.baselexicon = baselexicon
        self.baselexicon_mat = self.interpretation_matrix(self.baselexicon)
        
    def lexicon_iterator(self):
        words, refinements = zip(*self.get_all_refinements().items())
        for meaning_vector in itertools.product(*refinements):
            lex = dict(zip(words, meaning_vector))
            mat = self.interpretation_matrix(lex)
            # Lexica containing messages that denote {} need to be filtered on
            # the current formulation of the model:
            if 0.0 not in np.sum(mat, axis=1):
                yield mat

    def interpretation_matrix(self, lexicon):
        for word, sem in lexicon.items():
            setattr(sys.modules[__name__], word, sem)
        m = len(self.messages)
        n = len(self.worlds)           
        mat = np.zeros((m, n))
        for i, phi in enumerate(self.formulae):
            for j, w in enumerate(self.worlds):                
                if phi and eval(phi)(w):
                    mat[i,j] = 1.0
        if self.nullmsg:
            mat[-1] = np.ones(n)
        return mat
 
    def get_all_refinements(self):
        # Make sure we're in the baselexicon namespace:
        for word, sem in self.baselexicon.items():
            setattr(sys.modules[__name__], word, sem)
        # Refiments:
        refine = {}
        for word, semval in self.baselexicon.items():
            if word in self.refinable:
                if self.refinable[word]:
                    refine[word] = [semval] + [eval(phi) for phi in self.refinable[word]]
                else:
                    refine[word] = self.refinements(semval)
            else:
                refine[word] = [semval]
        return refine

    def refinements(self, semval):
        return powerset(semval, minsize=1)
       
######################################################################

if __name__ == '__main__':

    players = [a,b]
    shots = [s1,s2]
    basic_states = (0,1)
    worlds = get_worlds(basic_states=(0,1), length=2, increasing=False)
    baselexicon = define_lexicon(player=players, shot=shots, worlds=worlds)

    ug = UncertaintyGrammars(
        baselexicon=baselexicon,
        messages=[("PlayerA(scored)",       "iv(PlayerA, scored)"),
                  ("PlayerB(scored)",       "iv(PlayerB, scored)"),
                  ("every(player)(scored)", "iv(fa(every, player), scored)"),
                  ("no(player)(scored)",    "iv(fa(no, player), scored)"),
                  ("some_player(scored)",   "iv(some_player, scored)")],
        worlds=worlds,
        refinable={'scored':[]},
        nullmsg=True)

    worldnames = [worldname(w) for w in worlds]
    for lex in ug.lexicon_iterator():
        display_matrix(lex, rnames=ug.messages, cnames=worldnames)
