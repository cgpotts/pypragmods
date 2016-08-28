#!/usr/bin/env python

"""
Essentially a convenience function for analyzing complex lexical
spaces, especially those involving disjunctive closures. Use

python lexica.py

to see how a space of lexica is created from a simple baselexicon and
some useful specifications.
"""

__author__ = "Christopher Potts"
__version__ = "2.0"
__license__ = "GNU general public license, version 3"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"


from collections import defaultdict
from copy import copy
from functools import reduce
from itertools import product
import numpy as np

from pypragmods.utils import powerset, display_matrix


NULL_MSG = 'NULL'
DISJUNCTION_SIGN = ' v '
CONJUNCTION_SIGN = ' & '


class Lexica:
    def __init__(self,
            baselexicon,
            atomic_states=[],       
            nullsem=True,             
            join_closure=False,       
            block_ineffability=False, 
            costs=defaultdict(float), 
            disjunction_cost=0.01,    
            nullcost=5.0,            
            unknown_word=None):
        """
        baselexicon : dict
            Mapping strings to iteraables.

        atomic_states : list
            Usually the atomic states are the keys of `baselexicon`,
            but that set can be augmented if one wants states
            where no message is true (except `nullsem` perhaps).
            
        nullsem : bool
             If True, add the null message, true in all states in
             all lexica.

        join_closure : bool
            Close the messages and states under disjunction.

        block_ineffability : bool
            Block states without true messages; relevant only if
            `nullsem=False`.
        
        costs : defaultdict(float)
            `costs.keys()` must contain `baselexicon.keys()`

        disjunction_cost : float
            Cost of a disjunction.

        nullcost : float
            Should probably be higher than regular messages' costs.

        unknown_word : str or None
            A message constrained to have a singleton meaning in all lexica.


        Attributes
        ----------
        All of the parameters to __init__ become attributes.        

        self.messages : list
            The keys of `baselexicon`.

        self.atomic_states : list
            This is the union of `atomic_states` with all the elements
            of the list valued `self.baselexicon`.

        self.states : list
            Same as self.atomic_states` at first, but possibly
            expanded by closures later.

        self.lexica : list
            Created by `self.get_lexica`.
        
        """
        self.baselexicon = baselexicon
        self.messages = sorted(self.baselexicon.keys())
        self.atomic_states = \
          sorted(set(atomic_states) | \
                 set([x for X in self.baselexicon.values() for x in X]))
        self.states = copy(self.atomic_states)
        self.nullsem = nullsem
        self.join_closure = join_closure
        self.block_ineffability = block_ineffability
        self.costs = costs        
        self.disjunction_cost = disjunction_cost
        self.nullcost = nullcost
        self.unknown_word = unknown_word
        self.lexica = self.get_lexica()

    def cost_vector(self):
        """The numerical message cost vector in the same order
        as self.messages"""
        return np.array([self.costs[msg] for msg in self.messages])
        
    def get_lexica(self):
        """The main funtion for building lexicon sets from the user's specs."""
        lexica = []
        enrichments = [powerset(self.baselexicon[msg]) for msg in self.messages]
        for x in product(*enrichments):
            lexica.append(dict(list(zip(self.messages, x))))
        # If there's an unknown word, require it to have an
        # atomic meaning in each lexicon:
        new_lexica = []
        if self.unknown_word and self.unknown_word in self.messages:
            for lex in lexica:                
                if len(lex[self.unknown_word]) == 1:
                    new_lexica.append(lex)
            lexica = new_lexica
        # Close the lexica:
        if self.join_closure:
            lexica = self.add_join_closure(lexica)
            atomic_messages = sorted([x for x in list(self.baselexicon.keys())
                                      if CONJUNCTION_SIGN not in x])
            self.messages += [DISJUNCTION_SIGN.join(sorted(set(cm)))
                              for cm in powerset(atomic_messages, minsize=2)]
            self.states += [DISJUNCTION_SIGN.join(sorted(set(sem)))
                            for sem in powerset(self.atomic_states, minsize=2)]                   
        # Add nullsem last so that it doesn't participate in any
        # closures (and displays last in matrices):
        if self.nullsem:
            lexica = self.add_nullsem(lexica)
            self.messages.append(NULL_MSG)
            self.costs[NULL_MSG] = self.nullcost
        return lexica

    def add_join_closure(self, lexica):
        """Close the atomic messages and atomic states under joins"""
        return self.add_closure(
                lexica=lexica,
                connective=DISJUNCTION_SIGN,
                combo_func=(lambda x,y : x | y),
                cost_value=self.disjunction_cost)
   
    def add_closure(self,
            lexica=None,
            connective=None,
            combo_func=None,
            cost_value=None):
        """Generic function for adding closures."""
        atomic_messages = sorted([x for x in list(self.baselexicon.keys())
                                  if CONJUNCTION_SIGN not in x])
        complex_msgs = [connective.join(sorted(set(cm)))
                        for cm in powerset(atomic_messages, minsize=1)]
        for i, lex in enumerate(lexica):
            for cm in complex_msgs:
                # Get all the worlds consistent with the complex message:
                vals = reduce(combo_func, [set(lex[word])
                                           for word in cm.split(connective)])
                # Closure space:
                vals = powerset(vals, minsize=1)
                # Create the new value, containing worlds and "conjoined worlds":
                if cm not in lex: lex[cm] = set([])
                lex[cm] = list(set(lex[cm]) | set([connective.join(sorted(set(sem)))
                                                   for sem in vals]))
                args = cm.split(connective)
                # Costs for the complex messages:
                signs = len(args)-1                
                self.costs[cm] = (cost_value*signs) + sum(self.costs[word]
                                                          for word in args)
            lexica[i] = lex
        return lexica

    def add_nullsem(self, lexica):
        """Adds the null message to every lexicon"""
        for i, lex in enumerate(lexica):
            lex[NULL_MSG] = self.states
            lexica[i] = lex
        return lexica   

    def lexica2matrices(self):
        """Map the dict-based lexica to matrices for use with pragmods.py"""
        mats = []
        for lex in self.lexica:
            mat = np.zeros((len(self.messages), len(self.states)))
            for i, msg in enumerate(self.messages):
                for j, d in enumerate(self.states):
                    if d in lex[msg]:
                        mat[i,j] = 1.0
            minval = 1 if self.nullsem else 0
            # The models aren't defined for lexica where a message
            # denotes the emptyset:
            if 0.0 not in np.sum(mat, axis=1):
                # Option to ensure that every state can be named by some message:
                if not (self.block_ineffability and minval in np.sum(mat, axis=0)):
                    mats.append(mat)
        return mats    

    def display(self, digits=4):
        """Display all the lexica in a readable way"""
        for i, mat in enumerate(self.lexica2matrices()):      
            display_matrix(
                mat,
                rnames=self.messages,
                cnames=self.states,
                title="Lex{}".format(i),
                digits=digits)
 
    def __len__(self):
        """Number of lexica that are included after any filtering
        the user wanted."""
        return len(self.lexica2matrices())

    
if __name__ == '__main__':

    lexica = Lexica(
        baselexicon={'some': ['w_SOMENOTALL', 'w_ALL'], 'all': ['w_ALL']},
        costs={'some':0.0, 'all':0.0},
        join_closure=True,
        nullsem=True)
            
    lexica.display()
                             
