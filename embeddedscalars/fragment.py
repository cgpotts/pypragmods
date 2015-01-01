#!/usr/bin/env python

######################################################################
# The logical grammar (base lexicon) used throughout the paper. The
# code in grammar.py messes with the namespace that it establishes, in
# order to implement lexical uncertainty in an intuitive way.
######################################################################

import sys
from itertools import product
from utils import powerset

######################################################################

a = 'a'; b = 'b'; c = 'c'
s1 = 's1' ; s2 = 's2'

def define_lexicon(player=[], shot=[], worlds=[]):
    D_et = powerset(player+shot)
    relational_hit =  [[w, x, y] for w, x, y in product(worlds, player, shot) if y in shot[: w[player.index(x)]]]
    lex = {
        # Concessions to tractability -- these are defined extensionally (invariant across worlds):
        "some":        [[X, Y] for X, Y in product(D_et, repeat=2) if len(set(X) & set(Y)) > 0],
        "exactly_one": [[X, Y] for X, Y in product(D_et, repeat=2) if len(set(X) & set(Y)) == 1],
        "every":       [[X, Y] for X, Y in product(D_et, repeat=2) if set(X) <= set(Y)],
        "no":          [[X, Y] for X, Y in product(D_et, repeat=2) if len(set(X) & set(Y)) == 0],
        "PlayerA":     [X for X in powerset(player) if a in X],
        "PlayerB":     [X for X in powerset(player) if b in X],
        "PlayerC":     [X for X in powerset(player) if c in X],
        # Tempting to intensionalize these, but that means using intensional quantifiers,
        # which are intractable on this set-theoretic formulation. Our goal is to understand
        # refinement and lexical uncertainty, which we can study using verbs and extensional
        # quantifiers, so this limitation seems well worth it.
        "player":      player,
        "shot":        shot,
        # Intensional predicates:
        "scored":      [[w, x] for w, x in product(worlds, player) if len(shot[: w[player.index(x)]]) > 0],
        "aced":        [[w, x] for w, x in product(worlds, player) if len(shot[: w[player.index(x)]]) > 1],
        "missed":      [[w, x] for w, x in product(worlds, player) if len(shot[: w[player.index(x)]]) == 0],
        "hit" :        [[w, x, y] for w, x, y in product(worlds, player, shot) if y in shot[: w[player.index(x)]]],
        # More concessions to tractability -- we'll refine these rather than the determiners;
        # this should have no effect because of the limited class of predicates -- no predicate
        # is true of both players and shots, and player and shot have the same extensions in all
        # worlds.
        "some_player":        [Y for Y in powerset(player) if len(set(player) & set(Y)) > 0],
        "some_shot":          [Y for Y in powerset(shot)   if len(set(shot) & set(Y)) > 0],
        "exactly_one_player": [Y for Y in powerset(player) if len(set(player) & set(Y)) == 1],
        "exactly_one_shot":   [Y for Y in D_et if len(set(shot) & set(Y)) == 1],
        "every_player":       [Y for Y in D_et if set(player) <= set(Y)],
        "every_shot":         [Y for Y in D_et if set(shot) <= set(Y)],
        "no_player":          [Y for Y in D_et if len(set(player) & set(Y)) == 0],
        "no_shot":            [Y for Y in D_et if len(set(shot) & set(Y)) == 0],
        # Mainly for specifying refinements:
        "not_every_player":   [Y for Y in D_et if not(set(player) <= set(Y))],
        "not_every_shot":     [Y for Y in D_et if not(set(shot) <= set(Y))],
        "scored_not_aced":    [[w, x] for w, x in product(worlds, player) if len(shot[: w[player.index(x)]]) == 1],
        "only_PlayerA":       [X for X in powerset(player) if a in X and len(X) == 1],
        "only_PlayerB":       [X for X in powerset(player) if b in X and len(X) == 1],
        "only_PlayerC":       [X for X in powerset(player) if c in X and len(X) == 1]                
        }
    return lex

def fa(A, b):
    """Muskens-like function application -- in a list [(x,y), ...], we get
    back the second projection limited to the pairs where the first is b."""    
    return [y for x, y in A if x == b]
                    
def iv(Q, X):
    """Returns a proposition as function true of a world w iff the set of
    entities X-at-w is a member of the quantifier (set of sets) Q."""
    return (lambda w : fa(X, w) in Q)

def tv(V, Q, worlds, subjects):
    """Funcion composition taking the intensional relation on entities V
    and combining it with the set of sets Q to return an intensional
    property. The dependence on worlds and subjects is unfortunate but
    I don't see how to avoid it."""    
    return [[w,x] for w, x in product(worlds, subjects) if [y for w_prime, x_prime, y in V if w_prime == w and x_prime == x] in Q]

    
######################################################################

def get_worlds(basic_states=(0,1,2), length=3, increasing=False):
    worlds = list(product(basic_states, repeat=length))
    # Remove sequences in which the elements dom't appear in
    # increasing order. We don't care about order, so this just one
    # way of removing conceptual duplicates.
    if increasing:
        worlds = [w for w in worlds if tuple(sorted(w)) == w]
    return worlds

def worldname(w):
    return "".join(["NSA"[i] for i in w])

######################################################################

if __name__ == '__main__':

    # Domain set up:
    player = [a, b, c]
    shot = [s1, s2]
    worlds = get_worlds((0,1,2), length=len(player), increasing=True)    
    lex = define_lexicon(player=player, shot=shot, worlds=worlds)

    # Import the lexicon into this namespace:
    for word, sem in lex.items():
        setattr(sys.modules[__name__], word, sem)

    # Examples:
    for d1, d2 in product(("some", "exactly_one", "every", "no"), repeat=2):
        msg = "%s(player)(hit(%s(shot)))" % (d1, d2)
        formula = "iv(fa(%s, player), tv(hit, fa(%s, shot), worlds, player))" % (d1,  d2)       
        print msg, [worldname(w) for w in worlds if eval(formula)(w)]

    # Examples:
    for pn, pred in product(('PlayerA', 'PlayerB', 'PlayerC'), ("missed", "scored", "aced")):
        msg = "%s(%s)" % (pn, pred)
        formula = "iv(%s, %s)" % (pn, pred)
        print msg, [worldname(w) for w in worlds if eval(formula)(w)]
    
    
    
