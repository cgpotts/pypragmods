#/usr/bin/env python

"""
Implements the following pragmatic models

- The basic Rational Speech Acts model of Frank and Goodman 2012
- The lexical uncertainty model of Bergen et al. 2012
- The anxiety/uncertianty model of Smith et al. 2013
- The anxious experts model of Levy and Potts 2015
- A streaming lexical uncertainty implementation for large problems.

Use

python pragmods.py

to see an example involving the Levinson/Horn division of pragmatic
labor, using  all of the above models.

References:

Frank, Michael C. and Noah D. Goodman. 2012. Predicting pragmatic
reasoning in language games. Science 336(6084): 998.

Bergen, Leon; Noah D. Goodman; and Roger Levy. 2012. That's what she
(could have) said: how alternative utterances affect language
use. In Naomi Miyake, David Peebles, and Richard P. Cooper, eds.,
Proceedings of the 34th Annual Conference of the Cognitive Science
Society, 120-125. Austin, TX: Cognitive Science Society.

Smith, Nathaniel J.; Noah D. Goodman; and Michael C. Frank. 2013.
Learning and using language via recursive pragmatic reasoning about
other agents. In Advances in Neural Information Processing Systems
26, 3039-3047.

Levy, Roger and Christopher Potts. 2015. Negotiating lexical
uncertainty and expertise with disjunction. Poster presented at the
89th Meeting of the Linguistic Society of America, Portland, OR,
January 8-11.

Potts, Christopher; Daniel Lassiter; Roger Levy; Michael C. Frank.
2015. Embedded implicatures as pragmatic inferences under
compositional lexical uncertainty. Ms., Stanford and UCSD.
"""


__author__ = "Christopher Potts"
__version__ = "2.0"
__license__ = "GNU general public license, version 3"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"


from collections import defaultdict
from copy import copy
import numpy as np
import sys

from pypragmods.utils import rownorm, colnorm, safelog, display_matrix
from pypragmods.lexica import DISJUNCTION_SIGN, NULL_MSG


        
class Pragmod:    
    def __init__(self,
            name="",
            lexica=None,
            baselexicon=None,
            messages=None, 
            states=None, 
            costs=None, 
            prior=None, 
            lexprior=None,
            lexcount=None, 
            temperature=1.0,
            alpha=1.0,
            beta=1.0,
            nullmsg=True,
            nullcost=5.0):
        """
        Parameters
        ----------
        name : str
            Optional informal name for the model.

        lexica : list of np.array
            Dimension m x n.

        baselexicon : np.array
            The starting point for the space of lexica.

        messages : iterable
            Length m

        states : iterable
            Length n

        costs : np.array
            Length m, with float values. If this is `None`, then
            0.0 costs are assumed for all messages except 
            `nullmsg`, if there is one. If there is, it is given
            its own cost as specified by `nullcost`.

        prior : np.array
            Length n, with float values summing to 1.0. If `None`,
            then this becomes an even distribution over states.

        lexprior : np.array
            Length len(self.lexica) with float values summing to 1.0.
            If no lexicon prior is given, but we do know the number
            of lexica (`lexcount`), then we define a flat prior over
            lexica. If no `lexcount` is given, we lead this undefined
            and the lexicon prior is implicitly flat.

        lexcount : int
            Number of lexica if known ahead of time.

        temperature : float
            Usually \lambda, but lambda is a Python builtin;
            should be > 0.0.

        alpha : float
            Speaker value for the world state.

        beta : float
            Speaker value for the lexicon.

        nullmsg : bool
            Whether to assume the final message is null.

        nullcost : float
            Cost for the nullmsg if there is one. Should be positive.

        Attributes
        ----------
        All of the above become attributes. In addition:

        self.final_listener : initialized as an all 0s matrix
        self.final_speaker : initialized as `None`

        Both of these are filled in my model methods, allowing for
        easier access to the final agents.        
        """
        self.name = name
        self.lexica = lexica
        self.baselexicon = baselexicon
        self.messages = messages
        self.states = states
        self.costs = costs
        self.prior = prior
        self.lexprior = lexprior
        self.lexcount = lexcount
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta 
        self.nullmsg = nullmsg
        self.nullcost = nullcost
        if type(self.prior) == type(None):
            self.prior = np.repeat(1.0/len(self.states), len(self.states))                  
        if type(self.lexprior) == type(None) and self.lexcount != None:
            self.lexprior = np.repeat(1.0/self.lexcount, self.lexcount)
        else:
            self.lexprior = defaultdict(lambda : 1.0)
        if type(self.costs) == type(None):
            self.costs = np.zeros(len(self.messages))
            if self.nullmsg:
                self.costs[-1] = self.nullcost
        self.final_listener = np.zeros((len(self.messages), len(self.states)))
        self.final_speaker = None        

    ##################################################################
    ##### Iteration models

    def run_base_model(self, lex, n=2, display=True, digits=4):
        """Basic model with a specified messages x states matrix of
        truth values lex"""
        return self.run(
            n=n,
            display=display,
            digits=digits,
            initial_listener=self.l0(lex),
            start_level=0)

    def rsa(self, lex=None):
        if not lex:
            lex = self.baselexicon
        lit = self.l0(lex)
        spk = self.S(lit)
        lis = self.L(spk)
        return [lit, spk, lis]
    
    def run_uncertainty_model(self, n=2, display=True, digits=4):
        """The lexical uncertainty model of Bergen et al. 2012, 2014"""
        return self.run(
            n=n,
            display=display,
            digits=digits,
            initial_listener=self.UncertaintyListener(),
            start_level=1)    

    def run_anxiety_model(self, n=2, display=True, digits=4):
        """One-shot version of the social anxiety model of Smith et al. 2013"""        
        return self.run(
            n=n,
            display=display,
            digits=digits,
            initial_listener=self.UncertaintyAnxietyListener(marginalize=True),
            start_level=1)

    def run(self,
            initial_listener,
            n=2,
            display=True,
            start_level=0,
            digits=4):
        """Generic iterator.

        Parameters
        ----------
        initial_listener (model function)
        n : int (the depth of iteration)
        display : bool (whether to all matrices to standard output)
        start_level : int (controls which listener number to begin with for display)
        digits : int (rounding precision for intermediate displays)        
        """
        langs = [initial_listener]
        for i in range(1, (n-1)*2, 2):
            langs.append(self.S(langs[i-1]))
            langs.append(self.L(langs[i]))
        if len(langs) >= 2:
            self.final_speaker, self.final_listener = langs[-2: ]
        else:
            self.final_speaker = None
            self.final_listener = langs[-1]
        if display:
            self.display_iteration(
                langs, start_level=start_level, digits=digits)      
        return langs  

    def run_expertise_model(self, n=2, display=True, digits=4):
        """Iterator for the anxious experts model."""
        langs = [self.UncertaintyAnxietyListener(marginalize=False)]
        for i in range(1, (n-1)*2, 2):
            langs.append(self.ExpertiseSpeaker(langs[i-1]))
            langs.append(self.ExpertiseListener(langs[i]))
        if display:
            self.display_expertise_iteration(langs, digits=digits)
        if len(langs) >= 2:
            self.final_speaker, self.final_listener = langs[-2: ]
        else:
            self.final_speaker = None
            self.final_listener = langs[-1]
        return langs

    def stream_lexical_uncertainty(self, n=0, display_progress=True):
        """Separate interface to the lexical uncertainty model that doesn't
        hold all the lexica in memory -- essential for very large problem spaces."""        
        # If there is no lexicon prior, then this allows us to ignore it.
        lexprior_func = (lambda x : 1.0)
        # Where we have a lexicon prior, we can look up the value in self.lexprior:
        if self.lexprior != None:
            lexprior_func = (lambda lexindex : self.lexprior[lexindex])
        # Iterate through the lexica:
        for lexindex, lex in enumerate(self.lexica()):
            if display_progress and lexindex and lexindex % 10**2 == 0:
                sys.stderr.write('\r')
                sys.stderr.write('lexicon {}'.format(lexindex))
                sys.stderr.flush()
            self.final_listener += lexprior_func(lexindex) * self.s1(lex).T            
        # Update or fill in the lexcount based on the iteration:
        self.lexcount = lexindex + 1
        # Final normalization and state prior incorporation:
        self.final_listener = rownorm( self.prior * self.final_listener)
        # Optional further iteration of L and S with no lexical uncertainty:
        for i in range(n-1):
            self.final_speaker = self.S(self.final_listener)
            self.final_listener = self.L(self.final_speaker)
        
    ##################################################################
    ##### Agents

    def l0(self, lex):
        """Maps the truth-conditional lexicon lex to a probabilistic
        one incorporating priors."""
        return rownorm(lex * self.prior)  

    def L(self, spk):
        """The general listener differs from l0 only in transposing
        the incoming speaker matrix."""
        return self.l0(spk.T)

    def S(self, lis):
        """Bayesian speaker incorporating costs."""
        return rownorm(
            np.exp(self.temperature * ((self.alpha * safelog(lis.T)) - self.costs)))
    
    def s1(self, lex):
        """Convenience function for S(l0(lex))"""
        return self.S(self.l0(lex))

    def l1(self, lex):
        """Convenience function for L(S(l0(lex)))"""
        return self.L(self.s1(lex))        
    
    def UncertaintyListener(self): 
        """The lexical uncertainty listener reasons over the marginal
        of S(L0(lex)) for all lexicons lex."""
        result = np.array([self.lexprior[i] * self.prior * self.s1(lex).T
                           for i, lex in enumerate(self.lexica)])
        return rownorm(np.sum(result, axis=0))

    def UncertaintyAnxietyListener(self, marginalize=False):
        """Social anxiety listener of Smith et al. 2013."""        
        lik = self.lex_lik()                
        result = np.array([(self.l1(lex).T * lik[i]).T
                           for i, lex in enumerate(self.lexica)])
        if marginalize:
            return np.sum(result, axis=0)
        else:
            return np.transpose(result, axes=(1,0,2)) 
                       
    def lex_lik(self):
        """Creates a lexicon x utterance matrix, normalized
        columnwise for P(Lex|msg)."""
        p = np.array([np.sum(self.s1(lex), axis=0) * self.lexprior[i]
                      for i, lex in enumerate(self.lexica)])
        return colnorm(p)

    def ExpertiseSpeaker(self, listeners):
        """Expertise speaker: 3d array containing P(msg | meaning, lexicon)"""
        lis = np.sum(listeners, axis=1)
        lexprobs = np.sum(listeners, axis=2).T
        result = np.zeros((len(self.lexica), len(self.states), len(self.messages)))
        for l in range(len(self.lexica)):
            for m in range(len(self.states)):
                for u in range(len(self.messages)):
                    result[l,m,u] = \
                      np.exp(self.temperature * \
                             ((self.alpha*safelog(lis[u,m])) + \
                              (self.beta*safelog(lexprobs[l,u])) - \
                             self.costs[u]))
        return (result.T / np.sum(result.T, axis=0)).T

    def ExpertiseListener(self, speakers):
        """Expertise listener: for each message, a joint <lexicon, meaning> table"""            
        result = np.zeros((len(self.messages), len(self.lexica), len(self.states)))
        for u in range(len(self.messages)):
            for l in range(len(self.lexica)):                                
                for m in range(len(self.states)):
                    result[u,l,m] = speakers[l,m,u] * self.prior[m] * self.lexprior[l]
        totals = np.sum(result, axis=(1, 2))
        return (result.T / totals.T).T

    ##################################################################
    #### Return to simple signaling for joint models

    def listener_lexical_marginalization(self, lismat):
        """Return to state/message signaling by marginalizing over lexica"""
        return np.sum(lismat, axis=1)

    def speaker_lexical_marginalization(self, spkmat):
        """Return to state/message signaling by marginalizing over lexica"""
        return rownorm(np.sum(spkmat, axis=0))
        
    ##################################################################
    ##### Display functions

    def display_expertise_iteration(self, langs, digits=4):
        """Display the full iteration for any the expertise model"""       
        level = 1
        for index in range(0, len(langs)-1, 2):
            self.display_joint_listener_matrices(
                langs[index], level=level, digits=digits)
            self.display_listener_matrix(
                self.listener_lexical_marginalization(langs[index]),
                title="{} - marginalized".format(level),
                digits=digits)                        
            level += 1
            self.display_expert_speaker_matrices(
                langs[index+1], level=level, digits=digits)
            self.display_speaker_matrix(
                self.speaker_lexical_marginalization(langs[index+1]),
                title='{} - marginalized'.format(level),
                digits=digits)
            
    def display_iteration(self, langs, start_level=0, digits=4):
        """Display the full iteration for any model except expertise"""
        self.display_listener_matrix(
            langs[0], title=start_level, digits=digits)        
        start_level += 1
        display_funcs = (self.display_speaker_matrix,
                         self.display_listener_matrix)
        for i, lang in enumerate(langs[1: ]):
            display_funcs[i % 2](lang, title=start_level, digits=digits)
            if i % 2: start_level += 1

    def display_speaker_matrix(self, mat, title='', digits=4):
        """Pretty-printed (to stdout) speaker matrix to standard output"""
        display_matrix(
            mat,
            title='S{}'.format(title),
            rnames=self.states,
            cnames=self.messages,
            digits=digits)

    def display_listener_matrix(self, mat, title='', digits=4):
        """Pretty-printed (to stdout) listener matrix to standard output"""
        display_matrix(
            mat,
            title='L{}'.format(title),
            rnames=self.messages,
            cnames=self.states,
            digits=digits)

    def display_joint_listener(self, mat, title='', digits=4):
        """Pretty-printed (to stdout) lexicon x world joint probability
        table for a given message"""
        lexnames = ['Lex%s: %s' % (i, self.lex2str(lex))
                    for i, lex in enumerate(self.lexica)]
        display_matrix(
            mat,
            rnames=lexnames,
            cnames=self.states,
            title=title,
            digits=digits)        

    def display_joint_listener_matrices(self, mats, level=1, digits=4):
        """Pretty-printed (to stdout) lexicon x world joint probability
        table for all messages"""
        [self.display_joint_listener(
            mat,
            title='L{} - {}'.format(level, self.messages[i]),
            digits=digits)
         for i, mat in enumerate(mats)]
        
    def display_expert_speaker_matrices(self, mats, level=1, digits=4):
        """Pretty-printed (to stdout) list of world x message
        conditional probability tables, one for each lexicon"""
        [self.display_speaker_matrix(
            mat,
            title='{} - Lex{} {}'.format(level, i, self.lex2str(self.lexica[i])),
            digits=digits)
        for i, mat in enumerate(mats)]

    def lex2str(self, lex):
        """Format a lexicon for easy inspection"""
        def state_sorter(x):
            return sorted(x, key=len)
        entries = []
        for i, msg in enumerate(self.messages):
            if msg != NULL_MSG and DISJUNCTION_SIGN not in msg:
                sem = [w for j, w in enumerate(self.states)
                       if lex[i,j] > 0.0 if DISJUNCTION_SIGN not in w]
                entry = msg + "={" + ",".join(state_sorter(sem)) + "}"
                entries.append(entry)
        return "; ".join(entries)

    def listener_report(self, digits=4):
        print("=" * 70) # Divider bar.
        print('Lexica: {}'.format(self.lexcount))
        print('Final listener')
        display_matrix(
            self.final_listener,
            rnames=self.messages,
            cnames=self.states,
            digits=digits)
        print('\nBest inferences:')
        best_inferences = self.get_best_inferences(digits=digits)  
        for msg, val in sorted(best_inferences.items()):
            print("\t{} {}".format(msg, val))
        print("\nLaTeX table:\n")
        print(self.final_listener2latex())

    def get_best_inferences(self, digits=4):    
        best_inferences = {}
        # Round to avoid tiny distinctions that don't even display:
        mat = np.round(copy(self.final_listener), 10)
        for i, msg in enumerate(self.messages):
            best_inferences[msg] = [(w, str(np.round(mat[i,j], digits)))
                                    for j, w in enumerate(self.states)
                                    if mat[i,j] == np.max(mat[i])]             
        return best_inferences   

    def final_listener2latex(self, digits=2):
        mat = np.round(copy(self.final_listener), digits)
        rows = []
        rows.append([''] + self.states)
        for i in range(len(self.messages)):
            rowmax = np.max(mat[i])
            def highlighter(x):
                return r"\graycell{%s}" % x if x == rowmax else str(x)
            vals = [highlighter(x) for x in mat[i]]            
            rows.append([self.messages[i]] + vals)
        s = ""
        s += "\\begin{tabular}[c]{r *{%s}{r} }\n" % len(self.states)
        s += "\\toprule\n"
        s += "%s\\\\\n" % " & ".join(rows[0])
        s += "\\midrule\n"
        for row in rows[1: ]:
            s += "%s\\\\\n" % " & ".join(row)
        s += "\\bottomrule\n"
        s += "\\end{tabular}"
        return s       
        
if __name__ == '__main__':

    ##################################################################
    ##### Demo
    #
    # Example involving the division of pragmatic labor (marked forms
    # express marked meanings; unmarked forms express unmarked
    # meanings). This can be captured in the lexical uncertainty
    # models but not the fixed-lexicon ones.

    # The three non-contradictory propositions:
    TT = [1.0, 1.0]
    TF = [1.0, 0.0]
    FT = [0.0, 1.0]

    # Semantics for the null message fixed for all lexica:
    nullsem = [1.0, 1.0]

    # The nine logically distinct lexica -- message rows, world columns:
    lexica = [
        np.array([TT, TT, nullsem]),
        np.array([TT, TF, nullsem]),
        np.array([TT, FT, nullsem]),
        np.array([TF, TT, nullsem]),
        np.array([TF, TF, nullsem]),
        np.array([TF, FT, nullsem]),
        np.array([FT, TT, nullsem]),
        np.array([FT, TF, nullsem]),
        np.array([FT, FT, nullsem])]
   
    # General model with the temperature parameter (lambda) set aggressively:
    mod = Pragmod(
        lexica=lexica,
        messages=['normal-message', 'abnormal-message', 'null'], # Messsages and
        costs=np.array([1.0, 2.0, 5.0]),                         # their costs.
        states=['normal-world', 'abnormal-world'],               # World-types and      
        prior=np.array([2.0/3.0, 1.0/3.0]),                      # their prior.
        lexprior=np.repeat(1.0/len(lexica), len(lexica)),        # Flat lex prior.
        temperature=3.0,
        alpha=1.0, 
        beta=1.0) # Relevant only for the anxious experts model.

    # Compare the final listeners (display=True for full model output):
    # Iteration depth (sort of arbitrary here):
    n = 4
            
    # The base model on the first (true) lexicon:
    baselangs = mod.run_base_model(lexica[0], n=n, display=False)    
    mod.display_listener_matrix(
        baselangs[-1],
        title=" - Base model")
    
    # Basic lexical uncertainty model:
    lulangs = mod.run_uncertainty_model(n=n, display=False)
    mod.display_listener_matrix(
        lulangs[-1],
        title=" - Lexical uncertainty model")       

    # The Smith et al. uncertainty/anxiety listener:
    ualangs = mod.run_anxiety_model(n=n, display=False)
    mod.display_listener_matrix(
        ualangs[-1],
        title=" - The anxiety/uncertainty model")
    
    # Lexical uncertainty with anxious experts:
    expertlangs = mod.run_expertise_model(n=n, display=False)
    mod.display_listener_matrix(
        mod.listener_lexical_marginalization(expertlangs[-1]),
        title=" - The anxious experts model")

    ##################################################
    # Streaming lexical uncertainty model:
    def lexicon_iterator():
        for x in lexica:
            yield x    
    mod = Pragmod(
        lexica=lexicon_iterator,
        messages=['normal-message', 'abnormal-message', 'null'], # Messsages and
        costs=np.array([1.0, 2.0, 5.0]),                         # their costs.
        states=['normal-world', 'abnormal-world'],               # World-types and      
        prior=np.array([2.0/3.0, 1.0/3.0]),                      # their prior.
        temperature=3.0)
    mod.stream_lexical_uncertainty(n=n)
    mod.display_listener_matrix(
        mod.final_listener,
        title=" - Streaming lexical uncertainty model")   

    
