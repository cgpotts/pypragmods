#/usr/bin/env python

######################################################################
# Implements the following pragmatic models
#
# - The basic Rational Speech Acts model of Frank and Goodman 2012
# - The lexical uncertainty model of Bergen et al. 2012
# - The anxiety/uncertianty model of Smith et al. 2013
# - The anxious experts model of Levy and Potts 2015
#
# Use
#
# python pragmods.py
#
# to see an example involving the division of pragmatic labor, using
# all of the above models.
#
# References:
#
# Frank, Michael C. and Noah D. Goodman. 2012. Predicting pragmatic
# reasoning in language games. Science 336(6084): 998.
#
# Bergen, Leon; Noah D. Goodman; and Roger Levy. 2012. That's what she
# (could have) said: how alternative utterances affect language
# use. In Naomi Miyake, David Peebles, and Richard P. Cooper, eds.,
# Proceedings of the 34th Annual Conference of the Cognitive Science
# Society, 120-125. Austin, TX: Cognitive Science Society.
#
# Smith, Nathaniel J.; Noah D. Goodman; and Michael C. Frank. 2013.
# Learning and using language via recursive pragmatic reasoning about
# other agents. In Advances in Neural Information Processing Systems
# 26, 3039-3047.
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
from utils import rownorm, colnorm, safelog, display_matrix

######################################################################
        
class Pragmod:    
    def __init__(self,
            lexica=None, 
            messages=None, 
            meanings=None, 
            costs=None, 
            prior=None, 
            lexprior=None, 
            temperature=1.0,
            alpha=1.0,
            beta=1.0):
        self.lexica = lexica            # list of np.arrays of dimension m x n
        self.messages = messages        # list or tuple of strings of length m
        self.meanings = meanings        # list or tuple of strings of length n
        self.costs = costs              # np.array of length m
        self.prior = prior              # np.array of length n
        self.lexprior = lexprior        # np.array of length len(self.lexica)
        self.temperature = temperature  # usually \lambda, but lambda is a Python builtin; should be > 0.0
        self.alpha = alpha              # speaker value for the world state
        self.beta = beta                # speaker value for the lexicon

    ##################################################################
    ##### Iteration models

    def run_base_model(self, lex, n=2, display=True, digits=4):
        """Basic model with a specified messages x meanings matrix of truth values lex"""
        return self.run(n=n, display=display, digits=digits, initial_listener=self.l0(lex), start_level=0)
    
    def run_uncertainty_model(self, n=2, display=True, digits=4):
        """The lexical uncertainty model of Bergen et al. 2012, 2014"""
        return self.run(n=n, display=display, digits=digits, initial_listener=self.UncertaintyListener(), start_level=1)    

    def run_anxiety_model(self, n=2, display=True, digits=4):
        """One-shot version of the social anxiety model of Smith et al. 2013"""        
        return self.run(n=n, display=display, digits=digits, initial_listener=self.UncertaintyAnxietyListener(marginalize=True), start_level=1)

    def run(self, lex=None, n=2, display=True, initial_listener=None, start_level=0, digits=4):
        """Generic iterator. n is the depth of iteration. initial_listener is one of the
        listener methods, applied to a lexical argument in the case of the base model.
        display=True prints all matrices to standard output. start_level controls which 
        listener number to begin with for displaying the model."""
        langs = [initial_listener]
        for i in range(1, (n-1)*2, 2):
            langs.append(self.S(langs[i-1]))
            langs.append(self.L(langs[i]))
        if display:
            self.display_iteration(langs, start_level=start_level, digits=digits)        
        return langs  

    def run_expertise_model(self, n=2, display=True, digits=4):
        """Iterator for the anxious experts model."""
        langs = [self.UncertaintyAnxietyListener(marginalize=False)]
        for i in range(1, (n-1)*2, 2):
            langs.append(self.ExpertiseSpeaker(langs[i-1]))
            langs.append(self.ExpertiseListener(langs[i]))
        if display:
            self.display_expertise_iteration(langs, digits=digits)      
        return langs  
        
    ##################################################################
    ##### Agents

    def l0(self, lex):
        """Maps the truth-conditional lexicon lex to a probabilistic one incorporating priors."""
        return rownorm(lex * self.prior)  

    def L(self, spk):
        """The general listener differs from l0 only in transposing the incoming speaker matrix."""
        return self.l0(spk.T)

    def S(self, lis):
        """Bayesian speaker incorporating costs."""
        return rownorm(np.exp(self.temperature * ((self.alpha * safelog(lis.T)) - self.costs)))
    
    def s1(self, lex):
        """Convenience function for S(l0(lex))"""
        return self.S(self.l0(lex))

    def l1(self, lex):
        """Convenience function for L(S(l0(lex)))"""
        return self.L(self.s1(lex))        
    
    def UncertaintyListener(self): 
        """The lexical uncertainty listener reasons over the marginal of S(L0(lex)) for all lexicons lex."""
        result = [self.lexprior[i] * self.prior * self.s1(lex).T for i, lex in enumerate(self.lexica)]
        return rownorm(np.sum(result, axis=0))

    def UncertaintyAnxietyListener(self, marginalize=False):
        """Social anxiety listener of Smith et al. 2013."""
        lik = self.lex_lik()
        result = np.array([(self.l1(lex).T * lik[i]).T for i, lex in enumerate(self.lexica)])
        if marginalize:
            result = np.sum(result, axis=0)
        return result
                       
    def lex_lik(self):
        """Creates a lexicon x utterance matrix, normalized columnwise for P(Lex|u)."""
        p = np.array([np.sum(self.s1(lex), axis=0) * self.lexprior[i] for i, lex in enumerate(self.lexica)])
        return colnorm(p)

    def ExpertiseSpeaker(self, listeners):
        """Expertise speaker"""
        lis = np.sum(listeners, axis=0)
        lexprobs = np.sum(listeners, axis=2)                   
        result = np.zeros((len(self.messages), len(self.meanings), len(self.lexica)))
        for u in range(len(self.messages)):
            for m in range(len(self.meanings)):
                for l in range(len(self.lexica)):
                    result[u,m,l] = np.exp(self.temperature * ((self.alpha*safelog(lis[u,m])) + (self.beta*safelog(lexprobs[l,u])) - self.costs[u]))
        return result / np.sum(result, axis=0)

    def ExpertiseListener(self, speakers):
        """Expertise listener"""              
        result = np.zeros((len(self.lexica), len(self.messages), len(self.meanings)))
        for k in range(len(self.lexica)):                    
            for i in range(len(self.messages)):
                for j in range(len(self.meanings) ):
                    result[k,i,j] = speakers[i][j,k] * self.prior[j] * self.lexprior[k]
        totals = np.sum(result, axis=(0, 2))
        return np.array([(r.T / totals).T for r in result])

    ##################################################################
    #### Return to simple signaling for joint models

    def listener_lexical_marginalization(self, lismat):
        return np.sum(lismat, axis=0)

    def speaker_lexical_marginalization(self, spkmat):
        spk = np.zeros((len(self.messages), len(self.meanings)))
        for j, u in enumerate(spkmat): 
            spk[j] += np.sum(u, axis=1)
        return rownorm(spk.T)
        
    ##################################################################
    ##### Display functions

    def display_expertise_iteration(self, langs, digits=4):        
        level = 1
        for index in range(0, len(langs)-1, 2):
            self.display_joint_listener_matrices(langs[index], level=1, digits=digits)
            self.display_listener_matrix(self.listener_lexical_marginalization(langs[index]), title="%s - marginalized" % level, digits=digits)                        
            level += 1
            self.display_expert_speaker_matrices(langs[index+1], level=level, digits=digits)
            self.display_speaker_matrix(self.speaker_lexical_marginalization(langs[index+1]), title='%s - marginalized' % level, digits=digits)
            
    def display_iteration(self, langs, start_level=0, digits=4):
        self.display_listener_matrix(langs[0], title=start_level, digits=digits)        
        start_level += 1
        display_funcs = (self.display_speaker_matrix, self.display_listener_matrix)
        for i, lang in enumerate(langs[1: ]):
            display_funcs[i % 2](lang, title=start_level, digits=digits)
            if i % 2: start_level += 1

    def display_speaker_matrix(self, mat, title='', digits=4):
        """Pretty-printed speaker matrix to standard output"""
        display_matrix(mat, title='S%s' % title, rnames=self.meanings, cnames=self.messages, digits=digits)

    def display_listener_matrix(self, mat, title='', digits=4):
        """Pretty-printed listener matrix to standard output"""
        display_matrix(mat, title='L%s' % title, rnames=self.messages, cnames=self.meanings, digits=digits)

    def display_joint_listener(self, mat, title='', digits=4):
        lexnames = ['Lex%s' % i for i in range(len(self.lexica))]
        display_matrix(mat, rnames=lexnames, cnames=self.meanings, title=title, digits=digits)        

    def display_joint_listener_matrices(self, mats, level=1, digits=4):
        reconfig = np.transpose(mats, axes=(1,0,2)) 
        [self.display_joint_listener(mat, title='L%s - %s' % (level, self.messages[i]), digits=digits) for i, mat in enumerate(reconfig)]
        
    def display_expert_speaker_matrices(self, mats, level=1, digits=4):
        reconfig = np.transpose(mats, axes=(2,1,0))
        [self.display_speaker_matrix(mat, title='%s - Lex%s' % (level, i), digits=digits) for i, mat in enumerate(reconfig)]


        
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
        meanings=['normal-world', 'abnormal-world'],             # World-types and      
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
    mod.display_listener_matrix(baselangs[-1], title=" - Base model")
    
    # Basic lexical uncertainty model:
    lulangs = mod.run_uncertainty_model(n=n, display=False)
    mod.display_listener_matrix(lulangs[-1], title=" - Lexical uncertainty model") 

    # The Smith et al. uncertainty/anxiety listener:
    ualangs = mod.run_anxiety_model(n=n, display=False)
    mod.display_listener_matrix(ualangs[-1], title=" - The anxiety/uncertainty model")
    
    # Lexical uncertainty with anxious experts:
    expertlangs = mod.run_expertise_model(n=n, display=False)
    mod.display_listener_matrix(mod.listener_lexical_marginalization(expertlangs[-1]), title=" - The anxious experts model")

