Bayesian pragmatic models in Python
==========

## Models

* The basic Rational Speech Acts model of Frank and Goodman 2012
* The lexical uncertainty model of Bergen et al. 2012
* The anxiety/uncertainty model of Smith et al. 2013
* The anxious experts model of Levy and Potts 2015
* The streaming lexical uncertainty model useful for large problems like those in Potts et al. 2015

To see these models at work on an example involving the division of pragmatic labor, run

```
python -m pypragmods.pragmods
```

which runs the main method example given in full at the bottom of the
file. In essence, if one has created a set of lexica `lexica`, and
used it to instantiate a `Pragmod` called `mod`, then the different
models are accessible with

```
mod.run_base_model(lexica[0])
mod.run_uncertainty_model()
mod.run_anxiety_model()
mod.run_expertise_model()
```

The current version is compatible with Python 2 and Python 3.


## The disjunction code

For examples of the anxious experts model in action, see
`disjunction/bls41.py`.

```
python -m pypragmods.disjunction.bls41
```

It includes the code for the illustrative examples in Levy and Potts
2015 and Potts and Levy (reference below).  In particular, the
function `compositional_disjunction` shows how to use `lexica.py` to
create a space of lexica for analysis with `pragmod.py`.

## Embedded scalars code

The code in `embeddedscalars` implements the compositional lexical
uncertainty model of Potts et al. 2015. The core pragmatic models is
in `pragmods.py`; this code creates a logical grammar (`fragment.py`),
implements functions for refining that grammar (`grammar.py`),
analyzes our experimental data (`experiment.py`), and reproduces all
of the figures and tables in the paper (`paper.py`, making use of
`analysis.py` for the comparisons between model and experiment. For
examples, `paper.py` is the best place to start:

```
python -m pypragmods.embeddedscalars.paper
```

The subdirectory `experiment` contains the experimental code and
materials. For additional guidance on how to use these materials, see
the repository for
[Dan Lassiter's Submiterator](https://github.com/danlassiter/experiment-template).

## References

Bergen, Leon; Noah D. Goodman; and Roger Levy. 2012.
[That's what she (could have) said: how alternative utterances affect language use](https://mindmodeling.org/cogsci2012/papers/0034/).
In Naomi Miyake, David Peebles, and Richard P. Cooper,
eds., *Proceedings of the 34th Annual Conference of the Cognitive
Science Society*, 120&ndash;125. Austin, TX: Cognitive Science
Society.

Frank, Michael C. and Noah D. Goodman. 2012.
[Predicting pragmatic reasoning in language games](http://www.sciencemag.org/content/336/6084/998). *Science*
336(6084): 998.

Levy, Roger and Christopher Potts. 2015.
[Negotiating lexical uncertainty and expertise with disjunction](http://web.stanford.edu/~cgpotts/talks/levy-potts-pragdisj-poster.pdf).
Poster presented at the 89th Meeting of the Linguistic Society of
America, Portland, OR, January 8&ndash;11.

Potts, Christopher; Daniel Lassiter; Roger Levy; Michael C. Frank.
2015.
[Embedded implicatures as pragmatic inferences under compositional lexical uncertainty](http://web.stanford.edu/~cgpotts/papers/embedded-scalars.pdf).
Ms., Stanford and UCSD.

Potts, Christopher and Roger Levy. 2015. Negotiating lexical
uncertainty and speaker expertise with disjunction. To appear
in *Proceedings of the 41st Annual Meeting of the Berkeley Linguistics
Society*.

Smith, Nathaniel J.; Noah D. Goodman; and Michael C. Frank. 2013.
[Learning and using language via recursive pragmatic reasoning about other agents](http://papers.nips.cc/paper/4929-learning-and-using-language-via-recursive-pragmatic-reasoning-about-other-agents).
In *Advances in Neural Information Processing Systems* 26,
3039&ndash;3047.


