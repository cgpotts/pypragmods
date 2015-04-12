Embedded implicatures as pragmatic inferences under compositional lexical uncertainty
==========

Potts, Christopher; Daniel Lassiter; Roger Levy; Michael C. Frank. 2015. [Embedded implicatures as pragmatic inferences under compositional lexical uncertainty](http://web.stanford.edu/~cgpotts/papers/embedded-scalars.pdf). Ms., Stanford and UCSD. [[slides](http://web.stanford.edu/~cgpotts/talks/embedded-scalars-slides.pdf)]

## Code

`paper.py` reproduces all of the figures, tables, and analyses in the paper. It's a good source for illustrations and examples.

(`settings.py` contains basic set-up and plotting specs.)

### Models

* `../pragmods.py`: general implementations of the pragmatic models
* `fragment.py`: the core logical grammar
* `grammar.py`: compositional lexical uncertainty

### Experiment and analysis

* `experiment.py`: utilities for processing and studying the experimental data
* `analysis.py`: model assessment against the experimental data
* `embeddedscalars-experiment-results-binary.csv`: the experimental data from the binary version of the experiment
* `embeddedscalars-experiment-results-likert.csv`: the experimental data from the Likert-scale version of the experiment
* `embeddedscalars-paramexplore-binary.csv`: the results of the parameter exploration suggested by a reviewer (appendix A of the paper)
* `embeddedscalars-paramexplore-likert.csv`: corresponding parameter exploration for the Likert-scale experiment (not report in the paper)
* `experiment/`: JQuery-based experiment materials and code for both versions of the experiment
* `bootstrap.py`: version 0.3.2 of scikits.bootstrap (included here to ensure reproducibility)

In the two experiment files, the workerid values have been anonymized, but in a way that preserves identities across the two files. (31 people took both experiments, which were conducted about a year apart.)


