Phage proxy voting experiment
=============================

Graphs and analyses of proxy voting information for Phage.

### Setup

The code is pure python, but depends on

* [Graphviz](http://graphviz.org): Graph visualization
* [Graphviz python module](https://pypi.python.org/pypi/graphviz): Graphviz interface
* [Numpy](http://numpy.org): Efficient multidimensional arrays
* [Scipy](http://scipy.org): Scientific computation

On Mac with Homebrew, these can be installed via

    brew install graphviz
    pip install graphviz numpy scipy

The failed Bayesian network model also required

* [PyMC](https://github.com/pymc-devs/pymc): Markov chain Monte Carlo

but the current code does not.

## Model

We have a directed graph G = (V,E) of people with edges from people to people
"they trust in some capacity."  Let n = |V| and m = |E|.  There are at least
two possible goals:

1. *Vote*: Given a choice set C and an incomplete set of votes f : V -> C,
   determine a choice for the whole group.

2. *Represent*: Find a small number r of people that "best represent" the
   whole graph G.

The vote goal is easy: one simple approach is to define a Markov chain
on V which stops at range(f) and divides equally between outgoing edges
in V - range(f).  A vote for someone in V - range(f) can then be split
according to the limit distribution of the Markov chain starting at their
vertex.  This can be expressed as a single eigenvector computation.

The represent goal is harder, since we need a notion of what best
representation means.  I believe other people have come up with models for
this, in particular the Structured Deep Democracy (SD2) and Transitive
Mandatory Multiple Proxy (TMMP) mentioned in Jeremy's slides, but unfortunately
we lack papers with these algorithms.  Supposedly they're "PageRank", but
PageRank has various parameters, and they can't _just_ be PageRank if we want
to mimic proportional representation rather than winner take all.  Thus, we'll
derive our own simple model and hope it is reasonable.

Say we want to find r representatives capable of making representative binary
choices.  Let B = {-1,1}.  If we choose "an issue at random", we get a random
function f : V -> B with correlations between the different f(v) somehow
related to the edges.  Once we have a model for said correlations, we can choose
a subset R of V of size r that maximizes

  Pr(sign(sum(f(R))) = sign(sum(f(V))))

or some such.  For simplicity, assume both r and |V| are odd for now.

Note that it's important to minimize the above sign-based metric, rather than
something less nonlinear like

  E(sum(f(R)) sum(f(V)))

The latter doesn't get proportional representation right: if there are two
uniform voting blocs making up 49% and 51% of the population, we'd pick all
representatives from the 51% bloc.  Bloc: weirdly not a typo.

### Failed attempt 1: Rejection sampling

Here is the simplest model of correlations I can think of for our random function
f : V -> B.  Let X be a completely independent random function: one i.i.d. coin
flip per person.  Let d be the maximum out-degree of a vertex, and choose accept
probabilities aij = probability of acceptance if a degree i vertex is inconsistent
with exactly j of its proxies.  We have ai0 = 1, but the other parameters are
flexible.  Consider acceptance independently for each vertex, and let A be the event
that everyone accepts.  We can now define a random function Y by

  Pr(Y = f) = Pr(X = f | A)

This is well defined because Pr(A) > 0 since Pr(everyone wants +1) > 0.

Hmm, actually that model seems dangerous.  It's fine that Pr(A) is
exponentially low, but it's not fine that Pr(A) may end up weighting different
parts of the graph quite differently.  This isn't necessarily bad, but I don't
understand what properties this weighting would have, and thus don't trust this
model enough to use it.

Actually, maybe it's fine.  Let's implement this with Bayesian inference and
see what the results look like.  PyMC seems like a reasonable engine for doing
the necessary sampling.

Unfortunately, it doesn't seem to work: PyMC fails to converge.  I do not know
why this is, but given the limited development time I should switch to
something else.

### Attempt 2: Lazy voting

Different model: have each person vote with probability p and otherwise
distribute their votes among their proxies.  To handle people with no proxies
and ensure convergence if no one votes, add an extra sink proxy who votes randomly
and give the sink a fraction q of each person's proxy votes.

If we do that, it gives us a random stochastic matrix S, and the voting power
assignment is S^inf ones.  That seems easy enough to implement.

Actually, we can set q = 0 and specify that someone who doesn't find somewhere to
assign their vote simply assigns it via S^inf ones.
