Phage proxy voting experiment
=============================

Graphs and analyses of proxy voting information for Phage.

### Setup

The code is Python 3, but depends on

* [Graphviz](http://graphviz.org): Graph visualization
* [Graphviz python module](https://pypi.python.org/pypi/graphviz): Graphviz interface
* [Numpy](http://numpy.org): Efficient multidimensional arrays
* [Scipy](http://scipy.org): Scientific computation

Use your [virtual environment of choice][1], mine is [virtualenv][2], so I
create a new one using:

```
virtualenv -p $(which python3) env
```

Activate it:

```
source env/bin/activate
```

Then install the dependencies with pip:

```
pip install -r requirements.txt
```

## Model

We have a directed graph _G = (V,E)_ of people with edges from people to people
"they trust in some capacity." Let _n_ = |<em>V</em>| and _m_ = |<em>E</em>|.
There are at least two possible goals:

1. **Vote**: Given a choice set _C_ and an incomplete set of votes _f_ : _V_ -> <em>C</em>,
   determine a choice for the whole group.

2. **Represent**: Find a small number r of people that "best represent" the
   whole graph _G._

The vote goal is easy: one simple approach is to define a Markov chain on _V_
which stops at <em>range</em>(<em>f</em>) and divides equally between outgoing
edges in _V_ - <em>range</em>(<em>f</em>). A vote for someone in _V_ -
<em>range</em>(<em>f</em>) can then be split according to the limit distribution
of the Markov chain starting at their vertex. This can be expressed as a single
eigenvector computation.

The represent goal is harder, since we need a notion of what best
representation means. I believe other people have come up with models for
this, in particular the [Structured Deep Democracy (SD2)][3] and Transitive
Mandatory Multiple Proxy (TMMP) mentioned in Jeremy's slides, but unfortunately
we lack papers with these algorithms. Supposedly they're "PageRank", but
PageRank has various parameters, and they can't _just_ be PageRank if we want
to mimic proportional representation rather than winner take all. Thus, we'll
derive our own simple model and hope it is reasonable.

Say we want to find r representatives capable of making representative binary
choices. Let _B_ = {-1,1}. If we choose "an issue at random", we get a random
function _f_ : _V_ -> _B_ with correlations between the different
<em>f</em>(<em>v</em>) somehow related to the edges. Once we have a model for
said correlations, we can choose a subset _R_ of _V_ of size _r_ that maximizes

<p style="text-align:center;">Pr(<em>sign</em>(<em>sum</em>(<em>f</em>(<em>R</em>))) = <em>sign</em>(<em>sum</em>(<em>f</em>(<em>V</em>))))</p>

or some such. For simplicity, assume both _r_ and |<em>V</em>| are odd for now.

Note that it's important to minimize the above sign-based metric, rather than
something less nonlinear like

<p style="text-align: center;">E(<em>sum</em>(<em>f</em>(<em>R</em>)) <em>sum</em>(<em>f</em>(<em>V</em>)))</style>

The latter doesn't get proportional representation right: if there are two
uniform voting blocs making up 49% and 51% of the population, we'd pick all
representatives from the 51% bloc. Bloc: weirdly not a typo.

### Failed attempt 1: Rejection sampling

Here is the simplest model of correlations I can think of for our random function
_f_ : _V_ -> _B._ Let _X_ be a completely independent random function: one i.i.d. coin
flip per person. Let _d_ be the maximum out-degree of a vertex, and choose accept
probabilities <em>a<sub>ij</sub></em> = probability of acceptance if a degree
_i_ vertex is inconsistent with exactly _j_ of its proxies. We have
<em>a<sub>i0</sub></em> = 1, but the other parameters are flexible. Consider
acceptance independently for each vertex, and let _A_ be the event that everyone
accepts. We can now define a random function _Y_ by

<p style="text-align:center;">Pr(Y = f) = Pr(X = f | A)</p>

This is well defined because Pr(<em>A</em>) > 0 since Pr(everyone wants +1) > 0.

Hmm, actually that model seems dangerous. It's fine that Pr(<em>A</em>) is
exponentially low, but it's not fine that Pr(<em>A</em>) may end up weighting
different parts of the graph quite differently. This isn't necessarily bad, but
I don't understand what properties this weighting would have, and thus don't
trust this model enough to use it.

Actually, maybe it's fine. Let's implement this with Bayesian inference and
see what the results look like. PyMC seems like a reasonable engine for doing
the necessary sampling.

Unfortunately, it doesn't seem to work: PyMC fails to converge. I do not know
why this is, but given the limited development time I should switch to
something else.

### Attempt 2: Lazy voting

Different model: have each person vote with probability _p_ and otherwise
distribute their votes among their proxies. To handle people with no proxies
and ensure convergence if no one votes, add an extra sink proxy who votes randomly
and give the sink a fraction _q_ of each person's proxy votes.

If we do that, it gives us a random stochastic matrix _S_, and the voting power
assignment is <em>S<sup>inf</sup></em> ones. That seems easy enough to implement.

Actually, we can set _q_ = 0 and specify that someone who doesn't find somewhere to
assign their vote simply assigns it via <em>S<sup>inf</sup></em> ones.

   [1]: https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe
   [2]: https://virtualenv.pypa.io/en/stable/
   [3]: http://www.newciv.org/nl/newslog.php/_v45/__show_article/_a000009-000320.htm
