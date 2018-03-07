"""
Produce the graphs.
"""
from __future__ import division, print_function
from graphviz import Digraph
import numpy
import scipy.sparse
import optparse
import tempfile
import shutil
import csv
import re
import os


def sanitize(info):
    return re.sub(' - .*', '', info).strip()


# TODO: Some people have duplicate entries and should be pruned accordingly
def fields(entry, header, **kwds):
    def resolve(v):
        if isinstance(v, dict):
            (k, i), = list(v.items())
            return resolve(kwds[k]) + i
        try:
            return header.index(v)
        except ValueError:
            return -1

    d = {}
    for k, vs in list(kwds.items()):
        if isinstance(vs, tuple):
            v, v2 = vs
        else:
            v, v2 = vs, None
        i = resolve(v)
        if i >= 0:
            x = entry[i] if i < len(entry) else None
            if v2 and (not x or x == 'Name does not appear in this list'):
                i = resolve(v2)
                if i >= 0:
                    x = entry[i] if i < len(entry) else None
            x = sanitize(x) if x else None
            d[k] = x
    return d


def valid(d):
    for v in list(d.values()):
        if not v:
            return False
    return True


fs = {
    'name': ('Who are you?',
             "If your name isn't in the list, please enter it here.    "),
    'inviter': ('Who invited you to Phage?    If you are Original Phage, choose yourself. ',
                "If the person who invited you isn't in the menu above, please put their name here"),
    'resolver1': ('First ranked choice person for conflict resolution',
                  "1st choice conflict resolver: If the person who you chose isn't in the menu above, "
                  "please put their name here"),
    'resolver2': ('Second rank choice person for conflict resolution',
                  "2nd choice conflict resolver: If the person who you chose isn't in the menu above, "
                  "please put their name here"),
    'cook1': ('A Phageling you trust to enjoy cooking with!',
              "Cooking friend 1: If the person who you chose isn't in the menu above, please put their name here"),
    'cook2': ('A second Phageling you trust to enjoy cooking with!',
              "Cooking friend 2: If the person who you chose isn't in the menu above, please put their name here"),
    'cook3': ('A third Phageling you trust to enjoy cooking with!',
              "Cooking friend 3: If the person who you chose isn't in the menu above, please put their name here"),
    'sage1': ('A Phageling you would approach for life advice:',
              "Sage friend 1: If the person who you chose isn't in the menu above, please put their name here"),
    'sage2': ('A second Phageling you would approach for life advice:',
              "Sage friend 2: If the person who you chose isn't in the menu above, please put their name here"),
    'noob1': 'Who would you trust to (a) interview the phageling who has expressed concerns, (b) objectively listen to'
             ' both sides of the story,    (c) weigh the facts, and (d) cast a vote on your behalf.',
    'noob2': {'noob1': + 1},
    'noob3': {'noob1': + 2},
}


def graph(data, id, edges):
    """Extract a graph from data, given as a dictionary node -> out_edges.
    The graph is closed by introducing vertices with no outgoing edges if necessary."""
    G = {}

    def add(x):
        if x not in G:
            G[x] = []
        return x
    for d in data:
        if d[id]:
            x = add(d[id])
            for e in edges:
                if d[e]:
                    y = add(d[e])
                    G[x].append(y)
    return G


def vertices(G):
    return sorted(G.keys())


def rejection_sample(G, samples):
    """Build a probability model for voting given a proxy graph"""
    import pymc

    V = vertices(G)
    print('vertices = %d' % len(V))
    deg = numpy.asarray([len(G[v]) for v in V])
    print('max deg = %d' % deg.max())
    E = numpy.empty((len(V), deg.max()), dtype=int)
    for i, v in enumerate(V):
        es = [V.index(x) for x in G[v]]
        E[i, :len(es)] = es
        E[i, len(es):] = i  # Point remaining edges at ourselves for sentinel purposes

    if 0:
        print('V = %s' % V)
        print('deg = %s' % deg)
        print('E = \n%s' % E)

    # Each person's vote is an iid coin flip
    vote = pymc.Bernoulli('vote', p=0.5, size=len(V))

    # Each person's acceptance probability is a function of their
    # vote and the votes of their proxies (outgoing edges).    More
    # specifically, A[i,j] is the probability of acceptance for
    # a voter with i proxies, j of whom ver disagrees with.
    # For now, we pick A = fixed < 1 if a majority disagree, otherwise A = 1.
    i = numpy.arange(deg.max() + 1)
    A = 1 - .5 * (i[:, None] < 2 * i)
    print("A =\n%s" % A)

    # Acceptance *probability*, which is a deterministic function of votes
    @pymc.deterministic
    def accept_p(v=vote):
        return A[deg, sum(v[E] != v[:, None], axis=-1)]

    # Acceptance *events*, which are Bernoulli
    accept = pymc.Bernoulli('accept', size=len(V), p=accept_p, value=numpy.ones(len(V)), observed=True)

    # Sample model
    M = pymc.MCMC((vote, accept_p, accept))
    thin = 2
    burn = 1
    M.sample(iter=((burn + thin) * samples), burn=(burn * samples), thin=thin)
    return M.trace('vote')[:]


def green_to_red(G, x):
    V = vertices(G)
    assert x.shape == (len(V),)
    xmin = x.min()
    xmax = x.max()
    print('score range = %s %s' % (xmin, xmax))

    def scheme(x):
        x = (x - xmin) / (xmax - xmin)
        r = '%02x' % (255 * (1 - x))
        g = '%02x' % (255 * x)
        return '#%s%s00' % (r, g)
    return dict((v, scheme(x[i])) for i, v in enumerate(V))


def rejection_color(G, vote):
    print('votes = %s' % (vote.shape,))
    if 0:
        counts = numpy.zeros((2, 2, 2))
        for v in vote:
            counts[tuple(v)] += 1
        print('counts = \n%s' % counts)
    maj = numpy.mean(vote, axis=1) > 0.5
    score = numpy.asarray([numpy.corrcoef(maj, v) for v in vote.T])
    score = score[:, 0, 1]
    print('score shape = %s' % (score.shape,))
    return green_to_red(G, score)


def greedy_subset(energy, n, odd=False, r=5):
    """
    Choose a subset R of size r that approximately minimizes energy, using greedy plus 1-opt
    """

    def tweaked(R):
        if len(set(R)) < len(R):
            return numpy.inf
        return energy(R)

    def best(Rs):
        Rs = numpy.asarray(list(Rs))
        R = Rs[numpy.argmin(list(map(tweaked, Rs)))]
        return numpy.sort(R)

    K = range(n)
    if odd:
        # Find the best singleton
        R = best((i,) for i in K)
        # Greedily add two elements at a time
        while len(R) < r:
            R = best(tuple(R) + (i, j) for i in K for j in K if i < j)
            print('R = %s, energy = %g' % (R, energy(R)))
    else:
        R = []
        # Greedily add one element at a time
        while len(R) < r:
            R = best(tuple(R)+(i,) for i in K)

    # Search for 1-opt
    while 1:
        def replace(R, i, j):
            R = list(R)
            R[i] = j
            return R
        R2 = best(replace(R, i, j) for i in range(len(R)) for j in K)
        if all(R == R2):
            break
        R = R2
        print('R = %s, energy = %g' % (R, energy(R)))
    assert len(R) == r
    print('R = %s, energy = %g' % (R, energy(R)))
    return R


def rejection_represent(G, vote, r=5):
    """
    Choose r representatives that best reproduce the votes of the majority
    """
    assert (r % 2) == 1, "For now, assume an odd number of representatives"
    V = vertices(G)
    maj = numpy.mean(vote, axis=1) > 0.5  # Majority vote

    # We want a subset R of size r that maximizes
    #     sum((mean(vote[:,R]) > 1/2) == maj)
    # Doing this really well is hard, so we'll start with greedy plus 1-opt
    def energy(R):
        assert len(R) % 2 == 1
        if len(set(R)) < len(R):
            return -numpy.inf
        return -sum((numpy.mean(vote[:, numpy.asarray(R)], axis=1) > 0.5) == maj)
    R = greedy_subset(energy, n=len(V), r=r, odd=1)
    R = [V[i] for i in R]
    print('R = %s' % (R))
    return R


def lazy_vote(G, r=5, p=0.5, samples=1000, iters=1000, seed=8121, simple=False):
    V = vertices(G)
    deg = numpy.asarray([len(G[v]) for v in V])
    print('max deg = %d' % deg.max())

    # Compute a bunch of vote distributions
    dists = []
    numpy.random.seed(seed)
    for s in range(samples):
        # Choose randomly whether each person will vote directly
        active = numpy.random.uniform(size=len(V)) < p

        # Generate sparse transition matrix, very slowly
        T = numpy.eye(len(V))
        for i, v in enumerate(V):
            if not active[i]:
                es = list(set(V.index(x) for x in G[v]))
                if len(es):
                    T[i, i] = 0
                    T[es, i] = 1/len(es)
        T = scipy.sparse.csr_matrix(T)

        # Partially converge to a vote distribution
        x = numpy.ones(len(V))/len(V)
        for i in range(iters):
            x = T*x
        assert numpy.allclose(sum(x), 1)
        dists.append(x)
    dists = numpy.asarray(dists)

    # Color based on average amount of voting power
    score = numpy.mean(dists, axis=0)
    if 0:  # Transform to be a straight line
        score[numpy.argsort(score)] = numpy.arange(len(score))
    color = green_to_red(G, score)

    # In simple mode, just return the top scoring folk
    if simple:
        R = numpy.argsort(-score)[:r]
        if 1:
            print('The top few scores:')
            for i in R:
                print('    %.5g : %s' % (score[i], V[i]))
    else:
        # Given a vote distribution x with sum(x) = 1 and individual preferences v_i in {0,1},
        # we vote yes as a whole if sum(x_i*v_i) > 1/2. Ug, that's basically a knapsack
        # problem. I still want to do the majority cutoff method rather than simple
        # correlation, so let's apply random sampling again.
        votes = numpy.random.randint(2, size=(len(V) * samples)).reshape(-1, len(V))

        def energy(R, verbose=False):
            x = dists * votes
            if 0:  # Representatives vote according to their assembled power
                y = x[:, R]
                y /= sum(y, axis=-1)[:, None]
            else:  # Representatives vote equally
                y = votes[:, R] / len(R)
            similar = sum((sum(x, axis=-1) > 0.5) == (sum(y, axis=-1) > 0.5))
            return -similar
        R = greedy_subset(energy, n=len(V), r=r)

    R = [V[i] for i in R]
    return color, R


def draw(G, file, label=None, color=None, reps=None):
    print('writing %s' % (file))

    GV = Digraph(comment=file)
    nodes = {}

    def node(i):
        if i not in nodes:
            n = str(len(nodes))
            nodes[i] = n
            ks = {}
            ks['label'] = None if label is None else label(i)
            if color is not None:
                ks['style'] = 'filled'
                ks['fillcolor'] = color[i]
            if reps is not None and i in reps:
                ks['penwidth'] = '10'
            GV.node(n, **ks)
        return nodes[i]
    for i in vertices(G):
        x = node(i)
        for e in G[i]:
            GV.edge(x, node(e))

    # Render to a temporary directory, then copy the file back
    dir = tempfile.mkdtemp(prefix='phage-proxy')
    try:
        GV.render(filename=os.path.join(dir, 'graph'))
        os.rename(os.path.join(dir, 'graph.pdf'), file)
    finally:
        shutil.rmtree(dir)


def main():
    # Types of graphs to plot
    graphs = {
        'resolve': ('resolver1', 'resolver2'),
        'sage': ('sage1', 'sage2'),
        'cook': ('cook1', 'cook2', 'cook3'),
        'noob': ('noob1', 'noob2', 'noob3'),
        'invite': ('inviter',)
    }

    # Commnand line arguments
    usage = 'usage: %prog <options>'
    P = optparse.OptionParser(usage)
    P.add_option('-i', '--input', default='votes.csv', help='Input csv file from Wufoo')
    P.add_option('-o', '--output', default='', help='Output pdf file')
    P.add_option('-g', '--graph', default='resolve', choices=list(graphs)+['all'], help='Type of graph to analyze')
    P.add_option('-a', '--analyze', action='store_true', help='If true, perform voting analysis. If false, just draw.')
    P.add_option('-r', '--reps', default=5, type=int, help='Number of representatives to compute')
    P.add_option('-p', '--vote-prob', default=.4, type=float, help='Probability that each person votes')
    P.add_option('-n', '--samples', default=2000, type=int, help='Number of samples (more is more accurate and slower)')
    P.add_option('-s', '--seed', default=1827322, type=int, help='Random seed for sampling')
    P.add_option('-l', '--label', default='none', choices=['none', 'name'], help='Node label type')
    P.add_option('--simple', action='store_true', help='Skip representative analysis and just compute top scorers')
    P.add_option('--partial-label', action='store_true', help='Only label representatives')
    O, args = P.parse_args()

    if args:
        P.error('No arguments expected')
    if O.graph == 'all':
        assert not O.output, 'Specifying an output file with --graph all with cause files to be overwritten'
    if O.output:
        assert O.output.exdswith('.pdf'), '--output=%s should be a .pdf' % O.output

    print('reading input from %s' % O.input)
    lines = tuple(csv.reader(open(O.input, 'rb')))
    header = lines[0]
    if 0:
        print('fields:')
        for h in header:
            print('    '+repr(h))
        print()
    data = [fields(e, header, **fs) for e in lines[1:]]
    data = [d for d in data if d is not None]
    print('count = %d' % len(data))
    print('valid = %d' % len(list(filter(valid, data))))
    # data = filter(valid,data)
    print()

    def name(email):
        return re.sub('-.*', '', email).strip()

    def none(email):
        return ''

    label = {'name': name, 'none': none}[O.label]

    for g in list(graphs) if O.graph == 'all' else [O.graph]:
        print('analyzing %s' % g)
        G = graph(data, 'name', graphs[g])

        if O.analyze and g != 'invite':
            color, reps = lazy_vote(G, r=O.reps, p=O.vote_prob, samples=O.samples, seed=O.seed, simple=O.simple)
        else:
            color, reps = None, None

        def email_param(email):
            label(email) if email in reps else ''

        label_param = label
        if O.partial_label:
            label_param = email_param

        output = O.output or g+'.pdf'
        draw(G, file=output, label=label_param, color=color, reps=reps)
        print()


if __name__ == '__main__':
    main()
