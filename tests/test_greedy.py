from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'
import numpy
from hep_ml.losses import CompositeLossFunction, MSELossFunction
from pruning import greedy, utils


def test_pruner(mx_filename='../data/formula.mx', higgs_filename='../data/training.csv'):

    with open(mx_filename, 'rb') as mx:
        formula_mx = mx.read()

    X, y, w = utils.get_higgs_data(higgs_filename)
    X = numpy.array(X, dtype='float32')

    pruner = greedy.GreedyPruner(loss_function=CompositeLossFunction(), iterations=5, n_kept_best=0)
    pruner.prune(formula_mx, X, y, w, verbose=True)

    pruner = greedy.GreedyPruner(loss_function=CompositeLossFunction(), iterations=5, n_kept_best=5)
    pruner.prune(formula_mx, X, y, w, verbose=True)

    pruner = greedy.GreedyPruner(loss_function=MSELossFunction(), iterations=5, n_kept_best=5)
    pruner.prune(formula_mx, X, y, w, verbose=True)


def test_nesterov_pruner(mx_filename='../data/formula.mx', higgs_filename='../data/training.csv', iterations=30):

    with open(mx_filename, 'rb') as mx:
        formula_mx = mx.read()

    X, y, w = utils.get_higgs_data(higgs_filename)
    X = numpy.array(X, dtype='float32')

    pruner = greedy.NesterovPruner(loss_function=MSELossFunction(), iterations=iterations, n_nesterov_steps=0)
    pruner.prune(formula_mx, X, y, w, verbose=True)

    pruner = greedy.NesterovPruner(loss_function=MSELossFunction(), iterations=iterations, n_nesterov_steps=1)
    pruner.prune(formula_mx, X, y, w, verbose=True)

    pruner = greedy.NesterovPruner(loss_function=MSELossFunction(), iterations=iterations, n_nesterov_steps=2)
    pruner.prune(formula_mx, X, y, w, verbose=True)

    assert 0 == 1
