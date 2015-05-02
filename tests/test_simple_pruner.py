from __future__ import division, print_function, absolute_import


__author__ = 'Alex Rogozhnikov'
import numpy
from pruning import simple_pruner, utils


def test_pruner(mx_filename='../data/formula.mx',
                higgs_filename='../data/training.csv'):
    with open(mx_filename, 'rb') as mx:
        formula_mx = mx.read()

    X, y, w = utils.get_higgs_data(higgs_filename)
    X = numpy.array(X, dtype='float32')

    # checking workability, not quality
    simple_pruner.select_trees(X, y, w, initial_mx_formula=formula_mx,
                               iterations=4, learning_rate=0.5, selected_probability=0.5)

