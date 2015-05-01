from __future__ import division, print_function, absolute_import

from six import BytesIO
import numpy

from . import utils
from ._matrixnetapplier import MatrixnetClassifier

"""
About

this file contains alternative implementation of matrixnet applier,
which works event-by-event, vectorization goes over trees.

Implementation uses only numpy.

"""

__author__ = 'Alex Rogozhnikov'


class AlternativeOBDTClassifier(object):
    def __init__(self, features, bias, trees):
        """
        :param features: features used by classifier
        :param bias: float, general shift for all predictions (may be ignored)
        :param trees: list of trees represented as tuples (features, cuts, leaf_values)
        """
        self.trees = trees
        self.n_trees = len(trees)
        self.features = features

        feature_ids = []
        feature_cuts = []
        leaf_values = []
        for features, tree_cuts, tree_leaf_values in trees:
            feature_ids.append(features)
            feature_cuts.append(tree_cuts)
            leaf_values.append(tree_leaf_values)

        # repeating n times the last classifier
        zero_trees_added = self.n_trees - (self.n_trees // 8) * 8
        for _ in range(zero_trees_added):
            feature_ids.append(feature_ids[-1])
            feature_cuts.append(feature_cuts[-1])
            leaf_values.append(leaf_values[-1] * 0.)

        # packing bias to first tree (adding to all leaves)
        leaf_values[0] += bias

        self.n_trees += zero_trees_added

        self.feature_ids = numpy.array(feature_ids, order='F')
        self.feature_cuts = numpy.array(feature_cuts, order='F', dtype='float32')
        self.leaf_values = numpy.array(leaf_values)
        self.depth = self.feature_ids.shape[1]

        assert self.n_trees % 8 == 0, 'Number of trees is not divisible by 8.'

    def decision_function(self, X):
        # taking appropriate columns
        assert len(self.features) == X.shape[1]
        # X = X[self.features]
        X = numpy.array(X, dtype='float32')
        result = numpy.zeros(len(X), dtype=float)
        tree_dummy_indices = numpy.arange(self.n_trees)
        n_trees_by_8 = self.n_trees // 8

        for event_id in range(len(X)):
            event = X[event_id, :]

            leaf_indices = numpy.zeros(n_trees_by_8, dtype='int64')
            for depth in range(self.depth):
                leaf_indices |= (event[self.feature_ids[:, depth]] > self.feature_cuts[:, depth]).view('int64') << depth

            res = self.leaf_values[tree_dummy_indices, leaf_indices.view('int8')].sum()
            result[event_id] = res

        return result

    def decision_function_fast(self, X):
        # taking appropriate columns
        assert len(self.features) == X.shape[1]
        X = numpy.array(X, dtype='float32')
        result = numpy.zeros(len(X), dtype=float)

        feature_ids = self.feature_ids[:, ::-1].T
        feature_cuts = self.feature_cuts[:, ::-1].T
        raveled_leaf_values = numpy.zeros(256 * self.n_trees, dtype='float32')
        leaves_range = numpy.arange(self.n_trees * 2 ** self.depth)
        raveled_leaf_values[leaves_range << 2] = self.leaf_values.ravel()
        tree_shifts = numpy.arange(self.n_trees) * 256

        for event_id in range(len(X)):
            event = X[event_id, :]
            bits = event[feature_ids] > feature_cuts
            leaf_indices = numpy.packbits(bits.view('int8'), axis=0)
            result[event_id] = raveled_leaf_values[tree_shifts + leaf_indices].sum()

        return result


def convert_mx_to_alternative(formula_mx):
    clf = MatrixnetClassifier(BytesIO(formula_mx))
    trees = []
    for depth, ntrees, trees_of_depth in clf.iterate_trees():
        trees.extend(trees_of_depth)

    return AlternativeOBDTClassifier(clf.features, bias=clf.bias, trees=trees)


def test_alternative_classifier(mx_filename='../pruning/formula.mx',
                                higgs_filename='../../../datasets/higgs/training.csv'):
    with open(mx_filename, 'rb') as mx:
        formula_mx = mx.read()

    X, y, w = utils.get_higgs_data(higgs_filename)
    X = numpy.array(X)
    X_part = X[:50000]

    mn = MatrixnetClassifier(BytesIO(formula_mx))
    mn_predictions = mn.apply(X_part)

    new_mn = convert_mx_to_alternative(formula_mx)
    new_mn_predictions = new_mn.decision_function(X_part)

    assert numpy.allclose(mn_predictions, new_mn_predictions)
