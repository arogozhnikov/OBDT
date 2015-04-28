from __future__ import division, print_function, absolute_import


__author__ = 'Alex Rogozhnikov'

import numpy
import struct
import copy
from sklearn.metrics import roc_auc_score
from six import BytesIO
from collections import defaultdict, OrderedDict
from hep_ml.losses import BinomialDevianceLossFunction, AdaLossFunction, CompositeLossFunction

try:
    # To work both 
    from pruning import _matrixnetapplier
except:
    from rep.estimators import _matrixnetapplier


def compute_leaves(X, tree):
    assert len(X) % 8 == 0, 'for fast computations need len(X) be divisible by 8'
    leaf_indices = numpy.zeros(len(X) // 8, dtype='int64')
    for tree_level, (feature, cut) in enumerate(zip(tree[0], tree[1])):
        leaf_indices |= (X[:, feature] > cut).view('int64') << tree_level
    return leaf_indices.view('int8')


def predict_tree(X, tree):
    leaf_values = tree[2]
    return leaf_values[compute_leaves(X, tree)]


def predict_trees(X, trees):
    result = numpy.zeros(len(X), dtype=float)
    for tree in trees:
        result += predict_tree(X, tree)
    return result


def select_trees(X, y, sample_weight, initial_classifier,
                 loss_function=BinomialDevianceLossFunction(),
                 iterations=100,
                 n_candidates=100, learning_rate=0.1, regularization=3.):
    w = sample_weight  # for shortness
    features = initial_classifier.features
    # loss_function = copy.deepcopy(loss_function)

    old_trees = []
    mn_applier = _matrixnetapplier.MatrixnetClassifier(BytesIO(initial_classifier.formula_mx))
    for depth, n_trees, iterator_trees in mn_applier.iterate_trees():
        for tree in iterator_trees:
            old_trees.append(tree)

    # normalization of weight and regularization
    w[y == 0] /= numpy.sum(w[y == 0])
    w[y == 1] /= numpy.sum(w[y == 1])
    w /= numpy.mean(w)

    # fitting loss function
    loss_function.fit(X, y, w)

    new_trees = []
    pred = numpy.zeros(len(X), dtype=float)
    for iteration in range(iterations):
        selected = numpy.random.choice(len(old_trees), replace=False, size=n_candidates)
        candidates = [old_trees[i] for i in selected]

        grads = loss_function.negative_gradient(pred)
        hesss = loss_function.hessian(pred)

        candidate_losses = []
        candidate_new_trees = []
        for tree in candidates:
            leaves = compute_leaves(X, tree)
            new_leaf_values = numpy.bincount(leaves, weights=grads, minlength=2 ** 6) * learning_rate
            new_leaf_values /= numpy.bincount(leaves, weights=hesss, minlength=2 ** 6) + regularization
            new_tree = tree[0], tree[1], new_leaf_values
            new_preds = pred + predict_tree(X, new_tree)
            candidate_losses.append(loss_function(new_preds))
            candidate_new_trees.append(new_tree)

        # selecting one with minimal loss
        tree = candidate_new_trees[numpy.argmin(candidate_losses)]
        new_trees.append(tree)
        pred += predict_tree(X, tree)
        print(iteration, loss_function(pred), roc_auc_score(y, pred, sample_weight=w))

    # return ShortenedClassifier(features, new_trees)
    new_formula_mx = convert_trees_to_mx(new_trees, initial_classifier.formula_mx)
    # function returns features used in formula and new formula_mx
    return features, new_formula_mx, _matrixnetapplier.MatrixnetClassifier(BytesIO(new_formula_mx))


# class ShortenedClassifier(Classifier):
# def __init__(self, features, trees):
#         Classifier.__init__(self, features)
#         self.trees = trees
#
#     def fit(self, X, y, sample_weight=None):
#         pass
#
#     def staged_predict_proba(self, X):
#         X = self._get_train_features(X).values
#         result = numpy.zeros(len(X))
#         for features, cuts, leaf_values in self.trees:
#             indices = numpy.zeros(len(X), dtype=int)
#             for depth, (feature, cut) in enumerate(zip(features, cuts)):
#                 indices += (X[:, feature] > cut) << depth
#             result += leaf_values[indices]
#             yield utils.score_to_proba(result)
#
#     def predict_proba(self, X):
#         for p in self.staged_predict_proba(X):
#             pass
#         return p


def convert_trees_to_mx(trees, initial_mx):
    self = OrderedDict()
    # collecting possible thresholds
    n_features = max([max(features) for features, _, _ in trees]) + 1
    thresholds = [[]] * n_features
    for features, cuts, _ in trees:
        for feature, cut in zip(features, cuts):
            thresholds[feature].append(cut)
    thresholds_lengths = []

    # sorting, leaving only unique thresholds
    for feature, cuts in enumerate(thresholds):
        thresholds[feature] = numpy.unique(cuts)
        thresholds_lengths.append(len(thresholds[feature]))

    binary_feature_shifts = [0] + list(numpy.cumsum(thresholds_lengths))

    binary_feature_ids = []
    tree_table = []

    for features, cuts, leaf_values in trees:
        for feature, cut in zip(features, cuts):
            binary_feature_id = numpy.searchsorted(thresholds[feature], cut)
            binary_feature_id += binary_feature_shifts[feature]
            binary_feature_ids.append(binary_feature_id)
        tree_table.extend(leaf_values)

    formula_stream = BytesIO(initial_mx)
    result = BytesIO()

    self.features = []  # list of strings
    self.bins = []

    bytes = formula_stream.read(4)
    features_quantity = struct.unpack('i', bytes)[0]
    result.write(struct.pack('i', features_quantity))

    for index in range(0, features_quantity):
        bytes = formula_stream.read(4)
        factor_length = struct.unpack('i', bytes)[0]
        result.write(struct.pack('i', factor_length))

        self.features.append(formula_stream.read(factor_length))
        result.write(self.features[-1])

    _ = formula_stream.read(4)  # skip formula length
    part_result = ""

    used_features_quantity = struct.unpack('I', formula_stream.read(4))[0]
    part_result += struct.pack('I', used_features_quantity)
    assert used_features_quantity == len(thresholds)

    bins_quantities = struct.unpack(
        'I' * used_features_quantity,
        formula_stream.read(4 * used_features_quantity)
    )
    # putting new number of cuts
    part_result += struct.pack('I' * used_features_quantity, *thresholds_lengths)

    self.bins_total = struct.unpack('I', formula_stream.read(4))[0]
    part_result += struct.pack('I', sum(thresholds_lengths))

    for index in range(used_features_quantity):
        self.bins.append(
            struct.unpack(
                'f' * bins_quantities[index],
                formula_stream.read(4 * bins_quantities[index])
            )
        )
        part_result += struct.pack('f' * thresholds_lengths[index], *thresholds[index])

    _ = formula_stream.read(4)  # skip classes_count == 0
    part_result += struct.pack('I', 0)

    nf_counts_len = struct.unpack('I', formula_stream.read(4))[0]
    # leaving the same info on max depth
    part_result += struct.pack('I', nf_counts_len)

    self.nf_counts = struct.unpack('I' * nf_counts_len,
                                   formula_stream.read(4 * nf_counts_len)
    )
    new_nf_counts = numpy.zeros(nf_counts_len, dtype=int)
    new_nf_counts[5] = len(trees)
    part_result += struct.pack('I' * nf_counts_len, *new_nf_counts)

    ids_len = struct.unpack('I', formula_stream.read(4))[0]
    part_result += struct.pack('I', len(binary_feature_ids))

    self.feature_ids = struct.unpack(
        'I' * ids_len,
        formula_stream.read(4 * ids_len)
    )
    # writing new binary features
    part_result += struct.pack('I' * len(binary_feature_ids), *binary_feature_ids)

    self.feature_ids = numpy.array(self.feature_ids)
    tree_table_len = struct.unpack('I', formula_stream.read(4))[0]
    part_result += struct.pack('I', len(tree_table))

    self.tree_table = struct.unpack(
        'i' * tree_table_len,
        formula_stream.read(4 * tree_table_len)
    )

    # normalization here:
    new_delta_mult = 10000000.

    tree_table = new_delta_mult * numpy.array(tree_table)

    part_result += struct.pack('i' * len(tree_table), *tree_table.astype(int))
    self.tree_table = numpy.array(self.tree_table)

    self.bias = struct.unpack('d', formula_stream.read(8))[0]
    part_result += struct.pack('d', 0.)

    self.delta_mult = struct.unpack('d', formula_stream.read(8))[0]
    part_result += struct.pack('d', new_delta_mult)
    result.write(struct.pack('i', len(part_result)))
    result.write(part_result)
    return result.getvalue()