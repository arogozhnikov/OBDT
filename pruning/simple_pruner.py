from __future__ import division, print_function, absolute_import


__author__ = 'Alex Rogozhnikov'

import numpy
import copy
from sklearn.metrics import roc_auc_score
from six import BytesIO
from hep_ml.losses import BinomialDevianceLossFunction
from . import utils

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


def take_divisible(X, y, sample_weight):
    train_length = (len(X) // 8) * 8
    return X[:train_length], y[:train_length], sample_weight[:train_length]


def select_trees(X, y, sample_weight, initial_mx_formula,
                 loss_function=BinomialDevianceLossFunction(),
                 iterations=100, n_candidates=100, learning_rate=0.1, regularization=10.):
    # collecting information from formula
    old_trees = []
    mn_applier = _matrixnetapplier.MatrixnetClassifier(BytesIO(initial_mx_formula))
    for depth, n_trees, iterator_trees in mn_applier.iterate_trees():
        for tree in iterator_trees:
            old_trees.append(tree)

    features = list(mn_applier.features)

    # taking divisible by 8
    w = sample_weight  # for shortness
    X, y, w = take_divisible(X, y, sample_weight=w)

    # normalization of weight and regularization
    w[y == 0] /= numpy.sum(w[y == 0])
    w[y == 1] /= numpy.sum(w[y == 1])
    w /= numpy.mean(w)

    # fitting loss function
    loss_function = copy.deepcopy(loss_function)
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
    new_formula_mx = utils.convert_list_to_mx(new_trees, initial_mx_formula)
    # function returns features used in formula and new formula_mx
    return features, new_formula_mx

