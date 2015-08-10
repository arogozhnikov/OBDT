from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

import numpy
import copy
from six import BytesIO
from .utils import take_divisible, compute_leaves, predict_tree, OBDTListClassifier

try:
    # To work we need both
    from pruning import _matrixnetapplier
except:
    from rep.estimators import _matrixnetapplier


def compute_pessimistic_predictions(y_signed, predictions, tree_predictions, learning_rate, selected_probability):
    """
    Usually, one uses formula in GB:
    predictions += learning_rate * tree_predictions,
    but in pessimistic probabilistic setting (my term), the formula is
    predictions += - y_i * log( (1 - sp) + sp * exp( - y_i * learning_rate * tree_predictions) )
    where sp = selected_probability, probability with which the final classifier is included in GB,
          y_i are +1 or -1.

    :param y_signed: signed labels
    :param tree_predictions: predictions of new tree (without any normalization)
    :param learning_rate: float, shrinkage
    :param selected_probability: probability that particular tree is activated
    """
    sp = selected_probability
    return predictions - y_signed * numpy.log((1. - sp) + sp * numpy.exp(-y_signed * learning_rate * tree_predictions))


def canonical_pruning(X, y, sample_weight, initial_mx_formula,
                      loss_function,
                      iterations=100,
                      n_candidates=100,
                      n_kept_best=0,
                      learning_rate=0.1,
                      regularization=10.,
                      verbose=False):
    """
    Represents basic pruning algorithm, which greedily adds
    :param X: data
    :param y: binary labels (0 and 1)
    :param sample_weight: weights
    :param initial_mx_formula:
    :param loss_function: loss function (following hep_ml convention for losses)
    :param iterations: int, how many estimators we shall leave
    :param n_candidates: how many candidate_trees we check on each iteration
    :param n_kept_best: how many classifiers saved from previous iteration
    :param learning_rate: shrinkage, float
    :param regularization: roughly, it is amount of event of each class added to each leaf. Represents a penalty
    :param verbose: bool, print stats at each step?
    :return: new OBDT list classifier.
    """
    assert n_candidates > n_kept_best, "can't keep more then tested on each stage"
    X = numpy.array(X, dtype='float32', order='F')

    # collecting information from formula
    old_trees = []
    mn_applier = _matrixnetapplier.MatrixnetClassifier(BytesIO(initial_mx_formula))
    for depth, n_trees, iterator_trees in mn_applier.iterate_trees():
        for tree in iterator_trees:
            old_trees.append(tree)

    features = list(mn_applier.features)

    # taking divisible by 8
    X, y, w = take_divisible(X, y, sample_weight=sample_weight)

    # fitting loss function
    loss_function = copy.deepcopy(loss_function)
    loss_function.fit(X, y, w)

    new_trees = []
    pred = numpy.zeros(len(X), dtype=float)
    prev_iteration_best_ids = []  # ids of best trees at previous stages
    for iteration in range(iterations):
        new_candidate_trees = numpy.random.choice(len(old_trees), replace=False,
                                                  size=n_candidates - len(prev_iteration_best_ids))
        candidate_trees_indices = numpy.concatenate([new_candidate_trees, prev_iteration_best_ids]).astype(int)
        candidate_trees = [old_trees[i] for i in candidate_trees_indices]
        assert len(candidate_trees) == n_candidates

        grads = loss_function.negative_gradient(pred)
        hesss = loss_function.hessian(pred)

        candidate_losses = []
        candidate_new_trees = []
        for tree in candidate_trees:
            leaves = compute_leaves(X, tree)
            new_leaf_values = numpy.bincount(leaves, weights=grads, minlength=2 ** 6)
            new_leaf_values /= numpy.bincount(leaves, weights=hesss, minlength=2 ** 6) + regularization
            new_leaf_values *= learning_rate

            new_preds = pred + new_leaf_values[leaves]
            candidate_losses.append(loss_function(new_preds))
            new_tree = tree[0], tree[1], new_leaf_values
            candidate_new_trees.append(new_tree)

        # selecting one with minimal loss
        tree = candidate_new_trees[numpy.argmin(candidate_losses)]
        new_trees.append(tree)
        pred += predict_tree(X, tree)

        prev_iteration_best_ids = numpy.take(candidate_trees_indices, numpy.argsort(candidate_losses)[1:n_kept_best])

        if verbose:
            print(iteration, loss_function(pred))

    return OBDTListClassifier(features, trees=new_trees)
