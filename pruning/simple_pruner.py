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
    """
    :return: numpy.array of shape
    """
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


def select_trees(X, y, sample_weight, initial_mx_formula,
                 loss_function=BinomialDevianceLossFunction(),
                 iterations=100,
                 n_candidates=100, n_keptbest=0,
                 learning_rate=0.1, selected_probability=1.,
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
    :param n_candidates: how many candidates we check on each iteration
    :param n_keptbest: how many classifiers saved from previous iteration
    :param learning_rate: shrinkage, float
    :param selected_probability: almost the same as shrinkage, but makes different steps for guessed and wrong steps.
    :param regularization: roughly, it is amount of event of each class added to each leaf. Represents a penalty
    :param verbose: bool, print stats at each step?
    :return: new OBDT list classifier.
    """
    assert n_candidates > n_keptbest, "can't keep more then tested on each stage"

    # collecting information from formula
    old_trees = []
    mn_applier = _matrixnetapplier.MatrixnetClassifier(BytesIO(initial_mx_formula))
    for depth, n_trees, iterator_trees in mn_applier.iterate_trees():
        for tree in iterator_trees:
            old_trees.append(tree)

    features = list(mn_applier.features)

    # taking divisible by 8
    X, y, w = take_divisible(X, y, sample_weight=sample_weight)
    y_signed = 2 * y - 1

    # normalization of weight
    w[y == 0] /= numpy.sum(w[y == 0])
    w[y == 1] /= numpy.sum(w[y == 1])
    w /= numpy.mean(w)

    # fitting loss function
    loss_function = copy.deepcopy(loss_function)
    loss_function.fit(X, y, w)

    new_trees = []
    pred = numpy.zeros(len(X), dtype=float)
    prev_iteration_best_ids = []  # ids of best trees at previous stages
    for iteration in range(iterations):
        selected = numpy.random.choice(len(old_trees), replace=False, size=n_candidates - len(prev_iteration_best_ids))
        selected = numpy.concatenate([selected, prev_iteration_best_ids]).astype(int)
        candidates = [old_trees[i] for i in selected]
        assert len(candidates) == n_candidates

        grads = loss_function.negative_gradient(pred)
        hesss = loss_function.hessian(pred)

        candidate_losses = []
        candidate_new_trees = []
        for tree in candidates:
            leaves = compute_leaves(X, tree)
            new_leaf_values = numpy.bincount(leaves, weights=grads, minlength=2 ** 6)
            new_leaf_values /= numpy.bincount(leaves, weights=hesss, minlength=2 ** 6) + regularization
            new_tree = tree[0], tree[1], new_leaf_values
            # for the sake of speed, here we use approximate step
            effective_leaf_values = new_leaf_values * (learning_rate * selected_probability)
            new_preds = pred + effective_leaf_values[leaves]
            candidate_losses.append(loss_function(new_preds))
            candidate_new_trees.append(new_tree)

        # selecting one with minimal loss
        tree = candidate_new_trees[numpy.argmin(candidate_losses)]
        new_trees.append(tree)
        prev_iteration_best_ids = numpy.take(selected, numpy.argsort(candidate_losses)[1:n_keptbest])

        pred = compute_pessimistic_predictions(y_signed, predictions=pred, tree_predictions=predict_tree(X, tree),
                                               learning_rate=learning_rate, selected_probability=selected_probability)
        if verbose:
            print(iteration, loss_function(pred), roc_auc_score(y, pred, sample_weight=w))

    return utils.OBDTListClassifier(features, trees=new_trees)

