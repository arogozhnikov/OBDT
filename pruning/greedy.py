from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

import numpy
import copy
from .utils import take_divisible, compute_leaves, predict_tree, OBDTListClassifier
from six import BytesIO

try:
    # To work anywhere we need both
    from pruning import _matrixnetapplier
except:
    from rep.estimators import _matrixnetapplier


class GreedyPruner(object):
    def __init__(self, loss_function,
                 iterations=100,
                 n_candidates=100,
                 n_kept_best=0,
                 learning_rate=0.1,
                 regularization=10.,
                 verbose=False):
        """
        Represents basic pruning algorithm, which greedily adds trees
        :param loss_function: loss function (following hep_ml convention for losses)
        :param iterations: int, how many estimators we shall leave
        :param n_candidates: how many candidate_trees we check on each iteration
        :param n_kept_best: how many classifiers saved from previous iteration
        :param learning_rate: shrinkage, float
        :param regularization: roughly, it is amount of event of each class added to each leaf. Represents a penalty
        :return: new OBDT list classifier.
        """
        self.loss_function = loss_function
        self.iterations = iterations
        self.n_candidates = n_candidates
        self.n_kept_best = n_kept_best
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.verbose = verbose

    @staticmethod
    def _get_old_trees(mx_formula):
        # collecting information from formula
        old_trees = []
        mn_applier = _matrixnetapplier.MatrixnetClassifier(BytesIO(mx_formula))
        for depth, n_trees, iterator_trees in mn_applier.iterate_trees():
            for tree in iterator_trees:
                old_trees.append(tree)
        features = list(mn_applier.features)
        return old_trees, features

    def _compute_optimal_leaf_values(self, X, pred, tree, loss_function, regularization):
        grads = loss_function.negative_gradient(pred)
        hesss = loss_function.hessian(pred)
        leaf_indices = compute_leaves(X, tree)

        leaf_values = numpy.bincount(leaf_indices, weights=grads, minlength=2 ** 6)
        leaf_values /= numpy.bincount(leaf_indices, weights=hesss, minlength=2 ** 6) + regularization
        return leaf_indices, leaf_values

    def _compute_const(self, X, pred, loss_function, iterations=10):
        step = 0.
        for _ in range(iterations):
            temp_pred = pred + step
            grads = loss_function.negative_gradient(temp_pred)
            hesss = loss_function.hessian(temp_pred)
            step += 0.5 * numpy.sum(grads) / numpy.sum(hesss)
        depth = 6
        initial_tree = numpy.zeros(depth, dtype=int), \
                   numpy.zeros(depth, dtype=float), \
                   numpy.zeros(2 ** depth, dtype=float) + step
        return step, initial_tree

    def _test_trees(self, X, loss_function, pred, new_trees, candidate_trees):
        """
        Testing decision trees.
        :param X: dataset
        :param loss_function: fitted loss function
        :param pred: current predictions
        :param new_trees: trees already selected
        :param candidate_trees: trees to test at this stage
        :return: loss_values, trees_with_new_leaf_values
        """

        grads = loss_function.negative_gradient(pred)
        hesss = loss_function.hessian(pred)

        candidate_losses = []
        candidate_new_trees = []
        for tree in candidate_trees:
            leaves = compute_leaves(X, tree)
            new_leaf_values = numpy.bincount(leaves, weights=grads, minlength=2 ** 6)
            new_leaf_values /= numpy.bincount(leaves, weights=hesss, minlength=2 ** 6) + self.regularization
            new_leaf_values *= self.learning_rate

            new_preds = pred + new_leaf_values[leaves]
            candidate_losses.append(loss_function(new_preds))
            new_tree = tree[0], tree[1], new_leaf_values
            candidate_new_trees.append(new_tree)

        return candidate_losses, candidate_new_trees

    def prune(self, initial_mx_formula, X, y, sample_weight, verbose=False):
        """
        :param initial_mx_formula:
        :param X: data
        :param y: binary labels (0 and 1)
        :param sample_weight: weights
        :param verbose: bool, print stats at each step?
        """
        assert self.n_candidates > self.n_kept_best, "can't keep more then tested on each stage"
        X = numpy.array(X, dtype='float32', order='F')
        # taking divisible by 8
        X, y, w = take_divisible(X, y, sample_weight=sample_weight)

        # fitting loss function
        loss_function = copy.deepcopy(self.loss_function)
        loss_function.fit(X, y, w)

        # collecting old trees
        old_trees, features = self._get_old_trees(mx_formula=initial_mx_formula)

        # initial step
        pred = numpy.zeros(len(X), dtype=float)
        step, initial_tree = self._compute_const(X, pred, loss_function=loss_function, iterations=10)
        pred += step
        new_trees = [initial_tree]

        trees_ids = []  # ids of best trees at previous stages
        for iteration in range(self.iterations):
            _new_trees_ids = numpy.random.choice(len(old_trees), replace=False,
                                                 size=self.n_candidates - len(trees_ids))
            trees_ids = numpy.concatenate([_new_trees_ids, trees_ids]).astype(int)

            candidate_trees = [old_trees[i] for i in trees_ids]
            assert len(candidate_trees) == self.n_candidates

            candidate_losses, candidate_new_trees = self._test_trees(X, loss_function=loss_function,
                                                                     pred=pred, candidate_trees=candidate_trees,
                                                                     new_trees=new_trees)

            # selecting one with minimal loss
            tree = candidate_new_trees[numpy.argmin(candidate_losses)]
            new_trees.append(tree)
            pred += predict_tree(X, tree)
            trees_ids = numpy.take(trees_ids, numpy.argsort(candidate_losses)[1:self.n_kept_best])

            if verbose:
                print(iteration, loss_function(pred))

        return OBDTListClassifier(features, trees=new_trees)


class NesterovPruner(GreedyPruner):
    def __init__(self, loss_function, n_nesterov_steps=4,
                 iterations=100,
                 n_candidates=100,
                 n_kept_best=0,
                 learning_rate=0.1,
                 regularization=10.,
                 verbose=False):
        """
        :param n_nesterov_steps: number of steps checked backwards.
        We assume that for them optimal steps were done (though not done actually).
        """
        super(NesterovPruner, self).__init__(loss_function, iterations=iterations, n_candidates=n_candidates,
                                             n_kept_best=n_kept_best, learning_rate=learning_rate,
                                             regularization=regularization, verbose=verbose)
        self.n_nesterov_steps = n_nesterov_steps

    def _test_trees(self, X, loss_function, pred, new_trees, candidate_trees):
        pred = pred.copy()

        if self.n_nesterov_steps > 0:
            for tree in new_trees[-self.n_nesterov_steps:]:
                _leaf_indices, _leaf_values = self._compute_optimal_leaf_values(X, pred=pred, tree=tree,
                                                                                loss_function=loss_function,
                                                                                regularization=self.regularization)
                pred += _leaf_values[_leaf_indices]

        grads = loss_function.negative_gradient(pred)
        hesss = loss_function.hessian(pred)

        candidate_losses = []
        candidate_new_trees = []
        for tree in candidate_trees:
            leaves = compute_leaves(X, tree)
            leaf_values = numpy.bincount(leaves, weights=grads, minlength=2 ** 6)
            leaf_values /= numpy.bincount(leaves, weights=hesss, minlength=2 ** 6) + self.regularization
            leaf_values *= self.learning_rate

            new_preds = pred + leaf_values[leaves]
            candidate_losses.append(loss_function(new_preds))
            new_tree = tree[0], tree[1], leaf_values
            candidate_new_trees.append(new_tree)

        return candidate_losses, candidate_new_trees
