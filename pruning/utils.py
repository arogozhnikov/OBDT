from __future__ import division, print_function
import numpy
import pandas
from sklearn.metrics import roc_auc_score, roc_curve
import struct
from collections import OrderedDict
from rep.estimators import Classifier
from rep.estimators.utils import score_to_proba
from six import BytesIO

__author__ = 'Alex Rogozhnikov'


def compute_ams_on_cuts(answers, predictions, sample_weight):
    """ Predictions are probabilities"""
    b, s, thresholds = roc_curve(answers, predictions, sample_weight=sample_weight)
    # normalization constants
    real_s = 691.988607712
    real_b = 410999.847322
    s *= real_s
    b *= real_b
    br = 10.
    radicands = 2 * ((s + b + br) * numpy.log(1.0 + s / (b + br)) - s)
    return thresholds, radicands


def optimal_AMS(answers, predictions, sample_weight):
    """ Predictions are probabilities """
    cuts, radicands = compute_ams_on_cuts(answers, predictions, sample_weight)
    return numpy.sqrt(numpy.max(radicands))


def precisionAt15(answers, predictions, sample_weight, percent=0.15):
    n_passed = int(len(answers) * percent)
    RATIO = 50
    weight = sample_weight.copy()
    weight[answers == 0] /= weight[answers == 0].mean() / RATIO
    weight[answers == 1] /= weight[answers == 1].mean()
    order = numpy.argsort(-predictions)
    passed = order[:n_passed]
    return numpy.average(answers[passed], weights=weight[passed])


def get_higgs_data(train_file):
    """ reads the data and normalizes weights """
    data = pandas.read_csv(train_file, index_col='EventId')
    answers_bs = numpy.ravel(data.Label)
    weights = numpy.ravel(data.Weight)
    data = data.drop(['Label', 'Weight'], axis=1)
    answers = numpy.zeros(len(answers_bs), dtype=numpy.int)
    answers[answers_bs == 's'] = 1
    for label in [0, 1]:
        weights[answers == label] /= weights[answers == label].mean()
    return data, answers, weights


def print_control_metrics(trainY, proba_train, trainW, testY, proba_test, testW):
    for name, metrics in [('ROC', roc_auc_score), ('AMS', optimal_AMS), ('precision', precisionAt15)]:
        print(name,
              metrics(testY, proba_test, sample_weight=testW),
              metrics(trainY, proba_train, sample_weight=trainW))


class OBDTListClassifier(Classifier):
    def __init__(self, features, trees):
        """
        Boosted ODT, which follows REP conventions though cannot be fiited.
        :param features: list of strings, features used.
        :param trees: list of tuples, each tuple represents a tree.
        Tuple is features (features, cuts, leaf_values)
        """
        Classifier.__init__(self, features)
        self.trees = trees

    def fit(self, X, y, sample_weight=None):
        pass

    def staged_decision_function(self, X):
        X = self._get_train_features(X).values
        result = numpy.zeros(len(X))
        for features, cuts, leaf_values in self.trees:
            indices = numpy.zeros(len(X), dtype=int)
            for depth, (feature, cut) in enumerate(zip(features, cuts)):
                indices += (X[:, feature] > cut) << depth
            result += leaf_values[indices]
            yield result

    def staged_predict_proba(self, X):
        for score in self.staged_decision_function(X):
            yield score_to_proba(score)

    def decision_function(self, X):
        for score in self.staged_decision_function(X):
            pass
        return score

    def predict_proba(self, X):
        return score_to_proba(self.decision_function(X))


def convert_list_to_mx(obdt_classifier, formula_mx):
    """
    Reassembling list of tuples back to mx format
    """
    assert isinstance(obdt_classifier, OBDTListClassifier), ''
    trees = obdt_classifier.trees
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

    # transforming into indices of binary features.
    for features, cuts, leaf_values in trees:
        for feature, cut in zip(features, cuts):
            binary_feature_id = numpy.searchsorted(thresholds[feature], cut)
            binary_feature_id += binary_feature_shifts[feature]
            binary_feature_ids.append(binary_feature_id)
        tree_table.extend(leaf_values)

    formula_stream = BytesIO(formula_mx)
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

    max_depth = struct.unpack('I', formula_stream.read(4))[0]
    # leaving the same info on max depth
    part_result += struct.pack('I', max_depth)

    old_nf_counts = struct.unpack('I' * max_depth, formula_stream.read(4 * max_depth))

    new_nf_counts = numpy.zeros(max_depth, dtype=int)
    new_nf_counts[5] = len(trees)
    part_result += struct.pack('I' * max_depth, *new_nf_counts)

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


def convert_mx_to_list(formula_mx):
    import _matrixnetapplier
    clf = _matrixnetapplier.MatrixnetClassifier(BytesIO(formula_mx))
    features = clf.features

    new_trees = []
    for depth, n_trees, iterator_trees in clf.iterate_trees():
        for tree in iterator_trees:
            new_trees.append(tree)

    return OBDTListClassifier(features=features, trees=new_trees)