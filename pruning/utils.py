from __future__ import division, print_function, absolute_import
import numpy
import pandas
from sklearn.metrics import roc_auc_score, roc_curve

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