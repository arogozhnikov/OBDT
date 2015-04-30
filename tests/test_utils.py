from __future__ import division, print_function
import numpy

from pruning._matrixnetapplier import MatrixnetClassifier
from pruning import utils
from six import BytesIO

__author__ = 'Alex Rogozhnikov'


def test_converters(mx_filename='../pruning/formula.mx',
                    higgs_filename='../../../datasets/higgs/training.csv'):
    with open(mx_filename, 'rb') as mx:
        formula_mx = mx.read()

    X, y, w = utils.get_higgs_data(higgs_filename)
    X = numpy.array(X, dtype='float32')
    X_part = X[:5000]

    mn = MatrixnetClassifier(BytesIO(formula_mx))
    mn_predictions = mn.apply(X_part)

    obdt = utils.convert_mx_to_list(formula_mx)
    list_predictions = obdt.decision_function(X_part)

    assert numpy.allclose(mn_predictions, list_predictions), 'predictions are different'

    formula_new = utils.convert_list_to_mx(obdt, formula_mx=formula_mx)
    mn_new = MatrixnetClassifier(BytesIO(formula_new))
    mn_predictions2 = mn_new.apply(X_part)

    assert numpy.allclose(mn_predictions, mn_predictions2, atol=2e-4, rtol=1e-3), 'predictions are different'




