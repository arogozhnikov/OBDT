# coding=utf-8
from __future__ import print_function, division, absolute_import

"""
This class is used to build predictions of MatrixNet classifier
it uses .mx format of MatrixNet formula.
"""
__author__ = 'Alex Rogozhnikov, Egor Khairullin'

import struct
import numpy


class MatrixnetClassifier(object):
    def __init__(self, formula_stream):
        """Reading the data from .mx stream """
        self.features = []  # list of strings
        self.bins = []

        bytes = formula_stream.read(4)
        features_quantity = struct.unpack('i', bytes)[0]
        for index in range(0, features_quantity):
            bytes = formula_stream.read(4)
            factor_length = struct.unpack('i', bytes)[0]
            self.features.append(formula_stream.read(factor_length))

        _ = formula_stream.read(4)  # skip formula length
        used_features_quantity = struct.unpack('I', formula_stream.read(4))[0]
        bins_quantities = struct.unpack(
            'I' * used_features_quantity,
            formula_stream.read(4 * used_features_quantity)
        )

        self.bins_total = struct.unpack('I', formula_stream.read(4))[0]
        for index in range(used_features_quantity):
            self.bins.append(
                struct.unpack(
                    'f' * bins_quantities[index],
                    formula_stream.read(4 * bins_quantities[index])
                )
            )

        _ = formula_stream.read(4)  # skip classes_count == 0

        nf_counts_len = struct.unpack('I', formula_stream.read(4))[0]
        self.nf_counts = struct.unpack('I' * nf_counts_len,
                                       formula_stream.read(4 * nf_counts_len)
        )

        ids_len = struct.unpack('I', formula_stream.read(4))[0]
        self.feature_ids = struct.unpack(
            'I' * ids_len,
            formula_stream.read(4 * ids_len)
        )
        self.feature_ids = numpy.array(self.feature_ids)

        tree_table_len = struct.unpack('I', formula_stream.read(4))[0]
        self.tree_table = struct.unpack(
            'i' * tree_table_len,
            formula_stream.read(4 * tree_table_len)
        )
        self.tree_table = numpy.array(self.tree_table)

        self.bias = struct.unpack('d', formula_stream.read(8))[0]
        self.delta_mult = struct.unpack('d', formula_stream.read(8))[0]

    def get_stats(self):
        """
        Function returns different information about this formula as a dict
        """
        stats_dict = {}
        stats_dict['bias'] = self.bias
        stats_dict['bins(lenghts)'] = [len(x) for x in self.bins]
        stats_dict['total_bins'] = self.bins_total
        stats_dict['features'] = self.features
        stats_dict['delta_mult'] = self.delta_mult
        stats_dict['len(feature_ids)'] = len(self.feature_ids)
        stats_dict['nf_counts'] = self.nf_counts
        stats_dict['len(tree_table)'] = len(self.tree_table)

        return stats_dict

    def _prepare_features_and_cuts(self):
        """
        Returns tuple with information about binary features:
            numpy.array of shape [n_binary_features] with indices of (initial) features used
            numpy.array of shape [n_binary_features] with cuts used (float32)
        """
        n_binary_features = sum(len(x) for x in self.bins)
        features_ids = numpy.zeros(n_binary_features, dtype='int8')
        cuts = numpy.zeros(n_binary_features, dtype='float32')
        binary_feature_index = 0
        for feature_index in range(len(self.bins)):
            for cut in self.bins[feature_index]:
                features_ids[binary_feature_index] = feature_index
                cuts[binary_feature_index] = cut
                binary_feature_index += 1
        return features_ids, cuts

    def _prepare_2d_features_and_cuts(self, binary_feature_ids):
        """
        Provided indices of binary features used in trees, returns in the same format
        :param binary_feature_ids:
            numpy.array of shape [n_trees, tree_depth]
        :return: tuple,
            numpy.array of shape [n_trees, tree_depth] with indices of (initial features) used
            numpy.array of shape [n_trees, tree_depth] with cuts used
        """
        feature_ids, cuts = self._prepare_features_and_cuts()
        return feature_ids[binary_feature_ids], cuts[binary_feature_ids]

    def _iterate_over_trees_with_fixed_depth(self, tree_depth, n_trees,
                                             binary_features_index, leaves_table_index):

        leaves_in_tree = 1 << tree_depth
        binary_feature_ids = self.feature_ids[binary_features_index:binary_features_index + n_trees * tree_depth]
        binary_feature_ids = binary_feature_ids.reshape([-1, tree_depth])

        feature_ids, cuts = self._prepare_2d_features_and_cuts(binary_feature_ids)
        for tree in range(n_trees):
            leaf_values = self.tree_table[leaves_table_index:leaves_table_index + leaves_in_tree] / self.delta_mult
            leaves_table_index += leaves_in_tree
            yield feature_ids[tree, :], cuts[tree, :], leaf_values

    def iterate_trees(self):
        """
        :return: yields depth, n_trees, trees_iterator,
            trees_iterator yields feature_ids, feature_cuts, leaf_values
        """
        binary_features_index = 0
        leaves_table_index = 0

        for tree_depth, n_trees in enumerate(self.nf_counts, 1):
            if n_trees == 0:
                continue

            yield tree_depth, n_trees, self._iterate_over_trees_with_fixed_depth(
                tree_depth=tree_depth, n_trees=n_trees,
                binary_features_index=binary_features_index,
                leaves_table_index=leaves_table_index)
            binary_features_index += n_trees * tree_depth
            leaves_table_index += (1 << n_trees) * tree_depth

    def apply_separately(self, events):
        """
        :param events: numpy.array (or DataFrame) of shape [n_samples, n_features]
        :return: each time yields numpy.array predictions of shape [n_samples]
            which is output of a particular tree
        """

        # вначале возвращаем всем одинаковое константное предсказание (bias)
        yield numpy.zeros(len(events), dtype=float) + self.bias

        # число событий в предсказываемой выборке
        n_events = len(events)

        # features - матрица признаков [событие, признак]
        features = numpy.array(events, dtype='float32', order='F')

        # деревья разбиты на группы по глубине. В той формуле, что лежит в репозитории, есть только одна группа
        for tree_depth, nf_count, tree_iterator in self.iterate_trees():

            # каждое дерево описывается: номерами признаков (6 штук),
            # порогами по каждому из них (6 штук), значениями в листьях (2**6 = 64 штуки)
            # в общем случае глубина (tree_depth) будет не 6
            for tree_features, tree_cuts, leaf_values in tree_iterator:

                # используем numpy-векторизацию
                # в бинарной записи вычисляем номер листа для каждого события
                leaf_indices = numpy.zeros(n_events, dtype='uint64')
                for tree_level, (feature, cut) in enumerate(zip(tree_features, tree_cuts)):
                    leaf_indices |= ((features[:, feature] > cut) << tree_level).astype('uint64')

                # возвращаем для каждого события значение из соответствующего ему листа
                # это происходит дл каждого дерева
                yield leaf_values[leaf_indices]

    def apply(self, events):
        """
        :param events: numpy.array (or DataFrame) of shape [n_samples, n_features]
        :return: prediction of shape [n_samples]
        """
        result = numpy.zeros(len(events), dtype=float)
        for stage_predictions in self.apply_separately(events):
            # просто суммируем для каждого события предсказания всех деревьев. Происходит это дело векторно
            result += stage_predictions
        return result

    def compute_leaf_indices_separately(self, events):
        """for each tree yields leaf_indices of events """
        n_events = len(events)

        # using Fortran order (surprisingly doesn't seem to influence speed much)
        features = numpy.array(events, dtype='float32', order='F')

        for tree_depth, n_trees, tree_iterator in self.iterate_trees():
            for tree_features, tree_cuts, leaf_values in tree_iterator:
                leaf_indices = numpy.zeros(n_events, dtype='uint64')
                for tree_level, (feature, cut) in enumerate(zip(tree_features, tree_cuts)):
                    leaf_indices |= ((features[:, feature] > cut) << tree_level).astype('uint64')
                yield leaf_indices

    def compute_leaf_indices(self, events):
        """
        :param events: pandas.DataFrame of shape [n_events, n_features]
        :return: numpy.array of shape [n_events, n_trees]
        """
        result = numpy.zeros([len(events), sum(self.nf_counts)], dtype='int8', order='F')
        for tree, tree_leaves in enumerate(self.compute_leaf_indices_separately(events)):
            result[:, tree] = tree_leaves
        return result