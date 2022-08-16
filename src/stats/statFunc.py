import collections
import numpy as np
import pandas as pd


class Histogram(object):
    """represents a histogram, which ia a map from values to frequencies"""

    def __init__(self, obj, discrete=True, bins='auto', weights=None):
        """constructor"""
        self.d = {}
        self.discrete = discrete

        if obj is None:
            return

        if not self.discrete:
            self.bin_edges = np.histogram_bin_edges(obj, bins=bins, weights=weights)

        if isinstance(obj, dict):
            # use input freq dict
            if self.discrete:
                self.d.update(obj.items())

        elif isinstance(obj, pd.Series):
            # use pandas built in value count method to count freq
            if self.discrete:
                self.d.update(obj.value_counts().iteritems())
            else:
                self.d = obj.groupby(pd.cut(obj, self.bin_edges)).count().to_dict()

        elif isinstance(obj, list):
            # use input list to count value freq
            if self.discrete:
                self.d.update(collections.Counter(obj))
            else:
                temp = pd.Series(obj)

                for i in range(len(self.bin_edges) - 1):
                    h_margin = self.bin_edges[i + 1]
                    l_margin = self.bin_edges[i]

                    self.d[(l_margin, h_margin)] = len(temp[(l_margin <= temp) & (temp <= h_margin)])

    def incr(self, x, term=1):
        """increment the freq associate with the value x
        :param x:
        :param term:
        :return:
        """
        self.d[x] = self.d.get(x, 0) + term

    def freq(self, x):
        """gets the frequency associate with the value x
        :param x:
        :param bins:
        :param weights:
        :return:
        """
        if self.discrete:
            return self.d.get(x, 0)

    def freqs(self, xs):
        """return frequencies of input sequence
        :param xs:
        :return:
        """
        return [self.freq(x) for x in xs]

    def is_subset(self, hist):
        """check is this histogram subset of the given
        :param hist:
        :return:
        """
        for val, freq in self.d.items():
            if freq > hist.freq(val):
                return False

        return True

    def subtract(self, hist):
        """subtract given histogram values from this histogram
        :param hist:
        :return:
        """
        for val, freq in hist.items():
            self.incr(val, -freq)

    def smallest_k(self, k=1):
        return sorted(self.d.items(), reverse=False)[:k]

    def largest_k(self, k=1):
        return sorted(self.d.items(), reverse=True)[:k]


class Summary(object):
    """summary statistics about distribution"""

    def __init__(self, obj):
        self.obj = obj

    def sample_mean(self, axis):
        return np.mean(self.obj, axis=axis)

    def sample_variance(self, axis):
        return np.var(self.obj, axis=axis)

    def sample_std(self, axis):
        return np.std(self.obj, axis=axis)

    def effect_size(self, other):
        diff = self.sample_mean(axis=0) - np.mean(other)
        var1 = self.sample_variance(axis=0)
        var2 = np.var(other)
        n1, n2 = len(self.obj), len(other)
        pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

        return diff / np.sqrt(pooled_var)
