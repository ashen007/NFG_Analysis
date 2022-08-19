import collections
import copy

import numpy as np
import pandas as pd


class _DictWrapper(object):
    """dictionary container"""

    def __init__(self,
                 obj=None,
                 label=None,
                 discrete=True,
                 bins='auto', **options):
        """
        distribution
        :param obj:
        :param label:
        """
        self.label = label if label is not None else '_nolabel_'
        self.d = {}
        self._log = False
        self.discrete = discrete

        if obj is None:
            return

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.label = label if label is not None else obj.label

        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.d.update(obj.Items())
        elif isinstance(obj, pd.Series):
            if self.discrete:
                self.d.update(obj.value_counts().iteritems())
            else:
                bins = np.histogram_bin_edges(obj, bins=bins, **options)
                quantile = pd.cut(obj, bins)
                self.d.update(obj.groupby(quantile).count().to_dict().items())
        else:
            if self.discrete:
                self.d.update(collections.Counter(obj))
            else:
                bins = np.histogram_bin_edges(obj, bins=bins, **options)
                quantile = pd.cut(obj, bins)
                self.d.update(pd.Series(obj).groupby(quantile).count().to_dict().items())

        if len(self) > 0 and isinstance(self, Pmf):
            self.Normalize()

    def copy(self, label=None):
        """returns copy of dictionary"""
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label

        return new

    def scale(self, factor):
        """
        scale up or down by factor
        :param factor:
        :return:
        """
        new = self.copy()
        new.d.clear()

        for key, val in self.items():
            new.set(key * factor, val)

        return new

    def log(self, m=None):
        """
        log transform the frequencies
        :param m: maximum value
        :return:
        """
        if self._log:
            raise ValueError('already transformed.')

        self._log = True

        if m is None:
            m = self.max_()

        for x, p in self.d.items():
            if p:
                self.set(x, np.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """
        exponential transformation the frequencies
        :param m: maximum value
        :return:
        """
        if not self._log:
            raise ValueError('already transformed with log.')

        self._log = False

        if m is None:
            m = self.max_()

        for x, p in self.d.items():
            self.set(x, np.exp(p - m))

    @property
    def freq_table(self):
        return self.d

    @freq_table.setter
    def freq_table(self, d):
        self.d = d

    def max_(self):
        """
        return maximum frequency
        :return:
        """
        return np.max(self.d.values())

    def min_(self):
        """
        return minimum frequency
        :return:
        """
        return np.min(self.d.values())

    def sum_(self):
        """
        return total of frequencies
        :return:
        """
        return np.sum(self.d.values())

    def make_cdf(self, label=None):
        """create continues density function from this dictionary"""
        label = label if label is not None else self.label

        return Cdf(self, label)

    def increase(self, x, term=1):
        """increments the frequency associated with the values"""
        self.d[x] = self.d.get(x, 0) + term

    def mult(self, x, factor):
        self.d[x] = self.d.get(x, 0) * factor

    def set(self, x, y=0):
        """set value associate with y"""
        self.d[x] = y

    def remove(self, x):
        """remove a value"""
        del self.d[x]

    def items(self):
        """return items of dictionary"""
        return self.d.items()

    def values(self):
        """return keys of dictionary"""
        return self.d.keys()

    def sorted_items(self):
        """return sorted value frequency pairs"""
        if any([np.isnan(x) for x in self.values()]):
            raise ValueError('keys contain NaN values')

        try:
            return sorted(self.d.items())
        except TypeError:
            return self.d.items()

    def render(self):
        """generate a sequence of points"""
        return zip(*self.sorted_items())

    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__

        if self.label == '_nolabel_':
            return '%s(%s)' % (cls, str(self.d))
        else:
            return self.label

    def __eq__(self, other):
        try:
            return self.d == other.d
        except:
            return False

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def __contains__(self, item):
        return item in self.d

    def __getitem__(self, item):
        return self.d.get(item, 0)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]


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

    def sample_mode(self):
        val = np.max(list(self.d.values()))
        return list(self.d.keys())[list(self.d.values()).index(val)]

    def sample_allmods(self):
        return [i for i in list(self.d.items()) if i[1] == np.max(list(self.d.values()))]


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
