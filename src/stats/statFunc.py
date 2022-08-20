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
            self.normalize()

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


class Histogram(_DictWrapper):
    """represents a histogram, which ia a map from values to frequencies"""

    def freq(self, x):
        """get the frequency associated with the value x."""
        return self.d.get(x, 0)

    def freqs(self, xs):
        """get frequencies for a sequence of values"""
        return [self.freq(x) for x in xs]

    def is_subset(self, other):
        """check weather this histogram sub set of the given"""
        for val, freq in self.items():
            if freq > other.freq(val):
                return False

        return True

    def subtract(self, other):
        """subtract given histogram values from this"""
        for val, freq in other.items():
            self.increase(val, -freq)


class Pmf(_DictWrapper):
    def prob(self, x, default=0):
        """
        get probability associate with the x
        :param x:
        :param default:
        :return:
        """
        return self.d.get(x, default)

    def probs(self, xs):
        """
        get probabilities for sequence of x values
        :param xs:
        :return:
        """
        return [self.prob(x) for x in xs]

    def percentile(self, percentage):
        """compute a percentile of a given pmf"""
        p = percentage / 100
        total = 0

        for val, freq in sorted(self.items()):
            total += freq

            if total >= p:
                return val

    def prob_greater(self, x):
        """
        probability that a sample from this pmf exceeds x
        :param x:
        :return:
        """
        if isinstance(x, _DictWrapper):
            return pmf_prob_greater(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return np.sum(t)

    def prob_less(self, x):
        """
        probability that a sample from this pmf less than x
        :param x:
        :return:
        """
        if isinstance(x, _DictWrapper):
            return pmf_prob_less(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return np.sum(t)

    def prob_equal(self, x):
        """
        probability that a sample from this pmf is equal x
        :param x:
        :return:
        """
        if isinstance(x, _DictWrapper):
            return pmf_prob_equal(self, x)
        else:
            return self[x]

    def normalize(self, fraction=1):
        """
        normalize this pmf so the sum of all probabilities is fraction
        :param fraction:
        :return:
        """
        if self._log:
            raise ValueError('normalize: pmf is under a log transform')
        total = self.sum_()

        if total == 0:
            raise ValueError('normalize: total probability is zero')

        factor = fraction / total

        for x in self.d:
            self.d[x] *= factor

        return total

    def mean(self):
        """compute the mean of a pmf"""
        return np.sum(p * x for x, p in self.items())

    def median(self):
        """compute median of a pmf"""
        return

    def var(self):
        """compute the variance of a pmf"""
        mu = self.mean()
        return np.sum(((x ** 2) * p) - (mu ** 2) for x, p in self.items())

    def std(self, mu=None):
        """
        compute standard deviation of pmf
        :param mu:
        :return:
        """
        return np.sqrt(self.var())


class Summary(object):
    """summary statistics about distribution"""
    pass


def pmf_prob_greater(pmf1, pmf2):
    """
    probability that a value from pmf1 is less than a value from pmf2
    :param pmf1:
    :param pmf2:
    :return:
    """
    total = 0

    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 > v2:
                total += p1 * p2

    return total


def pmf_prob_less(pmf1, pmf2):
    """
    probability that a value from pmf1 is greater than a value from pmf2
    :param pmf1:
    :param pmf2:
    :return:
    """
    total = 0

    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 < v2:
                total += p1 * p2

    return total


def pmf_prob_equal(pmf1, pmf2):
    """
    probability that a value from pmf1 is greater than a value from pmf2
    :param pmf1:
    :param pmf2:
    :return:
    """
    total = 0

    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 == v2:
                total += p1 * p2

    return total
