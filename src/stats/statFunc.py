import collections
import copy
import decimal
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
                self.ps = obj.groupby(quantile).count().sort_index()
                self.ps = self.ps.cumsum()
        else:
            if self.discrete:
                self.d.update(collections.Counter(obj))
            else:
                bins = np.histogram_bin_edges(obj, bins=bins, **options)
                quantile = pd.cut(obj, bins)
                self.ps = pd.Series(obj).groupby(quantile).count().sort_index()
                self.ps = self.ps.cumsum()

        if len(self) > 0 and isinstance(self, Pmf):
            self.normalize()
        elif len(self) > 0 and isinstance(self, Cdf):
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
        return np.max((self.d.values()))

    def min_(self):
        """
        return minimum frequency
        :return:
        """
        return np.min((self.d.values()))

    def sum_(self):
        """
        return total of frequencies
        :return:
        """
        return np.sum(list(self.d.values()))

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


class Cdf(_DictWrapper):
    """represent a cumulative distribution function"""

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
            m = np.max(self.ps)

        self.ps = np.log(self.ps / m)

    def scale(self, factor):
        """multiply the xs by factor"""
        new = self.ps.copy()
        new = new * factor

        return new

    def normalize(self, fraction=1):
        """normalize cumulative density function"""
        if self._log:
            raise ValueError('normalize: pmf is under a log transform')
        total = self.ps[-1]

        if total == 0:
            raise ValueError('normalize: total probability is zero')

        factor = fraction / total

        self.ps = self.ps * factor

    def prob(self, interval):
        """return probability that correspond to value x"""
        if isinstance(interval, tuple):
            interval = pd.IntervalIndex.from_tuples([interval])[0]
        else:
            raise ValueError('interval need to be tuple')

        if interval in self.ps.index:
            return self.ps[interval]
        else:
            return None

    def probs(self, intervals):
        """return probabilities of sequence of intervals"""
        if isinstance(intervals, list):
            intervals = pd.IntervalIndex.from_tuples([intervals])
        else:
            raise ValueError('interval need to be list of tuple')

        return [self.prob(p) for p in intervals]

    def value(self, p):
        """returns the value that corresponds to probability p"""
        if p < 0 or p > 1:
            raise ValueError('p must be in range [0, 1]')

        f = abs(decimal.Decimal(str(p)).as_tuple().exponent)
        return self.ps[round(self.ps, f) == p].index[0]

    def values(self, p=None):
        """returns the value that corresponds to probability p"""
        if p is None:
            return self.ps.index

        p = np.asarray(p)

        if any(p < 0) or any(p > 1):
            raise ValueError('p must be in range [0, 1]')

        f = abs(decimal.Decimal(str(p)).as_tuple().exponent)
        return [self.ps[round(self.ps, f) == i].index for i in p]

    def percentile(self, p):
        """returns the value corresponds to percentile p"""
        return self.value(p / 100)

    def percentiles(self, p):
        """returns value that corresponds to percentile p"""
        p = np.asarray(p)
        return self.values(p / 100)

    def percentile_rank(self, interval):
        """returns the percentile rank of the value interval"""
        return self.prob(interval) * 100

    def percentile_ranks(self, intervals):
        """returns the percentile ranks of the value intervals"""
        return self.probs(intervals) * 100


class Pdf:
    pass


def convert_to_array(obj):
    """convert into array"""
    return np.asarray(obj)


def mean(obj, axis=0):
    """calculate average"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)

    return np.sum(obj, axis=axis) / len(obj)


def trimmed_mean(obj, n=10):
    """remove n elements from both ends of sorted values"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)
        return mean(obj.sort()[n: -n])

    elif isinstance(obj, pd.Series):
        return mean(obj.sort_values()[n: -n])

    elif isinstance(obj, pd.DataFrame):
        return obj.apply(lambda x: mean(x.sort_values()[n: -n]))


def weighted_mean(obj, weights):
    """calculate weighted"""
    if isinstance(weights, (pd.Series, list)):
        if len(obj) == len(weights):
            if isinstance(weights, list):
                return np.sum(obj * convert_to_array(weights)) / np.sum(weights)
            else:
                return np.sum(obj * weights) / np.sum(weights)
        else:
            raise AttributeError('obj and weights need to have same size')

    elif isinstance(weights, str):
        if isinstance(obj, pd.DataFrame):
            if weights in obj.columns:
                return np.sum(obj * obj[weights]) / np.sum(obj[weights])
            else:
                raise AttributeError('weights not in axis')
        else:
            raise AttributeError('obj is not a pandas dataframe')


def median(obj, axis=0):
    """calculate median"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)

        if len(obj) % 2 == 0:
            c = len(obj) // 2
            return np.sum(obj.sort()[c - 1: c]) / 2
        else:
            c = (len(obj) - 1) / 2
            return obj.sort()[c]

    elif isinstance(obj, pd.Series):
        if len(obj) % 2 == 0:
            c = len(obj) // 2
            return np.sum(obj.sort_values()[c - 1: c]) / 2
        else:
            c = (len(obj) - 1) / 2
            return obj.sort_values()[c]

    elif isinstance(obj, pd.DataFrame):
        return obj.apply(np.median, axis=axis)


def var(obj, axis=0):
    """calculate variance"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)
        return np.sum((obj - mean(obj)) ** 2) / (len(obj) - 1)

    elif isinstance(obj, pd.Series):
        return np.sum((obj - mean(obj)) ** 2) / (len(obj) - 1)

    elif isinstance(obj, pd.DataFrame):
        return obj.apply(lambda x: np.sum(((x - mean(x)) ** 2)) / (len(obj) - 1), axis=axis)


def std(obj, axis=0):
    """calculate standard deviation"""
    return np.sqrt(var(obj, axis))


def mean_abs_deviation(obj, axis=0):
    """calculate mean absolute deviation"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)
        return np.sum(np.abs(obj - mean(obj))) / (len(obj))

    elif isinstance(obj, pd.Series):
        return np.sum(np.abs(obj - mean(obj))) / (len(obj))

    elif isinstance(obj, pd.DataFrame):
        return obj.apply(lambda x: np.sum(np.abs(x - mean(x))) / (len(x)), axis=axis)


def mad_from_median(obj, axis=0):
    """calculate mean absolute deviation"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)
        return median(obj - median(obj))

    elif isinstance(obj, pd.Series):
        return median(obj - median(obj))

    elif isinstance(obj, pd.DataFrame):
        return obj.apply(lambda x: median(x - median(x)), axis=axis)


def iqr(obj, axis=0):
    """calculate inter quantile range"""
    if isinstance(obj, list):
        obj = convert_to_array(obj)
        return np.quantile(obj, q=0.75) - np.quantile(obj, q=0.25)

    elif isinstance(obj, pd.Series):
        return np.quantile(obj, q=0.75) - np.quantile(obj, q=0.25)

    elif isinstance(obj, pd.DataFrame):
        return obj.apply(lambda x: np.quantile(x, q=0.75) - np.quantile(x, q=0.25), axis=axis)


def cohen_effect_size(obj1, obj2):
    """
    calculate cohen's d effect size
    :param obj1:
    :param obj2:
    :return:
    """
    diff = mean(obj1) - mean(obj2)
    var1 = var(obj1)
    var2 = var(obj2)
    pooled_var = (var1 * len(obj1) + var2 * len(obj2)) / (len(obj1) + len(obj2))

    return diff / np.sqrt(pooled_var)


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
