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
