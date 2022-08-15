import collections
import numpy as np
import pandas as pd
from ..data import make_dataset


def read_fem_resp(dct_path, dat_path, nrows=None):
    """read NSFG female respondent data
    :param dct_path:
    :param dat_path:
    :param nrows:
    :return:
    """
    dct = make_dataset.read_stata_file(dct_path=dct_path)
    df = dct.read_fixed_width(dat_path, nrows=nrows)

    return df


def read_fem_preg(dct_path, dat_path, nrows=None):
    """read NSFG female pregnancy data
    :param dct_path:
    :param dat_path:
    :param nrows:
    :return:
    """
    dct = make_dataset.read_stata_file(dct_path=dct_path)
    df = dct.read_fixed_width(dat_path, nrows=nrows)

    return df


def read_male_resp(dct_path, dat_path, nrows=None):
    dct = make_dataset.read_stata_file(dct_path=dct_path)
    df = dct.read_fixed_width(dat_path, nrows=nrows)

    return df


def make_preg_map(df):
    d = collections.defaultdict(list)

    for index, case_id in df.caseid.iteritems():
        d[case_id].append(index)

    return d
