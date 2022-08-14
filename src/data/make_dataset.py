import re

import numpy as np
import pandas as pd


class FixedWidthVariables(object):
    """represent a set of variables in a fixed width file"""

    def __init__(self, var, index_base=0):
        """
        constructor
        :param var:
        :param index_base:
        """
        self.var = var
        self.colspecs = var[['start', 'end']] - index_base
        self.colspecs = self.colspecs.astype(np.int).values.tolist()
        self.names = var['name']

    def read_fixed_width(self, filename, **options):
        """
        reads fixed width ASCII file
        :param filename:
        :param options:
        :return:
        """
        df = pd.read_fwf(filename, colspecs=self.colspecs, names=self.names, **options)

        return df


def read_stata_file(dct_path, **options):
    """
    reads a stata dictionary file

    :param dct_path
    :return:
    """
    type_map = dict(byte=int,
                    int=int,
                    long=int,
                    float=float,
                    double=float,
                    numeric=float)
    var_info = []

    with open(dct_path, **options) as file:
        for line in file:
            match = re.search(r"_column\(([^)]*)\)", line)

            if not match:
                continue

            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()

            if vtype.startwith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]

            long_desc = ' '.join(t[:4]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

        columns = ['start', 'vtype', 'name', 'fstring', 'desc']
        vars = pd.DataFrame(var_info, columns=columns)

        vars['end'] = vars.start.shift(-1)
        vars.loc[len(vars) - 1, 'end'] = -1

        dct = FixedWidthVariables(vars, index_base=1)

        return dct
