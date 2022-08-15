import numpy as np
import pandas as pd


def clean_fem_preg(df):
    df['agepreg'] /= 100.0

    # these values represent special values in encoding
    # need to handle as null values otherwise maybe problematic
    na_values = [97, 98, 99]
    df.birthwgt_lb.replace(na_values, inplace=True)
    df.birthwgt_oz.replace(na_values, inplace=True)

    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0
