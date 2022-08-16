import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_hist(obj,
              xlabel=None,
              ylabel=None,
              title=None,
              label=None,
              discrete=True,
              **options):
    plt.figure(figsize=(12, 6), dpi=300)

    if isinstance(obj, list):
        sns.histplot(x=obj, discrete=discrete, label=label, **options)

    elif isinstance(obj, pd.Series):
        sns.histplot(x=obj, discrete=discrete, label=label, **options)

    elif isinstance(obj, pd.DataFrame):
        raise AttributeError('can not create plot for dataframe object need 1d object.')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.show()
