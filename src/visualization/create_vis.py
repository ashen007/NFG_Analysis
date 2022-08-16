import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_hist(obj, discrete=True, **options):
    plt.figure(figsize=(12, 6), dpi=300)

    if isinstance(obj, list):
        sns.histplot(x=obj, discrete=discrete, **options)

    elif isinstance(obj, pd.Series):
        sns.histplot(x=obj, discrete=discrete, **options)

    elif isinstance(obj, pd.DataFrame):
        raise AttributeError('can not create plot for dataframe object need 1d object.')

    plt.show()
