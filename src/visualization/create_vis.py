import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def colors(cmap, n):
    """get n hex color codes"""
    return plt.cm.get_cmap(cmap, n)


def plot_hist(obj,
              xlabel=None,
              ylabel=None,
              title=None,
              label=None,
              discrete=True,
              **options):
    """
    plot histogram
    :param obj:
    :param xlabel:
    :param ylabel:
    :param title:
    :param label:
    :param discrete:
    :param options:
    :return:
    """
    plt.figure(figsize=(12, 6), dpi=300)

    if isinstance(obj, list):
        sns.histplot(obj, discrete=discrete, label=label, **options)

    elif isinstance(obj, pd.Series):
        sns.histplot(obj, discrete=discrete, label=label, **options)

    elif isinstance(obj, pd.DataFrame):
        raise AttributeError('can not create plot for dataframe object need 1d object.')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.show()


def plot_pmf(obj,
             xlabel=None,
             ylabel=None,
             title=None,
             label=None,
             discrete=True,
             element='step',
             **options):
    """
    plot probability mass function
    :param obj:
    :param xlabel:
    :param ylabel:
    :param title:
    :param label:
    :param discrete:
    :param element:
    :param options:
    :return:
    """
    plt.figure(figsize=(12, 6), dpi=300)

    if isinstance(obj, list):
        sns.histplot(obj, stat='probability', element=element, discrete=discrete, label=label, **options)

    elif isinstance(obj, pd.Series):
        sns.histplot(obj, stat='probability', element=element, discrete=discrete, label=label, **options)

    elif isinstance(obj, pd.DataFrame):
        raise AttributeError('can not create plot for dataframe object need 1d object.')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.show()


def compare_hist(*objs,
                 labels,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 discrete=True,
                 l_title=None,
                 **options):
    """
    compare multiple histograms
    :param l_title:
    :param labels:
    :param objs:
    :param xlabel:
    :param ylabel:
    :param title:
    :param labels:
    :param discrete:
    :param options:
    :return:
    """
    plt.figure(figsize=(12, 6), dpi=300)
    c_codes = colors('winter', len(labels))

    if all([isinstance(obj, list) for obj in objs]):
        for i, obj in enumerate(objs):
            sns.histplot(x=obj, stat='frequency', color=c_codes(i), discrete=discrete, label=labels[i], **options)

    elif all([isinstance(obj, pd.Series) for obj in objs]):
        for i, obj in enumerate(objs):
            sns.histplot(x=obj, stat='frequency', color=c_codes(i), discrete=discrete, label=labels[i], **options)

    elif any([isinstance(obj, pd.DataFrame) for obj in objs]):
        raise AttributeError('can not create plot for dataframe object need 1d object.')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.legend(title=l_title, labels=labels)
    plt.show()


def compare_pmf(*objs,
                labels,
                xlabel=None,
                ylabel=None,
                title=None,
                element='step',
                discrete=True,
                l_title=None,
                **options):
    """
    compare multiple histograms
    :param l_title:
    :param element:
    :param labels:
    :param objs:
    :param xlabel:
    :param ylabel:
    :param title:
    :param labels:
    :param discrete:
    :param options:
    :return:
    """
    plt.figure(figsize=(12, 6), dpi=300)
    c_codes = colors('winter', len(labels))

    if all([isinstance(obj, list) for obj in objs]):
        for i, obj in enumerate(objs):
            sns.histplot(obj, stat='probability', color=c_codes(i), element=element, discrete=discrete, label=labels[i],
                         **options)

    elif all([isinstance(obj, pd.Series) for obj in objs]):
        for i, obj in enumerate(objs):
            sns.histplot(obj, stat='probability', color=c_codes(i), element=element, discrete=discrete, label=labels[i],
                         **options)

    elif any([isinstance(obj, pd.DataFrame) for obj in objs]):
        raise AttributeError('can not create plot for dataframe object need 1d object.')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.legend(title=l_title, labels=labels)
    plt.show()
