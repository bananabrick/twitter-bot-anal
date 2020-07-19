import matplotlib.pyplot as plt
from functools import lru_cache

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def apply_to_all(func, datasets):
    '''
    Applies func to each dataset in the dict of datasets.
    Result of the func must be a dataset.
    '''
    for key, dataset in list(datasets.items()):
        datasets[key] = func(key, dataset)
    return datasets


def accuracy(dataframe, field1, field2):
    '''
    field1, field2 must be columns in the dataframe.
    '''
    n = 0
    for e, row in dataframe.iterrows():
        if row[field1] == row[field2]:
            n += 1
    return n / len(dataframe.index)


def plot_bar(figname, title, x, y, tick_label, xlabel, ylabel):
    fig = plt.figure()
    plt.barh(x, y, tick_label=tick_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(figname)


def plot_feature_importances(
    figname, title, xlabel, importances
):
    '''
    Plots a horizontal bar graph for features, importance.

    title: figure title.
    figname: What you want to name the figure.
    xlabel: The type of importance.
    importances: 2-tup of (feature_name, importance).
    '''
    feat_names, imps = zip(*importances)
    plot_bar(
        figname, title, range(len(feat_names)), imps, feat_names,
        xlabel,
        "features",
    )


@lru_cache(maxsize=10000)
def edit_distance(word1, word2):
    if not word1 and not word2:
        return 0
    elif not word1 or not word2:
        return max(map(len, [word1, word2]))
    elif word1[0] == word2[0]:
        return edit_distance(word1[1:], word2[1:])
    else:  # word1[0] != word2[0]
        return min([
            1 + edit_distance(word1, word2[1:]),  # delete character
            1 + edit_distance(word1[1:], word2),  # insert character
            1 + edit_distance(word1, word1[0] + word2[1:])
        ])