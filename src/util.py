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
