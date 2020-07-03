def apply_to_all(func, datasets):
    '''
    Applies func to each dataset in the dict of datasets.
    Result of the func must be a dataset.
    '''
    for key, dataset in list(datasets.items()):
        datasets[key] = func(key, dataset)
    return datasets
