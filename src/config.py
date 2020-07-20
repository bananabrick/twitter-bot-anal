from typing import Dict, Set

import numpy
import pandas

import data

random_state = numpy.random.RandomState(12345)


class ConfigVars:
    def __init__(
        self, name, cols_to_build, cols_to_keep, classify_on
    ):
        '''
        name: self explanatory.

        cols_to_build: Custom columns which we might want to build
        from the existing columns.

        cols_to_keep: Every column in the dataframe which is not in
        cols_to_keep will be dropped.

        classify_on: The feature which we're classifying on.
        The feature must exist in
        '''
        self.name = name
        self.cols_to_build = cols_to_build
        self.cols_to_keep = cols_to_keep

        # The feature which we're classifying on.
        # In our case this is going to be `is_bot`.
        self.classify_on = classify_on

        self.datasets = self._build_datasets()
        for ds in self.datasets.values():
            if self.classify_on not in ds.columns.values:
                raise ValueError(
                    "Feature which is being classified on does not exist."
                )

    def _build_datasets(self):        
        raw_ds = data.get_data_sets()
        filtered_ds = data.build_filtered_datasets(
            raw_ds,
            self.cols_to_build
        )
        return filtered_ds

    def sample(self, to_sample, bucket_non_bool=False):
        '''
        Creates a pandas dataframe which we need to
        train on based on the config.

        to_sample: A key, value map from DataSets enum, to
        the number of rows we want to sample from that dataset.
        The dataset enum must have been passed in when initializing
        the config.

        Returns the dataframe with test feature and
        a column of the target feature defined by classify on.
        '''
        datasets = self.datasets
        sampled_datasets = []
        for key, num_samples in to_sample.items():
            num_samples = min(num_samples, len(datasets[key].index))
            sampled_datasets.append(
                datasets[key].copy(deep=True).sample(num_samples, random_state=random_state)
            )

        s = pandas.concat(sampled_datasets, sort=False)

        return self._split_to_XY(s, bucket_non_bool)

    def even_sample(self, test_datasets, bucket_non_bool=False, frac=1):
        """
        return a dataframe with 50% genuine and 50% test (evenly split across the test types in test_datasets)
        """
        assert test_datasets

        # we want 50% genuine, 50% test (evenly split across the test types)
        num_samples = len(self.datasets[data.DataSets.GENUINE])//len(test_datasets)
        for dataset_type in test_datasets:
            num_samples = min(num_samples, sum(map(lambda x: len(self.datasets[x]), dataset_type.value)))
        genuine = self.datasets[data.DataSets.GENUINE].sample(int(num_samples*len(test_datasets)*frac), random_state=random_state)
        test = pandas.DataFrame()
        for dataset_type in test_datasets:
            # Pass sort=True here, to get rid of the annoying ass warning.
            type_df = pandas.concat(map(lambda x: self.datasets[x], dataset_type.value), sort=True)
            test = pandas.concat([test, type_df.sample(int(num_samples*frac), random_state=random_state)], sort=True)

        # if this breaks, the num_samples logic is trash
        assert len(genuine) == len(test)
        df = pandas.concat([genuine, test], join='inner').reset_index(drop=True)

        return self._split_to_XY(df, bucket_non_bool)

    def _split_to_XY(self, df, bucket_non_bool):
        # Get rid of columns which are not needed.
        df = df.loc[:, self.cols_to_keep]
        # split
        target_column = getattr(df, self.classify_on)
        df = df.drop(columns=[self.classify_on])
        if bucket_non_bool:
            for key, dt in zip(df.keys(), df.dtypes):
                if dt != bool:
                    df[key] = pandas.cut(df[key], 5, labels=list(range(5)))
        return df, target_column
