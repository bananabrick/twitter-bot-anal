import pandas
import util
import data
import numpy


class ConfigVars:
    def __init__(
        self, to_sample,
        cols_to_build, cols_to_keep, classify_on
    ):
        '''
        to_sample: key, value map from data.DataSets enum to the
        number of samples which we need to take from that dataset.

        cols_to_build: Custom columns which we might want to build
        from the existing columns.

        cols_to_keep: Every column in the dataframe which is not in
        cols_to_keep will be dropped.

        classify_on: The feature which we're classifying on.
        The feature must exist in
        '''
        self.to_sample = to_sample
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
        raw_ds = data.get_data_sets(self.to_sample.keys())
        filtered_ds = data.build_filtered_datasets(
            raw_ds,
            self.cols_to_build
        )

        return filtered_ds

    def sample(self):
        '''
        Creates a pandas dataframe which we need to
        train on based on the config.

        Returns the dataframe with test feature and
        a column of the target feature defined by classify on.
        '''
        datasets = self.datasets
        for key, num_samples in self.to_sample.items():
            num_samples = min(num_samples, len(datasets[key].index))
            datasets[key] = datasets[key].sample(num_samples)

        s = pandas.concat(datasets.values(), sort=False)

        # Get rid of columns which are not needed.
        s = s.loc[:, self.cols_to_keep]

        # Split non-int/non-float columns
        # into separate binary columns.
        s = pandas.get_dummies(s)

        target_column = getattr(s, self.classify_on)
        s = s.drop(columns=[self.classify_on])
        return s, target_column
