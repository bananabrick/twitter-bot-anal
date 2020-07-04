import pandas
import util
import re
from enum import Enum, unique


@unique
class DataSets(Enum):
    GENUINE = "data/genuine_accounts.csv/users.csv"
    T_1 = "data/traditional_spambots_1.csv/users.csv"
    T_2 = "data/traditional_spambots_2.csv/users.csv"
    T_3 = "data/traditional_spambots_3.csv/users.csv"
    S_1 = "data/social_spambots_1.csv/users.csv"
    S_2 = "data/social_spambots_2.csv/users.csv"
    S_3 = "data/social_spambots_3.csv/users.csv"


def get_data_sets(data_set_enums=None):
    '''
    data_set_enums: [DataSets enums]. Loads all if this is None.
    '''
    data_set_enums = data_set_enums or list(DataSets)
    ds = {}
    for e in data_set_enums:
        ds[e] = pandas.read_csv(e.value)
    return ds


URL_REGEX = re.compile(
    "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]"
    "[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}"
    "|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}"
    "|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}"
    "|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
)


# Contains both features we want to build from other
# features and existing features we want to modify.
# Add features which fall into either of those cases
# here and add support in `build_filtered_datasets`.
TO_BUILD = [
    "is_bot",
    "has_default_profile_image", # 99% data seems to have this? Useless.
    "no_screen_name", # All data has this. Useless.
    "geo_enabled", # Super important feature. T1, T2, T2 rarely has this.
    "language_not_empty", # Sketchy cause it works well on our dataset.
    "description_contains_url", # Not that useful.
    "description_length", # Maybe we can use long lengths to classify genuine?
    # "lt_50_tweets", # Can't find a num tweets column.
]


TO_KEEP = TO_BUILD[:]
TO_KEEP.extend([
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count"
])


def add_final_classification(*args):
    '''
    Adds a field `is_bot` to a dataset depending on the
    data.DataSets enum.
    '''
    if args[0] == DataSets.GENUINE:
        args[1]["is_bot"] = 0
    elif args[0] in (
        DataSets.T_1, DataSets.T_2, DataSets.T_3,
        DataSets.S_1, DataSets.S_2, DataSets.S_3,
    ):

        args[1]["is_bot"] = 1
    else:
        raise NotImplementedError

    return args[1]


def profile_image(row):
    if row["default_profile_image"] != 1:
        return 0
    return 1


def no_screen_name(row):
    if pandas.isna(row["screen_name"]):
        return 1
    return 0


def geo_enabled(row):
    if row["geo_enabled"] == 1:
        return 1
    return 0


def lang_not_empty(row):
    if pandas.isna(row["lang"]):
        return 0
    return 1


def contains_url(row):
    if pandas.isna(row["description"]):
        return 0
    
    if URL_REGEX.match(row["description"]):
        return 1
    return 0


def desc_length(row):
    if pandas.isna(row["description"]):
        return 0
    return len(row["description"])


def add_col(raw_datasets, col_name, func):
    '''
    Adds col_name to dataset with values
    depending on func.
    '''
    for ds in raw_datasets.values():
        # print(key, ds["is_bot"])
        ds[col_name] = ds.apply(
            lambda row: func(row), axis=1
        )


def build_filtered_datasets(raw_datasets, to_build):
    '''
    raw_datasets: key, value map from DataSets enum to a
    pandas dataframe for the dataset. No preprocessing must
    have been done on the dataset.

    to_build: The list of features which we want to build.
    '''
    for feat in to_build:
        if feat not in TO_BUILD:
            raise ValueError("No support to build feature: ", feat)

    for data_filter in to_build:
        if data_filter == "is_bot":
            util.apply_to_all(add_final_classification, raw_datasets)
        elif data_filter == "has_default_profile_image":
            add_col(
                raw_datasets, data_filter, profile_image
            )
        elif data_filter == "no_screen_name":
            add_col(
                raw_datasets, data_filter, no_screen_name
            )
        elif data_filter == "geo_enabled":
            add_col(
                raw_datasets, data_filter, geo_enabled
            )
        elif data_filter == "language_not_empty":
            add_col(
                raw_datasets, data_filter, lang_not_empty
            )
        elif data_filter == "description_contains_url":
            add_col(
                raw_datasets, data_filter, contains_url
            )
        elif data_filter == "description_length":
            add_col(
                raw_datasets, data_filter, desc_length
            )

    return raw_datasets


def _col_type(datasets, col):
    '''
    Prints the column type in each of the datasets.
    '''
    for name, ds in datasets.items():
        print(name)
        print(ds[col].dtype)
