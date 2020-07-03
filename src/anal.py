import data
import util
import config


def mess():
    conf = config.ConfigVars(
        to_sample={
            # Let's try out these two datasets.
            data.DataSets.GENUINE: 1000,
            data.DataSets.T_1: 1000
        },
        num_trees=100,
        cols_to_build=data.TO_BUILD,
        cols_to_keep=data.TO_KEEP,
        classify_on="is_bot"
    )

    new_dataframe = conf.sample


conf = config.ConfigVars(
    to_sample={
        data.DataSets.GENUINE: 1000,
        data.DataSets.T_1: 1000,
        data.DataSets.T_2: 300
    },
    cols_to_build=[
        "is_bot",
        "has_default_profile_image",
        "no_screen_name",
        "geo_enabled",
        "language_not_empty",
    ],
    cols_to_keep=[
        "is_bot",
        "has_default_profile_image",
        "no_screen_name",
        "geo_enabled",
        "language_not_empty",
        "followers_count",
        "friends_count",
        "statuses_count",
        "favourites_count",
        "listed_count"
    ],
    classify_on="is_bot"
)

new_dataframe = conf.sample
new_dataframe.to_csv("test_me.csv")

if __name__ == "__main__":
    # mess()
    pass
