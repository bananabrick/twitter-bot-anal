import data
import util
import config
import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def base_test():
    base_config = config.ConfigVars(
        cols_to_build=[
            "is_bot",
            "has_default_profile_image", 
            "no_screen_name",
            "language_not_empty",
            "description_contains_url",
            "description_length",
            "geo_enabled",
            "has_name",
            "fr_fo_ratio_gt_100",
            "fr_fo_ratio_gt_50"
        ],
        cols_to_keep=[
            "is_bot",
            "has_default_profile_image", 
            "no_screen_name",
            "language_not_empty",
            "description_contains_url",
            "description_length",
            "fr_fo_ratio_gt_100",
            "fr_fo_ratio_gt_50",
            "has_name",
            "geo_enabled",
            "followers_count",
            "friends_count",
            "statuses_count",
            "listed_count"
        ],
        classify_on="is_bot"
    )

    sample, is_bot = base_config.sample(
        {
            data.DataSets.GENUINE: 300,
            data.DataSets.T_1: 100,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 100,
        }
    )

    tree = RandomForestClassifier()
    reg = LogisticRegression(max_iter=1000)
    reg.fit(sample, is_bot)
    tree.fit(sample, is_bot)
    print(list(zip(sample.columns.values, tree.feature_importances_)))

    test_sample, is_bot_test = base_config.sample(
        {
            data.DataSets.GENUINE: 2000,
            data.DataSets.T_1: 600,
            data.DataSets.T_2: 600,
            data.DataSets.T_3: 600,

        }
    )
    prediction = tree.predict(test_sample)
    compare = pandas.DataFrame(
        {'is_bot_original': is_bot_test, 'is_bot_predict': prediction}
    )

    print(util.accuracy(compare, "is_bot_original", "is_bot_predict"))


if __name__ == "__main__":
    base_test()
