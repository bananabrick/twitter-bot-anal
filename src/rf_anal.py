import data
import util
import config
import pandas

from sklearn.ensemble import RandomForestClassifier

def base_test():
    # Uses all custom columns, and keeps
    # all columns I think are relevant from
    # the old one.
    base_config = config.ConfigVars(
        cols_to_build=[
            "is_bot",
            "has_default_profile_image", 
            "no_screen_name",
            "language_not_empty",
            "description_contains_url",
            "description_length",
        ],
        cols_to_keep=[
            "is_bot",
            "has_default_profile_image", 
            "no_screen_name",
            "language_not_empty",
            "description_contains_url",
            "description_length",
            "followers_count",
            "friends_count",
            "statuses_count",
            "listed_count"
        ],
        classify_on="is_bot"
    )

    sample, is_bot = base_config.sample(
        {
            # Let's try out these two datasets.
            data.DataSets.GENUINE: 300,
            data.DataSets.T_1: 300,
        }
    )

    tree = RandomForestClassifier(max_depth=1)
    tree.fit(sample, is_bot)
    print(list(zip(sample.columns.values, tree.feature_importances_)))

    test_sample, is_bot_test = base_config.sample(
        {
            # Let's try out these two datasets.
            data.DataSets.GENUINE: 100,
            data.DataSets.T_1: 100,

        }
    )
    prediction = tree.predict(test_sample)
    compare = pandas.DataFrame(
        {'is_bot_original': is_bot_test, 'is_bot_predict': prediction}
    )

    print(util.accuracy(compare, "is_bot_original", "is_bot_predict"))

if __name__ == "__main__":
    base_test()
