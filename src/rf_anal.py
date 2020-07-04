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
        to_sample={
            # Let's try out these two datasets.
            data.DataSets.GENUINE: 500,
            data.DataSets.S_1: 500,
        },
        cols_to_build=data.TO_BUILD,
        cols_to_keep=data.TO_KEEP,
        classify_on="is_bot"
    )

    sample, is_bot = base_config.sample()
    print(sample)
    tree = RandomForestClassifier(max_depth=1)
    tree.fit(sample, is_bot)

    test_config = config.ConfigVars(
        to_sample={
            # Let's try out these two datasets.
            data.DataSets.GENUINE: 1000,
            data.DataSets.T_2: 1000,
        },
        cols_to_build=data.TO_BUILD,
        cols_to_keep=data.TO_KEEP,
        classify_on="is_bot"
    )

    test_sample, is_bot_test = test_config.sample()
    prediction = tree.predict(test_sample)
    compare = pandas.DataFrame(
        {'is_bot_original': is_bot_test, 'is_bot_predict': prediction}
    )

    print(util.accuracy(compare, "is_bot_original", "is_bot_predict"))
    # compare.to_csv("howdwedo.csv")

if __name__ == "__main__":
    base_test()
