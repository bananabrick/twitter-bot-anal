import math

import pandas
from sklearn.naive_bayes import MultinomialNB

import data
import useful_configs
import util


def base_test():
    base_config = useful_configs.ALL

    sample, is_bot = base_config.sample(
        {
            data.DataSets.GENUINE: 500,
            data.DataSets.T_1: 200,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 200,
        }, bucket_non_bool=True
    )

    # print(sample.dtypes)

    nb = MultinomialNB()
    nb.fit(sample, is_bot)

    # kinda feature importance?
    for s in (sorted(map(lambda x: (x[0], math.exp(x[1])), zip(sample.keys(), nb.feature_log_prob_[1])), key=lambda x: x[1])):
        print(s)

    test_sample, is_bot_test = base_config.sample(
        {
            data.DataSets.GENUINE: 1300,
            data.DataSets.T_1: 600,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 600,
        }, bucket_non_bool=True
    )

    prediction = nb.predict(test_sample)
    compare = pandas.DataFrame(
        {
            'is_bot': is_bot_test,
            'predict': prediction
        }
    )

    print(util.accuracy(compare, "is_bot", "predict"))


def cv_test():
    base_config = useful_configs.ALL

    print('\nTrain: Traditional bots, Test: Traditional bots')
    X, y = base_config.even_sample(test_datasets={data.TestDataSetType.TRADITIONAL_BOT}, bucket_non_bool=True)
    util.cv_test(MultinomialNB(), X, y)

    print('\nTrain: Social bots, Test: Social bots')
    X, y = base_config.even_sample(test_datasets={data.TestDataSetType.SOCIAL_BOT}, bucket_non_bool=True)
    util.cv_test(MultinomialNB(), X, y)

    print('\nTrain: Traditional + Social bots, Test: Traditional + Social bots')
    X, y = base_config.even_sample(test_datasets={data.TestDataSetType.SOCIAL_BOT, data.TestDataSetType.TRADITIONAL_BOT}, bucket_non_bool=True)
    util.cv_test(MultinomialNB(), X, y)


if __name__ == "__main__":
    # base_test()
    cv_test()
