import data
import util
import pandas
import useful_configs
import math

from sklearn.naive_bayes import MultinomialNB


def base_test():
    base_config = useful_configs.NB

    sample, is_bot = base_config.sample(
        {
            data.DataSets.GENUINE: 500,
            data.DataSets.T_1: 200,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 200,
        }, bucket_non_bool=False
        , random_state=12345
    )

    # print(sample.dtypes)

    nb = MultinomialNB()
    nb.fit(sample, is_bot)

    # kinda feature importance?
    for s in (sorted(map(lambda x: (x[0], math.exp(x[1])), zip(sample.keys(), nb.feature_log_prob_[0])), key=lambda x: x[1])):
        print(s)

    test_sample, is_bot_test = base_config.sample(
        {
            data.DataSets.GENUINE: 1300,
            data.DataSets.T_1: 600,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 600,
        }, bucket_non_bool=False
        , random_state=23456
    )

    prediction = nb.predict(test_sample)
    compare = pandas.DataFrame(
        {
            'is_bot': is_bot_test,
            'predict': prediction
        }
    )

    print(util.accuracy(compare, "is_bot", "predict"))


if __name__ == "__main__":
    base_test()
