import data
import util
import pandas
import useful_configs
import math
import statistics

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.model_selection import cross_validate


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
    X, y = base_config.even_sample(test_datasets={data.TestDataSetType.TRADITIONAL_BOT}, bucket_non_bool=True)
    nb = MultinomialNB()

    scores = cross_validate(nb, X, y, scoring=['precision', 'recall', 'accuracy'])
    print('precision: avg = {}, {}'.format(statistics.mean(scores['test_precision']), scores['test_precision']))
    print('recall: avg = {}, {}'.format(statistics.mean(scores['test_recall']), scores['test_recall']))
    print('accuracy: avg = {}, {}'.format(statistics.mean(scores['test_accuracy']), scores['test_accuracy']))


if __name__ == "__main__":
    # base_test()
    cv_test()
