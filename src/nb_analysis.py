from sklearn.naive_bayes import MultinomialNB

import useful_configs
import experiments
import util
import data
import math


def cv_test():
    experiments.run_cv("Multinomial Naive Bayes", MultinomialNB(), useful_configs.ALL, bucket_non_bool=True)


def random_test():
    experiments.run_random_sample("Multinomial Naive Bayes", MultinomialNB(), useful_configs.ALL, bucket_non_bool=True)


def graphs_and_data():
    print('\nTrain: Traditional + Social bots, Test: Traditional + Social bots')
    model = MultinomialNB()
    Xtrain, ytrain = useful_configs.ALL.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT, data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=True,
                                frac=0.4)
    util.random_sample_test(model,
                            Xtrain, ytrain,
                            *useful_configs.ALL.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT, data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=True,
                                frac=0.4))
    not_prob = dict(zip(Xtrain.columns, map(lambda x: math.exp(x), model.feature_log_prob_[0])))
    bot_prob = dict(zip(Xtrain.columns, map(lambda x: math.exp(x), model.feature_log_prob_[0])))
    feature_importances = {k : bot_prob[k] - v for (k,v) in not_prob.items()}
    for lol in sorted(feature_importances.items(), key=lambda x: x[1]):
        print(lol)


if __name__ == "__main__":
    # cv_test()
    # random_test()
    graphs_and_data()
