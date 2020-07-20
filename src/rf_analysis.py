import pandas
import numpy
import data
import useful_configs
import util
import experiments

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def cv_test():
    random_state_rf = numpy.random.RandomState(1234)

    tree = RandomForestClassifier(random_state=random_state_rf)
    experiments.run_cv("random forest", tree, useful_configs.ALL)


def random_test():
    random_state_rf = numpy.random.RandomState(1234)

    tree = RandomForestClassifier(random_state=random_state_rf)
    experiments.run_random_sample("random forest", tree, useful_configs.ALL)


def base_test():
    random_state_rf = numpy.random.RandomState(12345)
    base_config = useful_configs.ALL

    sample, is_bot = base_config.sample(
        {
            data.DataSets.GENUINE: 500,
            data.DataSets.T_1: 200,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 200,
        }
    )

    tree = RandomForestClassifier(random_state=random_state_rf)
    tree.fit(sample, is_bot)

    test_sample, is_bot_test = base_config.sample(
        {
            data.DataSets.GENUINE: 1300,
            data.DataSets.T_1: 600,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 600,

        }
    )

    tree_prediction = tree.predict(test_sample)
    compare = pandas.DataFrame(
        {
            'is_bot_original': is_bot_test,
            'tree_is_bot_predict': tree_prediction,
        }
    )

    rf_importances = list(zip(sample.columns.values, tree.feature_importances_))
    util.plot_feature_importances(
        "rf", "Random Forest feature importances",
        "Gini Importance", rf_importances
    )

    print(util.accuracy(compare, "is_bot_original", "tree_is_bot_predict"))


if __name__ == "__main__":
    base_test()
    # cv_test()
    # random_test()
