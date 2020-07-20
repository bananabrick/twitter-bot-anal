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
    tree = RandomForestClassifier(random_state=random_state_rf)

    importance_func = lambda tree: tree.feature_importances_
    experiments.plot_importances(
        "rf", "random forest", tree, "Gini Importance", importance_func,
        base_config
    )

    # util.plot_feature_importances(
    #     "rf", "Random Forest feature importances",
    #     "Gini Importance", rf_importances
    # )

    # print(util.accuracy(compare, "is_bot_original", "tree_is_bot_predict"))


if __name__ == "__main__":
    # base_test()
    # cv_test()
    random_test()
