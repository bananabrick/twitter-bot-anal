import pandas
import numpy
import data
import useful_configs
import util
import experiments

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


random_state_rf = numpy.random.RandomState(12345)
random_state_reg = numpy.random.RandomState(4)


def cv_test():
    tree = RandomForestClassifier(random_state=random_state_rf)
    reg = LogisticRegression(max_iter=1000, random_state=random_state_reg)
    experiments.run_cv("random forest", tree, useful_configs.ALL)
    experiments.run_cv("logistic regression", reg, useful_configs.ALL)


def base_test():
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
    reg = LogisticRegression(max_iter=1000, random_state=random_state_reg)

    reg.fit(sample, is_bot)
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
    reg_prediction = reg.predict(test_sample)
    compare = pandas.DataFrame(
        {
            'is_bot_original': is_bot_test,
            'tree_is_bot_predict': tree_prediction,
            'reg_is_bot_predict': reg_prediction
        }
    )

    rf_importances = list(zip(sample.columns.values, tree.feature_importances_))
    reg_importances = list(zip(sample.columns.values, reg.coef_[0]))
    util.plot_feature_importances(
        "rf", "Random Forest feature importances",
        "Gini Importance", rf_importances
    )
    util.plot_feature_importances(
        "reg", "Logistic Regression feature importances",
        "Feature Coefficients", reg_importances
    )

    print(util.accuracy(compare, "is_bot_original", "tree_is_bot_predict"))
    print(util.accuracy(compare, "is_bot_original", "reg_is_bot_predict"))


if __name__ == "__main__":
    # base_test()
    cv_test()
