import data
import util
import pandas
import useful_configs

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def base_test():
    base_config = useful_configs.ALL

    sample, is_bot = base_config.sample(
        {
            data.DataSets.GENUINE: 500,
            data.DataSets.T_1: 200,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 200,
        },
        random_state=12345
    )

    tree = RandomForestClassifier()
    reg = LogisticRegression(max_iter=1000)

    reg.fit(sample, is_bot)
    tree.fit(sample, is_bot)

    test_sample, is_bot_test = base_config.sample(
        {
            data.DataSets.GENUINE: 1300,
            data.DataSets.T_1: 600,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 600,

        },
        random_state=23456
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
    base_test()
