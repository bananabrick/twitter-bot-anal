import data
import util
import pandas
import useful_configs

from sklearn.ensemble import AdaBoostClassifier

def base_test():
    base_config = useful_configs.ADA

    sample, is_bot = base_config.sample(
        {
            data.DataSets.GENUINE: 500,
            data.DataSets.T_1: 200,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 200,
        }
    )

    ada = AdaBoostClassifier()
    
    ada.fit(sample, is_bot)

    test_sample, is_bot_test = base_config.sample(
        {
            data.DataSets.GENUINE: 1300,
            data.DataSets.T_1: 600,
            data.DataSets.T_2: 100,
            data.DataSets.T_3: 600,

        }
    )

    ada_prediction = ada.predict(test_sample)
    compare = pandas.DataFrame(
        {
            'is_bot_original': is_bot_test,
            'ada_is_bot_predict': ada_prediction
        }
    )

    ada_importances = list(zip(sample.columns.values, ada.feature_importances_))
    util.plot_feature_importances(
        "ada", "AdaBoost feature importances",
        "Gini Importance", ada_importances
    )

    print(util.accuracy(compare, "is_bot_original", "ada_is_bot_predict"))


if __name__ == "__main__":
    base_test()
