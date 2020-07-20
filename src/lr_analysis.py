import pandas
import numpy
import data
import useful_configs
import util
import experiments

from sklearn.linear_model import LogisticRegression


def cv_test():
    random_state_reg = numpy.random.RandomState(4)
    reg = LogisticRegression(max_iter=1000, random_state=random_state_reg)
    experiments.run_cv("logistic regression", reg, useful_configs.ALL)


def random_test():
    random_state_reg = numpy.random.RandomState(4)
    reg = LogisticRegression(max_iter=1000, random_state=random_state_reg)
    experiments.run_random_sample("logistic regression", reg, useful_configs.ALL)


def base_test():
    random_state_reg = numpy.random.RandomState(4)
    base_config = useful_configs.ALL
    reg = LogisticRegression(max_iter=1000, random_state=random_state_reg)

    importance_func = lambda reg: reg.coef_[0]
    experiments.plot_importances(
        "reg", "logistic regression", reg, "Feature Coefficients", importance_func,
        base_config
    )


if __name__ == "__main__":
    base_test()
    # cv_test()
    # random_test()
