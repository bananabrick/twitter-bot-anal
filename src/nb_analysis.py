from sklearn.naive_bayes import MultinomialNB

import useful_configs
import experiments


def cv_test():
    experiments.run_cv("Multinomial Naive Bayes", MultinomialNB(), useful_configs.ALL, bucket_non_bool=True)


def random_test():
    experiments.run_random_sample("Multinomial Naive Bayes", MultinomialNB(), useful_configs.ALL, bucket_non_bool=True)


if __name__ == "__main__":
    cv_test()
    random_test()
