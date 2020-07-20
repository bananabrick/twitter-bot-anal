'''
Put generic versions of common experiments here, so that
we can all run them.
'''

import data
import util
import config


def plot_importances(
    file_name, model_name, model, importance_name,
    model_to_imp, config, bucket_non_bool=False
):
    '''
    file_name: Name of the file you want to save to.
    model_name: Name of the model.
    model: Scikit model
    importance_name: Name of the feature importance benchmark like `GINI`.
    model_to_imp: Function which takes a model and produces its importances.
    config: config which you want to run the test on.
    bucket_non_bool: bucket non bool.
    '''
    xtrain, ytrain = config.even_sample(
        test_datasets=[
            data.TestDataSetType.TRADITIONAL_BOT,
            data.TestDataSetType.SOCIAL_BOT
        ],
        bucket_non_bool=bucket_non_bool,
        frac=0.4
    )

    util.random_sample_test(model,
        xtrain, ytrain,
        *config.even_sample(
            test_datasets=[
                data.TestDataSetType.TRADITIONAL_BOT,
                data.TestDataSetType.SOCIAL_BOT
            ],
            bucket_non_bool=bucket_non_bool,
            frac=0.4
        )
    )

    importances = list(zip(xtrain.columns.values, model_to_imp(model)))
    importances.sort(key=lambda x: x[1])
    util.plot_feature_importances(
        file_name, "{0} feature importances".format(model_name),
        importance_name, importances
    )

    return importances


def run_cv(model_name, model, config, bucket_non_bool=False):
    '''
    Runs cv experiment on traditional, social, traditional + social.
    '''
    print('\ncrossvalidation {0} experiment'.format(model_name))

    print('\nTrain: Traditional bots, Test: Traditional bots')
    X, y = config.even_sample(
        test_datasets=[data.TestDataSetType.TRADITIONAL_BOT], bucket_non_bool=bucket_non_bool
    )
    util.cv_test(model, X, y)

    print('\nTrain: Social bots, Test: Social bots')
    X, y = config.even_sample(
        test_datasets=[data.TestDataSetType.SOCIAL_BOT], bucket_non_bool=bucket_non_bool
    )
    util.cv_test(model, X, y)

    print('\nTrain: Traditional + Social bots, Test: Traditional + Social bots')
    X, y = config.even_sample(
        test_datasets=[data.TestDataSetType.SOCIAL_BOT, data.TestDataSetType.TRADITIONAL_BOT],
        bucket_non_bool=bucket_non_bool
    )
    util.cv_test(model, X, y)

    print("-" * 100)


def run_random_sample(model_name, model, config, bucket_non_bool=False):
    print('\nrandom sampling {0} experiment'.format(model_name))

    print('\nTrain: Traditional bots, Test: Social bots')
    util.random_sample_test(model,
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT],
                                bucket_non_bool=bucket_non_bool),
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=bucket_non_bool))

    print('\nTrain: Social bots, Test: Traditional bots')
    util.random_sample_test(model,
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=bucket_non_bool),
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT],
                                bucket_non_bool=bucket_non_bool))

    print('\nTrain: Traditional + Social bots, Test: Social bots')
    util.random_sample_test(model,
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT, data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=bucket_non_bool,
                                frac=0.4),
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=bucket_non_bool,
                                frac=0.4))

    print('\nTrain: Traditional + Social bots, Test: Traditional bots')
    util.random_sample_test(model,
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT, data.TestDataSetType.SOCIAL_BOT],
                                bucket_non_bool=bucket_non_bool,
                                frac=0.4),
                            *config.even_sample(
                                test_datasets=[data.TestDataSetType.TRADITIONAL_BOT],
                                bucket_non_bool=bucket_non_bool,
                                frac=0.4))


    print("-" * 100)
