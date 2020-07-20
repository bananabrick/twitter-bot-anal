'''
Put generic versions of common experiments here, so that
we can all run them.
'''

import data
import util
import config


def run_cv(model_name, model, config):
    '''
    Runs cv experiment on traditional, social, traditional + social.
    '''
    print('\ncrossvalidation {0} experiment'.format(model_name))

    print('\nTrain: Traditional bots, Test: Traditional bots')
    X, y = config.even_sample(
        test_datasets={data.TestDataSetType.TRADITIONAL_BOT}, bucket_non_bool=True
    )
    util.cv_test(model, X, y)

    print('\nTrain: Social bots, Test: Social bots')
    X, y = config.even_sample(
        test_datasets={data.TestDataSetType.SOCIAL_BOT}, bucket_non_bool=True
    )
    util.cv_test(model, X, y)

    print('\nTrain: Traditional + Social bots, Test: Traditional + Social bots')
    X, y = config.even_sample(
        test_datasets={data.TestDataSetType.SOCIAL_BOT, data.TestDataSetType.TRADITIONAL_BOT},
        bucket_non_bool=True
    )
    util.cv_test(model, X, y)

    print("-" * 100)
