from ConfigSpace.configuration_space import Configuration

from pipeline.classification import SimpleClassificationPipeline
from pipeline.regression import SimpleRegressionPipeline
from pipeline.dummy import MyDummyRegressor, MyDummyClassifier
from constants import *

def retrieve_template(task, config):
    if task in REGRESSION_TASKS:
        if config is None:
            model_class = MyDummyRegressor
        else:
            model_class = SimpleRegressionPipeline
    elif task in CLASSIFICATION_TASKS:
        if config is None:
            model_class = MyDummyClassifier
        else:
            model_class = SimpleClassificationPipeline
    else:
        raise NotImplementedError()

    model = model_class(config, task)
    return model


def get_default_configs(info, configuration_space):
    default_configs = []
    # == set default configurations
    # first enqueue the default configuration from our config space
    if info['task'] in CLASSIFICATION_TASKS:
        config_dict = {'balancing:strategy': 'weighting',
                       'classifier:__choice__': 'sgd',
                       'classifier:sgd:loss': 'hinge',
                       'classifier:sgd:penalty': 'l2',
                       'classifier:sgd:alpha': 0.0001,
                       'classifier:sgd:fit_intercept': 'True',
                       'classifier:sgd:n_iter': 5,
                       'classifier:sgd:learning_rate': 'optimal',
                       'classifier:sgd:eta0': 0.01,
                       'classifier:sgd:average': 'True',
                       'preprocessor:__choice__': 'no_preprocessing',
                       'rescaling:__choice__': 'min/max'}
        try:
            config = Configuration(configuration_space, config_dict)
            default_configs.append(config)
        except ValueError as e:
            print("Second default configurations %s cannot"
                                " be evaluated because of %s" %
                                (config_dict, e))

        if info["is_sparse"]:
            config_dict = {'classifier:__choice__': 'extra_trees',
                           'classifier:extra_trees:bootstrap': 'False',
                           'classifier:extra_trees:criterion': 'gini',
                           'classifier:extra_trees:max_depth': 'None',
                           'classifier:extra_trees:max_features': 1.0,
                           'classifier:extra_trees:min_samples_leaf': 5,
                           'classifier:extra_trees:min_samples_split': 5,
                           'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
                           'classifier:extra_trees:n_estimators': 100,
                           'balancing:strategy': 'weighting',
                           'preprocessor:__choice__': 'truncatedSVD',
                           'preprocessor:truncatedSVD:target_dim': 20,
                           'rescaling:__choice__': 'min/max'}
        else:
            n_data_points = info['train_num']
            percentile = 20. / n_data_points
            percentile = max(percentile, 2.)

            config_dict = {'classifier:__choice__': 'extra_trees',
                           'classifier:extra_trees:bootstrap': 'False',
                           'classifier:extra_trees:criterion': 'gini',
                           'classifier:extra_trees:max_depth': 'None',
                           'classifier:extra_trees:max_features': 1.0,
                           'classifier:extra_trees:min_samples_leaf': 5,
                           'classifier:extra_trees:min_samples_split': 5,
                           'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
                           'classifier:extra_trees:n_estimators': 100,
                           'balancing:strategy': 'weighting',
                           'preprocessor:__choice__': 'select_percentile_classification',
                           'preprocessor:select_percentile_classification:percentile': percentile,
                           'preprocessor:select_percentile_classification:score_func': 'chi2',
                           'rescaling:__choice__': 'min/max'}

        try:
            config = Configuration(configuration_space, config_dict)
            default_configs.append(config)
        except ValueError as e:
            print("Third default configurations %s cannot"
                                " be evaluated because of %s" %
                                (config_dict, e))

        if info["is_sparse"]:
            config_dict = {'balancing:strategy': 'weighting',
                           'classifier:__choice__': 'multinomial_nb',
                           'classifier:multinomial_nb:alpha': 1.0,
                           'classifier:multinomial_nb:fit_prior': 'True',
                           'preprocessor:__choice__': 'no_preprocessing',
                           'rescaling:__choice__': 'none'}
        else:
            config_dict = {'balancing:strategy': 'weighting',
                           'classifier:__choice__': 'gaussian_nb',
                           'preprocessor:__choice__': 'no_preprocessing',
                           'rescaling:__choice__': 'standardize'}
        try:
            config = Configuration(configuration_space, config_dict)
            default_configs.append(config)
        except ValueError as e:
            print("Forth default configurations %s cannot"
                                " be evaluated because of %s" %
                                (config_dict, e))

    elif info["task"] in REGRESSION_TASKS:
        config_dict = {'regressor:__choice__': 'sgd',
                       'regressor:sgd:loss': 'squared_loss',
                       'regressor:sgd:penalty': 'l2',
                       'regressor:sgd:alpha': 0.0001,
                       'regressor:sgd:fit_intercept': 'True',
                       'regressor:sgd:n_iter': 5,
                       'regressor:sgd:learning_rate': 'optimal',
                       'regressor:sgd:eta0': 0.01,
                       'regressor:sgd:average': 'True',
                       'preprocessor:__choice__': 'no_preprocessing',
                       'rescaling:__choice__': 'min/max'}
        try:
            config = Configuration(configuration_space, config_dict)
            default_configs.append(config)
        except ValueError as e:
            print("Second default configurations %s cannot"
                                " be evaluated because of %s" %
                                (config_dict, e))

        if info["is_sparse"]:
            config_dict = {'regressor:__choice__': 'extra_trees',
                           'regressor:extra_trees:bootstrap': 'False',
                           'regressor:extra_trees:criterion': 'mse',
                           'regressor:extra_trees:max_depth': 'None',
                           'regressor:extra_trees:max_features': 1.0,
                           'regressor:extra_trees:min_samples_leaf': 5,
                           'regressor:extra_trees:min_samples_split': 5,
                           'regressor:extra_trees:n_estimators': 100,
                           'preprocessor:__choice__': 'truncatedSVD',
                           'preprocessor:truncatedSVD:target_dim': 10,
                           'rescaling:__choice__': 'min/max'}
        else:
            config_dict = {'regressor:__choice__': 'extra_trees',
                           'regressor:extra_trees:bootstrap': 'False',
                           'regressor:extra_trees:criterion': 'mse',
                           'regressor:extra_trees:max_depth': 'None',
                           'regressor:extra_trees:max_features': 1.0,
                           'regressor:extra_trees:min_samples_leaf': 5,
                           'regressor:extra_trees:min_samples_split': 5,
                           'regressor:extra_trees:n_estimators': 100,
                           'preprocessor:__choice__': 'pca',
                           'preprocessor:pca:keep_variance': 0.9,
                           'preprocessor:pca:whiten': 'False',
                           'rescaling:__choice__': 'min/max'}

        try:
            config = Configuration(configuration_space, config_dict)
            default_configs.append(config)
        except ValueError as e:
            print("Third default configurations %s cannot"
                                " be evaluated because of %s" %
                                (config_dict, e))

    else:
        print("Tasktype unknown: %s" %
                         TASK_TYPES_TO_STRING[info["task"]])

    return default_configs