"""
This is the main module of the hyperparameter tuning for lightgbm using hyperopt
"""

import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
import lightgbm

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# Logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Hyperparemeter:
    """
    This is the main class of the hyperparameter framework.

    parameters
    ----------
    is_classifier : bool
        Determines if the hyperparameter object is used for classification, defaul is regression
    """

    def __init__(self, is_classifier=False, is_xgboost=False):
        if is_classifier:
            self.loss_metric = 'neg_log_loss'
            self.estimator = lightgbm.LGBMClassifier()
        else:
            self.loss_metric = 'neg_mean_squared_error'
            self.estimator = lightgbm.LGBMRegressor()
        self.params = None
        self.trials = None

    def optimize(self, trials, score, added_evals):
        """
        This function specifies the hyperparameter search space and minimises the score function

        :param trials: hyperopt.Trials
            hyperopt trials object responsible for the hyperparameter search
        :param mon_cons: str
            string of tuples specifying the monotonic constraint
        :param score: function
            the loss or score function to be minimised
        :param added_evals: int
            additional evals for hyperopt if previously tuned
        :return: best: dict
            the best hyperparameters
        """
        space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 3000, 5)),
            'learning_rate': hp.quniform('learning_rate', 0.05, 0.3, 0.025),
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 2, 1024, 2)),
            'min_child_samples': scope.int(hp.quniform('min_child_samples', 2, 100, 1)),
            'subsample': hp.quniform('subsample', 0.6, 1, 0.05),  # bagging_fraction
            'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.1),  # feature_fraction
            'min_sum_hessian_in_leaf': hp.quniform('min_sum_hessian_in_leaf', 0.001, 0.9, 0.001),
            'reg_lambda': hp.quniform('reg_lambda', 0.01, 1, 0.01),
            'reg_alpha': hp.quniform('reg_alpha', 1, 10, 0.01),
        }

        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=added_evals)
        logging.info('BEST_PARAMETERS')
        logging.info(best)
        return best


    def create_loss_func(self, X_train, y_train, folds):
        """
        This function is responsible for creating the score function based on training data and kfolds object

        :param X_train: Pandas.Dataframe
            The training data
        :param y_train: pandas.Dataframe
            The training label
        :param folds: sklearn.model_selection.Kfold
            the kfold crossvalidation object
        :return: score: function
            score function to be minimised
        """

        def loss_func(params):
            logging.info("Training with params : ")
            logging.info(params)
            model = self.estimator.set_params(**params)
            loss = cross_val_score(model, X_train, y_train, cv=folds, scoring=self.loss_metric).mean() * -1
            logging.info("\tLoss {0}\n".format(loss))
            return {'loss': loss, 'status': STATUS_OK}

        return loss_func

    def tune_model(self, ds_x, ds_y, folds, eval_rounds=100, trials=None):
        """
        Main function responsible for tuning hyperparameters

        :param ds_x: pandas.Dataframe
            Training data
        :param ds_y: pandas.Dataframe
            Training label
        :param folds: sklearn model_selection or cross_validation object
            fkolds for crossvaliation of hyper parameter
        :param trials: hyperopt.Trials object
            pretuned hyperopt trials object if available
        :return: parameters: dict
            the best hyperparameters
                 trials: hyperopt.Trials
            the hyperopt trials object post tuning for saving
        """
        # Create hyperopt Trials object
        if trials is None:
            trials = Trials()
            additional_evals = eval_rounds
        else:
            additional_evals = len(trials.losses()) + eval_rounds

        # Create the loss function
        loss_func = self.create_loss_func(ds_x, ds_y, folds)

        # Find optimal hyperparameters
        parameters = self.optimize(trials, loss_func, additional_evals)

        # Convert the relevant hyperparameters to int
        parameters['n_estimators'] = int(parameters['n_estimators'])
        parameters['max_depth'] = int(parameters['max_depth'])
        parameters['num_leaves'] = int(parameters['num_leaves'])

        self.params = parameters
        self.trials = trials

        return parameters
