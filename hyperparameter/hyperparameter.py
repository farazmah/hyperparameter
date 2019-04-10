"""
This is the main module of the hyperparameter tuning using hyperopt
"""

import logging
from abc import ABCMeta, abstractmethod

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
        Determines if the hyperparameter object is used for classification, default is regression

    attributes
    ----------
    loss_metric: str
        specifying the loss function to be minimised.  Dependent on is_classifier paramter
    estimator: lightgbm object
        lightgbm estimator object. Classifier or Regressor dependent on is_classifier parameter
    params: dict
        tuned paramters after tune_model method is called
    trials: hyperopt.Trials
        the hyperopt trials object after hyperparameter tuning.
    """

    __metaclass__ = ABCMeta

    def __init__(self, is_classifier=False):
        if is_classifier:
            self.loss_metric = 'neg_log_loss'
        else:
            self.loss_metric = 'neg_mean_squared_error'
        self.params = None
        self.trials = None

    @classmethod
    @abstractmethod
    def optimize(cls, trials, score, evals_rounds):
        raise NotImplementedError

    def create_loss_func(self, x_train, y_train, folds):
        """
        This function is responsible for creating the score function based on training data and kfolds object

        :param x_train: Pandas.Dataframe or numpy.array
            The training data
        :param y_train: pandas.Dataframe or numpy.array
            The training label
        :param folds: sklearn.model_selection
            the kfold crossvalidation object
        :return: loss_func: function
            loss function to be minimised
        """

        def loss_func(params):
            logging.info("Training with params : ")
            logging.info(params)
            model = self.estimator.set_params(**params)
            loss = cross_val_score(model, x_train, y_train, cv=folds, scoring=self.loss_metric).mean() * -1
            logging.info("\tLoss {0}\n".format(loss))
            return {'loss': loss, 'status': STATUS_OK}

        return loss_func

    def tune_model(self, ds_x, ds_y, folds, eval_rounds=100, trials=None, mon_cons=None, categorical=None):
        """
        Main function responsible for tuning hyperparameters

        :param ds_x: pandas.Dataframe or numpy.array
            Training data
        :param ds_y: pandas.Dataframe or numpy.array
            Training label
        :param folds: sklearn.model_selection or sklearn.cross_validation object
            fkolds for crossvaliation of hyperparameters
        :param eval_rounds: int
            number of iterations to run the hyperparameter turning
        :param trials: hyperopt.Trials object
            pretuned hyperopt trials object if available
        :return: parameters: dict
            the best hyperparameters
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
        parameters = self.optimize(trials, loss_func, additional_evals, mon_cons, categorical)

        # Convert the relevant hyperparameters to int
        parameters['n_estimators'] = int(parameters['n_estimators'])
        parameters['max_depth'] = int(parameters['max_depth'])
        parameters['num_leaves'] = int(parameters['num_leaves'])

        self.params = parameters
        self.trials = trials

        return parameters


class LightgbmHyper(Hyperparemeter):

    def __init__(self, is_classifier=False):
        super().__init__(is_classifier=False)
        if is_classifier:
            self.estimator = lightgbm.LGBMClassifier()
        else:
            self.estimator = lightgbm.LGBMRegressor()

    @classmethod
    def optimize(cls, trials, score, evals_rounds, mon_cons=None, categorical=None):
        """
        This function specifies the hyperparameter search space and minimises the score function

        :param trials: hyperopt.Trials
            hyperopt trials object responsible for the hyperparameter search
        :param score: function
            the loss or score function to be minimised
        :param evals_rounds: int
            number of evaluation rounds for hyperparameter tuning
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
            'monotone_constraints': mon_cons,
            'categorical_feature': categorical
        }

        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=evals_rounds)
        logging.info('BEST_PARAMETERS')
        logging.info(best)
        return best

