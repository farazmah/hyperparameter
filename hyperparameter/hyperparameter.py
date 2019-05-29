"""
This is the main module of the hyperparameter tuning using hyperopt
"""

import logging
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, log_loss
import lightgbm
import xgboost

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Hyperparameter:
    """
    This is the main class of the hyperparameter framework.

    parameters
    ----------
    is_classifier : bool
        Determines if the hyperparameter object is used for classification, default is regression

    attributes
    ----------
    loss_metric: sklearn.metrics object
        specifying the loss function to be minimised.  Dependent on is_classifier parameter
    params: dict
        tuned parameters after tune_model method is called
    trials: hyperopt.Trials
        the hyperopt trials object after hyperparameter tuning.
    """

    __metaclass__ = ABCMeta

    def __init__(self, is_classifier=False):
        self.is_classier = is_classifier
        if is_classifier:
            self.loss_metric = log_loss
        else:
            self.loss_metric = mean_squared_error
        self.params = None
        self.trials = None

    @classmethod
    @abstractmethod
    def optimize(cls, trials, score, evals_rounds, mon_cons, categorical):
        """
        abstract method for space and objective
        """
        raise NotImplementedError

    def cross_validation_score(self, model, x, y, cv, groups):
        """
        The purpose of this function is to calculate the crossvalidation accuracy measure

        :param model: xgboost.booster or lightgbm.booster
            estimator used to fit and predict
        :param x: numpy.array
            training dataset
        :param y: numpy.array
            training labels
        :param cv: sklearn.model_selection object
            sklearn cross validation object used for accuracy measure
        :param groups: numpy.array
            Used only for GroupKFolds and GroupShuffleSplit
        :return: losses: float
            Average loss for the folds
        """
        losses = []
        for train_idx, test_idx in cv.split(x, y, groups):
            x_tr, x_te = x[train_idx], x[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(x_tr, y_tr)
            if self.is_classier:
                test_preds = model.predict_proba(x_te)[:, 1]
            else:
                test_preds = model.predict(x_te)[:, ]
            loss = self.loss_metric(y_true=y_te, y_pred=test_preds)
            losses.append(loss)
        return np.mean(losses)

    def create_loss_func(self, x_train, y_train, folds, groups):
        """
        This function is responsible for creating the score function based on training data and kfolds object

        :param x_train: Pandas.Dataframe or numpy.array
            The training data
        :param y_train: pandas.Dataframe or numpy.array
            The training label
        :param folds: sklearn.model_selection
            the kfold crossvalidation object
        :param groups: numpy.array
            index of groups used for GroupKfold
        :return: loss_func: function
            loss function to be minimised
        """
        def loss_func(params):
            logging.info("Training with params : ")
            logging.info(params)
            model = self.estimator.set_params(**params)
            loss = self.cross_validation_score(model, x_train, y_train, folds, groups)
            logging.info("\tLoss {0}\n".format(loss))
            return {'loss': loss, 'status': STATUS_OK}

        return loss_func

    def tune_model(self, ds_x, ds_y, folds, eval_rounds=100, groups=None, trials=None, mon_cons=None,
                   categorical=None):
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
        :param groups numpy.array
            index of groups used for GroupKFold
        :param trials: hyperopt.Trials object
            pretuned hyperopt trials object if available
        :param mon_cons: str(tuple) for xgboost, tuple for lightgbm
            index of monotonic constraints
        :param categorical: list
            index of categorical feature for lightgbm
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
        loss_func = self.create_loss_func(ds_x, ds_y, folds, groups)

        # Find optimal hyperparameters
        parameters = self.optimize(trials, loss_func, additional_evals, mon_cons, categorical)

        self.params = parameters
        self.trials = trials

        return parameters


class LightgbmHyper(Hyperparameter):

    def __init__(self, is_classifier=False):
        super().__init__(is_classifier=False)
        if is_classifier:
            self.estimator = lightgbm.LGBMClassifier()
        else:
            self.estimator = lightgbm.LGBMRegressor()

    @classmethod
    def optimize(cls, trials, score, evals_rounds, mon_cons, categorical):
        """
        This function specifies the hyperparameter search space and minimises the score function

        :param trials: hyperopt.Trials
            hyperopt trials object responsible for the hyperparameter search
        :param score: function
            the loss or score function to be minimised
        :param evals_rounds: int
            number of evaluation rounds for hyperparameter tuning
        :param mon_cons: str(tuple) for xgboost, tuple for lightgbm
            index of monotonic constraints
        :param categorical: list
            index of categorical feature for lightgbm
        :return best: dict
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
            # 'categorical_feature': categorical
        }

        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=evals_rounds)

        # Convert the relevant hyperparameters to int
        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        best['num_leaves'] = int(best['num_leaves'])
        best['min_child_samples'] = int(best['min_child_samples'])

        logging.info('BEST_PARAMETERS')
        logging.info(best)
        return best


class XgboostHyper(Hyperparameter):

    def __init__(self, is_classifier=False):
        super().__init__(is_classifier=False)
        if is_classifier:
            self.estimator = xgboost.XGBClassifier()
        else:
            self.estimator = xgboost.XGBRegressor()

    @classmethod
    def optimize(cls, trials, score, evals_rounds, mon_cons, categorical):
        """
        This function specifies the hyperparameter search space and minimises the score function

        :param trials: hyperopt.Trials
            hyperopt trials object responsible for the hyperparameter search
        :param score: function
            the loss or score function to be minimised
        :param evals_rounds: int
            number of evaluation rounds for hyperparameter tuning
        :param mon_cons: str(tuple) for xgboost, tuple for lightgbm
            index of monotonic constraints
        :param categorical: list
            index of categorical feature for lightgbm
        :return best: dict
            the best hyperparameters
        """
        space = {
            'n_estimators': scope.int(hp.uniform('n_estimators', 10, 3000)),
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
            'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 1),
            'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.05, 3, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.05),
            'colsample_bylevel': hp.quniform('colsample_bylevel', 0.4, 1, 0.05),
            'reg_lambda': hp.quniform('reg_lambda', 0.01, 2, 0.01),
            'reg_alpha': hp.quniform('reg_alpha', 0, 10, 1),
            #'monotone_constraints': mon_cons,
        }

        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=evals_rounds)

        # Convert the relevant hyperparameters to int
        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        best['min_child_weight'] = int(best['min_child_weight'])

        logging.info('BEST_PARAMETERS')
        logging.info(best)
        return best
