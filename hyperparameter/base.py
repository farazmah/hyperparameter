"""
This is the base module of the hyperparameter tuner
"""
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, log_loss
from hyperopt import STATUS_OK, Trials

from .utils import logger


class Hyperparameter:
    """
    This is the base class of the hyperparameter framework.

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
                test_preds = model.predict(x_te)[:,]
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
            logger.info("Training with params : ")
            logger.info(params)
            model = self.estimator.set_params(**params)
            loss = self.cross_validation_score(model, x_train, y_train, folds, groups)
            logger.info("\tLoss {0}\n".format(loss))
            return {"loss": loss, "status": STATUS_OK}

        return loss_func

    def tune_model(
        self,
        ds_x,
        ds_y,
        folds,
        eval_rounds=100,
        groups=None,
        trials=None,
        mon_cons=None,
        categorical=None,
    ):
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
        parameters = self.optimize(
            trials, loss_func, additional_evals, mon_cons, categorical
        )

        self.params = parameters
        self.trials = trials

        return parameters
