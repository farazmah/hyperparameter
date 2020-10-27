"""
Xgboost hyperparameter tuning module
"""
import xgboost
from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope

from .base import Hyperparameter
from .utils import logger


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
            "n_estimators": scope.int(hp.uniform("n_estimators", 10, 3000)),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.3, 0.01),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
            "min_child_weight": hp.quniform("min_child_weight", 1, 9, 1),
            "subsample": hp.quniform("subsample", 0.6, 1, 0.05),
            "gamma": hp.quniform("gamma", 0.05, 3, 0.05),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 1, 0.05),
            "colsample_bylevel": hp.quniform("colsample_bylevel", 0.4, 1, 0.05),
            "reg_lambda": hp.quniform("reg_lambda", 0.01, 2, 0.01),
            "reg_alpha": hp.quniform("reg_alpha", 0, 10, 1),
            #'monotone_constraints': mon_cons,
        }

        best = fmin(
            score, space, algo=tpe.suggest, trials=trials, max_evals=evals_rounds
        )

        # Convert the relevant hyperparameters to int
        best["n_estimators"] = int(best["n_estimators"])
        best["max_depth"] = int(best["max_depth"])
        best["min_child_weight"] = int(best["min_child_weight"])

        logger.info("BEST_PARAMETERS")
        logger.info(best)
        return best
