"""
lightGBM hyperparameter tuning module
"""
import lightgbm
from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope

from .base import Hyperparameter
from .utils import logger

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
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 3000, 5)),
            "learning_rate": hp.quniform("learning_rate", 0.05, 0.3, 0.025),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
            "num_leaves": scope.int(hp.quniform("num_leaves", 2, 1024, 2)),
            "min_child_samples": scope.int(hp.quniform("min_child_samples", 2, 100, 1)),
            "subsample": hp.quniform("subsample", 0.6, 1, 0.05),  # bagging_fraction
            "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 1, 0.1),  # feature_fraction
            "min_sum_hessian_in_leaf": hp.quniform("min_sum_hessian_in_leaf", 0.001, 0.9, 0.001),
            "reg_lambda": hp.quniform("reg_lambda", 0.01, 1, 0.01),
            "reg_alpha": hp.quniform("reg_alpha", 1, 10, 0.01),
            "monotone_constraints": mon_cons,
            # 'categorical_feature': categorical
        }

        best = fmin(
            score, space, algo=tpe.suggest, trials=trials, max_evals=evals_rounds
        )

        # Convert the relevant hyperparameters to int
        best["n_estimators"] = int(best["n_estimators"])
        best["max_depth"] = int(best["max_depth"])
        best["num_leaves"] = int(best["num_leaves"])
        best["min_child_samples"] = int(best["min_child_samples"])

        logger.info("BEST_PARAMETERS")
        logger.info(best)
        return best
