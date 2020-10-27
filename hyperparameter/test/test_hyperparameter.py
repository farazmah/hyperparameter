"""
test module
"""
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

# Preparing test data
path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.split(path_to_current_file)[0]
path_to_file = os.path.join(current_directory, "titanic_train.csv")

df = pd.read_csv(path_to_file)
df = df.replace("", np.nan)
df = df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
df = pd.get_dummies(columns=['Embarked', 'Sex'],data=df)
y_train = df.Survived.values
X_train = df.drop(['Survived'], axis=1).values

skf = StratifiedKFold(n_splits=3)


def test_lightgbm_hyper():
    from ..lgbm import LightgbmHyper
    hpopt = LightgbmHyper(is_classifier=True)
    hpopt.tune_model(ds_x=X_train, ds_y=y_train, folds=skf, eval_rounds=2)


def test_xgboost_hyper():
    from ..xgb import XgboostHyper
    hpopt = XgboostHyper(is_classifier=True)
    hpopt.tune_model(ds_x=X_train, ds_y=y_train, folds=skf, eval_rounds=2)

