### Hyperparameter optimisation utility for [lightgbm](https://github.com/microsoft/LightGBM) and [xgboost](https://github.com/dmlc/xgboost) using [hyperopt](https://github.com/hyperopt/hyperopt)

Installation
```
git clone git@github.com:farazmah/hyperparameter.git
cd hyperparamter/hyperparameter
python ../setup.py install
```



Usage example for lightGBM:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("hyperparameter/test/titanic_train.csv")
df = df.replace("", np.nan)
df = df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
df = pd.get_dummies(columns=['Embarked', 'Sex'],data=df)
y_train = df.Survived.values
X_train = df.drop(['Survived'], axis=1).values

skf = StratifiedKFold(n_splits=3)

from hyperparameter.lgbm import LightgbmHyper
hpopt = LightgbmHyper(is_classifier=True)

hpopt.tune_model(ds_x=X_train, ds_y=y_train, folds=skf, eval_rounds = 20)
```

```python
Out[1]: {'colsample_bytree': 0.9,
         'learning_rate': 0.17500000000000002,
         'max_depth': 19,
         'min_child_samples': 68,
         'min_sum_hessian_in_leaf': 0.256,
         'n_estimators': 505,
         'num_leaves': 186,
         'reg_alpha': 1.84,
         'reg_lambda': 0.35000000000000003,
         'subsample': 0.7000000000000001}
```


Usage example for xgboost (last three lines from example above changes to):

```python
from hyperparameter.xgb import XgboostHyper
hpopt = XgboostHyper(is_classifier=True)

hpopt.tune_model(ds_x=X_train, ds_y=y_train, folds=skf, eval_rounds = 20)
```

```python
Out[2]: {'colsample_bylevel': 0.65,
         'colsample_bytree': 0.5,
         'gamma': 0.75,
         'learning_rate': 0.24,
         'max_depth': 4,
         'min_child_weight': 6,
         'n_estimators': 1290,
         'reg_alpha': 2.0,
         'reg_lambda': 1.48,
         'subsample': 0.7000000000000001}
```