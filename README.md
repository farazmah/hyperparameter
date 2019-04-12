### Hyperparameter optimisation framework for lightgbm and xgboost

Installation
```
git clone https://faraz_ma@bitbucket.org/wesaac/hyperparameter.git
cd hyperparamter/hyperparameter
python ../setup.py install
```



Usage example for lightGBM:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

df = pd.read_json("../titanic/train.json")
df = df.replace("", np.nan)
df = df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
df = pd.get_dummies(columns=['Embarked', 'Sex'],data=df)
y_train = df.Survived.values
X_train = df.drop(['Survived'], axis=1).values

skf = StratifiedKFold(n_splits=3)

from hyperparameter import LightgbmHyper
hpopt = LightgbmHyper(is_classifier=True)

hpopt.tune_model(ds_x=X_train, ds_y=y_train, folds=skf, eval_rounds = 20)

```

