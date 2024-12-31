import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("./resources/mobile_train.csv")
    # print(df.dtypes)

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs = -1)

    param_grid = {
        "n_estimators": [100, 200, 1500],
        "max_depth": [1,31],
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        verbose=20,
        n_jobs=1,
        cv=5
    )

    model.fit(X, y)
    print(f"Best Score: {model.best_score_}")

    print("Best Paramters Set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f'\t{param_name}: {best_parameters[param_name]}')