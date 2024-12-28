import pandas as pd

import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("./resources/adult_folds.csv")

    # drop numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    df = df.drop(num_cols, axis=1)

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0, 
        ">50K":1
    }
    df["income"] =df["income"].map(target_mapping)
    

    features = [
        f for f in df.columns if f not in ("income", "kfold")
    ]

    for col in features:
        df[col] = df[col].astype(str).fillna('NONE')

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df[col] = lbl.transform(df[col])



    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_val = df[df["kfold"] == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_val = df_val[features].values



    model = xgb.XGBClassifier(
        n_jobs = -1
        # ,max_depth=7,
        # n_estimators=200
    )
    model.fit(x_train, df_train["income"].values)

    valid_preds = model.predict_proba(x_val)[:, 1]

    auc = metrics.roc_auc_score(df_val['income'].values, valid_preds)

    print(f"Fold={fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

