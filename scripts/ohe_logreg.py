import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("./resources/cat_train_folds.csv")


    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        df[col] = df[col].astype(str).fillna('NONE')

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_val = df[df["kfold"] == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [df_train[features], df_val[features]],
        axis=0
    )

    ohe.fit(df[features])

    x_train = ohe.transform(df_train[features])
    x_val = ohe.transform(df_val[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train["target"].values)

    valid_preds = model.predict_proba(x_val)[:, 1]

    auc = metrics.roc_auc_score(df_val['target'].values, valid_preds)

    print(f"Fold={fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

