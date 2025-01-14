import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("./resources/cat_train_folds.csv")


    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
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



    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train["target"].values)

    valid_preds = model.predict_proba(x_val)[:, 1]

    auc = metrics.roc_auc_score(df_val['target'].values, valid_preds)

    print(f"Fold={fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

