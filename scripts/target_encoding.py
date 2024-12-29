import pandas as pd
import copy

import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def mean_target_encoding(data):
    df = copy.deepcopy(data)

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0, 
        ">50K":1
    }
    df["income"] =df["income"].map(target_mapping)


    # all columns are features except kfold & income columns
    features = [
        f for f in df.columns if f not in ("income", "kfold") 
        and f not in num_cols
    ]

    for col in features:
        if col not in num_cols:
            df[col] = df[col].astype(str).fillna('NONE')

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df[col] = lbl.transform(df[col])

    # a list to store 5 validation dataframes
    encoded_dfs = []

    for fold in range(5):
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_val = df[df["kfold"] == fold].reset_index(drop=True)

        for column in features:
            mapping_dict = dict(
                df_train.groupby(column)["income"].mean()
            )
            
            df_val.loc[
                :,
                column+"_enc"
            ] = df_val[column].map(mapping_dict)

        encoded_dfs.append(df_val)
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(df, fold):
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_val = df[df["kfold"] == fold].reset_index(drop=True)
    

    features = [
        f for f in df.columns if f not in ("income", "kfold") 
    ]

    x_train = df_train[features].values
    x_val = df_val[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
    )

    model.fit(x_train, df_train["income"].values)

    valid_preds = model.predict_proba(x_val)[:, 1]

    auc = metrics.roc_auc_score(df_val['income'].values, valid_preds)

    print(f"Fold={fold}, AUC = {auc}")






if __name__ == "__main__":
    df = pd.read_csv("./resources/adult_folds.csv")

    df = mean_target_encoding(df)
    df.to_csv("./resources/adult_folds_derived.csv", index=False)

    
    for fold_ in range(5):
        run(df, fold_)



