# src/train.py

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import argparse

PROJECT_ROOT = "/Users/ionahu/sources/MLStudy/project"

def run(fold):
    df = pd.read_csv(f"{PROJECT_ROOT}/input/mnist_train_folds.csv")

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_val = df[df["kfold"] == fold].reset_index(drop=True)

    x_train = df_train.drop("label", axis=1).values
    y_train = df_train["label"].values

    x_val = df_val.drop("label", axis=1).values
    y_val = df_val["label"].values

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)

    accuracy = metrics.accuracy_score(y_val, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    #save the model
    joblib.dump(clf, f"{PROJECT_ROOT}/models/dt_{fold}.bin")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    args = parser.parse_args()

    run(fold = args.fold)