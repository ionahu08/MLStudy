import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    df = pd.read_csv("./resources/imdb.csv")

    df["sentiment"] = df["sentiment"].apply(
        lambda x: 1 if x == "positive" else 0
    )

    df["kfold"] = -1

    df=df.sample(frac=1).reset_index(drop=True)

    y = df["sentiment"].values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (train_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, "kfold"] = f
    
    for fold_ in range(5):
        train_df = df[df["kfold"] != fold_].reset_index(drop=True)

        test_df = df[df["kfold"] == fold_].reset_index(drop=True)

        count_vec = CountVectorizer(
            tokenizer=word_tokenize, 
            token_pattern=None
        )

        count_vec.fit(train_df)

        xtrain = count_vec.transform(train_df["review"])
        xtest = count_vec.transform(test_df["review"])

        model = linear_model.LogisticRegression()

        model.fit(xtrain, train_df["sentiment"])
        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(test_df['sentiment'], preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy: {accuracy}")
        print("     ")

        