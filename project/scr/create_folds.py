#create_folds

# This py file will create a new file in the input/ folder called mnist_train_folds.csv, and it’s
# the same as mnist_train.csv. The only differences are that this CSV is shuffled and
# has a new column called kfold.
import pandas as pd
from sklearn import model_selection

def create_kfold_data():
    df = pd.read_csv('../input/mnist_train.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df['kfold'] = -1
    kf = model_selection.KFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    csv_file = "../input/mnist_train_folds.csv"
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    create_kfold_data