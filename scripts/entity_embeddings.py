import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from sklearn import preprocessing

def feature_engineering():
    df = pd.read_csv("./resources/adult.csv")

    numerical_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    
    target_mapping = {
        "<=50K": 0, 
        ">50K":1
    }
    df["income"] =df["income"].map(target_mapping)

    features = [
        f for f in df.columns if f not in ("income", "kfold")
    ]

    category_count = {}
    category_dim = {}
    for col in features:
        if col not in numerical_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df[col] = lbl.transform(df[col])
            category_count[col] = len(lbl.classes_)
            category_dim[col] = int(category_count[col]//2) if category_count[col]<=100 else 50

    return df, category_count, category_dim

class IncomeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()

        sample = {key: torch.tensor(value) for key, value in sample.items()}

        return sample

class IncomeClassifier(nn.Module):
    def __init__(self, category_count, category_dim):
        super(IncomeClassifier, self).__init__()
        self.category_count = category_count
        self.category_dim = category_dim

        # Embedding
        self.workclass_embd = nn.Embedding(num_embeddings=self.category_count["workclass"], embedding_dim=category_dim["workclass"])
        self.education_embd = nn.Embedding(num_embeddings=self.category_count["education"], embedding_dim=category_dim["education"])
        self.education_num_embd = nn.Embedding(num_embeddings=self.category_count["education.num"], embedding_dim=category_dim["education.num"])
        
        self.marital_status_embd = nn.Embedding(num_embeddings=self.category_count["marital.status"], embedding_dim=category_dim["marital.status"])
        self.occupation_embd = nn.Embedding(num_embeddings=self.category_count["occupation"], embedding_dim=category_dim["occupation"])

        self.relationship_embd = nn.Embedding(num_embeddings=self.category_count["relationship"], embedding_dim=category_dim["relationship"])
        self.race_embd = nn.Embedding(num_embeddings=self.category_count["race"], embedding_dim=category_dim["race"])
        self.sex_embd = nn.Embedding(num_embeddings=self.category_count["sex"], embedding_dim=category_dim["sex"])
        self.native_country_embd = nn.Embedding(num_embeddings=self.category_count["native.country"], embedding_dim=category_dim["native.country"])

        self.total_dim = sum(list(category_dim.values())) + 5

        # neural netwrok layers
        self.norm_0 = nn.BatchNorm1d(num_features=self.total_dim)
        self.drop_0 = nn.Dropout(p=0.3)

        self.linear_1 = nn.Linear(self.total_dim, 300)
        self.relu_1 = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(num_features=300)
        self.drop_1 = nn.Dropout(p=0.3)

        self.linear_2 = nn.Linear(300, 300)
        self.relu_2 = nn.ReLU()
        self.norm_2 = nn.BatchNorm1d(num_features=300)
        self.drop_2 = nn.Dropout(p=0.3)

        self.linear_3 = nn.Linear(300, 2)

    def forward(self, x):
        x_workclass = self.workclass_embd(x["workclass"])
        x_education = self.education_embd(x["education"])
        x_education_num = self.education_num_embd(x["education.num"])
        x_marital_status = self.marital_status_embd(x["marital.status"])
        x_occupation = self.occupation_embd(x["occupation"])
        x_relationship = self.relationship_embd(x["relationship"])
        x_race = self.race_embd(x["race"])
        x_sex = self.sex_embd(x["sex"])
        x_native_country = self.native_country_embd(x["native.country"])

        x_fnlwgt = x["fnlwgt"].unsqueeze(1)
        x_age = x["age"].unsqueeze(1)
        x_capital_gain = x["capital.gain"].unsqueeze(1)
        x_capital_loss = x["capital.loss"].unsqueeze(1)
        x_hours_per_week = x["hours.per.week"].unsqueeze(1)

        # concatenate everything above to create a 1d tensor
        x = torch.cat(
                (x_workclass, 
                x_education, 
                x_education_num, 
                x_marital_status, 
                x_occupation, 
                x_relationship, 
                x_race, 
                x_sex, 
                x_native_country,
                x_fnlwgt,
                x_age,
                x_capital_gain,
                x_capital_loss,
                x_hours_per_week), 
             dim=1)
        
        x = self.norm_0(x)
        x = self.drop_0(x)
        
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.norm_1(x)
        x = self.drop_1(x)

        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.norm_2(x)
        x = self.drop_2(x)

        x = self.linear_3(x)

        return x


def train(data_frame, category_count, category_dim):
    dataset = IncomeDataset(data_frame)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = IncomeClassifier(category_count, category_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs["income"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    df, cc, cd = feature_engineering()

    train(df, cc, cd)

    
    



    

    