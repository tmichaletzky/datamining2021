import numpy as np
import pandas as pd
from datetime import datetime as dt

from models import Embeddings, evaluate_classifiers, create_heatmap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


MODELS = dict([#("Nearest Neighbors", KNeighborsClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Neural Net", MLPClassifier()),
                ("Bernoulli Naive Bayes", BernoulliNB()),
                ("Logistic Regression", LogisticRegression())]
            )
MODELS["Dummy (stratified)"] = DummyClassifier(strategy='stratified')
MODELS["Dummy (uniform)"] = DummyClassifier(strategy='uniform')

df = pd.read_csv("data/Reddit_Data.csv")
columns = ["text", "score"]
df = df.rename(columns={'clean_comment':columns[0], 'category':columns[1]})
df.dropna(axis=0, inplace=True) #NA adatok eldobása

config = {
    "embedding": "tfidf",
    "drop_neutral": True,
    "dim": 250,
    "window": 5,
    "min_count": 1,
    "workers": 2,
    "epochs": 10
}

train_test_data = Embeddings(df, config["embedding"], columns, config["drop_neutral"], config=config)

is_multi = len(set(train_test_data[-2])) > 2
metrics_df = evaluate_classifiers(MODELS, train_test_data, is_multi)

date_str = dt.now().strftime("%Y-%m-%d_%H:%M")
df_name = "%s_%s_%s_%sdim" % (date_str, config["embedding"], config["drop_neutral"], config["dim"])
metrics_df.to_csv("results/%s"%df_name, index=False)

fig = create_heatmap(metrics_df, config["embedding"])
fig.savefig("results/%s"%df_name)

