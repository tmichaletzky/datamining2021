import numpy as np
import pandas as pd
from datetime import datetime as dt

from models import Embeddings, score2label, evaluate_classifiers, create_heatmap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.ensemble import *

MODELS = dict([ ("Random Forest", RandomForestClassifier()),
                ("Neural Net", MLPClassifier()),
                ("Bernoulli Naive Bayes", BernoulliNB()),
                ("Logistic Regression", LogisticRegression()),
                ("Nearest Neighbors", KNeighborsClassifier())
              ])

df = pd.read_csv("data/Reddit_Data.csv")
df["category"] = list(map(score2label, df["category"]))
df = df.rename(columns={'clean_comment':"text", 'category':"label"})
df.dropna(axis=0, inplace=True) #NA adatok eldobása

columns = ["text", "label"] # df.columns
print("\n", df.head(), "\n")

config = {
    "embedding": "word2vec",
    "drop_neutral": True,
    "dim": 250,
    "window": 5,
    "min_count": 1,
    "workers": 2,
    "epochs": 2,
    "num_times": 5
}

DEBUG = False
if DEBUG:
    df = df.loc[:100]
    MODELS = {}
    MODELS["Dummy (stratified)"] = DummyClassifier(strategy='stratified')
    MODELS["Dummy (uniform)"] = DummyClassifier(strategy='uniform')

is_multi = False if config["drop_neutral"] else True
instances = []
for _ in range(config["num_times"]):
    print("Multiclass: ", is_multi)
    train_test_data = Embeddings(df, config["embedding"], columns, config["drop_neutral"], config=config)
    metrics_df = evaluate_classifiers(MODELS, train_test_data, is_multi)
    instances.append(metrics_df)
if len(instances) > 1:
    metrics_df = pd.concat(instances, axis=0)
    group_cols = list(metrics_df.drop("score", axis=1).columns)
    metrics_df = metrics_df.groupby(group_cols)["score"].agg(["mean", "std"]).reset_index()

if not DEBUG:
    date_str = dt.now().strftime("%Y-%m-%d_%H:%M")
    df_name = "%s_%s_%s_%sdim_%itimes" % (date_str, config["embedding"], config["drop_neutral"], config["dim"], config["num_times"])
    metrics_df.to_csv("results/%s"%df_name, index=False)
    
    values = "mean" if config["num_times"] > 1 else "score"
    fig = create_heatmap(metrics_df, config["embedding"], values)
    fig.savefig("results/%s"%df_name)
else:
    print(metrics_df)

