import numpy as np
import pandas as pd

#For Preprocessing
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
my_stopwords = stopwords.words("english")

def text_preprocess(text):
    ''' Convert tweet text into a sequence of words '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [w for w in words if w not in my_stopwords]
    words = [stemmer.stem(w) for w in words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

# Models
from sklearn.model_selection import train_test_split

embeddings = ["tfidf", "word2vec", "doc2vec"] 
def Embeddings(df, model="tfidf", columns=["text", "score"], drop_neutral=False, config=None):
    if drop_neutral:
        df = df[df["score"] != 0]
    X = list(map(text_preprocess, df[columns[0]]))
    Y = list(df[columns[1]]) 
    X_tr, X_te, y_tr, y_te = train_test_split(X, Y)
    if model == "tfidf":
        X_train, X_test = TfIdf(X_tr, X_te, dim=config["dim"])
    elif model == "word2vec":
        X_train, X_test = Word2Vec(X_tr, X_te, dim=config["dim"], window=config["window"], min_count=config["min_count"], workers=config["workers"])
    elif model == "doc2vec":
        X_train, X_test = Doc2Vec(X_tr, X_te, dim=config["dim"], epochs=config["epochs"], window=config["window"], min_count=config["min_count"], workers=config["workers"])
    else:
        raise "Invalid embedding type, choose from %s"%(embeddings) 
    print(X_train.shape, X_test.shape, len(set(y_tr)), len(set(y_te)))      
    return X_train, X_test, y_tr, y_te   

    
from sklearn.feature_extraction.text import TfidfVectorizer
def TfIdf(X_tr, X_te, dim=250):
    X_train = np.array([' '.join(words) for words in X_tr])
    X_test = np.array([' '.join(words) for words in X_te])
    vectorizer = TfidfVectorizer(max_features=dim)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    return X_train, X_test

from gensim.models.word2vec import Word2Vec
def w2v_infer_vector(model, sentence):
    keys = [w for w in sentence if w in model.wv]
    if len(keys) > 0:
        return np.mean(model.wv[keys], axis=0)
    else:
        return np.zeros(model.vector_size)

def Word2Vec(X_tr, X_te, dim=100, window=5, min_count=1, workers=2):
    embedder = Word2Vec(X_tr, dim, window=window, min_count=min_count, workers=workers)
    X_train = [list(w2v_infer_vector(embedder, sentence)) for sentence in X_tr]
    X_test = [list(w2v_infer_vector(embedder, sentence)) for sentence in X_te]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    return X_train, X_test

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
def Doc2Vec(X_tr, X_te, dim=100, epochs=10, window=5, min_count=1, workers=2):
    tagged_doc = [TaggedDocument(doc, [idx]) for idx, doc in enumerate(X_tr)]
    embedder = Doc2Vec(tagged_doc, vector_size=dim, window=window, min_count=min_count, epochs=epochs, workers=workers)
    X_train = [list(embedder.infer_vector(sentence)) for sentence in X_tr]
    X_test = [list(embedder.infer_vector(sentence)) for sentence in X_te]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    
#Evaluate    
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, recall_score, precision_score, matthews_corrcoef
    
def metrics2df(metrics_dict):
    records = []
    for model in metrics_dict:
        for part in metrics_dict[model]:
            for metric in metrics_dict[model][part]:
                score = metrics_dict[model][part][metric]
                records.append([model, part, metric, score])
    return pd.DataFrame(records, columns=["model","part","metric","score"])

def calculate_metrics(model, X, y, multiclass=False, show_confusion_matrix=False, verbose=True):
    y_proba = model.predict_proba(X)
    y_pred = list(map(lambda x: x-1, np.argmax(y_proba, axis=1)))
    if show_confusion_matrix:
        cm = confusion_matrix(y, y_pred)
        print(model)
        print(cm)
    if multiclass is None:
        multiclass = len(set(y)) > 2
    average = "macro" if multiclass else "binary"
    metrics = {}
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average=average)
    rec = recall_score(y, y_pred, average=average)
    f1 = f1_score(y, y_pred, average=average)
    if multiclass:
        auc = roc_auc_score(y, y_proba, multi_class="ovo")
    else:
        auc = roc_auc_score(y, y_proba)
    metrics = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc
    }
    print(metrics)
    return metrics, y_pred, y_proba

def evaluate_classifier(model, data, multiclass=False, show_confusion_matrix=False, verbose=True):
    X_train, X_test, y_train, y_test = data
    model = model.fit(X_train, y_train)
    results = {}
    if verbose:
        print(model)
        print("TRAIN:")
    results["train"] = calculate_metrics(model, X_train, y_train, multiclass, show_confusion_matrix, verbose)[0]
    if verbose:
        print("TEST:")
    results["test"] = calculate_metrics(model, X_test, y_test, multiclass, show_confusion_matrix, verbose)[0]
    if verbose:
        print()
    return results

def evaluate_classifiers(model_dict, train_test_data, multiclass=True, show_confusion_matrix=True, verbose=True):
    names, classifiers = zip(*model_dict.items())
    results = {}
    for i in range(len(classifiers)):
        results[names[i]] = evaluate_classifier(classifiers[i], train_test_data, multiclass, show_confusion_matrix, verbose)
    metrics_df = metrics2df(results)
    metrics_df["dimension"] = train_test_data[0].shape[1]
    return metrics_df

import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap(metrics_df, embedding, part="test", split="metric", fsize=(8,4)):
    tmp_df = metrics_df[metrics_df["part"] == "test"]
    pivot_df = tmp_df.pivot(index="model", columns=split, values="score")    
    fig, ax = plt.subplots(1,1,figsize=fsize)
    sns.heatmap(pivot_df, cmap='RdYlGn', annot=True, fmt=".3", ax=ax)#, vmin=0.5, vmax=1.0)
    plt.title("Embedding: %s, Part: %s" % (embedding, part))
    return fig
    
def train_test_dim_performance(metrics_df, split, metric="f1", title=None, class_value=None, fsize=(6,4)):
    tmp_df = metrics_df[metrics_df["metric"]==metric]
    if class_value != None and "class" in metrics_df.columns:
        tmp_df = tmp_df[tmp_df["class"]==class_value]
    fig, ax = plt.subplots(1,1,figsize=fsize)
    if title != None:
        plt.title(title)
    sns.lineplot(data=tmp_df, x=split, y="score", hue="model", style="part", ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig

