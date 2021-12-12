# datamining2021
Adatbányászat 2021, Elte Matematikus MSc

Projekt cím: Sentiment analysis on Reddit dataset

Tagok: Forman Balázs, Michaletzky Tamás

Adathalmaz: Charan Gowda, Anirudh, Akshay Pai, and Chaithanya kumar A, “Twitter and Reddit Sentimental analysis Dataset.” Kaggle, 2019, doi: 10.34740/KAGGLE/DS/429085.

Cél: hate speech detection különböző NLP-módszerekkel
- TfIdf, word2vec, doc2vec + LogistisRegression (Tamás)
- GloVe + LogisticRegression / NaiveBayes / kNN / LSTM (Balázs)
- Bert HuggingFace (Tamás)

Python csomagok:

Anaconda:
- Numpy, Scipy, Pandas, Matplotlib

NLP:
- `from sklearn.feature_extraction.text import TfidfVectorizer`
- `from gensim.models.word2vec import Word2Vec`
- `from gensim.models.doc2vec import Doc2Vec`
- `from transformers import BertModel, BertForSequenceClassification`
- GloVe: http://nlp.stanford.edu/data/glove.6B.zip

Neural Models:
- Tensorflow, Keras
- PyTorch

Források:
- https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset
- http://nlp.stanford.edu/data/glove.6B.zip
