import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

f = pd.read_csv("fakeNews/Dataset/Fake.csv")
f.insert(4, 'label', "FAKE")
r = pd.read_csv("fakeNews/Dataset/True.csv")
r.insert(4, 'label', "REAL")
totdf = pd.concat([f,r])
labels = totdf.label
x_train, x_test, y_train, y_test = train_test_split(totdf['text'], labels, test_size = 0.2, random_state = 2)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)

# print(f'Accuracy: {round(score*100,2)}%')
def classifier(text1):
    txt_test_tfidf = tfidf_vectorizer.transform(np.array((text1,)))
    # print(txt_test_tfidf)
    return pac.predict(txt_test_tfidf)[0]

