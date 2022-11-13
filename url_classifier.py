import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

# Import phishing_site_urls dataset
urls_df = pd.read_csv(r'phishing_site_urls.csv')
urls_df.head()
urls_df.tail()
urls_df.info()
urls_df.isnull().sum()
label_counter = pd.DataFrame(urls_df.Label.value_counts())
print(label_counter.Label)
fig = px.bar(label_counter, x=label_counter.index, y=label_counter.Label)
fig.show()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
urls_df.URL[0]
tokenizer.tokenize(urls_df.URL[0])
urls_df['text_tokenized'] = urls_df.URL.map(lambda t: tokenizer.tokenize(t))
urls_df.sample(5)
stemmer = SnowballStemmer("english")
urls_df['text_stemmed'] = urls_df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
urls_df.sample(5)
urls_df['text_sent'] = urls_df['text_stemmed'].map(lambda l: ' '.join(l))
urls_df.sample(5)

# Visualization
bad_sites = urls_df[urls_df.Label == 'bad']
good_sites = urls_df[urls_df.Label == 'good']
bad_sites.head()
good_sites.head()

# Creating Model
cv = CountVectorizer()
feature = cv.fit_transform(urls_df.text_sent)
feature[:5].toarray()
trainX, testX, trainY, testY = train_test_split(feature, urls_df.Label)

# Logistic Regression
lr = LogisticRegression()
lr.fit(trainX,trainY)
lr.score(testX,testY)

Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)
print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")

# Multinomial Naive Bayes

mnb = MultinomialNB()
mnb.fit(trainX,trainY)
mnb.score(testX,testY)

Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)
print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")

# Pro e Cons between Logistic Regression and Multinomial Naive Bayes

acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(x=acc.index,y=acc.Accuracy)

# Support Vector Machine

from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(feature, urls_df.Label, test_size=0.25, random_state=27)

SVC_model = svm.SVC()
# KNN model requires you to specify n_neighbors,
# the number of points the classifier will look at to determine what class a new point belongs to
KNN_model = KNeighborsClassifier(n_neighbors=5)

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)

SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
# Accuracy score is the simplest way to evaluate
print(confusion_matrix(SVC_prediction, y_test))
print(classification_report(KNN_prediction, y_test))


# sklearn pipeline using Logistic Regression

from sklearn.pipeline import make_pipeline
pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
trainX, testX, trainY, testY = train_test_split(urls_df.URL, urls_df.Label)
pipeline_ls.fit(trainX,trainY)
pipeline_ls.score(testX,testY)

print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


import pickle
pickle.dump(pipeline_ls,open('phishing.pkl','wb'))
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)

# Test the model with some examples
predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
# predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)