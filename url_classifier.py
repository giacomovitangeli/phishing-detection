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


