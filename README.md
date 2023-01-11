# Phishing Detection System
Phishing Detection is an AI based software module that detect fraudolent and phishing e-mail.

### Idea
Phishing detection system combines two relevant elements of an e-mail: urls and text content.

### About the Datasets
The dataset is a composition of two labelled datasets, in particular urls and email content extracted from a bigger dataset called ENRON. 

### Preprocessing
In the preprocessing phase the data are tokenized and stammed in order to extract the most relevant words from each label (good or bad).
In order to exclude from the classification unuseful words there is the specified english stopwords functionality passed to the vectorizer.

### Cross Validation
In the performance evaluation there is a cross validation with K-Fold approach and K=10.
