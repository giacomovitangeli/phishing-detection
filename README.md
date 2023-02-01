# Phishing Detection System
Phishing Detection is an AI based software module that detect fraudolent and phishing e-mail.

### Idea
Phishing detection system combines two relevant elements of an e-mail: urls and text content.

### About the Datasets
The dataset is a composition of two labelled datasets, in particular phishing urls and spam email content extracted from a bigger dataset called ENRON. The dataset are merged in a unique composition of email content and phishing email, the goal is the extraction of the meaningful words, so this combination can improve the classification, making it more general. 

### Preprocessing
In the preprocessing phase the data are tokenized and stammed in order to extract the most relevant words from each label (good or bad). In order to do that were propose two different vectorization methods: TfidfVectorizer and CountVectorizer.

### Undersampling
To face with unbalanced dataset it's applied a random undersampling during the cross-validation, in this way it is possible to ensure that the undersampling is performed only on the training set of each dataset splitting.

### Pipeline
In the processing phase it's used the pipline to group different components and automate the learning process.

### Cross Validation
To evaluate the performance of our models was applied a Stratified K-fold cross-validation, that partition the dataset into ten folds and train the model, test it for different partitions of the dataset, maintaining a good representation of the real proportion of the original dataset in each fold.
