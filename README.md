# Phishing Detection System
Phishing Detection is an AI based software module that detect fraudolent and phishing e-mail.

## Idea
Implement a phishing detection system combining two different elements of an e-mail:
- url: using a dataset about 0.5M urls pointing to phishing websites and to legal ones, labeled with [good, bad], it's possible to extract some features from the urls, like the more common words that represent good or bad url. After fitting a model on this dataset it's possible to predict the nature of the feature urls, e.g. the ones found in an e-mail just recived.
- text content: sometimes urls analysis isn't sufficient to detect phishing e-mails, beacuse there are some misprediction. The idea is to combine, the e-mail content analysis to the urls classification in order to improve the system accuracy.

## About the Datasets
phshing_site_urls.csv is a collection of 0.5M urls useful to fit the url classifier model.


## Training phase

## Test phase

## Scores