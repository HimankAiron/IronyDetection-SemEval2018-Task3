# -*- coding: utf-8 -*-
# Classifiers 
# AUTHOR:Zhenduo Wang
# This script implements 5 classifiers(Logistic Regression,SVM, Random Forest, Naive Bayes and a voting system of classifiers)
# All of them takes the feature matrix and produce the prediction label.
# We keep tracking of these classifiers with different features in order to get the best results.
# This script also implements pca function.

import numpy
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from evaluate import *
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
'''
---------- Support Vector Machine Classifier ------------
This function uses SVM classifier for the classification problem based on the feature we select. 
The classifier takes feature matrix as input and outputs the categorical label vector (other classifiers as well).
We implent the classifier with sklearn package. (other classifiers as well).
We use cross validation to get accuracy of the classifier. (other classifiers as well).
We calculate precision, recall and F-measure score with the official evaluation function. (other classifiers as well).
We also compute confusion matrix with sklearn. (other classifiers as well).
'''
def SVMClf(X_train, y_train,X_test):
	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)
	clf = SVC()
	clf.fit(X_train, y_train)

	output = open("SVM_taskA.txt", 'w')

	for item in clf.predict(X_test).tolist():
		output.write("%d\n" % item)

	output.close()
		
''' Logistic Regression Classifier
Because logistic regression has linear kernel, 
we get the weight for all the features with a third party function show_most_informative_features.
The magnitude of a weight implies the importance of that feature.
'''
def LogisticRegressionClf(X_train, y_train,X_test):
	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)
	clf = LogisticRegression()
	clf.fit(X_train, y_train)

	output = open("LR_taskA.txt", 'w')
	for item in clf.predict(X_test).tolist():
		output.write("%d\n" % item)

	output.close()

'''
---------- Random Forest Classifier ------------
This function uses Random Forest classifier for the classification problem based on the feature we select. 
The classifier takes feature matrix as input and outputs the categorical label vector 
'''
# Random Forest Classifier 
def RandomForestClf(X_train, y_train,X_test):
	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)
	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)

	output = open("RF_taskA.txt", 'w')
	for item in clf.predict(X_test).tolist():
		output.write("%d\n" % item)

	output.close()

'''
---------- Voting Classifier ------------
This function uses SVM, Logistic Regression and Random Forest classifiers for the classification problem based on the feature we select. 
Because there are 3 classifiers in this voting system, it uses a 'hard voting' which means the decision is made by majority. 
'''
def VotedClf(X_train, y_train,X_test):
	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)

	clf1 = SVC()
	clf2 = LogisticRegression()
	clf3 = RandomForestClassifier()
	vclf = VotingClassifier(estimators=[('svm', clf1), ('lr', clf2), ('rf', clf3)], voting='hard')
	vclf.fit(X_train, y_train)

	output = open("VC_taskA.txt", 'w')
	for item in vclf.predict(X_test).tolist():
		output.write("%d\n" % item)
	output.close()
'''
---------- Principle Component Analysis ------------
This function is a pca module for dimension reduction.
We used dimension reduction for bag of words feature and for final feature list.(Not included in final version system)
We implent the function with sklearn package.
'''
def pcafunction(X,dim):
	pca = PCA(n_components=dim)
	pca.fit(X)
	return X
