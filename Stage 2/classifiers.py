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

'''
---------- Support Vector Machine Classifier ------------
This function uses SVM classifier for the classification problem based on the feature we select. 
The classifier takes feature matrix as input and outputs the categorical label vector (other classifiers as well).
We implent the classifier with sklearn package. (other classifiers as well).
We use cross validation to get accuracy of the classifier. (other classifiers as well).
We calculate precision, recall and F-measure score with the official evaluation function. (other classifiers as well).
We also compute confusion matrix with sklearn. (other classifiers as well).
'''
def SVMClf(X, y, output,class_num):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = SVC()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size])
	if class_num == 2:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	if class_num == 4:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1,2,3])
	output.write("Support Vector Machine Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
		
''' Logistic Regression Classifier
Because logistic regression has linear kernel, 
we get the weight for all the features with a third party function show_most_informative_features.
The magnitude of a weight implies the importance of that feature.
'''
def LogisticRegressionClf(X, y, output,class_num):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = LogisticRegression()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	if class_num == 2:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	if class_num == 4:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1,2,3])
	output.write("Logistic Regression Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
	# Call the feature importance function
	show_most_informative_features(clf,feature_num,output)


'''
---------- Random Forest Classifier ------------
This function uses Random Forest classifier for the classification problem based on the feature we select. 
The classifier takes feature matrix as input and outputs the categorical label vector 
'''
# Random Forest Classifier 
def RandomForestClf(X, y, output,class_num):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = RandomForestClassifier()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	if class_num == 2:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	if class_num == 4:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1,2,3])
	output.write("Random Forest Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	#output.write("predicted\tnon-ironic\tironic\n")
	#output.write("true\n")
	#output.write("non-ironic")
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[0,0])
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[0,1])
	#output.write("ironic")
	#print (confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist())
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[1,0])
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[1,1])
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
	
'''
---------- Naive Bayes Classifier ------------
This function uses Naive Bayes classifier for the classification problem based on the feature we select. 
This classifier is only used as a baseline. 
'''
def NBClf(X, y, output,class_num):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = GaussianNB()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	if class_num == 2:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	if class_num == 4:
		p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1,2,3])
	output.write("Naive Bayes Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
	print_cm(confusion_matrix(y[train_size:],clf.predict(X[train_size:])),labels=['non-ironic','ironic'])

'''
---------- Voting Classifier ------------
This function uses SVM, Logistic Regression and Random Forest classifiers for the classification problem based on the feature we select. 
Because there are 3 classifiers in this voting system, it uses a 'hard voting' which means the decision is made by majority. 
'''
def VotedClf(X,y,output,class_num):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf1 = SVC()
	clf2 = LogisticRegression()
	clf3 = RandomForestClassifier()
	vclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
	vclf.fit(X, y)
	# Cross validation accuracy
	scores = cross_val_score(vclf, X, y, cv=folds)
	# Call the evaluation function
	vclf.fit(X[:train_size], y[:train_size])
	if class_num == 2:
		p, r, f = precision_recall_fscore(y[train_size:], vclf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	if class_num == 4:
		p, r, f = precision_recall_fscore(y[train_size:], vclf.predict(X[train_size:]), beta=1, labels=[0,1,2,3])
	output.write("Voted Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:")
	output.write(str(p))
	output.write("\nr=:")
	output.write(str(r))
	output.write("\nf=:")
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:], vclf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)

'''
---------- Principle Component Analysis ------------
This function is a pca module for dimension reduction.
We used dimension reduction for bag of words feature and for final feature list.
We implent the function with sklearn package.
'''
def pcafunction(X,dim):
	pca = PCA(n_components=dim)
	pca.fit(X)
	return X
