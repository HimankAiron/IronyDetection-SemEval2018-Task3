# -*- coding: utf-8 -*-

# Main function
# AUTHOR: Zhenduo Wang
# The main function calls all the feature measuring function to get the measurements
#		and then concatenate the measurements into feature matrix. Then it calls
#       the pca function to do a dimensional redunction. Then it calls the
#       classifiers function to get the prediction label.

import numpy

from affectiveFeatures import *
from classifiers import *
from posfunctions import *
from preProcessing import *
from political import political_scorer
from sentencesimilarity import *
from structuralfeatures import *
from intensifiers import intensi_scorer
from celeb import *
from sklearn.feature_extraction.text import CountVectorizer

# the number of class in classification
# 2 for taskA, 4 for taskB
class_num = 2

# ----------MAIN RUN FUNCTION--------#
if class_num == 2:
    with open('SemEval2018-T4-train-taskA.txt', encoding='utf-8') as f:
        tweet_list_train = [line.split('\t')[2] for line in f]

    with open('SemEval2018-T4-train-taskA.txt', encoding='utf-8') as f:
        score_list = [line.split('\t')[1] for line in f]

    with open('SemEval2018-T3_input_test_taskA.txt', encoding='utf-8') as f:
        tweet_list_test = [line.split('\t')[1] for line in f]
if class_num == 4:

    with open('SemEval2018-T4-train-taskB.txt', encoding='utf-8') as f:
        tweet_list_train = [line.split('\t')[2] for line in f]

    with open('SemEval2018-T4-train-taskB.txt', encoding='utf-8') as f:
        score_list = [line.split('\t')[1] for line in f]

    with open('SemEval2018-T3_input_test_taskB.txt', encoding='utf-8') as f:
        tweet_list_test = [line.split('\t')[1] for line in f]

# Removes label from from the tweet list and the score list
tweet_list_train = tweet_list_train[1:]
tweet_list_test = tweet_list_test[1:]
score_list = score_list[1:]

# Prepare an empty array
cleaned_score_list = []

# This loop removes newline characters from the score list and converts the scores from strings to float values
for item in score_list:
    intscore = item.rstrip()
    intscore = float(intscore)
    cleaned_score_list.append(intscore)

print("Cleaning tweets...")
# Call Preprocess on the Tweet Data
cleanedtweets_train = preprocess(tweet_list_train)
cleanedtweets_test = preprocess(tweet_list_test)

print("Computing polarity and subjectivity...")
# Create an array containing the polarity and subjectivity scores
polarity_and_subjectivity_train = PolarityAndSubjectivity(cleanedtweets_train)
polarity_and_subjectivity_test = PolarityAndSubjectivity(cleanedtweets_test)

print("Computing similarities...")
# Collect the sentence similarity of all the cleaned tweet data
sent_sim_list_train = simrun(cleanedtweets_train)
sent_sim_list_test = simrun(cleanedtweets_test)

print("Computing discourse markers...")
# Collect the discourse marker scores of all the cleaned tweet data
disc_list_train = discourse_scorer(cleanedtweets_train)
disc_list_test = discourse_scorer(cleanedtweets_test)

print("Computing intensifiers...")
# Collect the intensifier scores of all the cleaned tweet data
inten_list_train = intensi_scorer(cleanedtweets_train)
inten_list_test = intensi_scorer(cleanedtweets_test)

print("Computing political...")
# Collect the political scores of all the cleaned tweet data
pol_list_train = political_scorer(cleanedtweets_train)
pol_list_test = political_scorer(cleanedtweets_test)

print("Computing celebrity...")
# Collect the celebrity mention scores of all the cleaned tweet data
celeb_list_train = celebrity_scorer(cleanedtweets_train)
celeb_list_test = celebrity_scorer(cleanedtweets_test)

print("Computing adjectives and adverbs...")
# Collect the adjective and adverb scores of all the cleaned tweet data
adj_adv_list_train = adj_adv_counter(cleanedtweets_train)
adj_adv_list_test = adj_adv_counter(cleanedtweets_test)

print("Computing Prepositions...")
prep_list_train = prep_scorer(cleanedtweets_train)
prep_list_test = prep_scorer(cleanedtweets_test)

print("Computing punctuation markers...")
# Collect the punctuation marker scores of all the cleaned tweet data
punc_list_train = punc_count(tweet_list_train)
punc_list_test = punc_count(tweet_list_test)

print("Computing word count...")
#Collect the word count of all tweet data
wc_list_train = word_counter(tweet_list_train)
wc_list_test = word_counter(tweet_list_test)

print("Computing laughter counts...")
laugh_list_train = laughter_scorer(tweet_list_train)
laugh_list_test = laughter_scorer(tweet_list_test)

print("Computing named entity count...")
# Collect the named entity scores of all the cleaned tweet data
ne_list_train = named_entity_count(tweet_list_train)
ne_list_test = named_entity_count(tweet_list_test)

print("Computing stopwords...")
stopword_list_train = stopwords_score(cleanedtweets_train)
stopword_list_test = stopwords_score(cleanedtweets_test)

print("Computing swearwords...")
swear_list_train = swear_scorer(cleanedtweets_train)
swear_list_test = swear_scorer(cleanedtweets_test)

print("Computing URLs...")
url_list_train = url_count(tweet_list_train)
url_list_test = url_count(tweet_list_test)

print("Computing bag of words features...")
# Choose the top words by frequence as bag of words features
vectorizer = CountVectorizer(stop_words='english',max_features=600)
bagofwordsfeature_train = vectorizer.fit_transform(tweet_list_train)
bagofwordsfeature_test = vectorizer.fit_transform(tweet_list_test)
bagofwordsfeature_train = pcafunction(bagofwordsfeature_train.toarray(),30)
bagofwordsfeature_test = pcafunction(bagofwordsfeature_test.toarray(),30)

print("Concatenating features...")
# Concatenate all the features together
feature_table_train = numpy.column_stack(
    [polarity_and_subjectivity_train, sent_sim_list_train, pol_list_train, disc_list_train, celeb_list_train, ne_list_train, inten_list_train, adj_adv_list_train, punc_list_train, wc_list_train, laugh_list_train, prep_list_train, stopword_list_train, swear_list_train, url_list_train,bagofwordsfeature_train])
feature_table_test = numpy.column_stack(
    [polarity_and_subjectivity_test, sent_sim_list_test, pol_list_test, disc_list_test, celeb_list_test, ne_list_test, inten_list_test, adj_adv_list_test, punc_list_test, wc_list_test, laugh_list_test, prep_list_test, stopword_list_test, swear_list_test, url_list_test,bagofwordsfeature_test])

# Use pca function to decrease the feature dimension
# feature_table = pcafunction(feature_table,10)
# Using classifiers and generate output
print("Training Classifiers...")
X_train = numpy.array(feature_table_train)
X_test = numpy.array(feature_table_test)
y_train = numpy.array(cleaned_score_list)

LogisticRegressionClf(X_train, y_train,X_test)
SVMClf(X_train, y_train,X_test)
RandomForestClf(X_train, y_train,X_test)
VotedClf(X_train, y_train,X_test)

