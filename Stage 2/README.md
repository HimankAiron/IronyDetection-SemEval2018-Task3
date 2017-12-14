
README
------------                                    
  
Author: Madiha Mirza


Project Contributions
==================

**Kevin Swanberg**: Wrote the code `sentencesimilarity.py`, `structuralfeatures.py`, `posfunctions.py`, and `__init__.py`. 

**Madiha Mirza**: Wrote the code `preprocessing.py`, `intensifiers.py`, `interjection.py`, `celeb.py`, `political.py`.  Created the documentation in `README`, `ORIGINS`.

**Zhenduo Wang**: Wrote the code `affectiveFeatures.py`, `classifiers.py`, `run.py`. Wrote the analysis in `RESULTS-1`, `RESULTS-2`.


Project Description
==================
SemEval-2018 Task 3 has two different subtasks for the automatic detection of irony on Twitter. Lovelace is participating in both subtasks, The first subtask (Task A) is a binary classification task in which our system predicts whether a tweet is 0 (non-ironic) or 1 (ironic). Task B is a multi-class classification task where our system has to predict if a tweet is: 1 (ironic by clash), 2 (situational irony), 3 (other irony), 0 (non-ironic).

We first researched about feature selection for irony in tweets. We read several papers (cited in `References`) about irony detection and summarized the features that were proved to be important. Based on our research, we explored two types of features, namely structual (syntactic) features and affective features. Syntactic features include sentence semantic similarity, discourse markers, named entity recognition, adjective/adverbs, punctuation, laughter, topical content like celebrities and politics, interjections, word count, stopwords, swear words, URLs, and prepositions. Affective features include sentiment polarity and subjectivity. We also include a Bag of Words model with the 30 most common words as an additional 30 features. Previous work on irony detection in tweets has proved the co-relation between sentiment polarity and subjectivity with irony in tweets. Ironic tweets tend to have positive sentiment words and are more subjective. After we selected the features, we assigned scores for each features and then we got a feature matrix. 

We have used a supervised learning approach. To establish a baseline performance, we train supervised models using the above described features.  We train a logistic regression classifier, a random forest classifier, and a support vector machine (SVM). The feature sets are measured separately and concatenated as one vector to train the classifiers. The classifiers take the feature vector as input and make a prediction using a voting system among the classifiers. For stage one, they predict a binary label 0 (non-ironic) or 1 (ironic) for each tweet. For stage two, they predict a 0 (non-ironic), 1 (Verbal polar irony), 2 (Other verbal irony), or 3 (Situational Irony). The baseline models are tested using 10-fold cross-validation.


Example Input and Output
-------------------------

| Input  |   Output |  
| ------------- | ------------- |
| Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion http://t.co/fej2v3OUBR  | 1.0 |
| @mrdahl87 We are rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)  | 1.0  |
| Hey there! Nice to see you Minnesota/ND Winter Weather   | 0.0 |
| 3 episodes left I'm dying over here  | 0.0 |
| "I can't breathe!" was chosen as the most notable quote of the year in an annual list released by a Yale University librarian   | 1.0 |



Part One: Collecting the Data
------------------------------
We used the training dataset `SemEval2018-T4-train-taskA.txt` provided by the task organizers. The training dataset contains 3,834 tweets, each tweet is marked as 1 for ironic and 0 for non-ironic.

Part Two: Preprocessing the Data
-------------------------------------------
Unstructured data such as tweets often contain shortened words, characters within words, over-use of punctuation, emojis, and may not conform to grammatical rules. We performed preprocessing on the training dataset to remove elements that serve no syntactic function. We cleaned tweets to remove quotation marks, contractions, Twitter @usernames (@-mentions), Twitter #hashtags, URLs, digits, emoticons, transport & map symbols, pictographs, flags (iOS), and then performed case normalization and tokenization. The code for data cleaning and preprocessing is found in `preProcessing.py`.

Part-of-Speech Tagging (POS tagging): After the data was cleaned, we tagged each sentence with a WordNet-compatible Part-of-Speech tag for calculating semantic similarity and discourse markers.


Part Three: Creating Feature Sets
---------------------------

* **Sentence Semantic Similarity** is the measurement of how similar the meaning of two sentences is. We employed WordNet   synsets (sets of synonyms) to evaluate sentence similarity between two sentences. The similarity of the sentences is computed based on the similarity of the pairs of words. The file `sentencesimilarity.py` includes the code for measuring sentence semantic similarity. This is a structural feature employed by Farias et al with some success, motivating our usage of this feature. Ironic tweets with multiple sentences should show a sharp change in meaning between sentences, which will be detected by a low similarity score. In stage 2, this feature has been updated to take the max similarity between sentences in a tweet and works for tweets of two sentences or more.

* **Discourse Markers** are phrases that are indicative of discourse segments. Examples: Something, Moreover, In addition, Additionally, Further, Also, Besides, However, On the other hand, Yet, Because, Since, As, Therefore, Consequently, Accordingly, Hence, Thus, etc. The file `structuralfeatures.py` takes the input from `discourse_list.txt` and scores discourse markers. These have been cited as being more common in Farias et al. In Stage 2, this list has been extended

* **Named Entity Recognition (NER)** is used to identify the proper names of things, such as people, organizations, locations, products, etc. from unstructured text. The file `posfunctions.py` includes function that tokenizes each tweet by separating it into sentences, then chunks sentences and creates an NLTK POS tree for each sentence. Named entities were also considered by some to be more common in ironic tweets, but our stage 1 results suggest otherwise. For stage 2, we initally tried making this more specific, identifying specific types of named entities (People, Places, Geopolitical Entities, Organizations, etc) but this reduced the accuracy of our model.

* **Word Count** The file `structuralfeatures.py` contains the code for counting the words in each tweet and append it to the list of wordcounts for every tweet. Farias et al suggest that ironic tweets are likely to contain fewer words than non-ironic tweets. For stage 2, we tried updating this to take the word count of each sentence within each tweet, but this reduced the accuracy of our model, so we kept this as a standard word count.

* **Adjective and Adverb Count** Adjectives and adverbs occur more frequently in ironic tweets than in non-ironic. The file `posfunctions.py` contains functions that require Part of Speech tagging, including adjective and adverb counting. Farias et al cite a 2007 paper from Kreuz and Caucci which suggests that adverbs and adjectives are used as lexical markers in expressions of irony, motivating the use of this feature.

* **Prepositions** It was hypothesized by our team that prepositions would occur more often in situational irony, since it is necessary to describe the situation, which often involves a preposition. For example, "I was on the bus," or "I was at school." Many irony detection functions also tag the parts of speech of all words, so this seemed like a logical step.

* **Punctuation** Ironic tweets tend to have excessive punctuation to catch the eyes of readers and to stress a point. Examples include "It is really worth it!!!" or "Okay...". Thus, heavy punctuation sometimes implies irony. This is discussed in detail, again, in Farias et al. they note that punctation has been used to detect irony in a number of experiments. Stage 2 will expand this to also identify all caps letters. The file `structuralfeatures.py` contains the code to check each important punctuation in a tweet and then append this to the count of punctuation in all tweets.

* **Affective Features** reflect the emotions expressed in the text. These include sadness, happiness, anger, fear, love, joy, and surprise, etc. The file `affectiveFeatures.py` contains the code to measure the affective score of the emoticons and other emotional words. This, along with sentiment polarity, is the primary focus of Farias et al's paper and they achieved strong results when applying these features.

* **Top Hashtags and Keywords** Twitter hashtags and keywords are a good measure of public opinion on treanding topics and current events. These words are used thousands of times daily and connect users around the global to discuss their interests. Through these hashtags, users express a wide variety of opinions ranging from sarcasm, irony, anger, joy, hope, among many others. The file `celebrity_list.txt` contains popular hashtags about entertainment, media, music, reality tv, sports, and fashion. Some examples of hashtags in that list are #justinbieberswag, #kkwfragrance, #IceBucketChallenge, #SFGiants. The file`celeb.py` takes the keywords and use it to score irony in tweets. The file `political_list.txt` contains hashtags related to elections, government, political leaders, countries and cities, climate change, human rights, economy. Some examples of hashtags in that list are #GOPDebate, #PrayforJapan, #Trump2020, #hillsquad, #alllivesmatter, #WomensRightsAreHumanRights. The file `political.py` takes these hashtags and use it to score irony in tweets. 

* **Intensifiers** Many users employ certain adverbs and adjectives to add more strength and emphasis to their opinion. Such words are called intensifers and they boost the ironic effect of the tweet. Some examples of intensifers are words such as absolutely, apparently, awful, hella, heck, for sure, wickedly. The file `intensifiers_list.txt` contains a list of such words. The file `intensifiers.py` uses these words to measure whether a tweet is ironic or not.

* **Interjections** Interjections are word that express feeling rather than meaning. These words aren't grammatically related to the rest of the sentence and are abruptly added to convey a sudden emotion or reaction. Interjections are more common in informal language and since tweets use informal language, interjections are a good measure of detecting whether a user modeled the tweet in an ironic way. Some examples of interjections are words such as wow, gosh, jeez, damn, blah, aww, boo, hmm, yay, yikes. The file `interjection_list.txt` contains a list of such words and the file `interjection.py` uses these words to measure the irony expressed in tweets.

* **Sentiment Polarity and Subjectivity** Subjectivity in a sentence expresses some personal feelings, views, or beliefs while polarity is a measure of whether the subjective text express a positive or negative opinion of the subject matter. The file `affectiveFeatures.py` inlcudes the code for measuring sentiment polarity and subjectivity of a tweet. As previously mentioned, Farias et al applied this feature with strong success in their paper.

* **Laughter** It was noted by Buschmeier et al that laughter is a common trait of ironic tweets, which makes sense since often irony is used to make a joke. Often, laughter is symbolized by terms like "LOL" or "Haha" or "LMAO," so terms like this were searched for.

* **Stopwords** Stopwords are words that search engines and NLP programs often ignore because they do not contain significant information often. However, they are very common in conversational English, which is a common feature of irony and so we thought this may be a significant feature - it did offer some improvement for our system

* **Swear Words** There is no academic basis behind this feature, but since ironic tweets are typically emotional in nature it was hypothesized that ironic tweets might include these "emotional words." The function works by checking if swear words are contained in each tweet. This did prove to have some benefit for our system.

* **URLs** This was based on the work found in "Detecting Sarcasm in Multimodal Social Platforms" by Schifanella et al, which found that ironic tweets often contain images and often the interpretation of the irony depends on the image. For example, a photo of a warm, sunny beach with the caption "Terrible weather we're having." However, our data does not immediately give us images. We did see when assessing the data though that any time there was an image, it was included in the link using a URL, and a majority of the URLs in the data were images, so the simplest way to identify this was to just check if a tweet had a URL.

Part Four: Building, Training, and Testing the Classifiers
-----------------------------------------------------
The file `classifiers.py` contains the code for building, training, and testing a support vector machine (SVM), a logistic regression classifier, and a random forest classifier.

SVM Classifier
---------------
Support vector machine(SVM) is one of the supervised learning methods used for classification. Classification is a process of identifying a category for an observation(e.g: 0 or 1 for each sentence in our training dataset). With the features matrix, we used SVM classifier for prediction task. SVM is a classification method that generally works on binary classification. It aims to find a hyperplane to linearly divide all the samples into two classes. The best hyperplane is found by maximizing the minimum distance among all the samples and the hyperplane subject to most samples are classified correctly. If the sample are not linearly separable, then SVM finds a kernel function. The kernel function projects the samples into higher dimensional space in which the samples are linearly separable. The SVM classifier takes the feature matrix as input and outputs binary label vector which indictes the categorical result. We seperated the data into training set and test set. We trained the SVM classifier with training set to optimize the parameters including the hyperplane and kernel. Then we used the trained classifier on test set and did several analysis.

**Usage**:  
#import svm from Scikit-learn tool  
from sklearn.svm import SVC

#create a SVM Classification instance  
clf = SVC()

#display the classifier  
print("Training classifiers...")

#output  
[sample_size = len(feature_table) train_size = int(0.8 * sample_size) feature_num = len(feature_table[0]) folds = 5]
 
#cross validation accuracy  
scores = cross_val_score(clf, X, y, cv=folds)


Logistic Regression (aka logit, MaxEnt) classifier
---------------
Logistic regression fits a logistic model to data and makes predictions about the probability of an event (between 0 and 1).


**Usage**:  
#import Logistic Regression from Scikit-learn tool  
from sklearn.linear_model import LogisticRegression  

#create a RandomForestClassifier instance  
clf = LogisticRegression()

#display the classifier  
print("Training classifiers...")

#output  
[sample_size = len(feature_table) train_size = int(0.8 * sample_size) feature_num = len(feature_table[0]) folds = 5]
 
#cross validation accuracy  
scores = cross_val_score(clf, X, y, cv=folds)


Random Forest Classifier
---------------
Random forest algorithm is a supervised classification algorithm. It creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.

**Usage**:  
#import RandomForestClassifier from Scikit-learn tool  
from sklearn.ensemble import RandomForestClassifier  

#create a RandomForestClassifier instance  
clf = RandomForestClassifier()  

#display the classifier  
print("Training classifiers...")


#output  
[sample_size = len(feature_table) train_size = int(0.8 * sample_size) feature_num = len(feature_table[0]) folds = 5]

#cross validation accuracy  
scores = cross_val_score(clf, X, y, cv=folds)


Cross-Validation:
---------------
If the parameters of an estimator method is learned and tested on the same data, it yields good results. 
But, the model would not predict the accurate labels if it is made to work on un-seen data. To avoid this, 
a good practice is to hold out the part of the original dataset by splitting the dataset into more than 
one part. To get more accurate results, the solution is to use statistical sampling. The aim of cross-validation 
is to make certain that every example from the original dataset has the same chance of appearing in the training 
and testing set.

We did cross-validation by calling the `cross_val_score helper` function on the SVM, MaxEnt, and Random Forest 
estimators along with the dataset. 



Results
==================
`evaluate.py` is the official evaluation script that takes as input a submission dir containing the system output and calculates accuracy, precision, recall and F1-score.

We evaluated our classifiers using standard evaluation metrics, including accuracy, precision, recall and F1-score. The classifiers produce quite a good result with these features. We incorporated more feature measurements to imporve the performance of the baseline models. A detailed description and analysis of results is found in `RESULTS-1` and `RESULTS-2`.


Dependencies
==================

The script is written in Python 3.

All the Python modules used in this project are specified below:

    1. NLTK
    2. Scikit-Learn
    3. NumPy
    4. SciPy
    5. TextBlob
    6. Openpyxl


`ORIGINS` outlines the 3rd party code and tools used in the project.


Setup
------------------

Run the bash script `install.sh` to install all the Python module dependencies found in `requirements.txt`. 

Running
------------------
Execute `runit.sh` to run the program and see the results.


Acknowledgements
==================
We sincerely thank our professor Dr. Ted Pedersen for encouraging us to participate in this task as part of our graduate coursework in Natural Language Processing. We are grateful for his guidance and support.


References
==================

1. SemEval-2018: http://alt.qcri.org/semeval2018/

2. We consulted the following research papers for Stage 1 and Stage 2.

  * **Clues for Detecting Irony in User-Generated Contents: Oh...!! It’s “so easy" ;-)**  
      Paula Carvalho, Luís Sarmento, Mário J. Silva, and Eugénio de Oliveira. 2009. 
      In Proceedings of the 1st international CIKM workshop on Topic-sentiment analysis for mass opinion (TSA '09). 
      ACM, New York, NY, USA, 53-56.  DOI: http://dx.doi.org/10.1145/1651461.1651471.
    
      *The paper provides information about using gestural clues (emoticons, onomatopoeic expressions for laughter, heavy    
      punctuation marks, quotation marks and positive interjections) to detect irony in user-generated online content.* 
    
 *  **Irony Detection in Twitter: The Role of Affective Content**  
      Delia Irazú Hernańdez Farías, Viviana Patti, and Paolo Rosso. 2016. 
      ACM Trans. Internet Technol. 16, 3, Article 19 (July 2016), 24 pages.
      DOI: http://dx.doi.org/10.1145/2930663.
    
      *This research paper presents a model that explores the use of affective information (Sentiment-Related Features and   
      Emotional Categories) based on a wide range of lexical resources available for English. It applies supervised machine  
      learning approach to detect irony in tweets and shows that affective information is effective in distinguishing among 
      ironic and nonironic tweets.*
      
   *  **Automatic Sarcasm Detection: A Survey.**  
      Joshi, Aditya, Pushpak Bhattacharyya, and Mark J. Carman. 2017. 
      ACM Comput. Surv, vol. 0, no. 0.
      DOI: https://arxiv.org/abs/1602.03426.
  
   *  **Are Word Embedding Based Features Useful for Sarcasm Detection?**  
      Joshi, Aditya, Vaibhav Tripathi, Kevin Patel, Pushpak Bhattacharyya, and Mark Carman.
      Conference on Empirical Methods in Natural Language Processing (EMNLP), November 2016.
      DOI: https://arxiv.org/pdf/1610.00883.pdf.
      
   *  **Detecting Sarcasm in Multimodal Social Platforms** 
      Rossano Schifanella, Paloma de Juan, Joel Tetreault, and LiangLiang Cao. 2016. In Proceedings of the 2016 ACM on
      Multimedia Conference (MM '16). ACM, New York, NY, USA, 1136-1145. DOI: https://doi.org/10.1145/2964284.2964321
      
   *  **An Impact Analysis of Features in a Classification Approach to Irony Detection in Product Reviews** 
      Konstantin Buschmeier, Philipp Cimiano and Roman Klinger. 2014. In Proceedings of the 5th Workshop on Computational 
      Approaches to Subjectivity, Sentiment and Social Media Analysis. ACL, Baltimore, MD, USA, 42-49. 
  
  

