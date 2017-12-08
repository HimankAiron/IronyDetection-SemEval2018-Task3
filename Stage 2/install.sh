#!/bin/sh

sudo apt-get install python3-pip

pip3 install nltk

python3 -m nltk.downloader punkt

python3 -m nltk.downloader averaged_perceptron_tagger

python3 -m nltk.downloader maxent_ne_chunker

python3 -m nltk.downloader words

pip3 install numpy

sudo apt-get build-dep python3-scipy

pip3 install scipy

pip3 install -U scikit-learn

pip3 install textblob

pip3 install openpyxl


