## Calculate TFIDF and Cosine Similarity

### Overview
1. Preprocess articles (word tokenize, remove stop words, remove punctuation, conduct stemming*)
2. Calculate tf-idf for each term
3. Calculate pairwise cosine similarity for the documents

\*Porter stemming was used for stemming

### How to use
1. place `cosine_similarity_tfidf_nltk.py` in a directory at the same level as `inputdata/`
2. run `python cosine_similarity_tfidf_nltk.py`
NOTE: you may need to install NLTK and download some of it's packages. You can do this by running a python script, importing nltk, then calling `nltk.download()` which will open a GUI.  This script is not intended for many or large files.

#### Source Code
* main source file can be found [/cosine_similarity_tfidf_nltk.py](https://github.com/JackBurdick/information_retrieval_TFIDFAndCosineSimilarity/blob/master/src/jburdick2015_hw1.py)
* step-by-step [jupyter notebook](https://github.com/JackBurdick/cosine_similarity_tfidf_nltk/blob/master/cosine_similarity_tfidf_nltk.ipynb)

#### Input information
* input files were assigned and can be found [/inputdata](https://github.com/JackBurdick/cosine_similarity_tfidf_nltk/tree/master/inputdata)

#### Results
* results can be viewed [/results](https://github.com/JackBurdick/cosine_similarity_tfidf_nltk/tree/master/results)
 * stepwise preprocessing [results](https://github.com/JackBurdick/cosine_similarity_tfidf_nltk/blob/master/results/process_text.txt)
 * tf-idf [results](https://github.com/JackBurdick/cosine_similarity_tfidf_nltk/blob/master/results/tfid.txt)
 * pairwise cosine_similarity [results](https://github.com/JackBurdick/cosine_similarity_tfidf_nltk/blob/master/results/cosine_similarity.txt)
