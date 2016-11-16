
# coding: utf-8

# In[99]:

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import random
import math


# In[100]:

random.seed(1149)
train_proportion = 0.8


# In[101]:

shuffled_indices


# In[102]:

data = pd.read_csv('IS602_Project_Data.csv')
data1 = data#[['benign_malignant','pathology']]
#print len(data1)
shuffled_indices = np.random.permutation(np.arange(len(data1)))
#print len(shuffled_indices)
#print min(shuffled_indices)
data1 = data1.reset_index(drop=True)
#print(data1)
shuffled_df = data1.reindex(np.random.permutation(data1.index))
shuffled_df = shuffled_df.reset_index(drop=True)
df_train = shuffled_df[0:int(math.floor(train_proportion*len(shuffled_df)))]
df_test = shuffled_df[int(math.floor(train_proportion*len(shuffled_df))+1):len(shuffled_df)]
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    #stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = words#[w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


# In[135]:

df_test


# In[104]:

clean_review = review_to_words( df_train["pathology"][0] )

num_paths


# In[105]:

num_paths_train = df_train["pathology"].size
num_paths_test = df_test["pathology"].size
print num_paths_test
clean_paths_train = []
clean_paths_test = []
for i in xrange(0, num_paths_train):
    clean_paths_train.append(review_to_words(df_train["pathology"][i]))

for i in xrange(0, num_paths_test):
    clean_paths_test.append(review_to_words(df_test["pathology"][i]))


# In[106]:

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000) 
train_data_features = vectorizer.fit_transform(clean_paths_train)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print count, tag


# ### Predicting benign_malignant

# In[143]:

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, df_train["benign_malignant"] )


# In[144]:

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)


# In[145]:

from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, roc_curve, jaccard_similarity_score, precision_score
accuracy_score(df_test['benign_malignant'], result)


# In[110]:

confusion_matrix(df_test['benign_malignant'], result)


# In[111]:

cohen_kappa_score(df_test['benign_malignant'], result)


# In[112]:

jaccard_similarity_score(df_test['benign_malignant'], result)


# ### Predicting path_diagnosis

# In[152]:

forest2 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest2 = forest.fit( train_data_features, df_train["path_diagnosis"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result2 = forest2.predict(test_data_features)


# In[153]:

accuracy_score(df_test['path_diagnosis'], result2)


# In[127]:

cohen_kappa_score(df_test['path_diagnosis'], result2)


# In[128]:

confusion_matrix(df_test['path_diagnosis'], result2)


# ### Predicting melanocytic

# In[136]:

forest3 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest3 = forest.fit( train_data_features, df_train["melanocytic"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result3 = forest3.predict(test_data_features)


# In[137]:

accuracy_score(df_test['melanocytic'], result3)


# In[138]:

cohen_kappa_score(df_test['melanocytic'], result3)


# In[139]:

confusion_matrix(df_test['melanocytic'], result3)


# In[ ]:




# ### Predicting nevus_type

# In[190]:

forest4 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest4 = forest.fit( train_data_features, df_train["nevus_type"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result4 = forest4.predict(test_data_features)


# In[191]:

accuracy_score(df_test['nevus_type'], result4)


# In[192]:

cohen_kappa_score(df_test['nevus_type'], result4)


# In[193]:

confusion_matrix(df_test['nevus_type'], result4)


# ### Predicting ulcer

# In[162]:

forest5 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest5 = forest.fit( train_data_features, df_train["nevus_type"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result5 = forest5.predict(test_data_features)


# In[166]:

accuracy_score(df_test['nevus_type'], result5)


# In[167]:

cohen_kappa_score(df_test['nevus_type'], result5)


# In[168]:

confusion_matrix(df_test['nevus_type'], result5)


# ### Predicting mel_class

# In[173]:

forest6 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest6 = forest.fit( train_data_features, df_train["mel_class"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result6 = forest6.predict(test_data_features)


# In[174]:

accuracy_score(df_test['mel_class'], result6)


# In[175]:

cohen_kappa_score(df_test['mel_class'], result6)


# In[176]:

confusion_matrix(df_test['mel_class'], result6)


# ### Predicting mel_type

# In[178]:

forest7 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest7 = forest.fit( train_data_features, df_train["mel_type"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result7 = forest7.predict(test_data_features)


# In[179]:

accuracy_score(df_test['mel_type'], result7)


# In[180]:

cohen_kappa_score(df_test['mel_type'], result7)


# In[181]:

confusion_matrix(df_test['mel_type'], result7)


# ### mel_assoc_nev

# In[194]:

forest8 = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest8 = forest.fit( train_data_features, df_train["mel_assoc_nev"] )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_paths_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result8 = forest8.predict(test_data_features)


# In[195]:

accuracy_score(df_test['mel_assoc_nev'], result8)


# In[196]:

cohen_kappa_score(df_test['mel_assoc_nev'], result8)


# In[197]:

confusion_matrix(df_test['mel_assoc_nev'], result8)


# In[ ]:



