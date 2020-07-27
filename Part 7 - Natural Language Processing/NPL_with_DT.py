# Natural Language Processing

# %%
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# %%
# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []        # Collection of texts
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [
            ps.stem(word) for word in review
            if word not in set(stopwords.words('english'))
            ]
    review = ' '.join(review)
    corpus.append(review)

# %%
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# %%
# Spliting dataset into trainning set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=0)



# %%
# Fitting the classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# %%
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# %%
# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# %%
# Evaluating performance of the model
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(accuracy)
precision = TP / (TP + FP)
print(precision)
recall = TP / (TP + FN)
print(recall)
F1_score = 2 * precision * recall / (precision + recall)
print(F1_score)
