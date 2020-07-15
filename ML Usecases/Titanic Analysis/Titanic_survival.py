# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Titanic Passenger Survival Analysis
# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Importing the Dataset

# %%
dataset = pd.read_csv('titanic.csv')
dataset


# %%
print("# of passengers in dataset are:", len(dataset))

# %% [markdown]
# ## Data Visualization
# Analyze the Data
# Plot to check how many passenger survived

# %%
sns.countplot(dataset['Survived'])


# %%
sns.countplot(x='Survived', hue='Sex', data=dataset)


# %%
# Passenger class
sns.countplot(dataset['Survived'], hue=dataset['Pclass'])


# %%
# Age distribution
dataset['Age'].plot.hist()


# %%
# Plotting Fare
dataset['Fare'].plot.hist(bins=20, figsize=(10, 5))


# %%
# Plot SibSp
sns.countplot(dataset['SibSp'])

# %% [markdown]
# ## Data Preprocessing
# Data Wrangling (Data cleaning)
# remove all Nan value

# %%
dataset.isnull()


# %%
dataset.isnull().sum()


# %%
# Plot Heatmap and visulize null values

sns.heatmap(dataset.isnull(), yticklabels=False, cmap='viridis')


# %%
sns.boxplot(x='Pclass', y='Age', data=dataset)

# %% [markdown]
# Passenger travelling in 1st class and 2nd class teds to
# be older than 3rd class
#
# %% [markdown]
# You can drop the missing value or fill some other value
# which is called 'cumputation'
#
# Cabin colomn has more Nan value so we can drop it

# %%
dataset.drop('Cabin', axis=1, inplace=True)


# %%
dataset.dropna(inplace=True)


# %%
sns.heatmap(dataset.isnull())

# %% [markdown]
# ### Encoding Categorical Features

# %%
pd.get_dummies(dataset['Sex'])


# %%
sex = pd.get_dummies(dataset['Sex'], drop_first=True)
sex


# %%
pclass = pd.get_dummies(dataset['Pclass'], drop_first=True)
pclass


# %%
embarked = pd.get_dummies(dataset['Embarked'], drop_first=True)
embarked


# %%
dataset.drop(['Sex', 'Pclass', 'PassengerId', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
dataset = pd.concat([dataset, sex, pclass, embarked], axis=1)
dataset


# %%
X = dataset.drop('Survived', axis=1).values
y = dataset['Survived'].values

# %% [markdown]
# ### Splitting into train and test sets

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %% [markdown]
# ### Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)

# %% [markdown]
# ## Analysis with Logistic Regression
# %% [markdown]
# ### Fitting logistic regression to training set

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# %% [markdown]
# ### Predicting Test set results

# %%
y_pred = model.predict(X_test)


# %%
# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# %%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# %%
# accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# %%


# %% [markdown]
# ## Analysis with DecisionTree Classification
# %% [markdown]
# ### Fitting DecisionTree to training set

# %%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# %% [markdown]
# ### Predicting Test set results

# %%
y_pred = classifier.predict(X_test)


# %%
# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# %%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# %%
# accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# %%


# %% [markdown]
# ## Analysis with Random Forest Classifier
# %% [markdown]
# ### Fitting Random Forest Classifier to training set

# %%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion='entropy',
                                    random_state = 0)
classifier.fit(X_train, y_train)

# %% [markdown]
# ### Predicting Test set results

# %%
y_pred = classifier.predict(X_test)


# %%
# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# %%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# %%
# accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# %%
# Feature Importance

features = list(dataset.drop('Survived', axis=1).columns)
importances = classifier.feature_importances_
for pair in zip(features, importances):
    print("{:<20} {}".format(*pair))

# %% [markdown]
# ## Analysis with SVM
# %% [markdown]
# ### Fitting SVM to training set

# %%
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# %% [markdown]
# ### Predicting Test set results

# %%
y_pred = classifier.predict(X_test)


# %%
# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# %%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# %%
# accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# %%
