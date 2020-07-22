# Apriori

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from apyori import apriori
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

# %% [markdown]
# # Association Rule Learning
# Apriori Algorithm
#
# %% [markdown]
# ## Importing the libraries

# %%
get_ipython().system('pip install apyori')


# %%

# %% [markdown]
# ## Data Preprocessing

# %%
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
dataset


# %%
dataset.isnull().sum()


# %%
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])


# %%
print(transactions[:5])
len(transactions)

# %% [markdown]
# ## Training Apriori model on the dataset

# %%
support = 3*7/7501
rules = apriori(transactions=transactions, min_support=support,
                min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# %% [markdown]
# ## Visualising the results
# %% [markdown]
# ### Displaying the first results coming directly from the output of the apriori function

# %%
results = list(rules)


# %%
[print(result) for result in results[:3]]

# %% [markdown]
# ### Putting the results well organised into a Pandas Dataframe

# %%


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


# %%
resultsinDataFrame = pd.DataFrame(inspect(results), columns=[
                                  'Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# %% [markdown]
# ### Displaying the results non sorted

# %%
resultsinDataFrame

# %% [markdown]
# ### Displaying the results sorted by descending lifts

# %%
resultsinDataFrame.nlargest(n=10, columns='Lift')

# %% [markdown]
# ### Displaying the results sorted by descending confidence

# %%
resultsinDataFrame.nlargest(n=10, columns='Confidence')

# %% [markdown]
# ### Displaying the results sorted by descending support

# %%
resultsinDataFrame.nlargest(n=10, columns='Support')


# %%
