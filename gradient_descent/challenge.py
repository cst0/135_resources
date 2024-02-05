import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

print(
    """
Now that we've seen classification of fake data, or of pre-curated data, let's
try to use some data that we've found somewhere on the internet.

When searching for data, I usually look in one of two places:
1. Kaggle (https://www.kaggle.com/datasets)
2. Google Dataset Search (https://datasetsearch.research.google.com/)

It's often required to sign up for a Kaggle account to download data, but it's
usually free. Google Dataset Search has more datasets, but they're not always
as clean (or free) as the ones on Kaggle.

For your convenience, I've just gone ahead and downloaded a dataset from Kaggle
for us to use. It's a .json file describing 311 US universities, which I
grabbed from here: https://www.kaggle.com/datasets/theriley106/university-statistics
"""
)

# Load the data
data = pd.read_json("data/schoolInfo.json")

print(
    """
There's a lot of possible columns here:
"""
    + "\n- ".join(data.columns)
)

print(
    """
Can we predict if a school will be in the top 10% of schools in terms of
admissions rate based on all (or any) of these features? Notice that by asking
``is it in the top 10%'', we've turned this into a binary classification
problem: either a school is in the top 10% or it isn't. We can
use logistic regression to solve this problem. First,
let's add a new column to the dataframe that tells us if a school is in the
top 10% or not.
"""
)

data["top_10"] = data["acceptance-rate"] > data["acceptance-rate"].quantile(0.9)

print(
    """
Now that we have our target variable, let's split the data into a training set
and a test set. We'll use the training set to train the model, and the test set
to evaluate its performance. There's also a lot of data in here that isn't
going to be useful to our model (like zip codes and cities). We can make a new
dataframe that only has a subset of the columns, for example:
"""
)

X = data[
    [
        "zip", "hs-gpa-avg", "enrollment"
    ]
]
print(X.columns)

print(
    """
Of course, for this particular problem, we might want to include more
relevant information than the zip code of the school... But determining
the most relevant features is a big part of the data science process,
and you'll have to use your own judgement to decide which features to
include in your model.
"""
)

y = data["top_10"]

print(X.head())

print(
    """
Uh-oh! It looks like there are some missing values in the data. We'll need to
fill these in before we can use it to train a model. We'll use the median value
for each column to fill in the missing values.
"""
)

X = X.fillna(X.median())
print(X.head())

print(
    """
And now: give it a try! Use the code from the previous example to train a
logistic regression model on the data. Then, use the model to make
predictions on the test set, and calculate the accuracy of the model.
"""
)
