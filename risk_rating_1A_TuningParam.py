# Feature: Forecasting New Risk Sources and associated values
# Scenario: Predicting the Value of New Risk Sources
# Given 73 participants have rated 125 existing risk sources
# And each existing risk source has a value ranging from -100 (safe) to +100 (risky)
# When a new risk source is identified
# And it has not been previously rated by any of the 73 participants
# Then the value of the new risk source can be predicted using a machine learning model trained on the existing risk sources and their values
# And the predicted value can be used to determine the level of risk people associate with the new risk source.

import gensim
from gensim.models import KeyedVectors
# import gensim.downloader as api
from Dataset import risk_ratings_1A
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  # suppress the warning line 171
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from tqdm import tqdm  # to show the progress bar
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns


"""To create a dictionary source_vectors (point 3 below) where the keys are the source names  and the values are the 
corresponding Word2Vec vectors, you can follow these steps:"""

# 1) Load the Word2Vec model using Gensim:
# model = api.load('word2vec-google-news-300')

# Save the model to disk
# model.save('word2vec-google-news-300.bin')

# Load the model from disk
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

# 2) Create a list of source names and a corresponding list of Word2Vec vectors for each source:

# how to create a list made of the values in the first row of risk_ratings_1A
risk_source_name_A = risk_ratings_1A.iloc[0].values.tolist()

# replace the space between each 2-words risk source with an underscore, e.g. clothes drier -> clothes_drier
risk_source_name_A = [word.replace(' ', '_') for word in risk_source_name_A]

# Check which risk sources are in the model's vocabulary
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_A = [word for word in risk_source_name_A if word in vocab]

# Create a list of omitted words
omitted_word_A = [word for word in risk_source_name_A if word not in valid_risk_source_name_A]

# Create a list of Word2Vec vectors for each source
# noinspection PyUnresolvedReferences
risk_source_vector_A = [model[word] for word in valid_risk_source_name_A]

# 3) Create a dictionary source_vectors where the keys are the risk_source_name and the values are the corresponding
# Word2Vec vectors using the dict() function:

dict_risk_source_vectors_A = dict(zip(valid_risk_source_name_A, risk_source_vector_A))

"""So now we have a list of source names:''risk_source_name'', and a dictionary: dict_risk_source_vectors
Where the keys are the source names and the values are the corresponding Word2Vec vectors. 
Hence, I can now define the get_vector_for_source_i(I) function as follows:"""

def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_A[a]
    return dict_risk_source_vectors_A[source_name]


# Get the Word2Vec vector for each risk source i
xi_vectors_A = []
for i in range(len(valid_risk_source_name_A)):  # Replace with the actual number of risk sources I have
    xi = get_vector_for_source_i(i)  # Replace with your own function to get the vector for source i
    xi_vectors_A.append(xi)

"""The above function to retrieve the Word2Vec vector for a specific risk source is needed because it provides a way to easily 
obtain the vector for a specific risk source given its name. The dictionary created is a useful data structure to have, 
but it does not provide an easy way to retrieve a specific vector given a specific risk source name."""

# Convert the list of vectors to a numpy array
# the rows represent the risk sources and the columns represent the 300 dimensions from Word2Vec
# (113 risk sources, 300 dimensions) (12 risk sources omitted)
mat_xi_300dim = np.array(xi_vectors_A)  # TODO: TRANSPOSE THIS MATRIX (✅) i think i transpose the ratings df!

# create a copy of the above array to use it for the analysis:
data_300dim = mat_xi_300dim.copy()

# how to change rows name of data_300dim with the words in valid_risk_source_name_A:
data_300dim = pd.DataFrame(data_300dim, index=valid_risk_source_name_A)

"""now we work on the ratings, first clean the data and then calculate the mean rate of each r.source (yi)of each row for the SVM regression"""
# create a copy of risk_ratings_1A dataframe for the analysis:
risk_ratings_1A_small = risk_ratings_1A.copy()

# name the columns with the risk source names and remove the first row (that are the r.source names):
risk_ratings_1A_small.columns = risk_source_name_A
risk_ratings_1A_small = risk_ratings_1A_small.drop(risk_ratings_1A_small.index[0])

# remove risk sources not included in word2vec model's vocabulary:
risk_ratings_1A_small = risk_ratings_1A_small.drop(omitted_word_A, axis=1)

# transpose the risk_ratings_1A_small dataframe:
risk_ratings_1A_small = risk_ratings_1A_small.T

"""at the beginning i transformed all the NA in zeros, but it is wrong bc zero can be in the ratings from -100 to +100"""
# remove all the columns of risk_ratings_1A_small that are composed of only NA values:
risk_ratings_1A_small = risk_ratings_1A_small.dropna(axis=1, how='all')

# the values in risk_ratings_1A_small are not float but objects, so I need to convert them to float:
risk_ratings_1A_small = risk_ratings_1A_small.astype(float)

# calculate the mean of each row (risk source) and add it to the data_300dim dataframe:
mean_rows = risk_ratings_1A_small.mean(axis=1)
data_300dim.insert(0, "mean_ratings", mean_rows)

"""now that I have the ratings and the vectors, I can start the analysis. first I need to find the right K and C parameters for the models.
to do this, I will split the data in 80% for training and 20% for testing. I run all the models on the 80% with the initial parameters
and then I will choose the best parameters for each model. then I will run the models on the 20% with the tuned param. and compare the results with Bhatia's"""

X = data_300dim.drop('mean_ratings', axis=1)
y = data_300dim['mean_ratings']

"""# retain only 80% of the data for training and 20% for testing
X_80 = X.iloc[:90, :]  # len(X) = 113 so 80% = 90
y_80 = y.iloc[:90]
X_20 = X.iloc[90:, :]
y_20 = y.iloc[90:]
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the values of the parameter C to evaluate (SVM, Lasso & Ridge) and K for KNN
C_values = [10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Define the models to evaluate
models = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'KNN Regression': KNeighborsRegressor()}

# Define the k-fold cross validation parameters
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in k_values:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)  # **{'n_neighbors': k}  --> to avoid this warning replace with the dic -.-
            # Perform cross-validation
            scores = cross_val_score(model, X=X_train, y=y_train, cv=kf, scoring='r2')
            # Store the results
            results_list.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})
            # results[(model_name, k)] = scores
    else:
        for C in tqdm(C_values):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(C=C)
            # Perform cross-validation
            scores = cross_val_score(model, X=X_train, y=y_train, cv=kf, scoring='r2')
            # Store the results
            results_list.append({'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})
            # results[(model_name, C)] = scores

# Convert the list of results to a pandas DataFrame with the appropriate index
results_df_defaultPar = pd.DataFrame(results_list, index=range(len(results_list)))

# Save the results dataframe to a csv file
# results_df.to_csv('results_df_1000_1A_AGG.csv', index=False)

"""
Personalize the parameters of the models based on the CV on the training set, the best parameters are:
SVM_RBF best performance: between 10 and 100 (most of the remaining R2 are negative)
SVM_Polynomial best performance: between 100 and 1000
SVM_Sigmoid best performance: between 10 and 100
Lasso best performance: between 0.1 and 1
Ridge best performance: between 1 and 10
KNN best performance: between 1 and 10
"""

parameter_values = {
    'SVR-RBF': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-polynomial': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-sigmoid': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Lasso Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'Ridge Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'KNN Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Define the models to evaluate
models = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'KNN Regression': KNeighborsRegressor()}

# Define the k-fold cross validation parameters
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list_A = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation
            scores = cross_val_score(model, X=X_test, y=y_test, cv=kf, scoring='r2')
            # Store the results
            results_list_A.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation
            scores = cross_val_score(model, X=X_test, y=y_test, cv=kf, scoring='r2')
            # Store the results
            results_list_A.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})


# Convert the list of results to a pandas DataFrame with the appropriate index
results_df_A = pd.DataFrame(results_list_A, index=range(len(results_list_A)))

# --------------------------------END-----------------------------------------------------------------------------------

