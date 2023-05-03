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
from sklearn.model_selection import cross_val_score, RepeatedKFold
from tqdm import tqdm  # to show the progress bar
# import matplotlib.pyplot as plt
# import seaborn as sns


"""To create a dictionary source_vectors (point 3 below) where the keys are the source names  and the values are the 
corresponding Word2Vec vectors, you can follow these steps:"""

# 1) Load the Word2Vec model using Gensim:
# model = api.load('word2vec-google-news-300')

# Save the model to disk
# model.save('word2vec-google-news-300.bin')

# Load the model from disk
model = gensim.models.KeyedVectors.load('word2vec-google-news-300.bin')

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

"""scrivere main description di quello che fai sotto (prendi word)"""

X = data_300dim.drop('mean_ratings', axis=1)
y = data_300dim['mean_ratings']


# Define the values of the parameter C to evaluate (SVM, Lasso & Ridge) and K for KNN
C_values = [10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7)]
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

"""
Personalize the parameters of the models based on the 1000 iteration, the best parameters are:
SVM_RBF : [10,20,30,40,50,60,70,80,90,100]
SVM_Polynomial : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
SVM_Sigmoid : [10,20,30,40,50,60,70,80,90,100]
Lasso : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Ridge : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10
KNN : [1,2,3,4,5,6,7,8,9,10]
"""
# Define the models to evaluate
models = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'KNN Regression': KNeighborsRegressor()}

# Define the k-fold cross validation parameters
n_splits = 10
n_repeats = 1000
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in k_values:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)  # **{'n_neighbors': k}  --> to avoid this warning replace with the dic -.-
            # Perform cross-validation
            scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Store the results
            results_list.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})
            # results[(model_name, k)] = scores
    else:
        for C in tqdm(C_values):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(C=C)
            # Perform cross-validation
            scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Store the results
            results_list.append({'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})
            # results[(model_name, C)] = scores
"""
# Calculate the mean and standard deviation of the R^2 scores for each model and parameter value
for (model_name, param), scores in results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    results_df = pd.concat([results_df, pd.DataFrame({'model': model_name, 'parameter': param, 'mean_score_R2': mean_score, 'std_score_R2': std_score})], ignore_index=True)
"""
# Convert the list of results to a pandas DataFrame with the appropriate index
results_df = pd.DataFrame(results_list, index=range(len(results_list)))

# Sort the results by the mean R^2 score
# results_df = results_df.sort_values(by=['mean_score_R2'], ascending=False)

# Save the results dataframe to a csv file
# results_df.to_csv('results_df_1000_1A_AGG.csv', index=False)


"""
# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='parameter', y='mean_score_R2', hue='model', data=results_df, ax=ax)
ax.set_xscale('log')
ax.set_xlabel('Parameter value')
ax.set_ylabel('Mean R^2 score')
ax.set_title('Mean R^2 score vs. parameter value')
plt.show()"""
"""
# Plot the R^2 results of all the models on the same plot using bar plots, on the x axis the model's name and on the y axis the mean R^2 score
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='model', y='mean_score_R2', data=results_df, ax=ax)
ax.set_ylabel('Mean R^2 score')
ax.set_title('Mean R^2 score vs. model')
plt.show()"""


""" ***CODE TO RUN SVM ALONE***
# Define the values of C to evaluate
C_values = [10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7)]

# Define the three types of SVM models
models_SVM = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid')}

# Define the k-fold cross validation parameters
n_splits = 10
n_repeats = 1000
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Create an empty dictionary to store the results
results = {}
results_df = pd.DataFrame(columns=['model', 'C', 'mean_score_R2', 'std_score_R2'])

# Loop through each model and each value of C for SVM
for kernel, model in models_SVM.items():
    for C in C_values:
        # Set the value of C for the model
        model.set_params(C=C)
        # Perform cross-validation
        scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
        # Store the results
        results[(kernel, C)] = scores


# Calculate the mean and standard deviation of the R^2 scores for each model and value of C
for (kernel, C), scores in results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    results_df = results_df.append({'model': kernel, 'C': C, 'mean_score_R2': mean_score, 'std_score_R2': std_score},
                                   ignore_index=True)
    # print(f'{kernel}, C={C}: Mean R^2 = {mean_score:.4f}, Standard Deviation = {std_score:.4f}')

results_df.to_csv('results_df.csv', index=False)
"""




""" ***CODE TO RUN LASSO AND RIDGE ALONE***
# Define the values of C to evaluate
C_values = [10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7)]

# Define the regression models
models = {'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge()}
          
# Define the k-fold cross validation parameters
n_splits = 10
n_repeats = 1000
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Create an empty dictionary to store the results
results = {}
results_df = pd.DataFrame(columns=['model', 'C', 'mean_score_R2', 'std_score_R2'])

# Loop through each model and each value of C
for model_name, model in models.items():
    for C in C_values:
        # Set the value of alpha for the model
        model.set_params(alpha=C)
        # Perform cross-validation
        scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
        # Store the results
        results[(model_name, C)] = scores

# Calculate the mean and standard deviation of the R^2 scores for each model and value of C
for (model_name, C), scores in results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    results_df = results_df.append({'model': model_name, 'C': C, 'mean_score_R2': mean_score, 'std_score_R2': std_score},
                                   ignore_index=True)
"""