# Feature: Predicting risk ratings for each risk source using Word2Vec Model and various Regression Techniques
# Scenario: Predicting Risk Sources' ratings through Word2Vec's vector representation for each risk source
# Given 73 participants have rated 125 existing risk sources
# And each existing risk source has a value ranging from -100 (safe) to +100 (risky)
# When the dataset is split in X and y to train and test the various Regression Techniques
# And it is chosen the appropriate parameters values K and C for the various Regression Techniques
# When looping through each model and each value of C or k for the model and performing cross-validation
# Then the value (rating) of each risk source can be predicted using a machine learning model trained on the existing risk sources and their values
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
from sklearn.model_selection import KFold
from tqdm import tqdm  # to show the progress bar
from sklearn.metrics import mean_squared_error, make_scorer
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
mat_xi_300dim_A = np.array(xi_vectors_A)

# create a copy of the above array to use it for the analysis:
data_300dim_A = mat_xi_300dim_A.copy()

# how to change rows name of data_300dim with the words in valid_risk_source_name_A:
data_300dim_A = pd.DataFrame(data_300dim_A, index=valid_risk_source_name_A)

"""now we work on the ratings, first clean the data and then calculate the mean rate of each r.source (yi) of each row"""
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
data_300dim_A.insert(0, "mean_ratings", mean_rows)

"""scrivere main description di quello che fai sotto (prendi word)"""

X = data_300dim_A.drop('mean_ratings', axis=1)
y = data_300dim_A['mean_ratings']

# Define the values of the parameter C to evaluate (SVM, Lasso & Ridge) and K for KNN
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

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list_A = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            model.set_params(n_neighbors=k)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores
            results_list_A.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                                   'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(mse_scores),
                                   'std_score_RMSE': np.std(mse_scores)})
    else:
        for C in tqdm(parameter_values[model_name]):
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores
            results_list_A.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

# Convert the list of results to a pandas DataFrame with the appropriate index
results_df_A = pd.DataFrame(results_list_A, index=range(len(results_list_A)))
max_min_resultsA = results_df_A.groupby('model').agg({'mean_score_R2': ['max', 'min'], 'mean_score_RMSE': ['max', 'min']})
max_min_resultsA.to_excel('/Users/ClaudioProiettiMercuri_1/Downloads/MAXmin_df_A.xlsx')
# Sort the results by the mean R^2 score
# results_df = results_df.sort_values(by=['mean_score_R2'], ascending=False)

# Save the results dataframe to a csv file
# results_df.to_csv('results_df_1000_customized_1A_AGG.csv', index=False)

# ----------------------------------descriptive statistics--------------------------------------------
# Below is the code to obtain the summary statistics for study 1A and a final dataset that combines the summary
# statistics of all the studies (both psychometric and semantic)
statistics_1A_X = X.describe()
mean_statistics_1A_X = statistics_1A_X.mean(axis=1)
statistics_1A_y = y.describe()

from risk_rating_1B_customized_K_C_AGG import mean_statistics_1B_X, statistics_1B_y
from risk_rating_2_customized_K_C_AGG import mean_statistics_2_X, statistics_2_y
from psychometric_1A import mean_psyco_1A_X
from psychometric_1B import mean_psyco_1B_X
from psychometric_2 import mean_psyco_2_X

sum_statistics = pd.concat([mean_statistics_1A_X, mean_statistics_1B_X, mean_statistics_2_X, mean_psyco_1A_X, mean_psyco_1B_X,
                            mean_psyco_2_X, statistics_1A_y, statistics_1B_y, statistics_2_y], axis=1)
col_names = ['study 1A (X)', 'study 1B (X)', 'study 2 (X)', 'psycho 1A (X)', 'psycho 1B (X)', 'psycho 2 (X)', 'study 1A (y)',
             'study 1B (y)', 'study 2 (y)']
sum_statistics.columns = col_names
T_sum_statistics = sum_statistics.T
T_sum_statistics.drop(columns=['25%', '75%'], inplace=True)
T_sum_statistics = T_sum_statistics.rename(columns={'50%': 'median'})
# T_sum_statistics.to_csv('Summary_statistics_Study1A1B2psy.csv')
