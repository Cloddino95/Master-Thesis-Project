# Feature: Forecasting New Risk Sources and associated values
# Scenario: Predicting the Value of New Risk Sources
# Given 79 participants have rated 125 existing risk sources
# And each existing risk source has a value ranging from -100 (safe) to +100 (risky)
# When a new risk source is identified
# And it has not been previously rated by any of the 73 participants
# Then the value of the new risk source can be predicted using a machine learning model trained on the existing risk sources and their values
# And the predicted value can be used to determine the level of risk people associate with the new risk source.

from gensim.models import KeyedVectors
from Dataset import risk_ratings_1B
import numpy as np
import gensim
# import gensim.downloader as api
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RepeatedKFold
from joblib import Parallel, delayed
import multiprocessing

model = gensim.models.KeyedVectors.load('word2vec-google-news-300.bin')

risk_source_name_B = risk_ratings_1B.iloc[0].values.tolist()
risk_source_name_B = [word.replace(' ', '_') for word in risk_source_name_B]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_B = [word for word in risk_source_name_B if word in vocab]

omitted_word_B = [word for word in risk_source_name_B if word not in valid_risk_source_name_B]

# noinspection PyUnresolvedReferences
risk_source_vector_B = [model[word] for word in valid_risk_source_name_B]

dict_risk_source_vectors_B = dict(zip(valid_risk_source_name_B, risk_source_vector_B))

def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_B[a]
    return dict_risk_source_vectors_B[source_name]


xi_vectors_B = []
for i in range(len(valid_risk_source_name_B)):
    xi = get_vector_for_source_i(i)
    xi_vectors_B.append(xi)

mat_xi_300dimB = np.array(xi_vectors_B)

data_300dimB = mat_xi_300dimB.copy()

data_300dimB = pd.DataFrame(data_300dimB, index=valid_risk_source_name_B)

risk_ratings_1B_small = risk_ratings_1B.copy()

risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])

risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)

risk_ratings_1B_small = risk_ratings_1B_small.T

risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')

risk_ratings_1B_small = risk_ratings_1B_small.astype(float)

mean_rows = risk_ratings_1B_small.mean(axis=1)
data_300dimB.insert(0, "mean_ratings", mean_rows)

X = data_300dimB.drop('mean_ratings', axis=1)
y = data_300dimB['mean_ratings']

# Define the values of the parameter C to evaluate (SVM, Lasso & Ridge) and K for KNN
parameter_values = {
    'SVR-RBF': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
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
n_repeats = 1000
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

results_list_B = []


# Function to perform cross-validation and store results
def process(model, X, y, parameter, rkf):
    # Set the value of parameter for the model
    if model.__class__.__name__ == 'KNeighborsRegressor':
        model.set_params(n_neighbors=parameter)
    elif model.__class__.__name__ in ['Lasso', 'Ridge']:
        model.set_params(alpha=parameter)
    else:
        model.set_params(C=parameter)

    # Perform cross-validation
    scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')

    # Return the results as a dictionary
    return {'model': model.__class__.__name__, 'parameter': parameter, 'mean_score_R2': scores.mean(), 'std_score_R2': scores.std()}


# Loop through each model and each value of C or k
for model_name, model in models.items():
    if model_name == 'KNN Regression':
        # Use parallel processing for KNN
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(process)(model, X, y, k, rkf) for k in parameter_values[model_name])
    else:
        # Use parallel processing for SVM, Lasso, and Ridge
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(process)(model, X, y, C, rkf) for C in parameter_values[model_name])

    results_list_B.extend(results)

# Join the results into a pandas dataframe
results_df_B = pd.DataFrame(results_list_B)
# print(results_df.index.duplicated().sum())

# Pivot the results to have each model as a column, and each parameter value as a row
pivoted_results_B = results_df_B.pivot(index='parameter', columns='model', values='mean_score_R2')

