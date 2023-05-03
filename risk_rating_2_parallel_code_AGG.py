# Feature: Forecasting new risk sources and associated values
# Scenario: Forecasting new risk sources and associated values for a large dataset
# Given there are 300 participants and 200 different risk sources
# And each participant was given 100 randomly selected risk sources
# And each risk source has an average of 150 ratings valued from -100 (safe) to +100 (risky)
# when I want to forecast a new risk source and associated value
# Then the value of the new risk source can be predicted using a machine learning model trained on the existing risk sources and their values
# And the predicted value can be used to determine the level of risk people associate with the new risk source.

from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RepeatedKFold
from joblib import Parallel, delayed
import multiprocessing

model = gensim.models.KeyedVectors.load('word2vec-google-news-300.bin')

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]

omitted_word_2 = [word for word in risk_source_name_2 if word not in valid_risk_source_name_2]

# noinspection PyUnresolvedReferences
risk_source_vector_2 = [model[word] for word in valid_risk_source_name_2]

dict_risk_source_vectors_2 = dict(zip(valid_risk_source_name_2, risk_source_vector_2))


def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_2[a]
    return dict_risk_source_vectors_2[source_name]


xi_vectors_2 = []
for i in range(len(valid_risk_source_name_2)):  # Replace with the actual number of risk sources I have
    xi = get_vector_for_source_i(i)  # Replace with your own function to get the vector for source i
    xi_vectors_2.append(xi)

mat_xi_300dim_2 = np.array(xi_vectors_2)

data_300dim_2 = mat_xi_300dim_2.copy()

data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

risk_ratings_2_small = risk_ratings_2.copy()

risk_ratings_2_small.columns = risk_source_name_2
risk_ratings_2_small = risk_ratings_2_small.drop(risk_ratings_2_small.columns[0], axis=1)

risk_ratings_2_small = risk_ratings_2_small.drop(omitted_word_2, axis=1)

risk_ratings_2_small = risk_ratings_2_small.T

risk_ratings_2_small = risk_ratings_2_small.dropna(axis=1, how='all')

risk_ratings_2_small = risk_ratings_2_small.astype(float)

mean_rows = risk_ratings_2_small.mean(axis=1)
data_300dim_2.insert(0, "mean_ratings", mean_rows)

X = data_300dim_2.drop('mean_ratings', axis=1)
y = data_300dim_2['mean_ratings']

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

results_list_2 = []


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

    results_list_2.extend(results)

# Join the results into a pandas dataframe
results_df_2 = pd.DataFrame(results_list_2)
# print(results_df.index.duplicated().sum())

# Pivot the results to have each model as a column, and each parameter value as a row
pivoted_results_2 = results_df_2.pivot(index='parameter', columns='model', values='mean_score_R2')

