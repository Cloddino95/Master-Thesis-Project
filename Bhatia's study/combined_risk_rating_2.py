# Feature: Predicting risk ratings for each risk source using Word2Vec Model and various Regression Techniques
# Scenario: Predicting Risk Sources' ratings through Word2Vec's vector representation for each risk source
# Given 73 participants have rated 125 existing risk sources
# And each existing risk source has a value ranging from -100 (safe) to +100 (risky)
# When the dataset is split in X and y to train and test the various Regression Techniques
# And it is chosen the appropriate parameters values K and C for the various Regression Techniques
# When looping through each model and each value of C or k for the model and performing cross-validation
# Then the value (rating) of each risk source can be predicted using a machine learning model trained on the existing risk sources and their values
# And the predicted value can be used to determine the level of risk people associate with the new risk source.

from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  #, RepeatedKFold
from tqdm import tqdm
from psychometric_2 import psy_df_2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer

model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

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
for i in range(len(valid_risk_source_name_2)):
    xi = get_vector_for_source_i(i)
    xi_vectors_2.append(xi)

mat_xi_300dim_2 = np.array(xi_vectors_2)
data_300dim_2 = mat_xi_300dim_2.copy()
data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

risk_ratings_2_small = risk_ratings_2.copy()
risk_ratings_2_small.columns = risk_source_name_2
risk_ratings_2_small = risk_ratings_2_small.drop(risk_ratings_2_small.index[0])
risk_ratings_2_small = risk_ratings_2_small.drop(omitted_word_2, axis=1)
risk_ratings_2_small = risk_ratings_2_small.T
risk_ratings_2_small = risk_ratings_2_small.dropna(axis=1, how='all')
risk_ratings_2_small = risk_ratings_2_small.astype(float)

mean_rows = risk_ratings_2_small.mean(axis=1)
data_300dim_2.insert(0, "mean_ratings", mean_rows)

# import the psychometric dataframe and drop the column 'mean_ratings'
psy_df1 = psy_df_2.copy()
psy_df1 = psy_df1.drop('mean_ratings', axis=1)

# concatenate the data_300dim dataframe with the psychometric dataframe in order to perform the analysis on 309 dimensions:
data_300dim_2 = pd.concat([data_300dim_2, psy_df1], axis=1)

X2 = data_300dim_2.drop('mean_ratings', axis=1)
y2 = data_300dim_2['mean_ratings']

parameter_values = {
    'SVR-RBF': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-polynomial': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-sigmoid': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Lasso Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'Ridge Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'KNN Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'KNN Regression': KNeighborsRegressor()}

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

results_list_2 = []

for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=X2, y=y2, cv=kf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=X2, y=y2, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_2.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=X2, y=y2, cv=kf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=X2, y=y2, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_2.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

results_df_2_combo = pd.DataFrame(results_list_2, index=range(len(results_list_2)))
max_min_results2comb = results_df_2_combo.groupby('model').agg({'mean_score_R2': ['max', 'min'], 'mean_score_RMSE': ['max', 'min']})
max_min_results2comb.to_excel('/Users/ClaudioProiettiMercuri_1/Downloads/MAXmin_df_2comb.xlsx')
# Save the results dataframe to a csv file
# results_df_2_combo.to_csv('results_df_2_COMBINED.csv', index=False)
