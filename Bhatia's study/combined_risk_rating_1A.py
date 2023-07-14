import gensim
from gensim.models import KeyedVectors
from Dataset import risk_ratings_1A
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from tqdm import tqdm  # to show the progress bar
from psychometric_1A import psy_df
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.options.mode.chained_assignment = None  # suppress the warning line 171
# import gensim.downloader as api


model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_A = risk_ratings_1A.iloc[0].values.tolist()
risk_source_name_A = [word.replace(' ', '_') for word in risk_source_name_A]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_A = [word for word in risk_source_name_A if word in vocab]
omitted_word_A = [word for word in risk_source_name_A if word not in valid_risk_source_name_A]

# noinspection PyUnresolvedReferences
risk_source_vector_A = [model[word] for word in valid_risk_source_name_A]
dict_risk_source_vectors_A = dict(zip(valid_risk_source_name_A, risk_source_vector_A))
def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_A[a]
    return dict_risk_source_vectors_A[source_name]


xi_vectors_A = []
for i in range(len(valid_risk_source_name_A)):
    xi = get_vector_for_source_i(i)
    xi_vectors_A.append(xi)

mat_xi_300dim_A = np.array(xi_vectors_A)
data_300dim_A = mat_xi_300dim_A.copy()
data_300dim_A = pd.DataFrame(data_300dim_A, index=valid_risk_source_name_A)
risk_ratings_1A_small = risk_ratings_1A.copy()
risk_ratings_1A_small.columns = risk_source_name_A
risk_ratings_1A_small = risk_ratings_1A_small.drop(risk_ratings_1A_small.index[0])
risk_ratings_1A_small = risk_ratings_1A_small.drop(omitted_word_A, axis=1)
risk_ratings_1A_small = risk_ratings_1A_small.T
risk_ratings_1A_small = risk_ratings_1A_small.dropna(axis=1, how='all')
risk_ratings_1A_small = risk_ratings_1A_small.astype(float)
mean_rows = risk_ratings_1A_small.mean(axis=1)
data_300dim_A.insert(0, "mean_ratings", mean_rows)

# import the psychometric dataframe and drop the column 'mean_ratings'
psy_df1 = psy_df.copy()
psy_df1 = psy_df1.drop('mean_ratings', axis=1)

# concatenate the data_300dim dataframe with the psychometric dataframe in order to perform the analysis on 309 dimensions:
data_300dim_A = pd.concat([data_300dim_A, psy_df1], axis=1)

Xa = data_300dim_A.drop('mean_ratings', axis=1)
ya = data_300dim_A['mean_ratings']

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

results_list_A = []

for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=Xa, y=ya, cv=kf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=Xa, y=ya, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_A.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=Xa, y=ya, cv=kf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=Xa, y=ya, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_A.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

results_df_A_combo = pd.DataFrame(results_list_A, index=range(len(results_list_A)))
max_min_resultsAcomb = results_df_A_combo.groupby('model').agg({'mean_score_R2': ['max', 'min'], 'mean_score_RMSE': ['max', 'min']})
max_min_resultsAcomb.to_excel('/Users/ClaudioProiettiMercuri_1/Downloads/MAXmin_df_Acomb.xlsx')
# Save the results dataframe to a csv file
# results_df_A_combo.to_csv('results_df_1A_COMBINED.csv', index=False)
