from sklearn.metrics import mean_squared_error, make_scorer
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
from tqdm import tqdm

model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Thesis_Clo/word2vec-google-news-300.bin')

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
# ---------------------------------------------------------------RATINGS------------------------------------------------
risk_ratings_1B_small = risk_ratings_1B.copy()

risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])

risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)

risk_ratings_1B_small = risk_ratings_1B_small.T

risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')

risk_ratings_1B_small = risk_ratings_1B_small.astype(float)

mean_rows = risk_ratings_1B_small.mean(axis=1)
# ---------------------------------------------------------------END------------------------------------------------
# Save the results dataframe to a csv file
# results_df_B.to_csv('results_df_1000_customized_1B_AGG.csv', index=False)

data_300dimB.insert(0, "mean_ratings", mean_rows)

X = data_300dimB.drop('mean_ratings', axis=1)
y = data_300dimB['mean_ratings']

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
n_repeats = 15
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Create a custom scoring function for MSE
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list_B = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_B.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(mse_scores),
                 'std_score_RMSE': np.std(mse_scores)})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_B.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})



# Convert the list of results to a pandas DataFrame with the appropriate index
results_df_B = pd.DataFrame(results_list_B, index=range(len(results_list_B)))
