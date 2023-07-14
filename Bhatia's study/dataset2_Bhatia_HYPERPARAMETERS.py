from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from tqdm import tqdm

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

X = data_300dim_2.drop('mean_ratings', axis=1)
y = data_300dim_2['mean_ratings']

statistics_2_X = X.describe()
mean_statistics_2_X = statistics_2_X.mean(axis=1)
statistics_2_y = y.describe()

parameter_values = {
    'SVR-RBF': [10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7],
    'SVR-polynomial': [10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7,],
    'SVR-sigmoid': [10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7],
    'Lasso Regression': [10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7],
    'Ridge Regression': [10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7],
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
            model.set_params(n_neighbors=k)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores
            results_list_2.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                                   'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(mse_scores),
                                   'std_score_RMSE': np.std(mse_scores)})
    else:
        for C in tqdm(parameter_values[model_name]):
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores
            results_list_2.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

results_df_DAFUALTMETRICS = pd.DataFrame(results_list_2, index=range(len(results_list_2)))
max_min_results = results_df_DAFUALTMETRICS.groupby('model').agg({'mean_score_R2': ['max', 'min'], 'mean_score_RMSE': ['max', 'min']})
results_df_DAFUALTMETRICS.to_excel('/Users/ClaudioProiettiMercuri_1/Downloads/results_df_2DAFAULMETRICS.xlsx')
max_min_results.to_excel('/Users/ClaudioProiettiMercuri_1/Downloads/max_min_2DAFAULMETRICS.xlsx')
# Save the results dataframe to a csv file
# results_df_2.to_csv('results_df_1000_customized_2_AGG_idk_differencewiththeotherone.csv', index=False)

