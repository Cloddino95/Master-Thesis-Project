"""here I am using the ratings from the 1B dataset to be forecasted by the 300-vector representation of RANDOM words"""

from gensim.models import KeyedVectors
from Dataset import risk_ratings_1B
import numpy as np
import gensim
# import gensim.downloader as api
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm
from risk_rating_2_Bhatia import tot_results_df_2_Bhatia, metrics_Bha
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import random

model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_B = risk_ratings_1B.iloc[0].values.tolist()
risk_source_name_B = [word.replace(' ', '_') for word in risk_source_name_B]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_B = [word for word in risk_source_name_B if word in vocab]
omitted_word_B = [word for word in risk_source_name_B if word not in valid_risk_source_name_B]
# noinspection PyUnresolvedReferences
risk_source_vector_B = [model[word] for word in valid_risk_source_name_B]
dict_risk_source_vectors_B = dict(zip(valid_risk_source_name_B, risk_source_vector_B))
# ---------------------------------------------------------------RATINGS------------------------------------------------
risk_ratings_1B_small = risk_ratings_1B.copy()
risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])
risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)
risk_ratings_1B_small = risk_ratings_1B_small.T
risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')
risk_ratings_1B_small = risk_ratings_1B_small.astype(float)

mean_rows = risk_ratings_1B_small.mean(axis=1)
# ---------------------------------------------------------------END----------------------------------------------------
# ----------------------------------------------------------RANDOM WORDS------------------------------------------------
# Get the list of words in the model
word_list = list(model.index_to_key)
random.seed(42)
# Create a vector of 114 random words
random_words = np.random.choice(word_list, size=114, replace=False)
# Get the vector representation of the random words
random_words_vectors = [model[word] for word in random_words]
# Create a dictionary with the random words and their vector representation
dict_random_words_vectors = dict(zip(random_words, random_words_vectors))

def get_vector_for_source_i(a):
    source_name = random_words[a]
    return dict_random_words_vectors[source_name]


rand_vectors = []
for i in range(len(random_words)):
    xi = get_vector_for_source_i(i)
    rand_vectors.append(xi)

mat_rand = np.array(rand_vectors)
df_mat_rand = mat_rand.copy()
df_mat_rand = pd.DataFrame(df_mat_rand, index=random_words)

mean_rows = mean_rows.values
df_mat_rand.insert(0, "mean_ratings", mean_rows)
# ---------------------------------------------------------------END----------------------------------------------------

X = df_mat_rand.drop('mean_ratings', axis=1)
y = df_mat_rand['mean_ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameter_values = {'SVR-RBF': [100],
                    'Ridge Regression': [10]}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Ridge Regression': Ridge()}


Ridge_rand = Ridge().set_params(alpha=10)
Ridge_rand.fit(X_train, y_train)
Ridge_pred_test_rand = Ridge_rand.predict(X_test)
Ridge_pred_train_rand = Ridge_rand.predict(X_train)

SVR_RBF_rand = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_rand.fit(X_train, y_train)
SVR_RBF_pred_test_rand = SVR_RBF_rand.predict(X_test)
SVR_RBF_pred_train_rand = SVR_RBF_rand.predict(X_train)


data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_test_rand.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_test_rand.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_test_rand.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_test_rand.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_rand = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)


# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_test_rand)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_train_rand)
SVR_RBF_r2_test2 = SVR_RBF_rand.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF_rand.score(X_train, y_train)

# Metrics for Ridge Regression

Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_test_rand)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_train_rand)
Ridge_r2_test2 = Ridge_rand.score(X_test, y_test)
Ridge_r2_train2 = Ridge_rand.score(X_train, y_train)

results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_rand = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_rand.index = ['Ridge', 'SVR-RBF']

diff_bh_rand = metrics_Bha - metrics_rand
diff_bh_rand.reset_index(inplace=True)
diff_bh_rand.rename(columns={'index': 'Model'}, inplace=True)

# ------------------------------------------------------ CV Analysis ---------------------------------------------------
"""Below is performed the usual cv analysis, (but it is not necessary since the best model is already known)
however not only is a bit useless but also it retrieves lower results due to the many splits. Furthermore, it is better if
i do the same procedure as per the psy script (psy_Ridge_SVR_Metric&PREDICTION) so i can compare the results"""

parameter_values2 = {'SVR-RBF': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                     'Ridge Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

models2 = {'SVR-RBF': SVR(kernel='rbf'), 'Ridge Regression': Ridge()}

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

results_list_2 = []

for model_name, model in tqdm(models2.items()):
    if model_name == 'Ridge Regression':
        for k in parameter_values2[model_name]:
            model.set_params(alpha=k)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                                   'std_score_R2': np.std(r2_scores),
                                   'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})
    else:
        for C in tqdm(parameter_values2[model_name]):
            model.set_params(C=C)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores),
                 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})


results_df2_rand = pd.DataFrame(results_list_2, index=range(len(results_list_2)))

tot_results_df2_rand = results_df2_rand.drop('parameter', axis=1).groupby('model').mean()
difference_Bh_rand = tot_results_df_2_Bhatia - tot_results_df2_rand  # WHEN NEGATIVE THE NEW MODEL IS BETTER
difference_Bh_rand.reset_index(inplace=True)

# Save the results dataframe to a csv file
# results_df2_1Closest.to_csv('results_df2_1Closest.csv', index=False)
