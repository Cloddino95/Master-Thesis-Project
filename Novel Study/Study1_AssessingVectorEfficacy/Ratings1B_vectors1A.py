"""here I am using the ratings from the 1B dataset that will be forecasted by
the 300-vector representation of the words in 1A dataset"""

from gensim.models import KeyedVectors
from Dataset import risk_ratings_1B, risk_ratings_1A
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

# this initial part is needed only to get the words in "risk_source_name_B" for the rating but the rest is not used (the 300-vec is not used)
model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_B = risk_ratings_1B.iloc[0].values.tolist()
risk_source_name_B = [word.replace(' ', '_') for word in risk_source_name_B]
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_B = [word for word in risk_source_name_B if word in vocab]
omitted_word_B = [word for word in risk_source_name_B if word not in valid_risk_source_name_B]
# ---------------------------------------------------------------RATINGS------------------------------------------------
# so in this part I get the ratings from the 1B dataset and I add them to the 300dim vector representation of the words in 1A
risk_ratings_1B_small = risk_ratings_1B.copy()
risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])
risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)
risk_ratings_1B_small = risk_ratings_1B_small.T
risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')
risk_ratings_1B_small = risk_ratings_1B_small.astype(float)

mean_rows = risk_ratings_1B_small.mean(axis=1)
# ---------------------------------------------------------------END------------------------------------------------
# ---------------------------------------------------------------1A vector source---------------------------------------
risk_source_name_A = risk_ratings_1A.iloc[0].values.tolist()
risk_source_name_A = [word.replace(' ', '_') for word in risk_source_name_A]
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_A = [word for word in risk_source_name_A if word in vocab]
omitted_word_A = [word for word in risk_source_name_A if word not in valid_risk_source_name_A]
risk_source_vector_A = [model[word] for word in valid_risk_source_name_A]
dict_risk_source_vectors_A = dict(zip(valid_risk_source_name_A, risk_source_vector_A))

def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_A[a]
    return dict_risk_source_vectors_A[source_name]


xi_vectors_A = []
for i in range(len(valid_risk_source_name_A)):  # Replace with the actual number of risk sources I have
    xi = get_vector_for_source_i(i)  # Replace with your own function to get the vector for source i
    xi_vectors_A.append(xi)

mat_xi_300dim_A = np.array(xi_vectors_A)
data_300dim_A = mat_xi_300dim_A.copy()
data_300dim_A = pd.DataFrame(data_300dim_A, index=valid_risk_source_name_A)
# ---------------------------------------------------------------END------------------------------------------------
# removing observation since dataset A is 113 and dataset B is 114
meanratingB = mean_rows
meanratingB = meanratingB.drop(meanratingB.index[-1])
meanratingB = meanratingB.values
data_300dim_A.insert(0, "mean_ratings_B", meanratingB.astype(float))

X = data_300dim_A.drop('mean_ratings_B', axis=1)
y = data_300dim_A['mean_ratings_B']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameter_values = {'SVR-RBF': [100],
                    'Ridge Regression': [10]}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Ridge Regression': Ridge()}


Ridge_1BA = Ridge().set_params(alpha=10)
Ridge_1BA.fit(X_train, y_train)
Ridge_pred_test_1BA = Ridge_1BA.predict(X_test)
Ridge_pred_train_1BA = Ridge_1BA.predict(X_train)

SVR_RBF_1BA = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_1BA.fit(X_train, y_train)
SVR_RBF_pred_test1BA = SVR_RBF_1BA.predict(X_test)
SVR_RBF_pred_trai1BA = SVR_RBF_1BA.predict(X_train)


data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_test1BA.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_test_1BA.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_test1BA.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_test_1BA.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_1BA = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)


# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_test1BA)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_trai1BA)
SVR_RBF_r2_test2 = SVR_RBF_1BA.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF_1BA.score(X_train, y_train)

# Metrics for Ridge Regression

Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_test_1BA)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_train_1BA)
Ridge_r2_test2 = Ridge_1BA.score(X_test, y_test)
Ridge_r2_train2 = Ridge_1BA.score(X_train, y_train)

results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_1BA = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_1BA.index = ['Ridge', 'SVR-RBF']

diff_bh_1BA = metrics_Bha - metrics_1BA
diff_bh_1BA.reset_index(inplace=True)
diff_bh_1BA.rename(columns={'index': 'Model'}, inplace=True)

# ------------------------------------------------------ CV Analysis ---------------------------------------------------
"""Below is performed the usual cv analysis, (but it is not necessary since the best model is already known)
however not only is a bit useless but also it retrieves lower results due to the many splits. Furthermore, it is better if
i do the same procedure as per the psy script (psy_Ridge_SVR_Metric&PREDICTION) so i can compare the results"""

parameter_values2 = {'SVR-RBF': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                     'Ridge Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

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


results_df2_1BA = pd.DataFrame(results_list_2, index=range(len(results_list_2)))

tot_results_df2_1BA = results_df2_1BA.drop('parameter', axis=1).groupby('model').mean()
difference_Bh_1BA = tot_results_df_2_Bhatia - tot_results_df2_1BA  # WHEN NEGATIVE THE NEW MODEL IS BETTER
difference_Bh_1BA.reset_index(inplace=True)

# Save the results dataframe to a csv file
# results_df2_1Closest.to_csv('results_df2_1Closest.csv', index=False)
