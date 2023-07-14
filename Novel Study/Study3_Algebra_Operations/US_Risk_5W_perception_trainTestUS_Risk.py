from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer
from risk_rating_2_Bhatia import metrics_Bha  # this is Bhatia study2 with the homogenized procedure
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tqdm import tqdm

model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]
omitted_word_2 = [word for word in risk_source_name_2 if word not in valid_risk_source_name_2]

risk_source_vectors_2 = []
modify_word_vectors_2 = []
for word in valid_risk_source_name_2:
    risk_source_vector = model[word]
    risk_source_vectors_2.append(risk_source_vector)
    modify_word = model[word] + model['USA'] + model['Risk'] + model['American'] + model['Danger'] + model['Perception']
    modify_word_vectors_2.append(modify_word)

""" IF I REMOVE MOST OF THE WORDS I GET A BETTER PREDICTION !!! ONLY FOR SVR, RIDGE DOES NOT CHANGE!!! --> PROBABLY BECAUSE I CHANGE TOO MUCH THE SEMANTIC MEANING 
eu_word = model[word] - model['US'] - model['USA'] - model['America'] - model['American'] - model['United_States'] \
              + model['Europe'] + model['EU'] + model['European_Union'] + model['EUROPE'] + model['European'] \
              + model['European_Union']"""


mat_xi_300dim_2 = np.array(risk_source_vectors_2)
mat_modify_vec_2 = np.array(modify_word_vectors_2)

modify_pd_2 = pd.DataFrame(mat_modify_vec_2, index=valid_risk_source_name_2)

data_300dim_2 = mat_xi_300dim_2.copy()
data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

risk_ratings_2_copy = risk_ratings_2.copy()
risk_ratings_2_copy.columns = risk_source_name_2
risk_ratings_2_copy = risk_ratings_2_copy.drop(risk_ratings_2_copy.index[0])
risk_ratings_2_copy = risk_ratings_2_copy.T
risk_ratings_2_copy = risk_ratings_2_copy.dropna(axis=1, how='all')
risk_ratings_2_copy = risk_ratings_2_copy.astype(float)

mean_rows = risk_ratings_2_copy.mean(axis=1)

data_300dim_2.insert(0, "mean_ratings", mean_rows)
modify_pd_2.insert(0, "mean_ratings", mean_rows)

# X = data_300dim_2.drop('mean_ratings', axis=1)
# y = data_300dim_2['mean_ratings']
X = modify_pd_2.drop('mean_ratings', axis=1)
y = modify_pd_2['mean_ratings']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# prediction Ridge
Ridge_norm = Ridge().set_params(alpha=4)
Ridge_norm.fit(X_train, y_train)
Ridge_pred_testnorm = Ridge_norm.predict(X_test)
Ridge_pred_trainnorm = Ridge_norm.predict(X_train)

# prediction SVR-RBF
SVR_RBF_norm = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_norm.fit(X_train, y_train)
SVR_RBF_pred_testnorm = SVR_RBF_norm.predict(X_test)
SVR_RBF_pred_trainorm = SVR_RBF_norm.predict(X_train)

# Dataframe for predictions
data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_testnorm.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_testnorm.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_testnorm.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_testnorm.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_norm = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)

# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_testnorm)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_trainorm)
SVR_RBF_r2_test2 = SVR_RBF_norm.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF_norm.score(X_train, y_train)

# Metrics for Ridge
Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_testnorm)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_trainnorm)
Ridge_r2_test2 = Ridge_norm.score(X_test, y_test)
Ridge_r2_train2 = Ridge_norm.score(X_train, y_train)

# Dataframe for metrics
results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_norm = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_norm.index = ['Ridge', 'SVR-RBF']

diff_bh_norm = metrics_Bha - metrics_norm
diff_bh_norm.reset_index(inplace=True)
diff_bh_norm.rename(columns={'index': 'Model'}, inplace=True)
