from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import shap
from tqdm import tqdm

model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
This study aims to leverage machine learning techniques to predict risk ratings using 300-dimensional vectors that represent each risk source. The principal objective is to enhance the prediction accuracy by focusing on the most influential dimensions that contribute to the risk ratings.

The research methodology unfolds in several key stages:

The original dataset consists of a 300-dimensional representation of risk sources (stored in variable 'X') and their associated risk ratings (stored in variable 'y'). A Support Vector Regressor (SVR) model with a Radial Basis Function (RBF) kernel is trained using this data.

The study employs Shapley Additive Explanations (SHAP), a game theory approach, to understand the contribution of each feature to the prediction. SHAP values are computed for the original dataset, which provides a measure of the importance of each dimension in predicting the risk rating.

To reduce the dimensionality and focus on the most influential features, the top 50 features from both absolute and positive SHAP values are selected for each risk source. This results in the creation of two new datasets, one retaining the top features with the highest absolute SHAP values (new_X_abs) and the other with the top features having positive SHAP values (new_X_positive).

After processing the datasets to remove NaN values and resetting the column indices, they are split into training and testing datasets, maintaining the original proportion of 80% training and 20% testing data.

The SVR model is then retrained on these reduced datasets. Predictions are made and evaluated using the Root Mean Square Error (RMSE) and R-squared (R2) metrics for both training and test sets. This process is conducted separately for new_X_abs and new_X_positive datasets.

The primary goal of this study is to create a streamlined yet effective risk rating prediction model. By focusing on the most influential features rather than using the entire 300-dimensional data, we hope to achieve better prediction performance while reducing computational complexity. The process also offers a meaningful interpretation of which features significantly drive the predictions, providing valuable insights into risk assessment and management.
"""

SVR_RBF = SVR(kernel='rbf').set_params(C=100)
SVR_RBF.fit(X, y)
# Create an explainer
explainer = shap.KernelExplainer(SVR_RBF.predict, X)
# Compute SHAP values
shap_values = explainer.shap_values(X)

# Compute absolute and positive SHAP values
shap_values_abs = np.abs(shap_values)
shap_values_pos = np.where(shap_values < 0, 0, shap_values)  # retain only positive values

n_features = 50  # The number of top features to keep

# Initialize new DataFrames
new_X_positive = pd.DataFrame()
new_X_abs = pd.DataFrame()

# For each risk source, get the indices of top N shap_values
for i in range(len(shap_values_abs)):
    # Get the indices of the top N absolute shap values
    top_n_indices_abs = np.argsort(shap_values_abs[i])[-n_features:]
    # Get the indices of the top N positive shap values
    top_n_indices_pos = np.argsort(shap_values_pos[i])[-n_features:]
    # Select the top N features from X
    reduced_X_abs = X.iloc[i, top_n_indices_abs]
    reduced_X_pos = X.iloc[i, top_n_indices_pos]
    # Append the reduced row to the new DataFrames
    new_X_abs = new_X_abs.append(reduced_X_abs)
    new_X_positive = new_X_positive.append(reduced_X_pos)

# Reset index for the new DataFrames
new_X_abs.reset_index(drop=True, inplace=True)
new_X_positive.reset_index(drop=True, inplace=True)

# Convert DataFrame to list of lists
data = new_X_abs.values.tolist()
# Remove NaN values from each row
data_without_nan = [[x for x in row if pd.notna(x)] for row in data[:200]]
# Convert back to DataFrame
new_X_abs = pd.DataFrame(data_without_nan)
# Use the original column names for the first n columns
new_X_abs.columns = new_X_abs.columns[:new_X_abs.shape[1]]

data2 = new_X_positive.values.tolist()
data2_without_nan = [[x for x in row if pd.notna(x)] for row in data2[:200]]
new_X_positive = pd.DataFrame(data2_without_nan)
new_X_positive.columns = new_X_positive.columns[:new_X_positive.shape[1]]


# Repeat the process for new_X_abs
X_train_abs, X_test_abs, y_train_abs, y_test_abs = train_test_split(new_X_abs, y, test_size=0.2, random_state=42)

SVR_RBF_abs = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_abs.fit(X_train_abs, y_train_abs)
SVR_RBF_abs_pred_test = SVR_RBF_abs.predict(X_test_abs)
SVR_RBF_abs_pred_train = SVR_RBF_abs.predict(X_train_abs)

rmse_train_abs = sqrt(mean_squared_error(y_train_abs, SVR_RBF_abs_pred_train))
rmse_test_abs = sqrt(mean_squared_error(y_test_abs, SVR_RBF_abs_pred_test))
r2_train_abs = r2_score(y_train_abs, SVR_RBF_abs_pred_train)
r2_test_abs = r2_score(y_test_abs, SVR_RBF_abs_pred_test)

# Repeat the process for new_X_positive
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(new_X_positive, y, test_size=0.2, random_state=42)

SVR_RBF_pos = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_pos.fit(X_train_pos, y_train_pos)
SVR_RBF_pos_pred_test = SVR_RBF_pos.predict(X_test_pos)
SVR_RBF_pos_pred_train = SVR_RBF_pos.predict(X_train_pos)

rmse_train_pos = sqrt(mean_squared_error(y_train_pos, SVR_RBF_pos_pred_train))
rmse_test_pos = sqrt(mean_squared_error(y_test_pos, SVR_RBF_pos_pred_test))
r2_train_pos = r2_score(y_train_pos, SVR_RBF_pos_pred_train)
r2_test_pos = r2_score(y_test_pos, SVR_RBF_pos_pred_test)


Ridge_pos = Ridge().set_params(alpha=4)
Ridge_pos.fit(X_train_pos, y_train_pos)
Ridge_pos_pred_test = Ridge_pos.predict(X_test_pos)
Ridge_pos_pred_train = Ridge_pos.predict(X_train_pos)

Ridgermse_train_pos = sqrt(mean_squared_error(y_train_pos, Ridge_pos_pred_train))
Ridgermse_test_pos = sqrt(mean_squared_error(y_test_pos, Ridge_pos_pred_test))
Ridger2_train_pos = r2_score(y_train_pos, Ridge_pos_pred_train)
Ridger2_test_pos = r2_score(y_test_pos, Ridge_pos_pred_test)

Ridge_abs = Ridge().set_params(alpha=4)
Ridge_abs.fit(X_train_abs, y_train_abs)
Ridge_abs_pred_test = Ridge_abs.predict(X_test_abs)
Ridge_abs_pred_train = Ridge_abs.predict(X_train_abs)

Ridgermse_train_abs = sqrt(mean_squared_error(y_train_abs, Ridge_abs_pred_train))
Ridgermse_test_abs = sqrt(mean_squared_error(y_test_abs, Ridge_abs_pred_test))
Ridger2_train_abs = r2_score(y_train_abs, Ridge_abs_pred_train)
Ridger2_test_abs = r2_score(y_test_abs, Ridge_abs_pred_test)

results_list2_ridge = {'R2 abs test': Ridger2_test_abs, 'RMSE abs test': Ridgermse_test_abs, 'R2 pos test': Ridger2_test_pos, 'RMSE pos test': Ridgermse_test_pos}
results_list2_SVR = {'R2 abs test': r2_test_abs, 'RMSE abs test': rmse_test_abs, 'R2 pos test': r2_test_pos, 'RMSE pos test': rmse_test_pos}
metrics_Bha = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_Bha.index = ['Ridge', 'SVR-RBF']

"""
!!! i tried to us n =5 peggiora n = 100 bene o male lo stesso, n=150 peggiora di nuovo
"""