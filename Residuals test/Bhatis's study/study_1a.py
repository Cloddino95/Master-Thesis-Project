from gensim.models import KeyedVectors
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
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from Dataset import risk_ratings_1A

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

X = data_300dim_A.drop('mean_ratings', axis=1)
y = data_300dim_A['mean_ratings']

parameter_values = {
    'SVR-RBF': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

residuals_df = pd.DataFrame()

for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            model.set_params(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            shapiro_test = stats.shapiro(residuals)
            temp_df = pd.DataFrame({
                'model': [model_name],
                'parameter': [k],
                'shapiro_statistic': [shapiro_test.statistic],
                'shapiro_p_value': [shapiro_test.pvalue]})
            residuals_df = pd.concat([residuals_df, temp_df], ignore_index=True)
    else:
        for C in tqdm(parameter_values[model_name]):
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            shapiro_test = stats.shapiro(residuals)
            temp_df = pd.DataFrame({
                'model': [model_name],
                'parameter': [C],
                'shapiro_statistic': [shapiro_test.statistic],
                'shapiro_p_value': [shapiro_test.pvalue]})
            residuals_df = pd.concat([residuals_df, temp_df], ignore_index=True)

# Plot residuals

specified_parameters = {
    'SVR-RBF': 100,
    'SVR-polynomial': 300,
    'SVR-sigmoid': 40,
    'Lasso Regression': 0.25,
    'Ridge Regression': 4,
    'KNN Regression': 3
}

for model_name, model in tqdm(models.items()):
    model_param = specified_parameters[model_name]

    if model_name == 'KNN Regression':
        model.set_params(n_neighbors=model_param)

    elif model_name in ['Lasso Regression', 'Ridge Regression']:
        model.set_params(alpha=model_param)

    else:
        model.set_params(C=model_param)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='-')  # this line represents no residuals
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'(Study 1A) Residuals vs Predicted Values for {model_name} with Parameter={model_param}')
    plt.show()
    plt.savefig(f'(Study 1A) Residuals vs Predicted Values for {model_name} with Parameter={model_param}.png')

    #plt.hist(residuals, bins=30)
    #plt.xlabel('Residual')
    #plt.title(f'(Study 1A) Histogram of residuals for {model_name} with Parameter={model_param}')
    #plt.show()

