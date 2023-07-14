import gensim
from gensim.models import KeyedVectors
from Dataset import risk_ratings_1A
from Dataset import risk_ratings_1B
from Dataset import risk_ratings_2
from Dataset import psychometric_1A
from Dataset import psychometric_1B
from Dataset import psychometric_2
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from tqdm import tqdm  # to show the progress bar
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression

# import gensim.downloader as api
# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.options.mode.chained_assignment = None  # suppress the warning line 171

model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

# ------------------------------------------------ study 1A ----------------------------------------
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
for i in range(len(valid_risk_source_name_A)):  # Replace with the actual number of risk sources I have
    xi = get_vector_for_source_i(i)  # Replace with your own function to get the vector for source i
    xi_vectors_A.append(xi)

mat_xi_300dim_A = np.array(xi_vectors_A)
data_300dim_A = mat_xi_300dim_A.copy()
data_300dim_A = pd.DataFrame(data_300dim_A, index=valid_risk_source_name_A)

"""now we work on the ratings"""
risk_ratings_1A_small = risk_ratings_1A.copy()
risk_ratings_1A_small.columns = risk_source_name_A
risk_ratings_1A_small = risk_ratings_1A_small.drop(risk_ratings_1A_small.index[0])
risk_ratings_1A_small = risk_ratings_1A_small.drop(omitted_word_A, axis=1)
risk_ratings_1A_small = risk_ratings_1A_small.T
risk_ratings_1A_small = risk_ratings_1A_small.dropna(axis=1, how='all')
risk_ratings_1A_small = risk_ratings_1A_small.astype(float)
# calculate the mean of each row (risk source) and add it to the data_300dim dataframe:
mean_rowsA = risk_ratings_1A_small.mean(axis=1)
data_300dim_A.insert(0, "mean_ratings", mean_rowsA)

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

results_df_A = pd.DataFrame(results_list_A, index=range(len(results_list_A)))
# results_df_A.to_csv('results_df_1A.csv', index=False)
# ------------------------------------------------ study 1B ----------------------------------------
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
"""now we work on the ratings"""
risk_ratings_1B_small = risk_ratings_1B.copy()
risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])
risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)
risk_ratings_1B_small = risk_ratings_1B_small.T
risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')
risk_ratings_1B_small = risk_ratings_1B_small.astype(float)

mean_rowsB = risk_ratings_1B_small.mean(axis=1)
data_300dimB.insert(0, "mean_ratings", mean_rowsB)

Xb = data_300dimB.drop('mean_ratings', axis=1)
yb = data_300dimB['mean_ratings']

results_list_B = []

for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=Xb, y=yb, cv=kf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=Xb, y=yb, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_B.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation for R2
            r2_scores = cross_val_score(model, X=Xb, y=yb, cv=kf, scoring='r2')
            # Perform cross-validation for MSE
            mse_scores = cross_val_score(model, X=Xb, y=yb, cv=kf, scoring=mse_scorer)
            mse_scores = -mse_scores  # Negate the scores to get actual MSE values
            # Store the results
            results_list_B.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores), 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)),
                 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

results_df_B = pd.DataFrame(results_list_B, index=range(len(results_list_B)))
# results_df_B.to_csv('results_df_1B.csv', index=False)
# ------------------------------------------------ study 2 ----------------------------------------

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
"""now we work on the ratings"""
risk_ratings_2_small = risk_ratings_2.copy()
risk_ratings_2_small.columns = risk_source_name_2
risk_ratings_2_small = risk_ratings_2_small.drop(risk_ratings_2_small.index[0])
risk_ratings_2_small = risk_ratings_2_small.drop(omitted_word_2, axis=1)
risk_ratings_2_small = risk_ratings_2_small.T
risk_ratings_2_small = risk_ratings_2_small.dropna(axis=1, how='all')
risk_ratings_2_small = risk_ratings_2_small.astype(float)

mean_rows2 = risk_ratings_2_small.mean(axis=1)
data_300dim_2.insert(0, "mean_ratings", mean_rows2)

X2 = data_300dim_2.drop('mean_ratings', axis=1)
y2 = data_300dim_2['mean_ratings']

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

results_df_2 = pd.DataFrame(results_list_2, index=range(len(results_list_2)))
# results_df_2.to_csv('results_df_2.csv', index=False)
# ------------------------------------------------ psychometric 1A ----------------------------------------

psychometric_1A = psychometric_1A.copy()

"""all the risk sources names in psychometric_1A matches with the risk sources names in risk_ratings_1A"""

# create a new vector of names which repeat each name 9 times in order to match the number of columns in psychometric_1A
risk_source_name_9dimA = [word for word in risk_source_name_A for i in range(9)]
# replace the first row of psychometric_1A with the new vector of names
psychometric_1A.iloc[0] = risk_source_name_9dimA
psychometric_1A_small = psychometric_1A.copy()
# name the columns with the risk source names and remove the first row (that are the r.source names):
psychometric_1A_small.columns = risk_source_name_9dimA
psychometric_1A_small = psychometric_1A_small.drop(psychometric_1A_small.index[0])
# remove risk sources not included in word2vec model's vocabulary:
psychometric_1A_small = psychometric_1A_small.drop(omitted_word_A, axis=1)
# remove NA
psychometric_1A_small = psychometric_1A_small.dropna(axis=0, how='all')
psychometric_1A_small = psychometric_1A_small.astype(float)
# do the mean of each column of psychometric_1A_small
mean_psyA = psychometric_1A_small.mean(axis=0)
# create a new dataframe with the mean of each row of psychometric_1A_small
new_dfA = pd.DataFrame(mean_psyA)
new_dfA = new_dfA.T
# reshape the df in a way that each row is a risk source and each column is a dimension
new_shapeA = (new_dfA.shape[1] // 9, 9)

psy_dfA = pd.DataFrame(new_dfA.values.reshape(new_shapeA), columns=["1 voluntariness", "2 immediacy of death",
                                                                    "3 known or unknown risk", "4 knowledge to science",
                                                                    "5 controllability", "6 novelty",
                                                                    "7 catastrophic potential of the risk",
                                                                    "8 the amount of dread",
                                                                    "9 potential for fatal consequences"])
psy_dfA.index = valid_risk_source_name_A

psy_dfA.insert(0, "mean_ratings", mean_rowsA)

XpsyA = psy_dfA.drop('mean_ratings', axis=1)
ypsyA = psy_dfA['mean_ratings']

psyco_1A_X = XpsyA.describe()
mean_psyco_1A_X = psyco_1A_X.mean(axis=1)

train_sizeA = int(0.8 * len(XpsyA))
X_trainA, X_testA, y_trainA, y_testA = XpsyA[:train_sizeA], XpsyA[train_sizeA:], ypsyA[:train_sizeA], ypsyA[
                                                                                                      train_sizeA:]

model_psyA = LinearRegression()
model_psyA.fit(X_trainA, y_trainA)

y_pred_testA = model_psyA.predict(X_testA)
y_pred_trainA = model_psyA.predict(X_trainA)

mse_testA = mean_squared_error(y_testA, y_pred_testA)
mse_trainA = mean_squared_error(y_trainA, y_pred_trainA)

r2_testA = model_psyA.score(X_testA, y_testA)
r2_trainA = model_psyA.score(X_trainA, y_trainA)

results_listA = {'(PsyA) R2 test': r2_testA, '(PsyA) R2 training': r2_trainA, '(PsyA) MSE test': mse_testA,
                 '(PsyA) MSE training': mse_trainA}
results_df_psyA = pd.DataFrame(results_listA, index=[0])
results_df_psyA.index = ['results A']
# results_df_psyA.to_csv('results_df_psy1A.csv', index=False)
# ------------------------------------------------ psychometric 1B ----------------------------------------

psychometric_1B = psychometric_1B.copy()

risk_source_name_9dimB = [word for word in risk_source_name_B for i in range(9)]

psychometric_1B.iloc[0] = risk_source_name_9dimB

psychometric_1B_small = psychometric_1B.copy()

psychometric_1B_small.columns = risk_source_name_9dimB
psychometric_1B_small = psychometric_1B_small.drop(psychometric_1B_small.index[0])
psychometric_1B_small = psychometric_1B_small.drop(omitted_word_B, axis=1)
psychometric_1B_small = psychometric_1B_small.astype(float)
psychometric_1B_small = psychometric_1B_small.dropna(axis=0, how='all')

mean_psyB = psychometric_1B_small.mean(axis=0)

new_dfB = pd.DataFrame(mean_psyB)
new_dfB = new_dfB.T
new_shapeB = (new_dfB.shape[1] // 9, 9)

psy_df_B = pd.DataFrame(new_dfB.values.reshape(new_shapeB), columns=["1 voluntariness", "2 immediacy of death",
                                                                     "3 known or unknown risk",
                                                                     "4 knowledge to science",
                                                                     "5 controllability", "6 novelty",
                                                                     "7 catastrophic potential of the risk",
                                                                     "8 the amount of dread",
                                                                     "9 potential for fatal consequences"])
psy_df_B.index = valid_risk_source_name_B

psy_df_B.insert(0, "mean_ratings", mean_rowsB)

# run a simple linear regression to predict the perceived riskiness of the risk sources, yi, from the corresponding xi.
XpsyB = psy_df_B.drop('mean_ratings', axis=1)
ypsyB = psy_df_B['mean_ratings']

psyco_1B_X = XpsyB.describe()
mean_psyco_1B_X = psyco_1B_X.mean(axis=1)

train_sizeB = int(0.8 * len(XpsyB))
X_trainB, X_testB, y_trainB, y_testB = XpsyB[:train_sizeB], XpsyB[train_sizeB:], ypsyB[:train_sizeB], ypsyB[
                                                                                                      train_sizeB:]

model_psyB = LinearRegression()
model_psyB.fit(X_trainB, y_trainB)

y_pred_testB = model_psyB.predict(X_testB)
y_pred_trainB = model_psyB.predict(X_trainB)

mse_testB = mean_squared_error(y_testB, y_pred_testB)
mse_trainB = mean_squared_error(y_trainB, y_pred_trainB)

r2_testB = model_psyB.score(X_testB, y_testB)
r2_trainB = model_psyB.score(X_trainB, y_trainB)

results_listB = {'(PsyB) R2 test': r2_testB, '(PsyB) R2 training': r2_trainB, '(PsyB) MSE test': mse_testB,
                 '(PsyB) MSE training': mse_trainB}
results_df_psyB = pd.DataFrame(results_listB, index=[0])
results_df_psyB.index = ['results B']
# results_df_psyB.to_csv('results_df_psy1B.csv', index=False)
# ------------------------------------------------ psychometric 2 ----------------------------------------

psychometric_2 = psychometric_2.copy()

risk_source_name_9dim2 = [word for word in risk_source_name_2 for i in range(9)]
psychometric_2.iloc[0] = risk_source_name_9dim2

psychometric_2.columns = risk_source_name_9dim2
psychometric_2 = psychometric_2.drop(psychometric_2.index[0])
psychometric_2 = psychometric_2.astype(float)
psychometric_2 = psychometric_2.dropna(axis=0, how='all')

mean_psy2 = psychometric_2.mean(axis=0)

new_df2 = pd.DataFrame(mean_psy2)
new_df2 = new_df2.T

new_shape2 = (new_df2.shape[1] // 9, 9)

psy_df_2 = pd.DataFrame(new_df2.values.reshape(new_shape2), columns=["1 voluntariness", "2 immediacy of death",
                                                                     "3 known or unknown risk",
                                                                     "4 knowledge to science",
                                                                     "5 controllability", "6 novelty",
                                                                     "7 catastrophic potential of the risk",
                                                                     "8 the amount of dread",
                                                                     "9 potential for fatal consequences"])
psy_df_2.index = valid_risk_source_name_2

psy_df_2.insert(0, "mean_ratings", mean_rows2)

Xpsy2 = psy_df_2.drop('mean_ratings', axis=1)
ypsy2 = psy_df_2['mean_ratings']

psyco_2_X = Xpsy2.describe()
mean_psyco_2_X = psyco_2_X.mean(axis=1)

train_size2 = int(0.8 * len(Xpsy2))
X_train2, X_test2, y_train2, y_test2 = Xpsy2[:train_size2], Xpsy2[train_size2:], ypsy2[:train_size2], ypsy2[train_size2:]

model_psy2 = LinearRegression()
model_psy2.fit(X_train2, y_train2)

y_pred_test2 = model_psy2.predict(X_test2)
y_pred_train2 = model_psy2.predict(X_train2)

mse_test2 = mean_squared_error(y_test2, y_pred_test2)
mse_train2 = mean_squared_error(y_train2, y_pred_train2)

r2_test2 = model_psy2.score(X_test2, y_test2)
r2_train2 = model_psy2.score(X_train2, y_train2)

results_list2 = {'(Psy2) R2 test': r2_test2, '(Psy2) R2 training': r2_train2, '(Psy2) MSE test': mse_test2,
                 '(Psy2) MSE training': mse_train2}
results_df_psy2 = pd.DataFrame(results_list2, index=[0])
results_df_psy2.index = ['results 2']
# results_df_psy2.to_csv('results_df_psy2.csv', index=False)
# ------------------------------------------------ FINAL DATASET ----------------------------------------

results_df_A_mean = results_df_A.drop("parameter", axis=1)
results_df_A_mean = results_df_A_mean.groupby('model').mean().reset_index()
results_df_A_mean.insert(0, "Study", "Study 1A")

results_df_B_mean = results_df_B.drop("parameter", axis=1)
results_df_B_mean = results_df_B_mean.groupby('model').mean().reset_index()
results_df_B_mean.insert(0, "Study", "Study 1B")

results_df_2_mean = results_df_2.drop("parameter", axis=1)
results_df_2_mean = results_df_2_mean.groupby('model').mean().reset_index()
results_df_2_mean.insert(0, "Study", "Study 2")

final_dataset = pd.concat([results_df_A_mean, results_df_B_mean, results_df_2_mean])
# this table can be assembled better but it is here just for your reference and double check with the results
final_dataset_psy = pd.concat([results_df_psyA, results_df_psyB, results_df_psy2])

